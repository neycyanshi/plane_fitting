import numpy as np
from PIL import Image
from ransac import *
import argparse
import os

focalLength = 365.668
centerX = 165.106
centerY = 120.599
scalingFactor = 1.0


def augment(xyzs):
    axyz = np.ones((len(xyzs), 4))
    axyz[:, :3] = xyzs
    return axyz


def estimate(xyzs):
    axyz = augment(xyzs[:3])
    return np.linalg.svd(axyz)[-1][-1, :]


def is_inlier(coeffs, xyz, threshold):
    return np.abs(coeffs.dot(augment([xyz]).T)) < threshold


def read_obj(obj_path):
    """
    return n_verts and 3d points mat(n_verts, 3)
    """
    pass


def depth2cloud(depth_path):
    """
    Generate a point cloud from a depth image.
    
    Input:
    depth_path -- filename of depth image

    Output: n_verts and 3d points mat(n_verts, 3)
    
    """
    depth = Image.open(depth_path)
    if depth.mode != "I":
        raise Exception("Depth image is not in intensity format")

    points = []
    for v in range(depth.size[1]):
        for u in range(depth.size[0]):
            Z = depth.getpixel((u, v)) / scalingFactor
            if Z == 0: continue
            X = (u - centerX) * Z / focalLength
            Y = (v - centerY) * Z / focalLength
            points.append(np.array([X, Y, Z], dtype=float))
    n_verts = len(points)
    xyzs = np.array(points)
    print("n_verts: {}".format(n_verts))
    return n_verts, xyzs


def save_cloud(xyzs, ply_path):
    lines = []
    for i in range(xyzs.shape[0]):
        lines.append("{:.6f} {:.6f} {:.6f}\n".format(*list(xyzs[i])))
    fout = open(ply_path, "w")
    fout.write('''ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
end_header
%s
''' % (len(lines), "".join(lines)))
    fout.close()
    print('cloud saved to {}, total {} vertexs.'.format(ply_path, len(lines)))


def calc_bias_jitter(xyzs, a, b, c, d):
    zs_plane = (-d - a * xyzs[:, 0] - b * xyzs[:, 1]) / c
    zs_diff = xyzs[:, 2] - zs_plane
    zs_mean = np.mean(zs_diff)
    zs_bias = np.mean(np.abs(zs_diff))
    zs_jitter = np.std(zs_diff)
    return zs_bias, zs_mean, zs_jitter


def save_plane(xyzs, a, b, c, d, ply_path):
    zs_plane = (-d - a * xyzs[:, 0:1] - b * xyzs[:, 1:2]) / c
    xyzs_plane = np.hstack((xyzs[:, :2], zs_plane))
    save_cloud(xyzs_plane, ply_path)


def rm_outliers(xyzs, coeffs, inlier_th):
    xyzs=list(xyzs)
    for i in reversed(range(len(xyzs))):
        if not is_inlier(coeffs, xyzs[i], inlier_th):
            del xyzs[i]
    return np.array(xyzs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('depth', type=str, help='input depth map path')
    parser.add_argument('--max_iter', type=int, help='max ransac iterations', default=100)
    parser.add_argument('--in_ratio', type=float, help='goal inlier ratio', default=0.95)
    parser.add_argument('--inlier_th', type=float, default=0.01, help='inlier threshold')
    args = parser.parse_args()

    # 3d obj format data
    n, xyzs = depth2cloud(args.depth)
    goal_inliers = n * args.in_ratio

    # RANSAC
    m, best_inliers = run_ransac(xyzs, estimate,
                                 lambda x, y: is_inlier(x, y, args.inlier_th), 3,
                                 goal_inliers, args.max_iter)
    a, b, c, d = m

    # remove outliers in origin point cloud
    xyzs_in = rm_outliers(xyzs, m, args.inlier_th)

    # save original and fitted plane point cloud
    depth_id = os.path.basename(args.depth).split('_d.png')[0]
    depth_dir = os.path.dirname(args.depth)
    save_cloud(xyzs, os.path.join(depth_dir, '{}.ply'.format(depth_id)))
    save_cloud(xyzs_in, os.path.join(depth_dir, '{}_in.ply'.format(depth_id)))
    save_plane(xyzs, a, b, c, d, os.path.join(depth_dir, '{}_plane.ply'.format(depth_id)))

    """
    # fit again without outliers
    m, best_inliers = run_ransac(xyzs_in, estimate,
                                 lambda x, y: is_inlier(x, y, args.inlier_th), 3,
                                 goal_inliers, args.max_iter)
    a, b, c, d = m
    xyzs_in = rm_outliers(xyzs, m, args.inlier_th)
    """


    # compute metrics
    z_bias, z_mean, z_jitter = calc_bias_jitter(xyzs_in, a, b, c, d)
    print("depth: {:.2f} | bias={}, mean={}, jitter={}".format(np.mean(xyzs_in[:,2]), z_bias, z_mean, z_jitter))
