import numpy as np
from scipy.spatial.transform import Rotation

def parse_pincam(path):
    """Parses wide-angle intrinsic files containing [w, h, fx, fy, cx, cy]."""
    with open(path, 'r') as f:
        w, h, fx, fy, cx, cy = (float(x) for x in f.read().split())
    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1]
    ])
    return int(w), int(h), K

def parse_traj(path):
    """Parses .traj files into a dict of 4x4 camera matrices (World -> Camera)."""
    poses = {}
    with open(path, 'r') as f:
        for line in f:
            v = [float(x) for x in line.split()]
            if len(v) != 7: continue
            T = np.eye(4)
            T[:3, :3] = Rotation.from_rotvec(v[1:4]).as_matrix()
            T[:3, 3]  = v[4:7]
            poses[f'{v[0]:.3f}'] = T
    return poses

def project_points(points, T_world_to_cam, K):
    """Projects NxD 3D points to 2D image coordinates."""
    pts_homo = np.c_[np.asarray(points, float), np.ones(len(points))]
    cam_coords = (T_world_to_cam @ pts_homo.T)[:3, :]
    img_coords = K @ cam_coords
    with np.errstate(divide='ignore', invalid='ignore'):
        z = img_coords[2, :]
        u = img_coords[0, :] / z
        v = img_coords[1, :] / z
    return u, v, z
