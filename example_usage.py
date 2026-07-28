import os
import cv2
import json
import numpy as np
from plyfile import PlyData
from projection_utils import parse_traj, project_points
from extraction_pipeline import build_view, MIN_DEPTH

# update these paths to point to your local SceneFun3D dataset directory
VISIT_ID = '420673'
VIDEO_ID = '42445198'
DATA_DIR = f'./data/{VISIT_ID}/{VIDEO_ID}'
FRAME_KEY = '10907.545' # Example frame key/timestamp from the trajectory

# 1. Load Camera Poses
poses = parse_traj(f'{DATA_DIR}/hires_poses.traj')
T_world_to_cam = poses[FRAME_KEY]

# 2. Load 3D points from the laser scan using annotation indices

# Read the point cloud
ply  = PlyData.read(f'./data/{VISIT_ID}/{VISIT_ID}_laser_scan.ply')['vertex']
pts  = np.stack([ply['x'], ply['y'], ply['z']], axis=1).astype(np.float64)

# Read annotations to get the indices for a specific object (e.g., a handle)
ann = json.load(open(f'./data/{VISIT_ID}/{VISIT_ID}_annotations.json'))['annotations']
target_annot = ann[0]  # Just grabbing the first annotated object as an example
points_3d = pts[target_annot['indices']]

# 3. Extract View (calculates visibility, area, and extracts crop + mask)
view_data = build_view(points_3d, DATA_DIR, FRAME_KEY, T_world_to_cam)
if view_data:
  print(f"Extraction successful for {target_annot['label']}! Crop shape:", view_data['crop'].shape)
  cv2.imwrite('output_crop.jpg', cv2.cvtColor(view_data['crop'], cv2.COLOR_RGB2BGR))
else:
  print("Points not visible in this frame.")
