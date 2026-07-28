import os
import cv2
import numpy as np
from projection_utils import parse_traj, project_points
from extraction_pipeline import build_view, MIN_DEPTH

# --- Setup Paths ---
# Teammates should update these paths to point to their local SceneFun3D dataset directory
DATA_DIR = './data/visit_123/video_456'
FRAME_KEY = '001' # Example frame key/timestamp

print("Modules imported successfully!")
print("Ready to run 3D-to-2D projection and extraction pipeline.")

# --- Example Pipeline Usage (Commented out until data paths are set) ---
# # 1. Load Camera Poses
# poses = parse_traj(f'{DATA_DIR}/hires_poses.traj')
# T_world_to_cam = poses[FRAME_KEY]
#
# # 2. Define some 3D points (e.g., bounding box center of an object)
# points_3d = np.array([[0.0, 0.0, 1.0], [0.1, 0.2, 1.5]])
#
# # 3. Extract View (calculates visibility, area, and extracts crop + mask)
# view_data = build_view(points_3d, DATA_DIR, FRAME_KEY, T_world_to_cam)
# if view_data:
#     print("Extraction successful! Crop shape:", view_data['crop'].shape)
#     cv2.imwrite('output_crop.jpg', cv2.cvtColor(view_data['crop'], cv2.COLOR_RGB2BGR))
# else:
#     print("Points not visible in this frame.")
