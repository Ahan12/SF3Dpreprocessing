# SceneFun3D 2D Extraction Pipeline

This repo provides an utility pipeline for converting the 3d point cloud annotations from the SceneFun3D dataset into 2D crops.

## Problem
In the SceneFun3D dataset, object annotations are provided as 3D point-index sets corresponding to the scene's laser scan. However, state-of-the-art vision models in our pipeline (such as **CLIP, DINOv3, and SAM**) strictly consume 2D images. 

Because a single scene can contain ~1,000 candidate video frames, there is no single "canonical" 2D view for any given object. Consequently, pre-computed 2D crops cannot be natively provided.

## Solution
This pipeline generates the necessary 2D data by:
1. Projecting the annotated 3D points into the 2D camera views using the provided trajectories and intrinsics.
2. Scoring candidate frames based on visibility, depth-checks, and pixel area.
3. Extracting the optimal 2D crops and masks on-the-fly for downstream 2D models.

## How to use
Please refer to [`example_usage.py`](example_usage.py) for a commented, step-by-step guide on how to run this extraction pipeline on a specific scene.

## Acknowledgements
The majority of the projection and scoring approach used in this repository is adapted from [FunGraph](https://github.com/DennisRotondi/FunGraph). 
