#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

import numpy as np
import matplotlib.pyplot as plt
from train import generate_orbiting_trajectory
from scene.cameras import Camera
from PIL import Image

import shutil
from pathlib import Path
from scene.cameras import Camera
from typing import List, Tuple
import csv

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def calculate_rotation_distance(R1: np.ndarray, R2: np.ndarray) -> float:
    """
    Calculates the angular difference (in radians) between two rotation matrices.
    
    The distance is the rotation angle theta of the relative rotation matrix R_rel.
    R_rel = R1 @ R2.T
    Trace(R_rel) = 1 + 2 * cos(theta)
    """
    # Calculate the relative rotation matrix R_rel = R1 * R2_transpose
    R_rel = R1 @ R2.T
    
    # Calculate the trace of the relative rotation matrix
    trace = np.trace(R_rel)
    
    # Calculate cos(theta) and clamp the value to [-1, 1] to prevent floating point errors
    # outside the valid domain of arccos
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # Calculate the rotation angle (in radians)
    angle = np.arccos(cos_theta)
    
    return angle

def calculate_pose_distance(pose1: Camera, pose2: Camera) -> float:
    """
    Calculates a weighted combination of translational and rotational distance.
    This metric defines the "similarity cost" between two adjacent views.
    """
    # 1. Translational Distance (Euclidean L2 norm)
    # T1 and T2 are 3x1 vectors.
    translation_diff = pose1.T - pose2.T
    translational_dist = np.linalg.norm(translation_diff)
    
    # 2. Rotational Distance (Angular difference in radians)
    rotational_dist = calculate_rotation_distance(pose1.R, pose2.R)
    
    # 3. Combined Weighted Distance
    combined_dist = (0.3 * translational_dist) + \
                    (0.7 * rotational_dist)

    return combined_dist

def nearest_neighbor_sort(poses: List[Camera], start_index: int = 0) -> Tuple[List[Camera], float]:
    """
    Sorts the camera poses using the Nearest Neighbor heuristic.
    This finds a near-optimal shortest path through all poses.
    
    Returns the sorted list of poses and the total path distance.
    """
    if not poses:
        return [], 0.0

    # Initialize
    num_poses = len(poses)
    
    # Start the path with the chosen pose
    current_pose = poses[start_index]
    sorted_sequence = [current_pose]
    resolution = (current_pose.image_width, current_pose.image_height)
    FoVx = current_pose.FoVx
    FoVy = current_pose.FoVy
    print(f"Image resolution: {resolution}")
    print(f"FoVx: {FoVx}, FoVy: {FoVy}")
    image = Image.open("data/mip-nerf360/bicycle/images/_DSC8679.JPG") # dummy image for interp frames
    
    # Keep track of which poses have been included in the sequence
    unvisited_indices = list(range(num_poses))
    unvisited_indices.pop(start_index)
    
    total_distance = 0.0
    
    print(f"Starting Nearest Neighbor path from: {current_pose.image_name}")
    
    # Build the path one pose at a time
    idx = 0
    for _ in range(num_poses - 1):
        min_dist = float('inf')
        best_neighbor_index = -1
        
        # 1. Find the nearest unvisited neighbor to the current pose
        for unvisited_idx in unvisited_indices:
            neighbor_pose = poses[unvisited_idx]
            dist = calculate_pose_distance(current_pose, neighbor_pose)
            
            if dist < min_dist:
                min_dist = dist
                best_neighbor_index = unvisited_idx

        # 2. Insert intermediate frames if the distance is too large
        print(min_dist)
        dist_threshold = 0.2
        inserted_frame_num = min_dist // dist_threshold
        for j in range(1, int(inserted_frame_num) + 1):
            alpha = j / (inserted_frame_num + 1)
            interp_T = (1 - alpha) * current_pose.T + alpha * poses[best_neighbor_index].T
            # interp_T = current_pose.T
            # Simple linear interpolation for rotation (not ideal, but works for small angles)
            interp_R = (1 - alpha) * current_pose.R + alpha * poses[best_neighbor_index].R
            # interp_R_ortho = current_pose.R
            # Re-orthogonalize the rotation matrix using SVD
            U, _, Vt = np.linalg.svd(interp_R)
            interp_R_ortho = U @ Vt
            
            # Create a new Camera pose for the interpolated frame
            interp_pose = Camera(resolution, colmap_id=-1, R=interp_R_ortho, T=interp_T,
                                   FoVx=FoVx, FoVy=FoVy, depth_params=None,
                                   image=image, invdepthmap=None,
                                   image_name=f"interp_{current_pose.image_name}_to_{poses[best_neighbor_index].image_name}_{j}",
                                   uid=num_poses + idx, data_device="cuda",
                                   train_test_exp=False, is_test_dataset=True, is_test_view=True)
            sorted_sequence.append(interp_pose)
            total_distance += dist_threshold
            idx += 1
            
        # 3. Update the path and state
        if best_neighbor_index != -1:
            next_pose = poses[best_neighbor_index]
            sorted_sequence.append(next_pose)
            total_distance += min_dist
            
            # Remove the chosen pose from the unvisited list
            unvisited_indices.remove(best_neighbor_index)
            current_pose = next_pose # Move to the new pose
        else:
            # Should not happen unless unvisited_indices is empty unexpectedly
            break

    return sorted_sequence, total_distance

def reorder_and_copy_images(
    sorted_image_names: List[str],
    views: List[Camera],
    source_dir: str,
    target_dir: str,
    file_extension: str = '.jpg', 
    gaussians: GaussianModel = None
):
    """
    Copies images from the source directory to the target directory 
    using a sequential naming convention based on the sorted list.

    Args:
        sorted_image_names (List[str]): The list of image filenames 
                                        in the desired temporal order.
        source_dir (str): The directory where the original images reside.
        target_dir (str): The directory where the reordered, numbered images will be saved.
        file_extension (str): The file extension of the images (e.g., '.jpg' or '.png').
    """
    
    # Use Path objects for clean, OS-independent path manipulation
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    

    # 1. Create the target directory if it does not exist
    try:
        target_path.mkdir(parents=True, exist_ok=True)
        print(f"Target directory created or already exists: {target_dir}")
    except Exception as e:
        print(f"Error creating directory {target_dir}: {e}")
        return

    num_frames = len(sorted_image_names)
    print(f"\nReordering and copying {num_frames} frames...")
    mapping_data = []
    # 2. Iterate through the sorted list and copy/rename
    for i, original_filename_with_ext in enumerate(sorted_image_names):
        # Ensure the filename has the correct extension for construction
        # Note: We assume the image names in the list ALREADY have the extension.
        padding_width = len(str(num_frames - 1)) 
        new_filename = f"frame_{i:0{padding_width}d}{file_extension}"
        
        dest_file = target_path / new_filename

        mapping_data.append({
            'frame_index': i,
            'original_filename': original_filename_with_ext,
            'sorted_filename': new_filename
        })

        
        # original name does not include "interp_"
        if not original_filename_with_ext.startswith("interp_"):
            # Determine paths
            source_file = source_path / original_filename_with_ext
            
            # 3. Perform the copy operation
            try:
                shutil.copy2(source_file, dest_file)
                print(f"Copied: {original_filename_with_ext} -> {new_filename}")
            except FileNotFoundError:
                print(f"ERROR: Source file not found: {source_file}. Skipping this frame.")
            except Exception as e:
                print(f"ERROR copying {original_filename_with_ext}: {e}")
        else:
            rendering = render(views[i], gaussians, pipeline, background, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
            
            torchvision.utils.save_image(rendering, dest_file)
            print(f"Rendered and saved: {new_filename}")
        
    csv_filename = target_path / "filename_mapping.csv"
    if mapping_data:
        fieldnames = ['frame_index', 'original_filename', 'sorted_filename']
        try:
            with open(csv_filename, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(mapping_data)
            print(f"\n✅ Filename mapping saved to: {csv_filename}")
        except Exception as e:
            print(f"\nERROR saving CSV mapping file: {e}")
                        
    print("\n" + "="*50)
    print(f"✅ Image reordering complete! Files saved in: {target_dir}")
    print(f"Ready for VTM encoding starting with frame_00000{file_extension}")
    print("="*50)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    dataset = model.extract(args)
    dataset.eval = True
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree) 
        scene = Scene(dataset, gaussians, load_iteration=30000, shuffle=False)

        cam_list = scene.getTrainCameras()
        print("Number of views: ", len(cam_list))
        
        # build a 
        sorted_cams, total_dist = nearest_neighbor_sort(cam_list, start_index=0)
        print("Total path distance: ", total_dist)
        for i, cam in enumerate(sorted_cams):
            print(f"Pose {i:03d}: {cam.image_name}")

    reorder_and_copy_images(
        sorted_image_names=[cam.image_name for cam in sorted_cams],
        views=sorted_cams,
        source_dir='data/mip-nerf360/bicycle/images',
        target_dir='data/mip-nerf360/bicycle/images_interp_0.2',
        file_extension=".jpg",
        gaussians=gaussians
    )