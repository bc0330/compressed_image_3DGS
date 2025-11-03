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

def calculate_pose_distance(pose1: Camera, pose2: Camera, alpha: float = 0.5, beta: float = 0.5, D_max: float = 1.0, max_ang: float = np.pi) -> float:
    """
    Calculates a weighted combination of translational and rotational distance.
    This metric defines the "similarity cost" between two adjacent views.
    Uses proper camera positions derived from R and T.
    Distances are normalized for balanced weighting.
    """
    # Compute camera positions in world coordinates
    pos1 = -np.dot(pose1.R.T, pose1.T)
    pos2 = -np.dot(pose2.R.T, pose2.T)
    
    # Translational Distance (Euclidean L2 norm)
    translational_dist = np.linalg.norm(pos1 - pos2)
    
    # Rotational Distance (Angular difference in radians)
    rotational_dist = calculate_rotation_distance(pose1.R, pose2.R)
    
    # Normalized and weighted combined distance
    norm_trans = translational_dist / D_max if D_max > 0 else 0.0
    norm_rot = rotational_dist / max_ang
    combined_dist = alpha * norm_trans + beta * norm_rot

    return combined_dist

def nearest_neighbor_sort(poses: List[Camera], start_index: int = 0, alpha: float = 0.5, beta: float = 0.5, D_max: float = 1.0) -> Tuple[List[Camera], float]:
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
    
    # Keep track of which poses have been included in the sequence
    unvisited_indices = set(range(num_poses))
    unvisited_indices.remove(start_index)
    
    total_distance = 0.0
    
    print(f"Starting Nearest Neighbor path from: {current_pose.image_name}")
    
    # Build the path one pose at a time
    for _ in range(num_poses - 1):
        min_dist = float('inf')
        best_neighbor_index = -1
        
        # 1. Find the nearest unvisited neighbor to the current pose
        for unvisited_idx in list(unvisited_indices):  # Copy to list to avoid runtime modification issues
            neighbor_pose = poses[unvisited_idx]
            dist = calculate_pose_distance(current_pose, neighbor_pose, alpha=alpha, beta=beta, D_max=D_max)
            
            if dist < min_dist:
                min_dist = dist
                best_neighbor_index = unvisited_idx

        # 2. Update the path and state
        if best_neighbor_index != -1:
            next_pose = poses[best_neighbor_index]
            sorted_sequence.append(next_pose)
            total_distance += min_dist
            
            # Remove the chosen pose from the unvisited set
            unvisited_indices.remove(best_neighbor_index)
            current_pose = next_pose  # Move to the new pose
        else:
            # Should not happen unless unvisited_indices is empty unexpectedly
            break

    return sorted_sequence, total_distance

def reorder_and_copy_images(
    sorted_image_names: List[str],
    source_dir: str,
    target_dir: str,
    file_extension: str = '.jpg'
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
        
        # Determine paths
        source_file = source_path / original_filename_with_ext
        
        # Create the new sequential filename, padded with zeros (e.g., 00000, 00001, ...)
        # We determine the padding width based on the number of frames (e.g., 1000 frames needs 4 zeros)
        padding_width = len(str(num_frames - 1)) 
        new_filename = f"frame_{i:0{padding_width}d}{file_extension}"
        
        dest_file = target_path / new_filename

        mapping_data.append({
            'frame_index': i,
            'original_filename': original_filename_with_ext,
            'sorted_filename': new_filename
        })

        # 3. Perform the copy operation
        try:
            shutil.copy2(source_file, dest_file)
            print(f"Copied: {original_filename_with_ext} -> {new_filename}")
        except FileNotFoundError:
            print(f"ERROR: Source file not found: {source_file}. Skipping this frame.")
        except Exception as e:
            print(f"ERROR copying {original_filename_with_ext}: {e}")
            
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
        scene = Scene(dataset, gaussians, shuffle=False)

        cam_list = scene.getTrainCameras()
        print("Number of views: ", len(cam_list))
        
        # Compute D_max (maximum pairwise translational distance) for normalization
        positions = [-np.dot(cam.R.T, cam.T) for cam in cam_list]
        D_max = max(np.linalg.norm(positions[i] - positions[j]) for i in range(len(cam_list)) for j in range(i+1, len(cam_list))) if len(cam_list) > 1 else 1.0
        print(f"Computed D_max: {D_max}")
        
        # Find the best starting index by trying all and selecting the path with minimal total distance
        best_sequence = None
        best_dist = float('inf')
        best_start = -1
        for start in range(len(cam_list)):
            seq, dist = nearest_neighbor_sort(cam_list, start_index=start, alpha=0.5, beta=0.5, D_max=D_max)
            if dist < best_dist:
                best_dist = dist
                best_sequence = seq
                best_start = start
        
        print(f"Best starting index: {best_start}")
        print(f"Total path distance (normalized): {best_dist}")
        for i, cam in enumerate(best_sequence):
            print(f"Pose {i:03d}: {cam.image_name}")

    reorder_and_copy_images(
        sorted_image_names=[cam.image_name for cam in best_sequence],
        source_dir='data/mip-nerf360/bicycle/images',
        target_dir='data/mip-nerf360/bicycle/images_sorted_grok',
        file_extension=".jpg"
    )