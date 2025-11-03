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

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh, dataset):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    
    
    # cam_pos = []
    # traj_cameras = []
    
    # for view in views:
    #     Rt = np.zeros((4, 4))
    #     Rt[:3, :3] = view.R.transpose()
    #     Rt[:3, 3] = view.T
    #     Rt[3, 3] = 1.0

    #     W2C = np.linalg.inv(Rt)
    #     pos = W2C[:3, 3]
    #     cam_pos.append(pos.tolist())
    # cam_pos = np.array(cam_pos)
    # x_data = cam_pos[:, 0]
    # y_data = cam_pos[:, 1]
    # z_data = cam_pos[:, 2]

    # x_center = np.array((x_data.max() + x_data.min()) / 2)
    # y_center = np.array((y_data.max() + y_data.min()) / 2)
    # z_center = np.array(z_data.mean())

    # trajectory_cam_pos, trajectory_cam_rot = generate_orbiting_trajectory([x_center, y_center, z_center], radius=4.5, num_cameras=60, axis='y')

    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')

    # # Create the scatter plot
    # ax.scatter(x_data, y_data, z_data, 
    #         c=z_data,         # Color points based on the Z value
    #         cmap='viridis',   # Colormap to use
    #         marker='o',       # Marker style
    #         s=50)             # Marker size
    # ax.scatter(x_center, y_center, z_center,
    #     c='red',          # Color red
    #     marker='o',       # Marker style
    #     s=300,            # Large marker size to stand out
    #     edgecolors='black', # Add a black edge for visibility
    #     label='Center (Mean)')
    # ax.scatter(trajectory_cam_pos[:, 0], trajectory_cam_pos[:, 1], trajectory_cam_pos[:, 2],
    #     c='blue',          # Color red
    #     marker='o',       # Marker style
    #     s=300,            # Large marker size to stand out
    #     edgecolors='black', # Add a black edge for visibility
    #     label='Center (Mean)')

    # image = Image.open("data/mip-nerf360/bicycle/images/_DSC8679.JPG")
    # orig_w, orig_h = image.size
    # global_down = orig_w / 1600
    # resolution = (int(orig_w / global_down), int(orig_h / global_down))
    # index = 0
    # for C, R in zip(trajectory_cam_pos, trajectory_cam_rot):
    #     ax.quiver(C[0], C[1], C[2], R[0, 2], R[1, 2], R[2, 2], length=0.5, color='blue', normalize=True)
    #     W2C = np.zeros((4, 4))
    #     W2C[:3, :3] = R
    #     W2C[:3, 3] = C
    #     W2C[3, 3] = 1.0
    #     RT = np.linalg.inv(W2C)
    #     _R = RT[:3, :3].transpose()
    #     _T = RT[:3, 3]
    #     traj_cameras.append(Camera(resolution, colmap_id=-1, R=_R, T=_T,
    #                                FoVx=views[0].FoVx, FoVy=views[0].FoVy, depth_params=None,
    #                                image=image, invdepthmap=None,
    #                                image_name=f"orbit_{index:03d}.png", uid=index, data_device="cuda",
    #                                train_test_exp=False, is_test_dataset=True, is_test_view=True))
    #     index += 1
    # # --- 3. Set Labels and Title ---
    # ax.set_xlabel('X Axis')
    # ax.set_ylabel('Y Axis')
    # ax.set_zlabel('Z Axis')
    # ax.set_title('3D Position Scatter Plot (Matplotlib)')

    # plt.savefig('bicycle_cam_pos.png')


    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
    # for idx, view in enumerate(tqdm(traj_cameras, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]
        # gt = view.original_image[0:3, :, :]
        gt_file = os.path.join(dataset.source_path, "images", view.image_name)
        gt = Image.open(gt_file)
        gt = torchvision.transforms.ToTensor()(gt).to("cuda")

        if args.train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            # gt = gt[..., gt.shape[-1] // 2:]

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh, dataset)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh, dataset)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE)