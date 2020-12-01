# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import numpy as np

import torch
from torch.utils.data import DataLoader

from layers import transformation_from_parameters,disp_to_depth, ScaleRecovery
from utils import readlines
from options import MonodepthOptions
from datasets import AirSimDataset
import networks
from AirSim_utils import read_pose


# from https://github.com/tinghuiz/SfMLearner
def dump_xyz(source_to_target_transformations):
    xyzs = []
    cam_to_world = np.eye(4)
    xyzs.append(cam_to_world[:3, 3])
    for source_to_target_transformation in source_to_target_transformations:
        cam_to_world = np.dot(cam_to_world, source_to_target_transformation)
        xyzs.append(cam_to_world[:3, 3])
    return xyzs


# from https://github.com/tinghuiz/SfMLearner
def compute_ate(gtruth_xyz, pred_xyz_o):
    # Make sure that the first matched frames align (no need for rotational alignment as
    # all the predicted/ground-truth snippets have been converted to use the same coordinate
    # system with the first frame of the snippet being the origin).
    offset = gtruth_xyz[0] - pred_xyz_o[0]
    pred_xyz = pred_xyz_o + offset[None, :]

    # Optimize the scaling factor
    scale = np.sum(gtruth_xyz * pred_xyz) / np.sum(pred_xyz ** 2)
    alignment_error = pred_xyz * scale - gtruth_xyz
    rmse = np.sqrt(np.sum(alignment_error ** 2)) / gtruth_xyz.shape[0]
    return rmse


def evaluate(opt):
    """Evaluate odometry on the KITTI dataset
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    K = np.array([[0.5, 0, 0.5, 0],
                  [0, 1.656, 0.5, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=np.float32)
    assert os.path.isdir(opt.load_weights_folder), \
        "Cannot find a folder at {}".format(opt.load_weights_folder)

    filenames = readlines(
        os.path.join(os.path.dirname(__file__), "splits", opt.eval_split,
                     "test_files.txt"))

    dataset = AirSimDataset(
        opt.data_path, filenames, opt.height, opt.width,
        [0, 1], 4, is_train=False)
    dataloader = DataLoader(
        dataset, opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True, drop_last=False)

    pose_encoder_path = os.path.join(opt.load_weights_folder, "pose_encoder.pth")
    pose_decoder_path = os.path.join(opt.load_weights_folder, "pose.pth")
    depth_encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    depth_decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
    

    pose_encoder = networks.ResnetEncoder(opt.num_layers, False, 2)
    pose_encoder.load_state_dict(torch.load(pose_encoder_path))
    depth_encoder = networks.ResnetEncoder(opt.num_layers, False)
    depth_encoder_dict = torch.load(depth_encoder_path)
    model_dict = depth_encoder.state_dict()
    depth_encoder.load_state_dict({k: v for k, v in depth_encoder_dict.items() if k in model_dict})

    
    
    pose_decoder = networks.PoseDecoder(pose_encoder.num_ch_enc, 1, 2)
    pose_decoder.load_state_dict(torch.load(pose_decoder_path))
    depth_decoder = networks.DepthDecoder(depth_encoder.num_ch_enc)
    depth_decoder.load_state_dict(torch.load(depth_decoder_path))


    pose_encoder.cuda()
    pose_encoder.eval()
    pose_decoder.cuda()
    pose_decoder.eval()
    depth_encoder.cuda()
    depth_encoder.eval()
    depth_decoder.cuda()
    depth_decoder.eval()

    pred_poses = []
    pred_disps = []

    print("-> Computing pose predictions")

    opt.frame_ids = [0, 1]  # pose network only takes two frames as input

    with torch.no_grad():
        for inputs in dataloader:
            input_color = inputs[("color", 0, 0)].cuda()
            depth_output = depth_decoder(depth_encoder(input_color))

            pred_disp, _ = disp_to_depth(depth_output[("disp", 0)], opt.min_depth, opt.max_depth)
            pred_disp = pred_disp.cpu()[:, 0].numpy()

            pred_disps.append(pred_disp)

            for key, ipt in inputs.items():
                inputs[key] = ipt.cuda()

            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in opt.frame_ids], 1)

            features = [pose_encoder(all_color_aug)]
            axisangle, translation = pose_decoder(features)

            pred_poses.append(
                transformation_from_parameters(axisangle[:, 0], translation[:, 0]).cpu().numpy())

    pred_poses = np.concatenate(pred_poses)
    pred_disps = np.concatenate(pred_disps)
    
    gt_norms_div = []
    gt_norms = []
    pred_norms = []
    trans_pred = pred_pose[:,:3,3]

    gt_poses_path = os.path.join(opt.data_path, "poses.txt")
    gt_local_poses = read_pose(gt_poses_path)
    num_frames = gt_local_poses.shape[0]
    for i in range(num_frames):
        local_xyzs = pred_poses[i, :3, 3]
        gt_local_xyzs = gt_local_poses[i, :3, 3]
        gt_norm_div = np.linalg.norm(gt_local_xyzs)/np.linalg.norm(local_xyzs)
        gt_norms_div.append(gt_norm_div)

    save_path = os.path.join(os.path.dirname(__file__), "gt_norms_div_AirSim.npy")
    np.save(save_path, gt_norms_div)

    print("-> Predictions saved to", save_path)


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
