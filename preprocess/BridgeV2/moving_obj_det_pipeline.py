# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import torch
import argparse
import cv2
import numpy as np
import sys
from glob import glob
from PIL import Image
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor
from tqdm import tqdm
DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_buffer_path",
        default="/apdcephfs_cq10/share_1150325/hongyanzhi/im2Flow2Act/bridge_orig/processed_tmp_zarr_debug",
    )
    parser.add_argument(
        "--data_path",
        default="/apdcephfs_cq10/share_1150325/hongyanzhi/im2Flow2Act/bridge_orig/processed_tmp",
    )
    parser.add_argument(
        "--checkpoint",
        # default="./checkpoints/cotracker.pth",
        default=None,
        help="CoTracker model parameters",
    )
    parser.add_argument("--grid_size", type=int, default=0, help="Regular grid size")
    parser.add_argument(
        "--grid_query_frame",
        type=int,
        default=0,
        help="Compute dense and grid tracks starting from this frame",
    )
    parser.add_argument(
        "--backward_tracking",
        action="store_true",
        help="Compute tracks in both directions, not only forward",
    )
    parser.add_argument(
        "--use_v2_model",
        action="store_true",
        help="Pass it if you wish to use CoTracker2, CoTracker++ is the default now",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Pass it if you would like to use the offline model, in case of online don't pass it",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Visualize the results in debug mode, which will save the visualization results to the disk",
    )
    parser.add_argument('--start_idx', type=int, default=None)
    parser.add_argument('--end_idx', type=int, default=None)
    args = parser.parse_args()

    VIZ_DIR = '< Your visualization path here >'

    if args.checkpoint is not None:
        if args.use_v2_model:
            model = CoTrackerPredictor(checkpoint=args.checkpoint, v2=args.use_v2_model)
        else:
            if args.offline:
                window_len = 60
            else:
                window_len = 16
            model = CoTrackerPredictor(
                checkpoint=args.checkpoint,
                v2=args.use_v2_model,
                offline=args.offline,
                window_len=window_len,
            )
    else:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
    model = model.to(DEFAULT_DEVICE)

    # =============== Process source and output path & Deal with multi-processes ==================
    os.makedirs(args.data_buffer_path, exist_ok=True)  
    data_buffer = zarr.open(args.data_buffer_path, mode="a")
    # We will heruistically filter the episodes that broken
    valid_num = 0  
    source = glob(f"{args.data_path}/*")
    source = sorted(source, key=lambda x: int(x.split('/')[-1]))
    if args.start_idx is None:
        args.start_idx = 0
    if args.end_idx is None:
        args.end_idx = len(source)
    print("Total Remain: ", len(source))
    print(f"Start from episode {args.start_idx} to episode {args.end_idx}")

    pbar = tqdm(total=len(source[args.start_idx:args.end_idx]))
    for ei, episode_dir in enumerate(source[args.start_idx:args.end_idx]):
        pbar.update(1)
        
        save_dir = f'episode_{episode_dir.split("/")[-1]}'
        if args.debug:
            viz_dir = f"{VIZ_DIR}/{save_dir}"

        data_buffer.create_group(save_dir)
        
        instruction_txt_path = f"{episode_dir}/instructions.txt"
        task_description = open(instruction_txt_path, 'r').read()
        if task_description == "":
            continue
        info_array = data_buffer[save_dir].create_dataset("info", shape=(1,), dtype=object, object_codec=zarr.codecs.JSON())  # 使用 JSON 编解码器
        info_array[0] = {"task_description": task_description}

        img_ps = glob(f"{episode_dir}/frames/*.jpg")
        img_ps = sorted(img_ps, key=lambda x: int(x.split('/')[-1].split('.')[0]))
        video = np.array([cv2.imread(img_p) for img_p in img_ps])
        # resize
        video = np.array([cv2.resize(frame, (256, 256)) for frame in video])
        H, W, C = video[0].shape
        data_buffer[f"{save_dir}/rgb_arr"] = video
        video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float() # [bsz, len, 3, h, w]
        video = video.to(DEFAULT_DEVICE)
        
        # =================== 1. First Use Cotracker3 to track the points of the entire first frame =================

        # We simple regard the moving points in the first few frame as the gripper or use GroundingSam2
        segm_mask = np.ones((H, W))
        
        # Parse bbox from mask
        h_coords, w_coords = torch.where(torch.from_numpy(segm_mask) > 0)
        sam_min_w = w_coords.min().item()
        sam_min_h = h_coords.min().item()
        sam_max_w = w_coords.max().item()
        sam_max_h = h_coords.max().item()
        segm_mask = torch.from_numpy(segm_mask)[None, None]           # [bsz, 1, h, w]
        
        min_w = 0
        min_h = 0
        max_w = W
        max_h = H

        num_points_per_side = 32
        x_coords = torch.linspace(min_w, max_w, num_points_per_side)
        y_coords = torch.linspace(min_h, max_h, num_points_per_side)

        grid_x, grid_y = torch.meshgrid(x_coords, y_coords)
        coordinates = torch.stack((grid_x.flatten(), grid_y.flatten()), dim=1).unsqueeze(0).cuda().float()

        # filter out points inside the mask
        new_coordinates = []
        for coord in coordinates[0]:
            if segm_mask[0, 0, int(coord[1]), int(coord[0])] == 0:
                new_coordinates.append(coord)
        coordinates = torch.stack(new_coordinates).unsqueeze(0).cuda().float()

        # pad 0 for dim -1
        coordinates = torch.cat((torch.zeros_like(coordinates[:, :, :1]), coordinates), dim=-1)
        
        pred_tracks, pred_visibility = model(                # [bsz, len, N, 2]
            video,
            grid_size=args.grid_size,
            grid_query_frame=args.grid_query_frame,
            backward_tracking=args.backward_tracking,
            # segm_mask=segm_mask,
            queries=coordinates
        )

        # Filter the pts that move initially (will be gripper）
        init_pts = pred_tracks[0, 0].cpu().numpy()
        intervel_pts = pred_tracks[0, 5].cpu().numpy()
        dist = np.linalg.norm(init_pts - intervel_pts, axis=-1)
        gripper_mask = dist < 1
        pred_tracks = pred_tracks[:, :, gripper_mask, :]
        pred_visibility = pred_visibility[:, :, gripper_mask]

        if args.debug:
            flow_save_dir = f"{viz_dir}/moving_object_det"
            os.makedirs(flow_save_dir, exist_ok=True)
            vis = Visualizer(save_dir=flow_save_dir, pad_value=120, linewidth=1, mode="rainbow", tracks_leave_trace=-1)
            vis.visualize(
                video.cpu(),
                pred_tracks[:, :, :, ...].cpu(),
                pred_visibility[:, :, :, ...].cpu(),
                query_frame=0 if args.backward_tracking else args.grid_query_frame,
            )

        # Filter the pts move noisy
        init_pts = pred_tracks[0, 0].cpu().numpy()
        final_pts = pred_tracks[0, -1].cpu().numpy()
        dist = np.linalg.norm(init_pts - final_pts, axis=-1)
      
        if dist.max() < 10:
            print("Filtered by max point move distance")
            continue
        move_mask = dist > (dist.max() - dist.min()) / 4
        pred_tracks = pred_tracks[:, :, move_mask, :]
        pred_visibility = pred_visibility[:, :, move_mask]

        if args.debug:
            flow_save_dir = f"{viz_dir}/moving_object_det_move_filter"
            os.makedirs(flow_save_dir, exist_ok=True)
            vis = Visualizer(save_dir=flow_save_dir, pad_value=120, linewidth=1, mode="rainbow", tracks_leave_trace=-1)
            vis.visualize(
                video.cpu(),
                pred_tracks[:, :, :, ...].cpu(),
                pred_visibility[:, :, :, ...].cpu(),
                query_frame=0 if args.backward_tracking else args.grid_query_frame,
            )

        tracking_point_sequence = np.concatenate(
            [pred_tracks.cpu().numpy(), np.expand_dims(pred_visibility.cpu().numpy(), axis=-1).astype(np.float32)],
                axis=-1,
            )

        pred_tracks = pred_tracks[0].cpu().numpy()   
        pred_visibility = pred_visibility.cpu().numpy()
        
        object_bboxs = []
        # (T, N, 2)
        obj_tracks = pred_tracks[0]
        obj_tracks = obj_tracks.reshape(-1, 2)
        obj_min_w = obj_tracks[:, 0].min()
        obj_min_h = obj_tracks[:, 1].min()
        obj_max_w = obj_tracks[:, 0].max()
        obj_max_h = obj_tracks[:, 1].max()

        # Filter by bbox size: samll than 1/4 of the whole image
        if (obj_max_w - obj_min_w) * (obj_max_h - obj_min_h) > H * W / 4:
            print("Filtered by bbox size")
            continue

        object_bboxs.append([obj_min_w, obj_min_h, obj_max_w, obj_max_h])
        object_bboxs = np.array(object_bboxs)
        data_buffer[f"{save_dir}/object_bboxs"] = object_bboxs

        # Get Object Mask
        object_mask = np.zeros((H, W))
        for bi, bbox in enumerate(object_bboxs):
            obj_min_w, obj_min_h, obj_max_w, obj_max_h = bbox
            object_mask[int(obj_min_h):int(obj_max_h), int(obj_min_w):int(obj_max_w)] = bi+1
        data_buffer[f"{save_dir}/object_masks"] = object_mask

        # Draw bbox on the first frame
        if args.debug:
            raw_img_t0 = cv2.imread(img_ps[0])
            for bbox in object_bboxs:
                # print(bbox)
                cv2.rectangle(raw_img_t0, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            os.makedirs(f"{episode_dir}/moving_objs", exist_ok=True)
            cv2.imwrite(f"{episode_dir}/moving_objs/bbox.jpg", raw_img_t0)

        # Parse Flow for each object
        all_object_tracks = []
        for obi, bbox in enumerate(object_bboxs):
            min_w, min_h, max_w, max_h = bbox

            obj_min_w = obj_min_w + (obj_max_w - obj_min_w) * 0.15
            obj_min_h = obj_min_h + (obj_max_h - obj_min_h) * 0.15
            obj_max_w = obj_max_w - (obj_max_w - obj_min_w) * 0.15
            obj_max_h = obj_max_h - (obj_max_h - obj_min_h) * 0.15

            num_points_per_side = 32
            x_coords = torch.linspace(min_w, max_w, num_points_per_side)
            y_coords = torch.linspace(min_h, max_h, num_points_per_side)

            # 创建网格坐标
            grid_x, grid_y = torch.meshgrid(x_coords, y_coords)
            coordinates = torch.stack((grid_x.flatten(), grid_y.flatten()), dim=1).unsqueeze(0).cuda().float()  # [1, 1024, 2]
            # pad 0 for dim -1
            coordinates = torch.cat((torch.zeros_like(coordinates[:, :, :1]), coordinates), dim=-1)

            pred_tracks, pred_visibility = model(                # [bsz, len, 900, 2]
                video,
                grid_size=args.grid_size,
                grid_query_frame=args.grid_query_frame,
                backward_tracking=args.backward_tracking,
                segm_mask=segm_mask,
                queries=coordinates
            )
            tracking_point_sequence = np.concatenate(
                [pred_tracks.cpu().numpy(), np.expand_dims(pred_visibility.cpu().numpy(), axis=-1).astype(np.float32)],
                    axis=-1,
                )
            all_object_tracks.append(tracking_point_sequence)

            if args.debug:
                flow_save_dir = f"{viz_dir}/flow_{obi}"
                os.makedirs(flow_save_dir, exist_ok=True)
                vis = Visualizer(save_dir=flow_save_dir, pad_value=120, linewidth=1, mode="rainbow", tracks_leave_trace=-1)
                vis.visualize(
                    video.cpu(),
                    pred_tracks[:, :, ::50, ...].cpu(),
                    pred_visibility[:, :, ::50, ...].cpu(),
                    query_frame=0 if args.backward_tracking else args.grid_query_frame,
                )

        all_object_tracks = np.array(all_object_tracks)
        data_buffer[f"{save_dir}/co_tracker_sequence_grid"] = (
                all_object_tracks
            )
        
        valid_num += 1
    
        print(f"Valid number of episodes: {valid_num} | {ei+1}")

        

        

    