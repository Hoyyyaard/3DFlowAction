import os

import numpy as np
import torch
from einops import rearrange
import cv2
from im2flow2act.common.utility.viz import convert_tensor_to_image, save_to_gif
from im2flow2act.flow_generation.inference import viz_generated_flow
from im2flow2act.flow_generation.constant import *

def get_intrinsics(H,W):
    """
    Intrinsics for a pinhole camera model.
    Assume fov of 55 degrees and central principal point.
    """
    f = 0.5 * W / np.tan(0.5 * 55 * np.pi / 180.0)
    cx = 0.5 * W
    cy = 0.5 * H
    return f, cx, cy

def flow_to_3d_points(depth_map, flow, fx, fy, cx, cy):
    '''
        flow: [T, N1, N2, 3]
        depth_map: [T, N1*N2]
    '''
    grid_size = flow.shape[1]
    flow = flow.reshape(flow.shape[0], -1, flow.shape[-1])[..., :2]
    N = flow.shape[0]
    pt_num = flow.shape[1]

    # 将 flow 投影到 3D
    # 3D 坐标计算: X = (u - cx) * z / fx, Y = (v - cy) * z / fy, Z = depth
    projected_3D = np.zeros((N, pt_num, 3))  # 初始化 3D 投影数组

    for pt in range(pt_num):
        seq_pts = flow[:, pt, :]
        for i in range(N):
            u = int(seq_pts[i, 0])
            v = int(seq_pts[i, 1])
            z = depth_map[i, pt]
            projected_3D[i, pt, 0] = (u - cx) * z / fx
            projected_3D[i, pt, 1] = (v - cy) * z / fy
            projected_3D[i, pt, 2] = z
    projected_3D = projected_3D.reshape(N, grid_size, grid_size, 3)

    return projected_3D

def save_to_mp4(frames, output_video_path):
    first_frame = frames[0]
    height, width, layers = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v 编码
    video = cv2.VideoWriter(output_video_path, fourcc, 10, (width, height))  # 30.0 是帧率

    # 遍历所有帧并写入视频
    for frame in frames:
        video.write(frame)  # 将帧写入视频

def ae_flow_inference(cfg, vae, dataset, wandb_log=False, prefix=""):
    vae.eval()
    for i in range(cfg.num_samples):
        data = dataset[i]
        point_tracking_sequence = data["seq_point_tracking_sequence"].cuda()  # (T,3,N1,N2)
        show_image = convert_tensor_to_image(data["global_image"])
        with torch.no_grad():
            recover_flows = []
            for step in range(len(point_tracking_sequence)):
                sample_flow_image = point_tracking_sequence[step]
                latents = vae.encode(
                    sample_flow_image.unsqueeze(0)
                ).latent_dist.sample()
                recover_flow_image = vae.decode(latents).sample
                recover_flows.append(recover_flow_image.detach().cpu().numpy())

        recover_flows = np.concatenate(
            recover_flows, axis=0
        )  # (n_sample_frames, 3, grid_size, grid_size)
        recover_flows = rearrange(recover_flows, "T C N1 N2 -> (N1 N2) T C")
        frames = viz_generated_flow(
            recover_flows,
            draw_line=True,
            initial_frame_uv=None,
            initial_frame=show_image,
            diff_flow=cfg.diff_flow,
            viz_n_points=256,
            is_gt=True,
        )
        save_to_mp4(
            frames,
            os.path.join(cfg.evaluation_save_path, f"sample_{i}_decoder_flow.mp4"),
        )
        frames = viz_generated_flow(
            rearrange(point_tracking_sequence.cpu().numpy(), "T C N1 N2 -> (N1 N2) T C"),
            draw_line=True,
            initial_frame_uv=None,
            initial_frame=show_image,
            diff_flow=cfg.diff_flow,
            viz_n_points=256,
            is_gt=True,
        )
        # Write to mp4 use cv2
        save_to_mp4(
            frames,
            os.path.join(cfg.evaluation_save_path, f"sample_{i}_gt_flow.mp4"),
        )
        if wandb_log:
            import wandb

            frames = np.array(frames)
            frames = np.transpose(frames, [0, 3, 1, 2])
            wandb.log(
                {f"{prefix}/eval_video_{i}": wandb.Video(frames, fps=5, format="gif")}
            )
    vae.train()


def ae_flow_inference_3d(cfg, vae, dataset, wandb_log=False, prefix=""):
    vae.eval()
    for i in range(len(dataset)):
        data = dataset[i]
        point_tracking_sequence = data["seq_point_tracking_sequence"].cuda()  # (T,C,N1,N2)
        # point_tracking_sequence = point_tracking_sequence.view(-1, *point_tracking_sequence.shape[2:])
        with torch.no_grad():
            recover_flows = []
            for step in range(len(point_tracking_sequence)):
                sample_flow_image = point_tracking_sequence[step]
                latents = vae.encode(
                    sample_flow_image.unsqueeze(0)
                ).latent_dist.sample()
                recover_flow_image = vae.decode(latents).sample
                recover_flows.append(recover_flow_image.detach().cpu().numpy())

        recover_flows = np.concatenate(
            recover_flows, axis=0
        )  # (n_sample_frames, 3, grid_size, grid_size)
        flow = rearrange(recover_flows, "T C N1 N2 -> T N1 N2 C")
        
        if not dataset.dataset_name.find('realworld') != -1:
            H = 464
            W = 720
            fx, fy = 748.41846, 748.1292
            cy = 358.26108 - 256
            cx = 640.1674 - 256
        else:
            H = W = 256
            fx, cx, cy = get_intrinsics(H, W)
            fy = fx

        # unnormalize flow
        norm_scale = data["norm_scale"]
        norm_offset = data["norm_offset"]
        flow = (flow + 1) / 2
        flow[..., -1] = flow[..., -1] * norm_scale[-1] + norm_offset[-1]
        flow[..., 0] *= W
        flow[..., 1] *= H
        depths = flow[..., 2].copy().reshape(flow.shape[0], -1)
        flow = flow_to_3d_points(depths, flow, fx, fy, cx, cy)
        flow = rearrange(flow, "f h w c -> c f h w")

        gt_flow = data["seq_point_tracking_sequence"].cpu().numpy()
        gt_flow = rearrange(gt_flow, "f c h w -> f h w c")
        gt_flow = (gt_flow + 1) / 2
        gt_flow[..., -1] = gt_flow[..., -1] * norm_scale[-1] + norm_offset[-1]
        gt_flow[..., 0] *= W
        gt_flow[..., 1] *= H
        depths = gt_flow[..., 2].copy().reshape(gt_flow.shape[0], -1)
        gt_flow = flow_to_3d_points(depths, gt_flow, fx, fy, cx, cy)
        gt_flow = rearrange(gt_flow, "f h w c -> f c h w")

        global_image = data["global_image"]
        global_image[..., :3] = (global_image[..., :3] + 1) / 2
        global_image[..., :3] = global_image[..., :3] * norm_scale + norm_offset
        
        text = data['text']

        save_dict = {
            'pred_flow': flow,
            'gt_flow': gt_flow,
            'global_image': global_image,
            'norm_scale': norm_scale,
            'norm_offset': norm_offset
        }
        # print(cfg.evaluation_save_path)
        np.save(os.path.join(cfg.evaluation_save_path, f"sample_{i}_{text}.npy"), save_dict)


    vae.train()
