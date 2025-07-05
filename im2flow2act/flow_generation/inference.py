import os
from tqdm import tqdm
import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from einops import rearrange
from transformers import CLIPTextModel, CLIPTokenizer
import cv2
import random
import pdb 
from im2flow2act.common.utility.arr import uniform_sampling
from im2flow2act.common.utility.model import load_config
from im2flow2act.common.utility.viz import convert_tensor_to_image, save_to_gif
from im2flow2act.flow_generation.animatediff.models.unet import UNet3DConditionModel
from im2flow2act.flow_generation.AnimateFlow import AnimateFlow
from im2flow2act.flow_generation.dataloader.animateflow_dataset import process_image
from im2flow2act.tapnet.utility.viz import draw_point_tracking_sequence
from im2flow2act.flow_generation.constant import *
import im2flow2act.flow_generation.constant as CONST
import imageio

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
    # first_frame = frames[0]
    # height, width, layers = first_frame.shape

    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v 编码
    # video = cv2.VideoWriter(output_video_path, fourcc, 10, (width, height))  # 30.0 是帧率

    # # 遍历所有帧并写入视频
    # for frame in frames:
    #     video.write(frame)  # 将帧写入视频
    imageio.mimsave(output_video_path.replace('mp4','gif'), frames, duration=0.3)


def load_model(cfg):
    model_cfg = load_config(cfg.model_path)

    noise_scheduler = DDIMScheduler(**model_cfg.noise_scheduler_kwargs)
    print(f"loading vae from {model_cfg.vae_pretrained_model_path}")
    vae = AutoencoderKL.from_pretrained(
        model_cfg.vae_pretrained_model_path, subfolder="vae"
    ).to("cuda")
    print(f"loading tokenizer and text_encoder from {model_cfg.pretrained_model_path}")
    tokenizer = CLIPTokenizer.from_pretrained(
        model_cfg.pretrained_model_path, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        model_cfg.pretrained_model_path, subfolder="text_encoder"
    ).to("cuda")
    print(f"loading unet from {model_cfg.pretrained_model_path}")
    unet = UNet3DConditionModel.from_pretrained_2d(
        model_cfg.pretrained_model_path,
        subfolder="unet",
        unet_additional_kwargs=model_cfg.unet_additional_kwargs,
    )
    # load the model
    model = AnimateFlow(unet=unet, **model_cfg.animateflow_kwargs)
    model.load_model(
        os.path.join(
            cfg.model_path,
            "checkpoints",
            f"epoch_{cfg.model_ckpt}",
            f"epoch_{cfg.model_ckpt}.ckpt",
        )
    )
    model = model.to("cuda")
    model.eval()
    return vae, text_encoder, tokenizer, noise_scheduler, model


def viz_generated_flow(
    flows,
    initial_frame_uv,
    initial_frame,
    diff_flow=True,
    draw_line=True,
    viz_n_points=-1,
    is_gt=False,
):
    # flows: (N,T,3)
    frame_shape = initial_frame.shape[0]
    if is_gt:
        flows = np.clip(flows / 2 + 0.5, a_min=0, a_max=1)
    video_length = flows.shape[1]
    if diff_flow:
        for i in range(video_length):
            step_sequence = (
                initial_frame_uv + flows[:, i, :2] * (2 * frame_shape) - frame_shape
            )
            flows[:, i, :2] = step_sequence
    else:
        flows[:, :, :2] = flows[:, :, :2] * frame_shape
    flows = np.round(flows).astype(np.int32)
    flows = np.clip(flows, 0, frame_shape - 1)
    frames = []
    if viz_n_points == -1:
        viz_indicies = np.arange(flows.shape[0])
    else:
        _, viz_indicies = uniform_sampling(flows, viz_n_points, return_indices=True)
    
    for j in range(flows.shape[1]):
        frame = draw_point_tracking_sequence(
            initial_frame.copy(),
            flows[viz_indicies, :j],
            draw_line=draw_line,
        )
        frames.append(frame)
    return frames


def inference_from_dataset(
    pipeline,
    evalulation_datasets,
    evaluation_save_path,
    num_inference_steps,
    num_samples,
    guidance_scale,
    viz_n_points,
    draw_line=True,
    wandb_log=False,
):
    for k, evalulation_dataset in enumerate(evalulation_datasets):
        num_samples = len(evalulation_dataset)
        data_set = evalulation_dataset.set
        dataset_name = evalulation_dataset.dataset_name
        dataset_result_save_path = os.path.join(evaluation_save_path, f"{dataset_name}_{data_set}")
        video_length = evalulation_dataset.n_sample_frames
        os.makedirs(dataset_result_save_path, exist_ok=True)
        for i in range(num_samples):
            sample = evalulation_dataset[i]
            text = sample["text"]
            global_image = sample["global_image"].unsqueeze(0).cuda()
            point_uv = (
                torch.from_numpy(sample["first_frame_point_uv"]).unsqueeze(0).cuda()
            )
            gt_flow = sample["point_tracking_sequence"].unsqueeze(0).cuda()
            flows = pipeline(
                prompt=text,
                global_image=global_image,
                point_uv=point_uv,
                video_length=video_length,
                height=evalulation_dataset.grid_size,
                width=evalulation_dataset.grid_size,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                output_type="numpy",
            )  # (B,T,C,N1,N2) [0,1]
            
            flows_2d = flows[0].copy()  # (T,3,N,N)
            flows_2d[:, -1, ...] = 1.
            flows_2d = rearrange(flows_2d, "T C N1 N2 -> (N1 N2) T C")
            viz_gt_flow = gt_flow[0].cpu().numpy()
            viz_gt_flow[:, -1, ...] = 1.
            viz_gt_flow = rearrange(viz_gt_flow, "T C N1 N2 -> (N1 N2) T C")
            frames = viz_generated_flow(
                flows=viz_gt_flow,
                initial_frame_uv=sample["first_frame_point_uv"],
                initial_frame=convert_tensor_to_image(global_image[0]),
                diff_flow=evalulation_dataset.diff_flow,
                draw_line=draw_line,
                viz_n_points=viz_n_points,
                is_gt=True,
            )
            save_to_mp4(
                frames,
                os.path.join(dataset_result_save_path, f"gt_flow_{i}_{text}.mp4"),
            )

            frames = viz_generated_flow(
                flows=flows_2d,
                initial_frame_uv=sample["first_frame_point_uv"],
                initial_frame=convert_tensor_to_image(global_image[0]),
                diff_flow=evalulation_dataset.diff_flow,
                draw_line=draw_line,
                viz_n_points=viz_n_points,
            )
            save_to_mp4(
                frames,
                os.path.join(dataset_result_save_path, f"generated_flow_{i}_{text}.mp4"),
            )

            #! Debug exp
            flow = flows[0].copy()
            flow = rearrange(flow, "f c h w -> f h w c")
            # unnormalize flow
            norm_scale = sample["norm_scale"]
            norm_offset = sample["norm_offset"]
            # flow = (flow + 1) / 2
            flow = flow * norm_scale + norm_offset
            flow = rearrange(flow, "f h w c -> c f h w")

            gt_flow = gt_flow[0].cpu().numpy()
            gt_flow = rearrange(gt_flow, "f c h w -> f h w c")
            gt_flow = (gt_flow + 1) / 2
            gt_flow = gt_flow * norm_scale + norm_offset
            gt_flow = rearrange(gt_flow, "f h w c -> f c h w")

            global_image = sample["global_image_3d"]
            
            save_dict = {
                'text': text,
                'pred_flow': flow,
                'gt_flow': gt_flow,
                'global_image': global_image,
            }
            np.save(os.path.join(dataset_result_save_path, f"sample_{i}_{text}.npy"), save_dict)

def inference_from_dataset_3d(
    pipeline,
    evalulation_datasets,
    evaluation_save_path,
    num_inference_steps,
    num_samples,
    guidance_scale,
    viz_n_points,
    draw_line=True,
    wandb_log=False,
):  
    
    for k, evalulation_dataset in enumerate(evalulation_datasets):
        num_samples = len(evalulation_dataset)
        pbar = tqdm(total=num_samples, desc="Datasets")
        data_set = evalulation_dataset.set
        dataset_name = evalulation_dataset.dataset_name
        dataset_result_save_path = os.path.join(evaluation_save_path, f"{dataset_name}_{data_set}")
        video_length = evalulation_dataset.n_sample_frames
        os.makedirs(dataset_result_save_path, exist_ok=True)
        for i in range(num_samples):
            pbar.update(1)
            sample = evalulation_dataset[i]
            text = sample["text"]
            global_image = sample["global_image"].unsqueeze(0).cuda()
            global_image_2d = sample["global_image_2d"].unsqueeze(0).cuda()
            point_uv = (torch.from_numpy(sample["first_frame_point_uv"]).unsqueeze(0).cuda())
            gt_flow = sample["point_tracking_sequence"].unsqueeze(0).cuda()
            
            if os.environ.get("USE_VIDEO_CONDITION", 'False') == "True":
                video_conditions = sample["video_conditions"]
                video_conditions = torch.zeros_like(video_conditions)
                # Sum will be zero
                assert video_conditions.sum() == 0, f"video_conditions should not be all zeros while inference: {video_conditions}"
                video_conditions = video_conditions.unsqueeze(0).cuda()
            else:
                video_conditions = None

            flows = pipeline(
                prompt=text,
                global_image=global_image,
                global_image_2d=global_image_2d,
                video_conditions=video_conditions,
                point_uv=point_uv,
                video_length=video_length,
                height=evalulation_dataset.grid_size,
                width=evalulation_dataset.grid_size,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                output_type="numpy",
            )  # (B,T,C,N1,N2) [0,1]

            if os.environ.get("USE_CLIP") == "True":
                flows_2d = flows[0].copy()  # (T,3,N,N)
                flows_2d[:, -1, ...] = 1.
                flows_2d = rearrange(flows_2d, "T C N1 N2 -> (N1 N2) T C")
                viz_gt_flow = gt_flow[0].cpu().numpy()
                viz_gt_flow[:, -1, ...] = 1.
                viz_gt_flow = rearrange(viz_gt_flow, "T C N1 N2 -> (N1 N2) T C")
                gt_frames = viz_generated_flow(
                    flows=viz_gt_flow,
                    initial_frame_uv=sample["first_frame_point_uv"],
                    initial_frame=convert_tensor_to_image(global_image_2d[0]),
                    diff_flow=evalulation_dataset.diff_flow,
                    draw_line=draw_line,
                    viz_n_points=viz_n_points,
                    is_gt=True,
                )
                # save_to_mp4(
                #     frames,
                #     os.path.join(dataset_result_save_path, f"gt_flow_{i}_{text}.mp4"),
                # )

                frames = viz_generated_flow(
                    flows=flows_2d,
                    initial_frame_uv=sample["first_frame_point_uv"],
                    initial_frame=convert_tensor_to_image(global_image_2d[0]),
                    diff_flow=evalulation_dataset.diff_flow,
                    draw_line=draw_line,
                    viz_n_points=viz_n_points,
                )

                # Concatenate gt_frames and frames
                frames = np.concatenate((gt_frames, frames), axis=2)

                save_to_mp4(
                    frames,
                    os.path.join(dataset_result_save_path, f"generated_flow_{i}_{text}.mp4"),
                )
            
            ''' Normal 3D version'''
            # flow = flows[0].copy()
            # flow = rearrange(flow, "f c h w -> f h w c")
            # # unnormalize flow
            # norm_scale = sample["norm_scale"]
            # norm_offset = sample["norm_offset"]
            # # flow = (flow + 1) / 2
            # flow = flow * norm_scale + norm_offset
            # flow = rearrange(flow, "f h w c -> c f h w")

            # gt_flow = gt_flow[0].cpu().numpy()
            # gt_flow = rearrange(gt_flow, "f c h w -> f h w c")
            # gt_flow = (gt_flow + 1) / 2
            # gt_flow = gt_flow * norm_scale + norm_offset
            # gt_flow = rearrange(gt_flow, "f h w c -> f c h w")

            # global_image = global_image[0].cpu().numpy()
            # point_uv = point_uv[0].cpu().numpy()
            # global_image[..., :3] = (global_image[..., :3] + 1) / 2
            # global_image[..., :3] = global_image[..., :3] * norm_scale + norm_offset
            # point_uv = (point_uv + 1) / 2
            # # point_uv = point_uv / POS3D_GRID_SIZE
            # point_uv = point_uv * norm_scale + norm_offset

            ''' Use 2D xy and 3D z: xy need project to 3D'''
            if dataset_name.find('realworld') != -1:
                H = CONST.REALWORLD_H
                W = CONST.REALWORLD_W
                fx, fy = CONST.FX, CONST.FY
                cy = CONST.RCY
                cx = CONST.RCX
            else:
                H = W = 256
                fx, cx, cy = get_intrinsics(H, W)
                fy = fx

            flow = flows[0].copy()
            flow = rearrange(flow, "f c h w -> f h w c")
            # unnormalize flow
            norm_scale = sample["norm_scale"]
            norm_offset = sample["norm_offset"]
            # flow = (flow + 1) / 2
            flow[..., -1] = flow[..., -1] * norm_scale[-1] + norm_offset[-1]
            flow[..., 0] *= W
            flow[..., 1] *= H
            depths = flow[..., 2].copy().reshape(flow.shape[0], -1)
            flow = flow_to_3d_points(depths, flow, fx, fy, cx, cy)
            flow = rearrange(flow, "f h w c -> c f h w")

            gt_flow = gt_flow[0].cpu().numpy()
            gt_flow = rearrange(gt_flow, "f c h w -> f h w c")
            gt_flow = (gt_flow + 1) / 2
            gt_flow[..., -1] = gt_flow[..., -1] * norm_scale[-1] + norm_offset[-1]
            gt_flow[..., 0] *= W
            gt_flow[..., 1] *= H
            depths = gt_flow[..., 2].copy().reshape(gt_flow.shape[0], -1)
            gt_flow = flow_to_3d_points(depths, gt_flow, fx, fy, cx, cy)
            gt_flow = rearrange(gt_flow, "f h w c -> f c h w")

            global_image = global_image[0].cpu().numpy()
            point_uv = point_uv[0].cpu().numpy()
            global_image[..., :3] = (global_image[..., :3] + 1) / 2
            global_image[..., :3] = global_image[..., :3] * norm_scale + norm_offset
            
            # point_uv[..., -1] = (point_uv[..., -1] + 1) / 2
            point_uv[..., -1] = point_uv[..., -1] / POS3D_GRID_SIZE[-1]
            point_uv[..., -1] = point_uv[..., -1] * norm_scale[-1] + norm_offset[-1]
            point_uv[..., 0] *= W / 224
            point_uv[..., 1] *= H / 224 
            point_uv_2d = point_uv[..., :2].copy()

            depths = point_uv[..., 2].copy()[None, ...]
            point_uv = flow_to_3d_points(depths, point_uv[None, ...].reshape(1, gt_flow.shape[-1], gt_flow.shape[-2], 3), fx, fy, cx, cy).reshape(-1, 3)
            
            save_dict = {
                'text': text,
                'pred_flow': flow,
                'gt_flow': gt_flow,
                'global_image': global_image,
                'point_uv': point_uv,
                'point_uv_2d': point_uv_2d,
                'global_image_2d': convert_tensor_to_image(global_image_2d[0]),
            }
            np.save(os.path.join(dataset_result_save_path, f"sample_{i}_{text}.npy"), save_dict)

        
def inference(
    pipeline,
    global_image,
    point_uv,
    text,
    height,
    width,
    video_length,
    diff_flow,
    num_inference_steps,
    guidance_scale,
    evaluation_save_path=None,
    gif_name=None,
    viz_n_points=-1,
    draw_line=True,
    wandb_log=False,
):
    point_uv = point_uv.astype(int)
    initial_flow = point_uv.copy()
    initial_flow = initial_flow / global_image.shape[0]
    initial_flow = np.concatenate(
        [initial_flow, np.ones((initial_flow.shape[0], 1))], axis=-1
    )
    initial_flow = initial_flow[:, None, :]  # (N1*N2,1,3)
    global_image = process_image(global_image).unsqueeze(0).cuda()
    point_uv_tensor = torch.from_numpy(point_uv).unsqueeze(0).cuda()
    flows = pipeline(
        prompt=text,
        global_image=global_image,
        point_uv=point_uv_tensor,
        video_length=video_length,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        output_type="numpy",
    )  # (B,T,C,N1,N2) [0,1]
    flows = flows[0]  # (T,3,N,N)
    flows = rearrange(flows, "T C N1 N2 -> (N1 N2) T C")
    print("final flows shape", flows.shape)
    if evaluation_save_path is not None and gif_name is not None:
        frames = viz_generated_flow(
            flows=flows.copy(),
            initial_frame_uv=point_uv,
            initial_frame=convert_tensor_to_image(global_image[0]),
            diff_flow=diff_flow,
            draw_line=draw_line,
            viz_n_points=viz_n_points,
        )
        save_to_gif(
            frames,
            os.path.join(evaluation_save_path, gif_name),
        )
        if wandb_log:
            import wandb

            frames = np.array(frames)
            frames = np.transpose(frames, [0, 3, 1, 2])
            wandb.log(
                {gif_name: wandb.Video(frames, fps=5, format="gif", caption=text)}
            )

    return flows  # (N,T,3)
