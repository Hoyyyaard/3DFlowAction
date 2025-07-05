import random

import cv2
import numpy as np
import os
import omegaconf
import torch
import zarr
from copy import deepcopy
from einops import rearrange, repeat
import json
import glob
from torchvision.transforms import v2
from tqdm import tqdm
from im2flow2act.flow_generation.constant import *
import im2flow2act.flow_generation.constant as CONST
from im2flow2act.common.imagecodecs_numcodecs import register_codecs
from im2flow2act.common.utility.arr import (
    random_sampling,
    stratified_random_sampling,
    uniform_sampling,
)
import matplotlib.pyplot as plt
from im2flow2act.tapnet.utility.utility import get_buffer_size
register_codecs()
from scipy.ndimage import median_filter
from termcolor import cprint

def apply_median_filter(flow, spatial_size=3, temporal_size=8):
    '''
        Input:
            flow: (Frames, H, W, 3)
        Return:
            filtered_flow: (Frames, H, W, 3)
    '''
    # Filter at frame level
    filtered_flow = np.zeros_like(flow)
    for i in range(flow.shape[0]):  # 遍历每一帧
        # for j in range(3):  # 对每个分量
        #     filtered_flow[i, :, :, j] = median_filter(flow[i, :, :, j], size=size)
        filtered_flow[i] = median_filter(flow[i], size=spatial_size)
    flow = filtered_flow
    # Filter at point level
    filtered_flow = np.zeros_like(flow)
    for j in range(3):  # 对每个分量
        filtered_flow[..., j] = median_filter(flow[..., j], size=temporal_size)
    return filtered_flow


def process_image(image, optional_transforms=[]):
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)  # (3,H,W)

    transform_list = [
        v2.ToDtype(torch.float32, scale=True),  # convert to float32 and scale to [0,1]
    ]

    for transform_name in optional_transforms:
        if transform_name == "ColorJitter":
            transform_list.append(
                # v2.RandomApply(
                #     [
                #         v2.ColorJitter(
                #             brightness=0.32, contrast=0.32, saturation=0.32, hue=0.08
                #         )
                #     ],
                #     p=0.8,
                # )
                v2.RandomApply(
                    [
                        v2.ColorJitter(
                            brightness=0.1, contrast=0.1, saturation=0.1, hue=0.08
                        )
                    ],
                    p=0.5,
                )
            )
        elif transform_name == "RandomGrayscale":
            transform_list.append(v2.RandomGrayscale(p=0.2))
        elif transform_name == 'RandomCrop':
            transform_list.append(
                v2.RandomCrop(
                    size=(200, 200),
            ))

    transform_list.append(
        v2.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )
    )

    transforms = v2.Compose(transform_list)
    return transforms(image)


class AnimateFlowDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_pathes,
        n_sample_frames=16,
        grid_size=32,
        frame_sampling_method="uniform",
        frame_resize_shape=[224, 224],
        point_tracking_img_size=[256, 256],
        diff_flow=True,
        optional_transforms=[],
        max_episode=None,
        start_episode=0,
        seed=0,
        preprocess=None,
        proc_cache=None,
        **kwargs
    ):
        self.data_pathes = data_pathes
        self.n_sample_frames = n_sample_frames
        self.grid_size = grid_size
        self.frame_sampling_method = frame_sampling_method
        self.frame_resize_shape = frame_resize_shape
        self.point_tracking_img_size = point_tracking_img_size
        self.diff_flow = diff_flow
        self.optional_transforms = optional_transforms
        
        # Print all args
        print(">> data_pathes", data_pathes)
        print(">> n_sample_frames", n_sample_frames)
        print(">> grid_size", grid_size)
        print(">> frame_sampling_method", frame_sampling_method)
        print(">> frame_resize_shape", frame_resize_shape)
        print(">> point_tracking_img_size", point_tracking_img_size)
        print(">> diff_flow", diff_flow)
        print(">> optional_transforms", optional_transforms)
        
        self.max_episode = (
            max_episode
            if isinstance(max_episode, (list, omegaconf.listconfig.ListConfig))
            else [max_episode] * len(data_pathes)
        )
        print(">> max_episode", self.max_episode)
        self.start_episode = start_episode
        self.set_seed(seed)
        self.train_data = []

        if preprocess:
            proc_dataset_name = kwargs.get("proc_dataset_name", "")
            print(">> Preprocess the dataset")
            getattr(self, f'preprocess_dataset_{proc_dataset_name}')(proc_cache)
            exit()

        if not proc_cache == 'None':
            print(">> Load the dataset from cache")
            if hasattr(self, 'construct_dataset_from_cache'):
                self.construct_dataset_from_cache(proc_cache)
            else:
                self.construct_dataset()
        else:
            print(">> Load the dataset from raw data")
            self.construct_dataset()

        self.flow_transforms = v2.Compose(
            [
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    @staticmethod
    def sample_point_tracking_frame(
        point_tracking_sequence,
        frame_sampling_method="uniform",
        n_sample_frames=16,
    ):
        sequence = point_tracking_sequence.copy()
        if frame_sampling_method == "uniform":
            sampled_sequence = uniform_sampling(sequence, n_sample_frames)
        elif frame_sampling_method == "random":
            sampled_sequence = random_sampling(
                sequence, n_sample_frames, zero_include=True, replace=False
            )
        elif frame_sampling_method == "stratified_random":
            sampled_sequence = stratified_random_sampling(sequence, n_sample_frames)
        return sampled_sequence

    def set_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    def construct_dataset(self):
        for i, data_path in enumerate(self.data_pathes):
            buffer = zarr.open(data_path, mode="r")
            num_episodes = get_buffer_size(buffer)
            data_path_discard_num = 0
            data_path_append_num = 0
            for j in tqdm(
                range(self.start_episode, num_episodes), desc=f"Loading {data_path}"
            ):
                if (
                    self.max_episode[i] is not None
                    and data_path_append_num >= self.max_episode[i]
                ):
                    break

                episode = buffer[f"episode_{j}"]
                if "bbox_point_tracking_sequence" not in episode:
                    data_path_discard_num += 1
                    continue
                # load point tracking sequence

                point_tracking_sequence = episode["bbox_point_tracking_sequence"][
                    :
                ].copy()  # (N^2, T, 3)
                point_tracking_sequence = np.clip(        # clip to image size
                    point_tracking_sequence,
                    a_min=0,
                    a_max=self.point_tracking_img_size[0] - 1,
                )
                data_grid_size = np.sqrt(point_tracking_sequence.shape[0]).astype(int) # 32
                # if no sufficient point, simply repeat the first sample
                if data_grid_size < self.grid_size:
                    num_samples = (
                        self.grid_size * self.grid_size
                        - point_tracking_sequence.shape[0]
                    )
                    slack_point_tracking_sequence = repeat(
                        point_tracking_sequence[0], "t c -> s t c", s=num_samples
                    )
                    point_tracking_sequence = np.concatenate(
                        [point_tracking_sequence, slack_point_tracking_sequence], axis=0
                    )
                    data_grid_size = np.sqrt(point_tracking_sequence.shape[0]).astype(
                        int
                    )
                assert data_grid_size == self.grid_size
                point_tracking_sequence = rearrange(
                    point_tracking_sequence,
                    "(N1 N2) T C -> T C N1 N2",
                    N1=self.grid_size,
                )
                point_tracking_sequence = self.sample_point_tracking_frame(
                    point_tracking_sequence,
                    self.frame_sampling_method,
                    self.n_sample_frames + 1,
                )  # (n_sample_frames+1, 3, grid_size, grid_size)
                first_frame_point_uv = point_tracking_sequence[0, :2, :, :].copy()
                first_frame_point_uv = rearrange(
                    first_frame_point_uv, "C N1 N2 -> (N1 N2) C"
                )  # (N^2, 2)
                if self.diff_flow:
                    # (n_sample_frames, 3, grid_size, grid_size)
                    diff_point_tracking_sequence = point_tracking_sequence.copy()[1:]
                    diff_point_tracking_sequence[:, :2, :, :] = (
                        diff_point_tracking_sequence[:, :2, :, :]
                        - point_tracking_sequence[:1, :2, :, :]
                    )
                    # This make sure the flow is in the range of [0,1]
                    diff_point_tracking_sequence[:, :2, :, :] = (
                        diff_point_tracking_sequence[:, :2, :, :]
                        + self.point_tracking_img_size[0]
                    ) / (2 * self.point_tracking_img_size[0])
                    point_tracking_sequence = diff_point_tracking_sequence
                else:
                    # discard the first frame points
                    point_tracking_sequence = point_tracking_sequence[1:]
                    # normalize the flow
                    point_tracking_sequence[:, :2, :, :] = (
                        point_tracking_sequence[:, :2, :, :]
                        / self.point_tracking_img_size[0]
                    )  # assume the image size is square

                point_tracking_sequence = point_tracking_sequence.astype(np.float32)
                global_image = episode["rgb_arr"][0].copy()
                # resize the global image
                global_image = cv2.resize(global_image, self.frame_resize_shape)
                # get u,v under the resize shape
                first_frame_point_uv = (
                    first_frame_point_uv
                    / self.point_tracking_img_size[0]
                    * self.frame_resize_shape[0]
                ).astype(int)
                first_frame_point_uv = np.clip(
                    first_frame_point_uv, a_min=0, a_max=self.frame_resize_shape[0] - 1
                )
                text = episode["task_description"][0]

                data_path_append_num += 1

                self.train_data.append(
                    {
                        "global_image": global_image,
                        "point_tracking_sequence": point_tracking_sequence,
                        "first_frame_point_uv": first_frame_point_uv,
                        "text": text,
                    }
                )
            print(f">> Loaded {data_path_append_num} data from {data_path}")

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        data = self.train_data[idx]
        global_image = data["global_image"]

        global_image = process_image(
            global_image, optional_transforms=self.optional_transforms
        )
        point_tracking_sequence = torch.from_numpy(data["point_tracking_sequence"])
        point_tracking_sequence = self.flow_transforms(point_tracking_sequence)

        point_tracking_sequence = rearrange(point_tracking_sequence, "T C N1 N2 -> T N1 N2 C")
        point_tracking_sequence = np.clip((point_tracking_sequence - offset) / scale, 0, 1)
        point_tracking_sequence = point_tracking_sequence * 2 - 1
        point_tracking_sequence = rearrange(point_tracking_sequence, "T N1 N2 C -> T C N1 N2")
        point_tracking_sequence = torch.from_numpy(point_tracking_sequence)

        first_frame_point_uv = data["first_frame_point_uv"]
        text = data["text"]

        sample = dict(
            global_image=global_image,
            point_tracking_sequence=point_tracking_sequence,
            first_frame_point_uv=first_frame_point_uv,
            text=text,

            global_image_3d=global_image_3d,
            norm_scale=scale,
            norm_offset=offset
        )
        return sample


def calculate_movement(flow, threshold, interval=5):
    """
    计算每个点的相对移动较远的起止时间
    :param flow: 光流数据，形状为 (Frames, N, 2)
    :param threshold: 移动变化的阈值
    :param interval: 计算移动距离的帧间隔
    :return: 每个点的平均起始帧和结束帧
    """
    frames, N, _ = flow.shape
    start_frames = np.full(N, -1)  # 初始化起始帧
    end_frames = np.full(N, -1)    # 初始化结束帧

    # 计算每个点在每一帧的移动距离
    for j in range(frames - interval):
        # 计算当前帧和 interval 后的帧之间的移动距离
        movement = np.linalg.norm(flow[j + interval] - flow[j], axis=1)

        # 找到移动距离大于阈值的点
        significant_movement = movement > threshold

        # 更新起始帧和结束帧
        start_frames[significant_movement] = np.where(start_frames[significant_movement] == -1, j, start_frames[significant_movement])
        end_frames[significant_movement] = j + interval

    # 计算平均起始帧和结束帧
    valid_start_frames = start_frames[start_frames != -1]
    valid_end_frames = end_frames[end_frames != -1]

    avg_start_frame = int(np.mean(valid_start_frames) if valid_start_frames.size > 0 else 0)
    avg_end_frame = int(np.mean(valid_end_frames) if valid_end_frames.size > 0 else -1)

    return avg_start_frame, avg_end_frame

def filter_moving_points(pred_tracks):
    
    lengths = np.zeros((pred_tracks.shape[-2], pred_tracks.shape[0] - 1))
    # 遍历时间帧，计算每个点的光流长度
    for t in range(1, pred_tracks.shape[0]):  # 从第二帧开始
        # 提取前一帧和当前帧的光流
        flow_prev = pred_tracks[t - 1][:, :2]  # 前一帧的光流，形状为 (num_points, 2)
        flow_next = pred_tracks[t][:, :2]      # 当前帧的光流，形状为 (num_points, 2)
        # 计算位置变化
        position_change = flow_next - flow_prev  # 计算位置变化，形状为 (num_points, 2)
        # 计算变化的长度
        lengths[:, t - 1] = np.linalg.norm(position_change, axis=1)

    avg_lengths = np.mean(lengths, axis=1)
    threshold = avg_lengths.mean() # if avg_lengths.mean() * 2 < avg_lengths.max() else avg_lengths.mean() * 2
    # threshold = 0.4
    pred_tracks = pred_tracks[:, avg_lengths > threshold, ...].reshape(pred_tracks.shape[0], -1, pred_tracks.shape[2])
    # 判断N是否是单数 填充最后一个光流
    if pred_tracks.shape[1] % np.sqrt(pred_tracks.shape[1]) != 0:
        # pad to can be sqrt
        pad_num = int((np.ceil(np.sqrt(pred_tracks.shape[1])))**2 - pred_tracks.shape[1])
        pred_tracks = np.concatenate([pred_tracks, pred_tracks[:, -pad_num:, :]], axis=1)
    grid_size = int(np.sqrt(pred_tracks.shape[1]))
    pred_tracks = pred_tracks.reshape(pred_tracks.shape[0], grid_size*grid_size, pred_tracks.shape[-1])

    return pred_tracks


class AnimateFlowDataset_Libero(AnimateFlowDataset):
    def __init__(self, 
        data_pathes,
        n_sample_frames=32,
        grid_size=32,
        frame_sampling_method="uniform",
        frame_resize_shape=[224, 224],
        point_tracking_img_size=[256, 256],
        diff_flow=True,
        optional_transforms=[],
        max_episode=None,
        start_episode=0,
        seed=0,
        preprocess=None,
        proc_cache=None,
        **kwargs):

        # Define the dataset split
        self.set = kwargs['set'] if 'set' in kwargs else 'train'
        self.dataset_name = kwargs['dataset_name'] if 'dataset_name' in kwargs else 'libero'
        # self.training = kwargs['training'] if 'training' in kwargs else 
        # self.max_val = LIBERO_MAX_VALS if self.dataset_name == 'libero' else BRIDGE_MAX_VALS
        # self.min_val = LIBERO_MIN_VALS if self.dataset_name == 'libero' else BRIDGE_MIN_VALS
        if not self.dataset_name == 'mix':
            if self.dataset_name.find('realworld') != -1:
                self.max_val = CONST.BRIDGE_MAX_VALS
                self.min_val = CONST.BRIDGE_MIN_VALS
            else:
                self.max_val = getattr(CONST, f'{self.dataset_name.upper()}_MAX_VALS')
                self.min_val = getattr(CONST, f'{self.dataset_name.upper()}_MIN_VALS')
        self.pc_aug = kwargs['pc_aug'] if 'pc_aug' in kwargs else False

        super().__init__(
            data_pathes,
            n_sample_frames=n_sample_frames,
            grid_size=grid_size,
            frame_sampling_method=frame_sampling_method,
            frame_resize_shape=frame_resize_shape,
            point_tracking_img_size=point_tracking_img_size,
            diff_flow=diff_flow,
            optional_transforms=optional_transforms,
            max_episode=max_episode,
            start_episode=start_episode,
            seed=seed,
            preprocess=preprocess,
            proc_cache=proc_cache,
            **kwargs
        )
        
    def construct_dataset_from_cache(self, proc_cache):
        load_pbar = tqdm(
            range(len(glob.glob(f'{proc_cache}/*'))),
            desc="Loading from cache",
        )
        cprint(f"Construc dataset for split {self.set}", 'green')

        proc_caches = [proc_cache] if not isinstance(proc_cache, (list, omegaconf.listconfig.ListConfig)) else proc_cache
        dataset_names = [self.dataset_name] if not isinstance(self.dataset_name, (list, omegaconf.listconfig.ListConfig)) else self.dataset_name
        data_sets = [self.set] if not isinstance(self.set, (list, omegaconf.listconfig.ListConfig)) else self.set

        print(f"proc_caches: {proc_caches}")
        print(f"dataset_names: {dataset_names}")
        print(f"data_sets: {data_sets}")

        for pi, proc_cache in enumerate(proc_caches):
            print(f">>> Load the dataset from {proc_cache}")

            dataset_name = dataset_names[pi] if len(dataset_names) > 1 else dataset_names[0]
            data_set = data_sets[pi]
            train_data = []

            source = sorted(glob.glob(f'{proc_cache}/*'), key=lambda x: int(x.split('/')[-1].split('_')[-1].split('.')[0].split('_')[0]))
            total_num = len(source)
            
            for i, data_path in enumerate(source):
                load_pbar.update(1)
                train_data.append(data_path)
                if self.max_episode[pi] is not None:
                    if len(train_data) >= self.max_episode[pi]:
                        break

            self.train_data.extend(train_data)
        print(f">> Loaded {len(self.train_data)}")

    def preprocess_dataset_bridge(self, proc_cache):
        bridge_zarr_cahce_dir = 'example_bridge_data/processed_tmp_zarr'
        raw_data_cache_dir = 'example_bridge_data/processed_tmp'
        
        os.makedirs(proc_cache, exist_ok=True)
        # Bound of xyz
        global_max_bound = [-1e9] * 3
        global_min_bound = [1e9] * 3

        num_episodes = len(os.listdir(raw_data_cache_dir))
        data_path_discard_num = 0
        data_path_append_num = 0
        pbar = tqdm(total=num_episodes, desc="Loading data")
        source = os.listdir(raw_data_cache_dir)
        sorted(source, key=lambda x: int(x))
        for i, data_id in enumerate(source):
            pbar.update(1)

            # Contain instruction, rgb, flow
            buffer = zarr.open(os.path.join(bridge_zarr_cahce_dir, f'episode_{data_id}'), mode="r")
            depths_dir = os.path.join(raw_data_cache_dir, data_id, 'depth')
            depth_source = sorted(glob.glob(f'{depths_dir}/*.npz'), key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[0]))
            depths = np.array([np.load(dep)['arr_0'] for dep in depth_source])

            # rgbs = np.array(buffer['rgb_arr'])
            rgbs_dir = os.path.join(raw_data_cache_dir, data_id, 'frames')
            rgb_source = sorted(glob.glob(f'{rgbs_dir}/*.jpg'), key=lambda x: int(x.split('/')[-1].split('.')[0]))
            rgbs = np.array([cv2.imread(rgb) for rgb in rgb_source])

            if not len(depths) == len(rgbs):
                print(f"Error: {len(depths)} != {len(rgbs)}")
                continue

            if not 'co_tracker_sequence_grid' in buffer:
                print(f"Error: No co_tracker_sequence_grid")
                continue
            
            point_tracking_sequence = buffer['co_tracker_sequence_grid'][0][0] # (T, N^2, 3)

            init_rgb = rgbs[0]
            init_depth = depths[0]

            point_tracking_sequence = np.transpose(point_tracking_sequence, (1, 0, 2)) # (N^2, T, 3)
            point_tracking_sequence = np.clip(        # clip to image size
                point_tracking_sequence,
                a_min=0,
                a_max=init_rgb.shape[0] - 1,
            )
            
            # 32*32 > N*N
            N2, T, C = point_tracking_sequence.shape
            N = np.sqrt(N2).astype(int)
            point_tracking_sequence = point_tracking_sequence.reshape(N, N, T, C)
            point_tracking_sequence = point_tracking_sequence[::2, ::2, ...].reshape(-1, T, C)
            data_grid_size = np.sqrt(point_tracking_sequence.shape[0]).astype(int) # 8
            
            point_tracking_sequence = rearrange(
                point_tracking_sequence,
                "(N1 N2) T C -> T C N1 N2",
                N1=self.grid_size,
            )
            point_tracking_sequence = self.sample_point_tracking_frame(
                point_tracking_sequence,
                self.frame_sampling_method,
                self.n_sample_frames + 1,
            )  # (n_sample_frames+1, 3, grid_size, grid_size)

            depths = self.sample_point_tracking_frame(
                depths,
                self.frame_sampling_method,
                self.n_sample_frames + 1,
            ) 
            rgbs = self.sample_point_tracking_frame(
                rgbs,
                self.frame_sampling_method,
                self.n_sample_frames + 1,
            )

            point_tracking_sequence_2d = point_tracking_sequence.copy()
            first_frame_point_uv_2d = point_tracking_sequence_2d[0, :2, :, :].copy()
            first_frame_point_uv_2d = rearrange(
                first_frame_point_uv_2d, "C N1 N2 -> (N1 N2) C"
            )  # (N^2, 2)
            first_frame_point_uv_2d = (
                first_frame_point_uv_2d
                / init_rgb.shape[0]
                * self.frame_resize_shape[0]
            ).astype(int)
            first_frame_point_uv_2d = np.clip(
                first_frame_point_uv_2d, a_min=0, a_max=self.frame_resize_shape[0] - 1
            )
            point_tracking_sequence_2d = point_tracking_sequence_2d[1:]
            # normalize the flow
            point_tracking_sequence_2d[:, :2, :, :] = (
                point_tracking_sequence_2d[:, :2, :, :]
                / init_rgb.shape[0]
            )  # assume the image size is square

            point_tracking_sequence_2d = point_tracking_sequence_2d.astype(np.float32)
            global_image_2d = cv2.resize(init_rgb.copy(), self.frame_resize_shape)

            # Project the uv to 3D space
            point_tracking_sequence = rearrange(
                point_tracking_sequence,
                "T C N1 N2 -> T N1 N2 C",
            )
            from rgbd2pointcloud import flow_to_3d_points_bridge, pixel_to_3d_points_bridge
            first_frame_world_3d = pixel_to_3d_points_bridge(init_depth, init_rgb)
            point_tracking_sequence_3d = flow_to_3d_points_bridge(depths, point_tracking_sequence, rgbs)
            
            point_tracking_sequence_visible = point_tracking_sequence[:, :, :, 2]
            point_tracking_sequence_3d = rearrange(
                point_tracking_sequence_3d,
                "T N1 N2 C -> T C N1 N2",
            )
            point_tracking_sequence_visible = rearrange(
                point_tracking_sequence_visible,
                "T N1 N2 -> T 1 N1 N2",
            )

            first_frame_point_3d = point_tracking_sequence_3d[0, :3, :, :].copy()
            first_frame_point_3d = rearrange(
                first_frame_point_3d, "C N1 N2 -> (N1 N2) C"
            )  # (N^2, 3)

            max_bound = np.max(first_frame_world_3d, axis=0)[:3]
            for i in range(len(global_max_bound)):
                if max_bound[i] > global_max_bound[i]:
                    global_max_bound[i] = max_bound[i]
            min_bound = np.min(first_frame_world_3d, axis=0)[:3]
            for i in range(len(global_min_bound)):
                if min_bound[i] < global_min_bound[i]:
                    global_min_bound[i] = min_bound[i]
                    
            # Drop first frame
            point_tracking_sequence_3d = point_tracking_sequence_3d[1:]
            point_tracking_sequence_visible = point_tracking_sequence_visible[1:]
            video_condition = rgbs[1:]
            video_depth_condition = depths[1:]
            # Resize
            video_condition = np.array([cv2.resize(rgb, (self.frame_resize_shape[0], self.frame_resize_shape[0])) for rgb in video_condition])
            video_depth_condition = np.array([cv2.resize(depth, (self.frame_resize_shape[0], self.frame_resize_shape[0])) for depth in video_depth_condition])

            assert len(point_tracking_sequence_3d) == len(video_condition)

            # point_tracking_sequence_3d = np.concatenate([point_tracking_sequence_3d, point_tracking_sequence_visible], axis=1)

            point_tracking_sequence_3d = point_tracking_sequence_3d.astype(np.float32)
            text = buffer['info'][0]['task_description']

            data_path_append_num += 1

            save_dict = {
                    "global_image": first_frame_world_3d.reshape(-1, 6),
                    "point_tracking_sequence": point_tracking_sequence_3d,
                    "first_frame_point_uv": first_frame_point_3d,
                    "text": text,

                    'global_image_2d': global_image_2d,
                    'point_tracking_sequence_2d':point_tracking_sequence_2d,
                    'first_frame_point_uv_2d':first_frame_point_uv_2d,

                    'rgb_stream': video_condition,
                    'depth_stream': video_depth_condition,
                }
            torch.save(save_dict, f"{proc_cache}/episode_{data_id}.pt")
    
            print(">>> Max Bound: ", global_max_bound)
            print(">>> Min Bound: ", global_min_bound)

    def smooth_optical_flow(self, flow, sigma=2):
        from scipy.ndimage import gaussian_filter
        smoothed_flow = gaussian_filter(flow, sigma=(sigma, 0, 0), mode='nearest')
        return smoothed_flow

    def _viz_flow_3d(self, flow_points, first_frame_world_3d, save_path):
        def linear_interpolate(points, num_interpolated=10):
            interpolated_points = []
            for i in range(len(points) - 1):
                p1 = points[i]
                p2 = points[i + 1]
                
                # Generate interpolated points between p1 and p2
                for t in range(num_interpolated):
                    alpha = t / (num_interpolated - 1)  # Normalized parameter
                    interpolated_point = (1 - alpha) * p1 + alpha * p2
                    interpolated_points.append(interpolated_point)
            return np.array(interpolated_points)
        
        def write_ply(filename, points, colors, additional_points, additional_colors):
            colors = (colors * 255).astype(np.uint8)
            additional_colors = additional_colors.astype(np.uint8)
            additional_points_num = additional_points.reshape(-1, 3).shape[0]
            total_pts_num = additional_points_num + points.shape[0]
            # Create a PLY file
            with open(filename, 'w') as ply_file:
                # Write the PLY header
                ply_file.write("ply\n")
                ply_file.write("format ascii 1.0\n")
                ply_file.write(f"element vertex {total_pts_num}\n")
                ply_file.write("property float x\n")
                ply_file.write("property float y\n")
                ply_file.write("property float z\n")
                ply_file.write("property uchar red\n")
                ply_file.write("property uchar green\n")
                ply_file.write("property uchar blue\n")
                ply_file.write("end_header\n")

                # Write the original point cloud data
                for i in range(points.shape[0]):
                    x, y, z = points[i]
                    r, g, b = colors[i]
                    ply_file.write(f"{x} {y} {z} {r} {g} {b}\n")

                # Write the additional points data
                for i in range(additional_points.shape[0]):
                    for j in range(additional_points.shape[1]):
                        x, y, z = additional_points[i, j]
                        r, g, b = additional_colors[i, j]
                        ply_file.write(f"{x} {y} {z} {r} {g} {b}\n")

        # F, N, C
        flow_points = flow_points[..., :3].reshape(flow_points.shape[0], -1 ,3)
        interpolated_flow_points = []
        interpolated_flow_colors = []

        from matplotlib import cm
        import colorsys
        color_map = cm.get_cmap("jet") 
        for pt in range(flow_points.shape[1]):
            flow = flow_points[:,pt]
            interpolated_flow = []
            interpolated_colors = []
            for i in range(flow.shape[0] - 1):
                p1 = flow[i]
                p2 = flow[i + 1]
                interpolated_points = linear_interpolate(np.array([p1, p2]), num_interpolated=10)
                
                # Same timestep use the same color
                # interpolated_colors.extend([color_map[i]] * interpolated_points.shape[0])
                
                # Same point use the same color    
                num_points = flow_points.shape[1]  
                color = np.array(color_map(pt / max(1, float(num_points - 1)))[:3]) * 255
                color_alpha = 1
                hsv = colorsys.rgb_to_hsv(color[0], color[1], color[2])
                color = colorsys.hsv_to_rgb(hsv[0], hsv[1] * color_alpha, hsv[2])
                interpolated_colors.extend([np.array(list(color))] * interpolated_points.shape[0])
                
                interpolated_flow.extend(interpolated_points)
            interpolated_flow_colors.append(np.array(interpolated_colors))
            interpolated_flow_points.append(np.array(interpolated_flow))

        scene_points = first_frame_world_3d[..., :3].reshape(-1, 3)
        scene_colors = first_frame_world_3d[..., 3:].reshape(-1, 3)
        
        write_ply(save_path, scene_points, scene_colors, np.array(interpolated_flow_points), np.array(interpolated_flow_colors))

    def _normalize_point_cloud(self, point_cloud):
        # [-1, 1]
        # min_vals = point_cloud.min(axis=0)
        # max_vals = point_cloud.max(axis=0)
        
        # Dataset constant
        max_vals = self.max_val
        min_vals = self.min_val
        
        scale = max_vals - min_vals
        offset = min_vals
        
        normalized_point_cloud = (point_cloud - offset) / scale
        normalized_point_cloud = normalized_point_cloud * 2 - 1
        
        return normalized_point_cloud, scale, offset

    def __getitem__(self, idx):
        sample_data_p = self.train_data[idx]
        data = torch.load(sample_data_p)
        global_image = data["global_image"]

        norm_xyz, norm_scale, norm_offset = self._normalize_point_cloud(global_image[..., :3])
        global_image[..., :3] = norm_xyz

        point_tracking_sequence_3d = rearrange(data["point_tracking_sequence"], "T C N1 N2 -> T N1 N2 C")[..., :3]

        # Downsample N to self.grid_size
        N = point_tracking_sequence_3d.shape[1]
        if N > self.grid_size:
            sample_interval = N // self.grid_size
            point_tracking_sequence_3d = point_tracking_sequence_3d[:, ::sample_interval, ::sample_interval, :]
            
        norm_point_tracking_sequence_3d = np.clip((point_tracking_sequence_3d - norm_offset) / norm_scale, 0, 1)
        norm_point_tracking_sequence_3d = norm_point_tracking_sequence_3d * 2 - 1
        point_tracking_sequence = rearrange(norm_point_tracking_sequence_3d, "T N1 N2 C -> T C N1 N2")

        first_frame_point_uv = data["first_frame_point_uv"]

        first_frame_point_uv = (first_frame_point_uv - norm_offset) / norm_scale
        # For 3D pos sincos: Norm to 0-128
        first_frame_point_uv[..., -1] =  first_frame_point_uv[..., -1] * POS3D_GRID_SIZE[-1]
        first_frame_point_uv[..., -1] = np.clip(first_frame_point_uv[..., -1], a_min=0, a_max=POS3D_GRID_SIZE[-1]-1)
        
        text = data["text"]

        global_image = global_image.reshape(-1, 6)

        raw_3d_first_frame_point_uv = first_frame_point_uv.copy()
        
        ''' Here we use the 2D xy init pts and target flow'''
        episode_idx = sample_data_p.split('/')[-1]

        first_frame_point_uv_2d = data["first_frame_point_uv_2d"]

        point_tracking_sequence_2d = torch.from_numpy(data["point_tracking_sequence_2d"])
        
        point_tracking_sequence_2d = self.flow_transforms(point_tracking_sequence_2d).numpy()

        # Replace the z channel of the 2D flow with the 3D flow
        point_tracking_sequence[:, :2, ...] = point_tracking_sequence_2d[:, :2, ...]
        first_frame_point_uv[:, :2] = first_frame_point_uv_2d

        global_image = torch.from_numpy(global_image.astype(np.float32))

        sample = dict(
            global_image=global_image,
            point_tracking_sequence=torch.from_numpy(point_tracking_sequence.astype(np.float32)),
            first_frame_point_uv=first_frame_point_uv.astype(np.float32),
            text=text.lower(),
            norm_scale=norm_scale,
            norm_offset=norm_offset
        )

        global_image_2d = data['global_image_2d']
        global_image_2d = cv2.resize(global_image_2d, self.frame_resize_shape)
        global_image_2d = process_image(
            global_image_2d, optional_transforms=self.optional_transforms
        )
        sample["global_image_2d"] = global_image_2d

        del data

        return sample
