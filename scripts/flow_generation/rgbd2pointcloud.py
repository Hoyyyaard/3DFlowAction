import numpy as np
import cv2
import open3d
import os
from im2flow2act.flow_generation.constant import *

MAX_DEPTH = 1.5

def get_intrinsics(H,W):
    """
    Intrinsics for a pinhole camera model.
    Assume fov of 55 degrees and central principal point.
    """
    f = 0.5 * W / np.tan(0.5 * 55 * np.pi / 180.0)
    cx = 0.5 * W
    cy = 0.5 * H
    return f, cx, cy

def flow_to_3d_points_bridge(depth_map, flow, rgbs):

    grid_size = flow.shape[1]
    flow = flow.reshape(flow.shape[0], -1, flow.shape[-1])[..., :2]
    depth_width = depth_map.shape[-1]
    depth_height = depth_map.shape[-2]
    fx, cx, cy = get_intrinsics(depth_height, depth_width)
    fy = fx
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
            z = depth_map[i, v, u]
            projected_3D[i, pt, 0] = (u - cx) * z / fx
            projected_3D[i, pt, 1] = (v - cy) * z / fy
            projected_3D[i, pt, 2] = z
    projected_3D = projected_3D.reshape(N, grid_size, grid_size, 3)

    return projected_3D

def pixel_to_3d_points_bridge(depth_image, rgb_image):
    
    depth_image[depth_image >= MAX_DEPTH] = MAX_DEPTH

    img_width = depth_image.shape[1]
    img_height = depth_image.shape[0]
    focal_len_in_pixel, cx, cy = get_intrinsics(img_height, img_width)

    intrinsics = np.eye(3)
    intrinsics[0,0] = focal_len_in_pixel
    intrinsics[1,1] = focal_len_in_pixel
    intrinsics[0,2] = cx
    intrinsics[1,2] = cy

    # Get the shape of the depth image
    H, W = depth_image.shape

    # Create a grid of (x, y) coordinates corresponding to each pixel in the image
    i, j = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')

    # Unpack the intrinsic parameters
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Convert pixel coordinates to normalized camera coordinates
    z = depth_image
    x = (i - cx) * z / fx
    y = (j - cy) * z / fy

    # Stack the coordinates to form (H, W, 3)
    camera_coordinates = np.stack((x, y, z), axis=-1)

    # Reshape to (H*W, 3) for matrix multiplication
    camera_coordinates = camera_coordinates.reshape(-1, 3)
    
    world_coordinates = camera_coordinates.reshape(H, W, 3)

    # Concatenate the world coordinates with the original image color
    world_coordinates = np.concatenate((world_coordinates, rgb_image/255), axis=-1).reshape(-1, 6)

    return world_coordinates

