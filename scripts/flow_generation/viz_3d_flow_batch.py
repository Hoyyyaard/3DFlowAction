import numpy as np
import torch
from matplotlib import cm
import os
from glob import glob
from tqdm import tqdm
from einops import rearrange
import colorsys
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
dbscan = DBSCAN(eps=0.05, min_samples=5)  # eps 是邻域的半径，min_samples 是形成一个簇所需的最小点数

import sys
sys.path.append('< YOUR PATH TO Grounded-SAM-2 >')
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2
SAM2_CHECKPOINT = "< YOUR PATH TO sam2.1_hiera_large.pt >"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device='cuda')
sam2_predictor = SAM2ImagePredictor(sam2_model)

def compute_transformation(A, B):
    """
    计算从 A 到 B 的刚性变换（旋转和平移）
    :param A: 物体上的点 (N1, 3)
    :param B: 运动变化后的点 (N1, 3)
    :return: 旋转矩阵 R 和平移向量 t
    """
    # 计算 A 和 B 的中心
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    # 中心化点
    AA = A - centroid_A
    BB = B - centroid_B
    # 计算协方差矩阵
    H = AA.T @ BB
    # SVD 分解
    U, S, Vt = np.linalg.svd(H)
    # 计算旋转矩阵 R
    R = Vt.T @ U.T
    # 确保 R 是一个有效的旋转矩阵
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    # 计算平移向量 t
    t = centroid_B - R @ centroid_A
    return R, t

def reconstruct_point_cloud(P, A, B, radius=0.1):
    """
    重建物体点云
    :param P: 场景点云 (N, 6)
    :param A: 物体上的点 (N1, 3)
    :param B: 运动变化后的点 (N1, 3)
    :param radius: 聚合的半径
    :return: 重建的点云
    """
    # 提取场景点云的坐标
    points = P[:, :3]  # (N, 3)
    # 创建 KDTree 以加速邻近点查找
    tree = cKDTree(points)
    # 计算从 A 到 B 的变换
    R, t = compute_transformation(A, B)
    # 存储重建的点云
    reconstructed_points = []
    # 对于每个点 A，查找周围的点
    for a in A:
        # 查找在 radius 范围内的点
        indices = tree.query_ball_point(a, radius)
        
        # 获取这些点
        nearby_points = points[indices]
        nearby_colors = P[indices, 3:]

        # 计算聚合后的点（可以是均值、最大值等）
        for ni,nearby_point in enumerate(nearby_points):
            transformed_point = R @ nearby_point + t
            reconstructed_points.append(np.concatenate([transformed_point, nearby_colors[ni]]))
    
    # 将重建的点云转换为 numpy 数组
    reconstructed_points = np.array(reconstructed_points)
    
    return reconstructed_points

def reconstruct_point_cloud_object_mask(P, A, B, radius=0.1, object_points=None):
    """
    重建物体点云
    :param P: 场景点云 (N, 6)
    :param A: 物体上的点 (N1, 3)
    :param B: 运动变化后的点 (N1, 3)
    :param radius: 聚合的半径
    :return: 重建的点云
    """
    # 提取场景点云的坐标
    points = P[:, :3]  # (N, 3)
    # 创建 KDTree 以加速邻近点查找
    tree = cKDTree(points)
    # 计算从 A 到 B 的变换
    R, t = compute_transformation(A, B)
    # 存储重建的点云
    reconstructed_points = []
    # 对于每个点 A，查找周围的点
    for a in object_points:

        # 获取这些点
        nearby_points = a[...,:3].reshape(-1, 3)
        nearby_colors = a[...,3:].reshape(-1, 3)

        # 计算聚合后的点（可以是均值、最大值等）
        for ni,nearby_point in enumerate(nearby_points):
            transformed_point = R @ nearby_point + t
            reconstructed_points.append(np.concatenate([transformed_point, nearby_colors[ni]]))
    
    # 将重建的点云转换为 numpy 数组
    reconstructed_points = np.array(reconstructed_points)
    
    return reconstructed_points


def write_ply(filename, points, colors, additional_points, additional_colors):
    """
    Write points and colors to a PLY file.
    
    :param filename: The name of the output PLY file.
    :param points: A numpy array of shape (M, 3) representing the original point cloud.
    :param colors: A numpy array of shape (M, 3) representing the colors of the original points.
    :param additional_points: A numpy array of shape (N, Frame, 3) representing the additional points.
    :param additional_colors: A numpy array of shape (N, Frame, 3) representing the colors of the additional points.
    """
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

def get_colors(start_color, end_color, num_colors):
    """
    Generate a list of colors transitioning from start_color to end_color.
    
    :param start_color: The starting color (RGB tuple).
    :param end_color: The ending color (RGB tuple).
    :param num_colors: The number of colors to generate.
    :return: A numpy array of shape (num_colors, 3) representing the colors.
    """
    # Create a colormap from start_color to end_color
    cmap = cm.colors.LinearSegmentedColormap.from_list("custom_cmap", [start_color, end_color])
    
    # Generate colors
    colors = cmap(np.linspace(0, 1, num_colors))
    
    # Convert to uint8
    return (colors[:, :3] * 255).astype(np.uint8)

def linear_interpolate(points, num_interpolated=10):
    """
    Linearly interpolate between given points.
    
    :param points: A numpy array of shape (N, 3) representing the points to interpolate.
    :param num_interpolated: Number of interpolated points to generate between each pair of points.
    :return: Interpolated points.
    """
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

Exp_dir = '<YOUR FINETUNE PROJECT PATH>/evaluations/'
for epoch in tqdm(os.listdir(Exp_dir), desc='Epochs'):
    for dataset in os.listdir(os.path.join(Exp_dir, epoch)):
        epoch_dir = os.path.join(Exp_dir, epoch, dataset)
        save_dir = os.path.join(epoch_dir, 'viz')
        # if os.path.exists(save_dir):
        #     continue
        os.makedirs(save_dir, exist_ok=True)
        # use os.listdir instead of glob to avoid hidden files
        source = [os.path.join(epoch_dir, file) for file in os.listdir(epoch_dir) if file.endswith('.npy')]
        # source = glob(f'{epoch_dir}/*.npy')
        for file in tqdm(source, desc='Batches'):
            buffer = np.load(os.path.join(epoch_dir, file), allow_pickle=True).item()
            scene_points = buffer["global_image"][:,:3]
            if 'norm_scale' in buffer:
                scene_points = (scene_points + 1) / 2
                scene_points = scene_points * buffer['norm_scale'] + buffer['norm_offset']
            print(scene_points.min(axis=0), scene_points.max(axis=0))
            
            scene_colors = buffer["global_image"][:, 3:][..., ::-1]
            
            if "text" in buffer.keys() and "point_uv" in buffer.keys():
                text = buffer["text"]
                init_points = buffer["point_uv"]
                init_2d_points = buffer["point_uv_2d"]
                image = buffer['global_image_2d']
                ############### get object bbox ###############
                # get max_x, min_x, max_y, min_y from init_2d_points
                obj_max_w = np.max(init_2d_points[:, 0])
                obj_min_w = np.min(init_2d_points[:, 0])
                obj_max_h = np.max(init_2d_points[:, 1])
                obj_min_h = np.min(init_2d_points[:, 1])

                # Scale the bbox for 15%
                obj_max_w = obj_max_w * 1.05
                obj_min_w = obj_min_w * 0.95
                obj_max_h = obj_max_h * 1.05
                obj_min_h = obj_min_h * 0.95

                bbox = [obj_min_w, obj_min_h, obj_max_w, obj_max_h]
                # Draw the bbox on the image
                import cv2
                image = cv2.resize(image, (256,256))
                # cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                # cv2.imwrite('bbox.png', image)
                # import pdb; pdb.set_trace()
                # final_bboxs.append(bbox)
                sam2_predictor.set_image(image)
                masks, scores, logits = sam2_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=np.array([bbox]),
                    multimask_output=False,
                )
                object_mask = masks[0]
                object_mask = (object_mask > 0).astype(bool)

                # Draw the mask on the image
                # mask_image = np.zeros_like(image)
                # mask_image[object_mask] = [0, 255, 0]
                # image = cv2.addWeighted(image, 0.5, mask_image, 0.5, 0)
                # cv2.imwrite('bbox.png', image)
                # assert False
                object_pcd = buffer["global_image"].copy().reshape(256, 256, 6)
                object_pcd = object_pcd[object_mask]
                object_pcd[..., 3:] = object_pcd[..., 3:][..., ::-1]
                
                # 使用 DBSCAN 进行聚类
                # init_labels = dbscan.fit_predict(init_points) + 1
                # labels 中的 -1 表示噪声点
                # Get the most frequent label and not -1
                # indexes = np.argmax(np.bincount(init_labels))

                init_colors = np.ones_like(init_points) * [0, 1, 0]
                # scene_points = np.concatenate([scene_points, init_points], axis=0)
                # scene_colors = np.concatenate([scene_colors, init_colors], axis=0)
                # scene_points = init_points
                # scene_colors = init_colors
                init_colors = (init_colors * 255).astype(np.uint8)
                total_pts_num = init_points.shape[0]
                # Create a PLY file
                with open(os.path.join(save_dir, f'{file.split("/")[-1].split(".")[0]}_{text}_init.ply'), 'w') as ply_file:
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
                    for i in range(init_points.shape[0]):
                        x, y, z = init_points[i]
                        r, g, b = init_colors[i]
                        ply_file.write(f"{x} {y} {z} {r} {g} {b}\n")
            else:
                text = ''

            # GT Flow
            flow_points = rearrange(buffer["gt_flow"], 'F C N1 N2 -> F N1 N2 C')
            flow_points = flow_points[..., :3].reshape(flow_points.shape[0], -1 ,3)

            # Downsample
            flow_points = flow_points[:, ::4]

            if 'norm_scale' in buffer:
                flow_points = (flow_points + 1) / 2
                flow_points = flow_points * buffer['norm_scale'] + buffer['norm_offset']
            print(flow_points.reshape(-1,3).min(axis=0), flow_points.reshape(-1,3).max(axis=0))
            
            interpolated_flow_points = []
            interpolated_flow_colors = []
            
            # color_map = get_colors([1, 1, 0], [1, 0, 0], flow_points.shape[0])
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

            # Calculate the final object pose
            # A = buffer["point_uv"]
            A = flow_points[0]
            B = flow_points[-1]
            # Filter points has small distance in flow[0] and flow[-1] and return indexes
            first_flow = flow_points[0]
            last_flow = flow_points[-1]
            distance = np.linalg.norm(first_flow - last_flow, axis=-1)

            # 去除噪声点
            # take 20-80% longest points
            # indexes = np.where(distance > np.percentile(distance, 70))[0]
            mask1 = distance > np.percentile(distance, 10)
            mask2 = distance < np.percentile(distance, 90)
            indexes = np.where(mask1 & mask2)[0]

            A = A[indexes]
            B = B[indexes]

            mask = object_mask.reshape(-1)
            raw_scene_points = scene_points.copy()
            raw_scene_colors = scene_colors.copy()
            scene_points = scene_points[~mask]
            scene_colors = scene_colors[~mask]
            P = np.concatenate([scene_points, scene_colors], axis=-1)
            # reconstructed_points = reconstruct_point_cloud(P, A, B)
            reconstructed_points = reconstruct_point_cloud_object_mask(P, A, B, object_points=object_pcd)
            if len(reconstructed_points) > 0:
            #     scene_points = np.concatenate([scene_points, reconstructed_points[:,:3]], axis=0)
            #     scene_colors = np.concatenate([scene_colors, reconstructed_points[:,3:]], axis=0)
                reconstructed_points[:, 3:] = (reconstructed_points[:, 3:] * 255).astype(np.uint8)
                # Append end keypoints
                # end_pts = np.concatenate([B, (np.ones_like(B) * [0, 255, 0]).astype(np.uint8)], axis=-1)
                # reconstructed_points = np.concatenate([reconstructed_points, end_pts], axis=0)
                scene_pcd = np.concatenate([scene_points, scene_colors*255], axis=-1)
                reconstructed_points = np.concatenate([reconstructed_points, scene_pcd], axis=0)
                with open(os.path.join(save_dir, f'{file.split("/")[-1].split(".")[0]}_{text}_trans_gt.ply'), 'w') as ply_file:
                    # Write the PLY header
                    ply_file.write("ply\n")
                    ply_file.write("format ascii 1.0\n")
                    ply_file.write(f"element vertex {reconstructed_points.shape[0]}\n")
                    ply_file.write("property float x\n")
                    ply_file.write("property float y\n")
                    ply_file.write("property float z\n")
                    ply_file.write("property uchar red\n")
                    ply_file.write("property uchar green\n")
                    ply_file.write("property uchar blue\n")
                    ply_file.write("end_header\n")

                    # Write the original point cloud data
                    for i in range(reconstructed_points.shape[0]):
                        x, y, z = reconstructed_points[i, :3]
                        r, g, b = reconstructed_points[i, 3:].astype(np.uint8)
                        ply_file.write(f"{x} {y} {z} {r} {g} {b}\n")

            write_ply(os.path.join(save_dir, f'{file.split("/")[-1].split(".")[0]}_{text}.ply'), raw_scene_points, raw_scene_colors, np.array(interpolated_flow_points), np.array(interpolated_flow_colors))

            # Predicted Flow
            flow_points = rearrange(buffer["pred_flow"], 'C F N1 N2 -> F N1 N2 C')
            flow_points = flow_points[..., :3].reshape(flow_points.shape[0], -1 ,3)

            # Downsample
            flow_points = flow_points[:, ::4]

            if 'norm_scale' in buffer:
                flow_points = (flow_points + 1) / 2
                flow_points = flow_points * buffer['norm_scale'] + buffer['norm_offset']
            print(flow_points.reshape(-1,3).min(axis=0), flow_points.reshape(-1,3).max(axis=0))
            # assert False
            interpolated_flow_points = []
            interpolated_flow_colors = []
            
            # color_map = get_colors([1, 1, 0], [1, 0, 0], flow_points.shape[0])
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

            # Calculate the final object pose
            # A = buffer["point_uv"]
            A = flow_points[0]
            B = flow_points[-1]
            # Filter points has small distance in flow[0] and flow[-1] and return indexes
            first_flow = flow_points[0]
            last_flow = flow_points[-1]
            distance = np.linalg.norm(first_flow - last_flow, axis=-1)

            # labels 中的 -1 表示噪声点
            # A = A[init_labels == indexes]
            # B = B[init_labels == indexes]

            # take 20-80% longest points
            # indexes = np.where(distance > np.percentile(distance, 70))[0]
            mask1 = distance > np.percentile(distance, 10)
            mask2 = distance < np.percentile(distance, 90)
            indexes = np.where(mask1 & mask2)[0]

            A = A[indexes]
            B = B[indexes]

            # mask = object_mask.reshape(-1)
            # import pdb; pdb.set_trace()
            # scene_points = scene_points[mask]
            # scene_colors = scene_colors[mask]
            P = np.concatenate([scene_points, scene_colors], axis=-1)
            reconstructed_points = reconstruct_point_cloud_object_mask(P, A, B, object_points=object_pcd)
            if len(reconstructed_points) > 0:
            #     scene_points = np.concatenate([scene_points, reconstructed_points[:,:3]], axis=0)
            #     scene_colors = np.concatenate([scene_colors, reconstructed_points[:,3:]], axis=0)
                reconstructed_points[:, 3:] = (reconstructed_points[:, 3:] * 255).astype(np.uint8)
                # Append end keypoints
                # end_pts = np.concatenate([B, (np.ones_like(B) * [0, 255, 0]).astype(np.uint8)], axis=-1)
                # reconstructed_points = np.concatenate([reconstructed_points, end_pts], axis=0)
                scene_pcd = np.concatenate([scene_points, scene_colors*255], axis=-1)
                reconstructed_points = np.concatenate([reconstructed_points, scene_pcd], axis=0)
                with open(os.path.join(save_dir, f'{file.split("/")[-1].split(".")[0]}_{text}_trans_red.ply'), 'w') as ply_file:
                    # Write the PLY header
                    ply_file.write("ply\n")
                    ply_file.write("format ascii 1.0\n")
                    ply_file.write(f"element vertex {reconstructed_points.shape[0]}\n")
                    ply_file.write("property float x\n")
                    ply_file.write("property float y\n")
                    ply_file.write("property float z\n")
                    ply_file.write("property uchar red\n")
                    ply_file.write("property uchar green\n")
                    ply_file.write("property uchar blue\n")
                    ply_file.write("end_header\n")

                    # Write the original point cloud data
                    for i in range(reconstructed_points.shape[0]):
                        x, y, z = reconstructed_points[i, :3]
                        r, g, b = reconstructed_points[i, 3:].astype(np.uint8)
                        ply_file.write(f"{x} {y} {z} {r} {g} {b}\n")
            
            write_ply(os.path.join(save_dir, f'{file.split("/")[-1].split(".")[0]}_{text}_pred.ply'), raw_scene_points, raw_scene_colors, np.array(interpolated_flow_points), np.array(interpolated_flow_colors))
