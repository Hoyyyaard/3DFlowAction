import numpy as np
import torch
from torch import nn
from transformers import CLIPVisionModel
import argparse
from im2flow2act.common.utility.model import freeze_model
import json
import time
import os
import sys
import accelerate
from torch.nn.functional import interpolate
from im2flow2act.flow_generation.constant import *
import torch.nn.functional as F

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature**omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

def posemb_sincos_3d(h, w, d, dim, temperature: int = 10000, dtype=torch.float32):
    z, y, x = torch.meshgrid(torch.arange(d), torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 6) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 6) / (dim //  - 1)
    omega = 1.0 / (temperature**omega)

    z = z.flatten()[:, None] * omega[None, :]
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos(), z.sin(), z.cos()), dim=1)
    return pe.type(dtype)

class AnimateFlow(nn.Module):
    def __init__(
        self,
        unet,
        clip_model,
        global_image_size,
        freeze_visual_encoder=True,
        global_condition_type="cls_token",
        emb_dim=768,
        other_class=False,
    ) -> None:
        super().__init__()
        if other_class:
            return
        self.freeze_visual_encoder = freeze_visual_encoder
        self.global_condition_type = global_condition_type
        self.visual_encoder = CLIPVisionModel.from_pretrained(clip_model)
        # if self.freeze_visual_encoder:
        #     freeze_model(self.visual_encoder)

        if not self.freeze_visual_encoder and (
            self.global_condition_type != "cls_token"
            or self.global_condition_type != "all"
        ):
            self.visual_encoder.vision_model.post_layernorm = nn.Identity()

        self.unet = unet

        with torch.no_grad():
            reference_image = torch.zeros(1, 3, *global_image_size)
            reference_last_hidden = self.visual_encoder(reference_image)[0]
            token_num, clip_dim = reference_last_hidden.shape[-2:]

        self.vit_grid_size = np.sqrt(token_num).astype(int)
        self.global_projection_head = nn.Linear(clip_dim, emb_dim)
        self.local_projection_head = nn.Linear(clip_dim, emb_dim)
        # self.local_projection_head = nn.Linear(768, emb_dim)

        zero_module(self.global_projection_head)
        zero_module(self.local_projection_head)

        pos_2d = posemb_sincos_2d(*global_image_size, clip_dim).reshape(
            *global_image_size, clip_dim
        )
        self.register_buffer("pos_2d", pos_2d)

    def prepare_condition(self, global_image, point_uv):
        B, _, H, W = global_image.shape
        vision_output = self.visual_encoder(global_image)
        last_hidden_states = vision_output["last_hidden_state"]
        vit_patch_embedding = last_hidden_states[:, 1:]
        # get global feature
        if self.global_condition_type == "cls_token":
            vit_cls_token = vision_output["pooler_output"]
            global_condition = self.global_projection_head(vit_cls_token).unsqueeze(
                1
            )  # (B,1,C)
        elif self.global_condition_type == "patch":
            global_condition = self.global_projection_head(
                vit_patch_embedding
            )  # (B,P^2,C)
        elif self.global_condition_type == "all":
            vit_cls_token = vision_output["pooler_output"]
            global_condition = self.global_projection_head(
                torch.cat([vit_cls_token.unsqueeze(1), vit_patch_embedding], axis=1)
            )  # (B,1+P^2,C)
        else:
            raise ValueError(
                f"global_condition_type {self.global_condition_type} not supported"
            )
        uv_discriptor = self.pos_2d[point_uv[:, :, 1], point_uv[:, :, 0]]
        local_condition = self.local_projection_head(uv_discriptor)  # (B,N,C)
        return global_condition, local_condition # (bs, 257, 768). (bs, 1024, 768)

    def forward(
        self,
        noisy_latents,
        timesteps,
        encoder_hidden_states,
        global_image,
        point_uv,
    ):
        global_condition, local_condition = self.prepare_condition(
            global_image, point_uv
        )
        # concat with text feature
        encoder_hidden_states = torch.cat(
            [
                encoder_hidden_states,
                global_condition,   #  Test experiment
                local_condition,
            ],
            axis=1,
        )  # (B,77+1+N,C)
        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        return model_pred

    def load_model(self, path):
        print("Loading complete model...")
        self.load_state_dict(torch.load(path))
        print(">> loaded model")

class SelfAttentionLayer(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8):
        super(SelfAttentionLayer, self).__init__()
        
        # Multi-head attention layer
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, embed_dim)
        
        # Self-attention
        attn_output, _ = self.attention(x, x, x)  # Query, Key, Value 都是 x
        x = self.layer_norm1(x + attn_output)  # Residual connection and layer normalization
        
        # Feed-forward network
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + ff_output)  # Residual connection and layer normalization
        
        return x

class AnimateFlow3D(AnimateFlow):
    def __init__(
        self,
        unet,
        clip_model,
        global_image_size,
        freeze_visual_encoder=True,
        global_condition_type="cls_token",
        emb_dim=768,
    ) -> None:
        # parent parent class init
        super(AnimateFlow3D, self).__init__(
            unet,
            clip_model,
            global_image_size,
            freeze_visual_encoder,
            global_condition_type,
            emb_dim,
            other_class=True,
        )
        self.freeze_visual_encoder = freeze_visual_encoder
        self.global_condition_type = global_condition_type
        self.training = True

        # Clip
        print(">>> Use CLIP as visual encoder")
        self.visual_encoder = CLIPVisionModel.from_pretrained(clip_model)
        ''' Concat to 2D features '''
        pos_dim = 768

        pos_3d = posemb_sincos_3d(*(global_image_size+[POS3D_GRID_SIZE[-1]]), 72).reshape(
            *(global_image_size+[POS3D_GRID_SIZE[-1]]), 72
        )
        self.register_buffer("pos_3d", pos_3d)
        
        if self.freeze_visual_encoder:
            freeze_model(self.visual_encoder)

        self.unet = unet

        self.global_projection_head = nn.Linear(1024, emb_dim)
        self.local_projection_head = nn.Linear(72, emb_dim)

        zero_module(self.global_projection_head)
        zero_module(self.local_projection_head)

    def _extract_non_overlapping_patches(self, tensor, patch_size):
        """
        从给定的张量中提取不重叠的补丁，并调整输出形状为 (H1, W1, N)。

        参数:
        tensor: 输入张量，形状为 (B, H, W)
        patch_size: 补丁的大小 (H1, W1)

        返回:
        patches: 形状为 (H1, W1, N) 的张量
        """
        B, H, W = tensor.shape
        H1, W1 = patch_size
        # 计算补丁的数量
        num_patches_h = H // H1
        num_patches_w = W // W1
        # 提取补丁
        patches = tensor.unfold(1, H1, H1).unfold(2, W1, W1)
        # 调整形状为 (B, num_patches_h, num_patches_w, H1, W1)
        patches = patches.contiguous().view(B, num_patches_h, num_patches_w, H1, W1)
        # 转置为 (B, H1, W1, N) 形式
        patches = patches.permute(0, 3, 4, 1, 2).contiguous().view(B, H1*W1, -1)

        return patches


    def prepare_condition(self, global_image, point_uv, global_image_2d=None, video_conditions=None):

        # Clip2D
        B, _, H, W = global_image_2d.shape
        
        vision_output = self.visual_encoder(global_image_2d)
        last_hidden_states = vision_output["last_hidden_state"]
        vit_patch_embedding = last_hidden_states[:, 1:] # (B,P^2,C)
        grid_size = np.sqrt(vit_patch_embedding.shape[1]).astype(int)
        vit_patch_embedding = vit_patch_embedding.view(B, grid_size, grid_size, -1) # (B,P,P,C)
        vit_cls_token = vision_output["pooler_output"] # (B,C)
        
        vit_patch_embedding = vit_patch_embedding.view(B, grid_size**2, -1) # (B,P2,C)
        visual_tokens = torch.cat([vit_cls_token.unsqueeze(1), vit_patch_embedding], dim=1)  # (B, 1+128, C)

        uv_discriptor = self.pos_3d[point_uv[:, :, 1].int(), point_uv[:, :, 0].int(), point_uv[:, :, 2].int()]

        global_condition = self.global_projection_head(visual_tokens)
        local_condition = self.local_projection_head(uv_discriptor)  # (B,N,C)

        return global_condition, local_condition # (bs, 257, 768). (bs, 1024, 768)

    def forward(
        self,
        noisy_latents,
        timesteps,
        encoder_hidden_states,
        global_image,
        point_uv,
        global_image_2d=None,
        video_conditions=None,
    ):
        global_condition, local_condition = self.prepare_condition(
            global_image, point_uv, global_image_2d, video_conditions
        )
        # concat with text feature
        encoder_hidden_states = torch.cat(
            [
                encoder_hidden_states,
                global_condition,   #  Test experiment
                local_condition,
            ],
            axis=1,
        )  # (B,77+1+N,C)


        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        return model_pred
