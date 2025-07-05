import copy
import gc
import math
import os
from accelerate import DistributedDataParallelKwargs
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_P2P_DISABLE"] = "1"
import random
import tqdm
import hydra
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model
from transformers import CLIPTextModel, CLIPTokenizer

from im2flow2act.flow_generation.animatediff.models.unet import UNet3DConditionModel
from im2flow2act.flow_generation.AnimateFlow import AnimateFlow, AnimateFlow3D
from im2flow2act.flow_generation.AnimationFlowPipeline import AnimationFlowPipeline, AnimationFlowPipeline3D
from im2flow2act.flow_generation.inference import inference_from_dataset, inference_from_dataset_3d

#     backend="nccl", init_method="env://", timeout=datetime.timedelta(seconds=5400)
# )


def cast_training_params(model, dtype=torch.float32):
    if not isinstance(model, list):
        model = [model]
    for m in model:
        for param in m.parameters():
            # only upcast trainable parameters into fp32
            if param.requires_grad:
                param.data = param.to(dtype)


@hydra.main(
    version_base=None,
    config_path="../../config/flow_generation",
    config_name="train_flow_generation",
)
def train(cfg: DictConfig):
    output_dir = HydraConfig.get().runtime.output_dir

    dataset = hydra.utils.instantiate(cfg.dataset)
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        shuffle=cfg.training.shuffle,
        pin_memory=cfg.training.pin_memory,
        persistent_workers=cfg.training.persistent_workers,
        drop_last=cfg.training.drop_last,
    )
    evalulation_datasets = [
        hydra.utils.instantiate(dataset) for dataset in cfg.evaluation.datasets
    ]

    # os.makedirs(state_save_dir, exist_ok=True)
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDIMScheduler(**cfg.noise_scheduler_kwargs)

    vae = AutoencoderKL.from_pretrained(cfg.vae_pretrained_model_path)
    tokenizer = CLIPTokenizer.from_pretrained(
        cfg.pretrained_model_path, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        cfg.pretrained_model_path, subfolder="text_encoder"
    )
    unet = UNet3DConditionModel.from_pretrained_2d(
        cfg.pretrained_model_path,
        subfolder="unet",
        unet_additional_kwargs=cfg.unet_additional_kwargs,
        load_pretrain_weight=cfg.training.load_pretrain_weight,
    )

    # freeze parameters of models to save more memory
    # vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    weight_dtype = torch.float32
    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to('cuda')
    vae.to('cuda')
    text_encoder.to('cuda')

    if not hasattr(cfg, "use_vae"):
        os.environ["USE_VAE"] = "True"  
    elif cfg.use_vae:
        os.environ["USE_VAE"] = "True" if cfg.use_vae else "False"

    if not hasattr(cfg, "use_clip"):
        os.environ["USE_CLIP"] = "False"  
    elif cfg.use_clip:
        os.environ["USE_CLIP"] = "True" if cfg.use_clip else "False"

    model = AnimateFlow3D(unet=unet, **cfg.animateflow_kwargs)
    model.to('cuda')
    
    exp_dir = cfg.evaluation.epoch_dir
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    eval_save_dir = os.path.join(exp_dir, "evaluations")

    for epoch in tqdm.tqdm(os.listdir(ckpt_dir), desc="Epochs"): 

        ckpt_path = os.path.join(ckpt_dir, epoch, f"{epoch}.ckpt")
        print("Evaluating model at", ckpt_path)
        msg = model.load_state_dict(torch.load(ckpt_path))
        print(msg)
        evaluation_save_path = os.path.join(eval_save_dir, f"{epoch}")
        os.makedirs(eval_save_dir, exist_ok=True)

        pipeline = AnimationFlowPipeline3D(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            model=model,
            scheduler=noise_scheduler,
        )
        inference_from_dataset_3d(
            pipeline=pipeline,
            evalulation_datasets=evalulation_datasets,
            evaluation_save_path=evaluation_save_path,
            num_inference_steps=cfg.evaluation.num_inference_steps,
            num_samples=cfg.evaluation.num_samples,
            guidance_scale=cfg.evaluation.guidance_scale,
            viz_n_points=cfg.evaluation.viz_n_points,
            draw_line=cfg.evaluation.draw_line,
            wandb_log=True,
        )



if __name__ == "__main__":
    train()
