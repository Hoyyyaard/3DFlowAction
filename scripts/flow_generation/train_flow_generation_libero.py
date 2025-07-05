import copy
import gc
import math
import os
from accelerate import DistributedDataParallelKwargs
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_P2P_DISABLE"] = "1"
import time
import random
import tqdm
import hydra
import json
import numpy as np
import torch
import torch.nn.functional as F
import wandb
# from memory_profiler import profile
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
# @profile
def train(cfg: DictConfig):
    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        # kwargs_handlers=[ddp_kwargs]
    )
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

    ckpt_save_dir = os.path.join(output_dir, "checkpoints")
    eval_save_dir = os.path.join(output_dir, "evaluations")
    if accelerator.is_main_process:
        wandb.init(project=cfg.project_name, name=cfg.entity_name)
        # wandb.config.update(OmegaConf.to_container(cfg))
        accelerator.print("Logging dir", output_dir)
        # state_save_dir = os.path.join(output_dir, "state")
        os.makedirs(ckpt_save_dir, exist_ok=True)
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
    if cfg.unet_pretrained_model_path is not None:
        ckpt = torch.load(cfg.unet_pretrained_model_path, map_location="cpu")
        for k, v in list(ckpt.items()):
            ckpt[k.replace("unet.", "")] = ckpt.pop(k)
        msg = unet.load_state_dict(
            ckpt, strict=False
        )
        if accelerator.is_local_main_process:
            print("### Load pretrained model for unet: ", msg)

    # freeze parameters of models to save more memory
    vae.requires_grad_(False)

    if not hasattr(cfg, "use_vae"):
        os.environ["USE_VAE"] = "True"  
    elif cfg.use_vae:
        os.environ["USE_VAE"] = "True" if cfg.use_vae else "False"
    else:
        os.environ["USE_VAE"] = "False"

    if not hasattr(cfg, "use_clip"):
        os.environ["USE_CLIP"] = "False"  
    elif cfg.use_clip:
        os.environ["USE_CLIP"] = "True" if cfg.use_clip else "False"
    else:
        os.environ["USE_CLIP"] = "False"

    print(f">>> USE_VAE: ", os.environ['USE_VAE'])

    text_encoder.requires_grad_(False)
    
    fully_finetune = cfg.training.get("fully_finetune", False)

    if cfg.training.load_pretrain_weight and not fully_finetune:
        unet.requires_grad_(False)
        unet_lora_config = LoraConfig(
            r=cfg.lora.rank,
            lora_alpha=cfg.lora.rank,
            init_lora_weights="gaussian",
            target_modules=r"(?!.*motion_modules).*?(to_k|to_q|to_v|to_out\.0)",
        )
        unet = get_peft_model(unet, unet_lora_config)
        print(">> Before enable motion modules")
        unet.print_trainable_parameters()
    else:
        print(">> Fully trainable")
        unet.requires_grad_(True)

    for name, param in unet.named_parameters():
        if "motion_modules." in name:
            param.requires_grad = True

    weight_dtype = torch.float32
    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device)
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)
    if cfg.training.mixed_precision == "fp16":
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(unet, dtype=torch.float32)

    # Enable xformers
    if cfg.training.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    # Enable gradient checkpointing
    if cfg.training.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # AnimateFlow
    model = AnimateFlow3D(unet=unet, **cfg.animateflow_kwargs)
    trainable_parameters = filter(lambda p: p.requires_grad, model.parameters())
    
    trained_para_list = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            # print(name)
            trained_para_list.append(name)
    # Write to file
    if accelerator.is_main_process:
        with open(f'{output_dir}/train_parameters.json', "w") as f:
            json.dump(trained_para_list, f)

    if hasattr(cfg.training, "resume_training"):
        if cfg.training.resume_training:
            # Auot resume the lastest checkpoint
            ckpts = os.listdir(ckpt_save_dir)
            ckpts = [int(ckpt.split("_")[1]) for ckpt in ckpts]
            ckpt = max(ckpts)
            ckpt_path = os.path.join(ckpt_save_dir, f"epoch_{ckpt}", f"epoch_{ckpt}.ckpt")
            ckpt = torch.load(ckpt_path, map_location="cpu")
            for k, v in list(ckpt.items()):
                nk = k.replace("unet.", "unet.base_model.model.")
                if not k.find("motion_modules") != -1:       
                    nk = nk.replace("to_q.weight", "to_q.base_layer.weight")
                    nk = nk.replace("to_k.weight", "to_k.base_layer.weight")
                    nk = nk.replace("to_v.weight", "to_v.base_layer.weight")
                    nk = nk.replace("to_out.0.weight", "to_out.0.base_layer.weight")
                    nk = nk.replace("to_q.bias", "to_q.base_layer.bias")
                    nk = nk.replace("to_k.bias", "to_k.base_layer.bias")
                    nk = nk.replace("to_v.bias", "to_v.base_layer.bias")
                    nk = nk.replace("to_out.0.bias", "to_out.0.base_layer.bias")
                ckpt[nk] = ckpt.pop(k)
            msg = model.load_state_dict(ckpt, strict=False)
            print("### Resume training from checkpoint: ", msg)
        time.sleep(5)
    
    # Fine-tune the model
    if hasattr(cfg.training, "flow_pretrained_model_path"):
        if cfg.training.flow_pretrained_model_path is not None:
            ckpt = torch.load(cfg.training.flow_pretrained_model_path, map_location="cpu")
            if not fully_finetune:
                for k, v in list(ckpt.items()):
                    nk = k.replace("unet.", "unet.base_model.model.")
                    if not k.find("motion_modules") != -1:       
                        nk = nk.replace("to_q.weight", "to_q.base_layer.weight")
                        nk = nk.replace("to_k.weight", "to_k.base_layer.weight")
                        nk = nk.replace("to_v.weight", "to_v.base_layer.weight")
                        nk = nk.replace("to_out.0.weight", "to_out.0.base_layer.weight")
                        nk = nk.replace("to_q.bias", "to_q.base_layer.bias")
                        nk = nk.replace("to_k.bias", "to_k.base_layer.bias")
                        nk = nk.replace("to_v.bias", "to_v.base_layer.bias")
                        nk = nk.replace("to_out.0.bias", "to_out.0.base_layer.bias")
                    ckpt[nk] = ckpt.pop(k)
            msg = model.load_state_dict(ckpt, strict=False)
            # print("### Fine-tune the model from checkpoint: ", msg)
            if accelerator.is_main_process:
                # print("### Fine-tune the model from checkpoint: ", msg)
                for name in msg.missing_keys:
                    if not name.find("lora") != -1:
                        assert False, f"### {name} not in the pretrained model"

    optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        model.parameters(),
        lr=cfg.optimizer.learning_rate,
        betas=(cfg.optimizer.adam_beta1, cfg.optimizer.adam_beta2),
        weight_decay=cfg.optimizer.adam_weight_decay,
        eps=cfg.optimizer.adam_epsilon,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.training.gradient_accumulation_steps
    )
    if cfg.training.max_train_steps is None:
        max_train_steps = cfg.training.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        cfg.training.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.training.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )
    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.training.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        max_train_steps = cfg.training.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    cfg.training.num_train_epochs = math.ceil(
        max_train_steps / num_update_steps_per_epoch
    )
    # Train!
    total_batch_size = (
        cfg.training.batch_size
        * accelerator.num_processes
        * cfg.training.gradient_accumulation_steps
    )

    accelerator.print("***** Running training *****")
    accelerator.print(f"  Num examples = {len(dataset)}")
    accelerator.print(f"  Num Epochs = {cfg.training.num_train_epochs}")
    accelerator.print(
        f"  Instantaneous batch size per device = {cfg.training.batch_size}"
    )
    accelerator.print(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    accelerator.print(
        f"  Gradient Accumulation steps = {cfg.training.gradient_accumulation_steps}"
    )
    accelerator.print(f"  Total optimization steps = {max_train_steps}")
    total_time = 0
    # for epoch in range(cfg.training.num_train_epochs):
    for epoch in tqdm.tqdm(range(cfg.training.num_train_epochs), desc="Epoch"):   
        model.train()
        epoch_loss = []
        pabr = tqdm.tqdm(total=len(train_dataloader))
        for step, batch in enumerate(train_dataloader):
            pabr.update(1)
            if cfg.cfg_random_null_text:
                batch["text"] = [
                    name if random.random() > cfg.cfg_random_null_text_ratio else ""
                    for name in batch["text"]
                ]
            ### >>>> Training >>>> ###
            global_image = batch["global_image"]
            point_uv = batch["first_frame_point_uv"]

            ############ Random mask point_uv as all zero ##########
            # if random.random() < 0.5:
            #     point_uv = point_uv * 0

            # Convert flows to latent space
            point_tracking_sequence = batch["point_tracking_sequence"].to(
                dtype=weight_dtype
            )
            video_length = point_tracking_sequence.shape[1]
            global_image_2d = batch["global_image_2d"] if os.environ.get("USE_CLIP") == "True" else None
            video_conditions = None
            if not video_conditions is None:
                if random.random() > 0.3:
                    video_conditions = torch.zeros_like(video_conditions)

            with torch.no_grad():
                if os.environ.get("USE_VAE") == "True":
                    point_tracking_sequence = rearrange(
                        point_tracking_sequence, "b f c h w -> (b f) c h w"
                    )
                    latents = vae.encode(point_tracking_sequence).latent_dist
                    latents = latents.sample()
                    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length) # c = 4
                    latents = latents * 0.18215

                else:
                    # [-1, 1]
                    latents = point_tracking_sequence
                    # Concat 1 to the channel dimension
                    latents = torch.cat([latents, torch.ones_like(latents[:, :, :1])], dim=2)
                    latents = rearrange(latents, "b f c h w -> b c f h w")
                
            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]

            # Sample a random timestep for each video
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=latents.device,
            )
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            with torch.no_grad():
                prompt_ids = tokenizer(
                    batch["text"],
                    max_length=tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.to(latents.device)
                encoder_hidden_states = text_encoder(prompt_ids)[0]

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                raise NotImplementedError
            else:
                raise ValueError(
                    f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                )
            
            start_time = time.time()
            model_pred = model(
                noisy_latents, timesteps, encoder_hidden_states, global_image, point_uv, global_image_2d, video_conditions
            )
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            # Backpropagate
            accelerator.backward(loss)
            duration = time.time() - start_time
            total_time += duration
            if accelerator.sync_gradients:
                params_to_clip = trainable_parameters
                accelerator.clip_grad_norm_(params_to_clip, cfg.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            # logging
            epoch_loss.append(loss.item())

            if accelerator.is_main_process and step % cfg.training.log_frequency == 0:
                wandb.log(
                    {
                        "loss": loss.item(),
                        "step": (epoch+1)*len(train_dataloader) + step,
                    }
                )
                print(f"######### [Epoch] {epoch}, [Step] {step}, [Loss] {loss.item()} [Memory] {torch.cuda.max_memory_reserved(0) / (1024**3)}GB [Time] {duration}s|{total_time}s ########")

        if accelerator.is_main_process:
            wandb.log(
                {
                    "epoch loss": np.mean(epoch_loss),
                    "epoch": epoch,
                }
            )
        if epoch % cfg.training.ckpt_frequency == 0 or cfg.debug:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                ckpt_save_path = os.path.join(ckpt_save_dir, f"epoch_{epoch}")
                os.makedirs(ckpt_save_path, exist_ok=True)
                ckpt_model = accelerator.unwrap_model(model)
                if cfg.training.load_pretrain_weight:
                    model_to_save = copy.deepcopy(ckpt_model)
                    if not fully_finetune:
                        model_to_save.unet = model_to_save.unet.merge_and_unload()
                    accelerator.save(
                        model_to_save.state_dict(),
                        os.path.join(ckpt_save_path, f"epoch_{epoch}.ckpt"),
                    )
                    del model_to_save
                    torch.cuda.empty_cache()
                    gc.collect()
                    accelerator.print(f"Saved checkpoint at epoch {epoch}.")
                else:
                    accelerator.save(
                        ckpt_model.state_dict(),
                        os.path.join(ckpt_save_path, f"epoch_{epoch}.ckpt"),
                    )
                    accelerator.print(f"Saved checkpoint at epoch {epoch}.")

        if (epoch % cfg.evaluation.eval_frequency == 0 and epoch != 0)or cfg.debug:
            accelerator.wait_for_everyone()
            accelerator.print(f"Evaluate at epoch {epoch}.")
            if accelerator.is_local_main_process:
                eval_model = accelerator.unwrap_model(model)
                evaluation_save_path = os.path.join(eval_save_dir, f"epoch_{epoch}")
                os.makedirs(eval_save_dir, exist_ok=True)
                pipeline = AnimationFlowPipeline3D(
                    vae=vae,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    model=eval_model,
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
                    wandb_log=False,
                )
                # it should be fine but set the timestep back to the training timesteps
                noise_scheduler.set_timesteps(
                    cfg.noise_scheduler_kwargs.num_train_timesteps
                )
                eval_model.train()


if __name__ == "__main__":
    train()
