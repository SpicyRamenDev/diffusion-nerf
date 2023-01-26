# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


import os
import logging as log
from tqdm import tqdm
import random
import pandas as pd
import torch
from lpips import LPIPS
from torch.utils.tensorboard import SummaryWriter
from wisp.trainers import BaseTrainer, log_metric_to_wandb, log_images_to_wandb
from wisp.ops.image import write_png, write_exr
from wisp.ops.image.metrics import psnr, lpips, ssim
from wisp.core import Rays, RenderBuffer

import wandb
import numpy as np
from tqdm import tqdm
from PIL import Image

import time

from torch.utils.data import DataLoader
from wisp.datasets import default_collate
from kaolin.render.camera import Camera
from wisp.ops.raygen.raygen import generate_centered_pixel_coords, generate_pinhole_rays

from .utils import sample, spherical_to_cartesian, l2_normalize, sample_polar, sample_spherical_uniform, get_rotation_matrix
import math


from PIL import Image
import numpy as np

class SDSTrainer(BaseTrainer):

    def __init__(self, *args,
                 diffusion=None,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.aug_bg_color = self.extra_args["aug_bg_color"]

        self.diffusion = diffusion
        if self.diffusion is not None:
            self.init_text_embeddings()

        if self.dataset is not None and hasattr(self.dataset, "data"):
            self.scene_state.graph.cameras = self.dataset.data.get("cameras", dict())

        def scheduler_function(epoch):
            iteration = epoch * self.iterations_per_epoch
            if iteration < self.extra_args["warmup_iterations"]:
                return self.extra_args["init_lr"] + (1 - self.extra_args["init_lr"]) * iteration / self.extra_args["warmup_iterations"]
            else:
                t = (iteration - self.extra_args["warmup_iterations"]) / (self.max_iterations - self.extra_args["warmup_iterations"])
                return self.extra_args["end_lr"] + 0.5 * (1 - self.extra_args["end_lr"]) * (1 + math.cos(t * math.pi))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, scheduler_function)

    def init_optimizer(self):
        params_dict = {name: param for name, param in self.pipeline.nef.named_parameters()}

        params = []
        decoder_params = []
        decoder_bg_params = []
        decoder_normal_params = []
        grid_params = []
        rest_params = []

        for name in params_dict:
            if name == 'decoder_background':
                decoder_bg_params.append(params_dict[name])

            if name == 'decoder_normal':
                decoder_normal_params.append(params_dict[name])

            elif 'decoder' in name:
                decoder_params.append(params_dict[name])

            elif 'grid' in name:
                grid_params.append(params_dict[name])

            else:
                rest_params.append(params_dict[name])

        params.append({"params": decoder_normal_params,
                       "lr": self.lr,
                       "weight_decay": self.weight_decay})

        params.append({"params": decoder_bg_params,
                       "lr": self.lr * 0.1,
                       "weight_decay": self.weight_decay})

        params.append({"params": decoder_params,
                       "lr": self.lr,
                       "weight_decay": self.weight_decay})

        params.append({"params": grid_params,
                       "lr": self.lr * self.grid_lr_weight})

        params.append({"params": rest_params,
                       "lr": self.lr})

        self.optimizer = self.optim_cls(params, **self.optim_params)

    def init_dataloader(self):
        self.iterations_per_epoch = self.extra_args['iterations_per_epoch']
        self.train_data_loader = [None] * self.iterations_per_epoch

        if self.dataset is not None:
            self.train_gt_data_loader = DataLoader(self.dataset,
                                                   batch_size=self.batch_size,
                                                   collate_fn=default_collate,
                                                   shuffle=True, pin_memory=True,
                                                   num_workers=self.extra_args['dataloader_num_workers'])

    def init_text_embeddings(self):
        prompt, negative_prompt = self.extra_args["prompt"], self.extra_args["negative_prompt"]

        view_prompts = {
            '': ('', ''),
            'overhead': ('overhead view', ''),
            'front': ('front view', 'back view'),
            'side': ('side view', ''),
            'back': ('back view', 'front view')
        }
        self.text_embeddings = dict()
        for view, (positive, negative) in view_prompts.items():
            positive_cond = prompt if positive == '' else prompt + ', ' + positive
            negative_cond = negative_prompt  # if negative == '' else negative_prompt + ', ' + negative
            text_embedding = self.diffusion.get_text_embedding(positive_cond, negative_cond)
            self.text_embeddings[view] = text_embedding

    def get_text_embeddings(self, azimuth, polar):
        if not self.extra_args["use_view_prompt"]:
            return self.text_embeddings['']
        view_prompt = ''
        azimuth = (azimuth - 45) % 360
        if polar < 30:
            view_prompt = 'overhead'
        elif azimuth < 90:
            view_prompt = 'front'
        elif azimuth < 180:
            view_prompt = 'side'
        elif azimuth < 270:
            view_prompt = 'back'
        else:
            view_prompt = 'side'

        text_embeddings = self.text_embeddings[view_prompt]

        return text_embeddings, view_prompt

    def pre_step(self):
        super().pre_step()

        if self.extra_args["prune_every"] > -1 and self.iteration > 0 and self.iteration % self.extra_args["prune_every"] == 0:
            self.pipeline.nef.prune()

    def iterate(self):
        if self.is_optimization_running:
            if self.is_first_iteration():
                self.pre_training()
            iter_start_time = time.time()
            if self.iteration == 1:
                self.begin_epoch()
            self.iteration += 1
            self.pre_step()
            self.step()
            self.post_step()
            iter_end_time = time.time()
            if self.iteration > self.iterations_per_epoch:
                self.end_epoch()
                if not self.is_any_iterations_remaining():
                    self.post_training()
            self.scene_state.optimization.elapsed_time += iter_end_time - iter_start_time

    def init_log_dict(self):
        super().init_log_dict()
        self.log_dict['rgb_loss'] = 0.0

    def generate_camera_rays(self, camera, resolution):
        ray_grid = generate_centered_pixel_coords(resolution, resolution,
                                                  resolution, resolution,
                                                  device='cuda')
        return generate_pinhole_rays(camera, ray_grid).reshape(resolution, resolution, 3)

    def step_gt(self, data, phase=0):
        rays = data['rays'].to(self.device).squeeze(0)
        img_gts = data['imgs'].to(self.device).squeeze(0)

        loss = 0

        # with torch.cuda.amp.autocast():
        if True:
            nef_parameters = dict(phase=phase,
                                  use_light=False)
            rb = self.pipeline(rays=rays,
                               lod_idx=None,
                               channels=["rgb"],
                               bg_color=self.extra_args["bg_color"],
                               nef_parameters=nef_parameters)

            rgb_loss = torch.abs(rb.rgb[..., :3] - img_gts[..., :3])
            rgb_loss = rgb_loss.sum(-1)
            rgb_loss = rgb_loss.sum()
            rgb_loss *= 1.0 / rays.shape[0]
            rgb_loss *= 1.0 / len(self.train_gt_data_loader)

            loss += rgb_loss

            self.scaler.scale(loss).backward()

        return loss.item()

    def prepare_novel_view(self, phase=0):
        is_uniform = random.random() < 0.5
        camera_dir, azimuth, polar = sample_spherical_uniform(azimuth_range=self.extra_args["azimuth_range"],
                                                              polar_range=self.extra_args["polar_range"],
                                                              uniform=is_uniform)
        camera_offset = (2 * torch.rand(3) - 1) * self.extra_args["camera_offset"]
        camera_distance = sample(self.extra_args["camera_distance_range"])
        focal_length_multiplier = sample(self.extra_args["focal_length_multiplier_range"])
        camera_up = l2_normalize(torch.tensor([0., 1., 0.]) + torch.randn(3) * self.extra_args["camera_up_std"])
        look_at = torch.randn(3) * self.extra_args["look_at_std"]

        resolution = self.extra_args["resolutions"][phase]
        camera_coords = camera_dir * camera_distance  # + camera_offset
        focal_length = 0.5 * resolution * (camera_distance - 0.) * focal_length_multiplier
        # focal_length = resolution * focal_length_multiplier

        centroid = torch.zeros(3)
        # centroid = torch.tensor([0., 0.2, 0.])
        camera_coords += centroid
        look_at += centroid

        camera = Camera.from_args(
            eye=camera_coords,
            at=look_at,
            up=camera_up,
            focal_x=focal_length,
            width=resolution, height=resolution,
            near=max(camera_distance-1.74, 0.0),
            far=camera_distance+1.74,
            dtype=torch.float32,
            device='cuda'
        )

        light_dir, light_azimuth, light_polar = sample_spherical_uniform(polar_range=(0, 60), uniform=True)
        camera_rot = get_rotation_matrix(azimuth, polar)
        light_dir = camera_rot @ light_dir

        one_eps = 1 - torch.finfo(torch.float32).eps
        dot_prod = (light_dir * camera_dir).sum(-1, keepdim=True)
        dot_prod = dot_prod.clamp(-one_eps, one_eps)
        print( torch.arccos(dot_prod) * 180.0 / torch.pi)

        light_distance = sample(self.extra_args["light_distance_range"])
        light = light_dir * light_distance
        light = light.to(self.device)

        text_embeddings, view = self.get_text_embeddings(azimuth, polar)

        rays = self.generate_camera_rays(camera, resolution)
        rays = rays.reshape(resolution ** 2, -1)

        return dict(rays=rays, camera=camera, light=light, text_embeddings=text_embeddings, azimuth=azimuth, polar=polar, view=view)

    def sample_rays_gaussian(self, rays, num_samples):
        idx = torch.multinomial(self.pgrid, num_samples)
        output = rays[idx].contiguous()
        return output

    def get_novel_view_render_parameters(self, phase=0):
        if self.total_iterations < self.extra_args["albedo_steps"]:
            shading = 'albedo'
            ambient_ratio = 1.0
        else:
            rand = random.random()
            if rand < 0.8:
                if rand < 0.4:
                    ambient_ratio = 0.1
                    shading = 'lambertian'
                else:
                    ambient_ratio = 0.1
                    shading = 'textureless'
            else:
                shading = 'albedo'
                ambient_ratio = 1.0
            # shading = 'lambertian'
            # ambient_ratio = 0.1 + 0.9 * (random.random() ** 0.5)
        return dict(shading=shading, ambient_ratio=ambient_ratio)

    def get_diffusion_parameters(self, phase=0):
        if phase == 0:
            weight_type = 'constant'
            min_ratio = 0.40
            max_ratio = 0.98
        elif phase == 1:
            weight_type = 'linear'
            min_ratio = 0.02
            max_ratio = 0.98
        else:
            weight_type = 'quadratic'
            min_ratio = 0.02
            max_ratio = 0.60
        guidance_scale = self.extra_args["guidance_scale"]

        return dict(weight_type=weight_type, min_ratio=min_ratio, max_ratio=max_ratio, guidance_scale=guidance_scale)

    def step_novel_view(self, phase=0):
        lod_idx = None

        render_parameters = self.get_novel_view_render_parameters(phase)
        diffusion_parameters = self.get_diffusion_parameters(phase)

        scene = self.prepare_novel_view(phase=phase)
        rays = scene['rays']
        light = scene['light']
        camera = scene['camera']
        text_embeddings = scene['text_embeddings']
        if render_parameters['shading'] == 'textureless':
            bg_color_value = torch.rand(1, device='cuda').repeat(3)
        else:
            bg_color_value = torch.rand(3, device='cuda')
        bg_color_value *= 0.25

        if self.total_iterations % 50 == 0:
            self.scene_state.graph.cameras["sample_camera"] = camera
            print(f"Camera: {scene['azimuth']}, {scene['polar']}, {scene['view']}")

        nef_parameters = dict(phase=phase,
                              use_light=True,
                              light=light,
                              perturb_density=False,
                              perturb_density_std=0.1,
                              **render_parameters)

        bg_color = random.choice(['noise', 'decoder'])
        bg_color = self.aug_bg_color
        trace_parameters = dict(lod_idx=lod_idx,
                                channels=["rgb"],
                                total_variation_loss=self.extra_args["total_variation_loss"],
                                orientation_loss=self.extra_args["orientation_loss"],
                                entropy_loss=self.extra_args["entropy_loss"],
                                bg_color=bg_color,
                                bg_color_value=bg_color_value,
                                nef_parameters=nef_parameters)
        render_only_trace_parameters = dict(lod_idx=lod_idx,
                                            channels=["rgb"],
                                            bg_color=bg_color,
                                            bg_color_value=bg_color_value,
                                            nef_parameters=nef_parameters)

        diffusion_parameters = dict(text_embeddings=text_embeddings, use_decoder=False, **diffusion_parameters)

        total_loss_value = 0

        if self.total_iterations >= self.extra_args["reg_warmup_iterations"]:
            reg_factor = 1
        else:
            reg_factor = self.extra_args["reg_init_lr"] + (1 - self.extra_args["reg_init_lr"]) * self.total_iterations / self.extra_args["reg_warmup_iterations"]

        if 0 < self.extra_args["render_batch"] < rays.shape[0]:
            image_batches = []
            rays_cache = []
            for ray_pack in rays.split(self.extra_args["render_batch"]):
                # with torch.cuda.amp.autocast():
                if True:
                    with torch.no_grad():
                        rb = self.pipeline(rays=ray_pack, out_rays=rays_cache, **render_only_trace_parameters)
                        image_batches.append(rb.rgb[..., :3])
                        del rb

            image = torch.cat(image_batches, dim=0)
            image = image.reshape(1, camera.width, camera.height, 3)
            image = image.permute(0, 3, 1, 2).contiguous()

            diffusion_grad = self.diffusion.step(image=image, scaler=self.scaler, **diffusion_parameters)
            diffusion_grad = diffusion_grad.reshape(3, -1).contiguous()

            for i, (ray_pack, ray_cache) in enumerate(zip(rays.split(self.extra_args["render_batch"]), rays_cache)):
                # with torch.cuda.amp.autocast():
                if True:
                    rb = self.pipeline(rays=ray_pack, raymarch_results=ray_cache, **trace_parameters)

                pack_image = rb.rgb[..., :3]
                pack_image = pack_image.reshape(self.extra_args["render_batch"], 3)
                pack_image = pack_image.permute(1, 0).contiguous()
                start, end = i * self.extra_args["render_batch"], (i + 1) * self.extra_args["render_batch"]
                pack_diffusion_grad = diffusion_grad[..., start:end]
                pack_diffusion_loss = (pack_diffusion_grad * pack_image).sum(0)
                pack_diffusion_loss = pack_diffusion_loss.sum()
                pack_diffusion_loss = self.extra_args["diffusion_loss"] * pack_diffusion_loss

                pack_reg_loss = 0
                orientation_loss = rb.orientation_reg.sum()
                # opacity_loss = torch.sqrt(rb.alpha ** 2 + 0.01).sum()
                alphas = rb.alpha.clamp(1e-5, 1 - 1e-5)
                opacity_loss = (-alphas * torch.log2(alphas) - (1 - alphas) * torch.log2(1 - alphas)).sum()
                entropy_loss = rb.entropy.sum()
                pack_reg_loss += reg_factor * self.extra_args["orientation_loss"] * orientation_loss
                pack_reg_loss += self.extra_args["opacity_loss"] * opacity_loss
                pack_reg_loss += self.extra_args["entropy_loss"] * entropy_loss
                pack_reg_loss *= 1.0 / rays.shape[0]
                pack_reg_loss *= 1.0 / self.extra_args["minibatch_size"]

                pack_loss = pack_diffusion_loss + pack_reg_loss
                pack_loss *= 1.0 / self.extra_args["rgb_loss"]
                self.scaler.scale(pack_loss).backward()
                total_loss_value += pack_loss.item()

                del rb
        else:
            # with torch.cuda.amp.autocast():
            if True:
                rb = self.pipeline(rays=rays, **trace_parameters)

            image = rb.rgb[..., :3]
            image = image.reshape(1, camera.width, camera.height, 3)
            image = image.permute(0, 3, 1, 2).contiguous()

            diffusion_grad = self.diffusion.step(image=image, scaler=self.scaler, **diffusion_parameters)

            diffusion_loss = (diffusion_grad * image).sum(1)
            diffusion_loss = diffusion_loss.sum()
            diffusion_loss = self.extra_args["diffusion_loss"] * diffusion_loss
            diffusion_loss *= 1.0 / self.extra_args["minibatch_size"]
            diffusion_loss *= 64 * 64

            reg_loss = 0
            if self.extra_args["orientation_loss"] > 0:
                orientation_loss = rb.orientation_loss.sum()
                reg_loss += reg_factor * self.extra_args["orientation_loss"] * orientation_loss
            if self.extra_args["opacity_loss"] > 0:
                # opacity_loss = torch.sqrt(rb.alpha ** 2 + 0.01).sum()
                alphas = rb.alpha.clamp(1e-5, 1 - 1e-5)
                opacity_loss = (-alphas * torch.log2(alphas) - (1 - alphas) * torch.log2(1 - alphas)).sum()
                reg_loss += self.extra_args["opacity_loss"] * opacity_loss
            if self.extra_args["entropy_loss"] > 0:
                entropy_loss = rb.entropy.sum()
                reg_loss += self.extra_args["entropy_loss"] * entropy_loss
            reg_loss *= 1.0 / rays.shape[0]
            reg_loss *= 1.0 / self.extra_args["minibatch_size"]

            loss = diffusion_loss + reg_loss
            loss *= 1.0 / self.extra_args["rgb_loss"]
            self.scaler.scale(loss).backward()
            total_loss_value += loss.item()

        print("Iteration:", self.total_iterations,
              "phase:", phase,
              "Learning rate:", self.optimizer.param_groups[0]['lr'],
              "Diffusion loss: ", total_loss_value)

        if self.total_iterations % 10 == 0:
            # image = diffusion_grad.reshape(1, 3, camera.width, camera.height)
            # image *= 0.5 / image.abs().mean()
            # image = 0.5 * (image + 1)
            image = image.clamp(0, 1)
            image = image.permute(0, 2, 3, 1).contiguous()
            pil_image = Image.fromarray((image[0].detach().cpu().numpy() * 255).astype(np.uint8))
            pil_image = pil_image.resize((512, 512))
            pil_image.save("image.png")

        return total_loss_value

    def step(self):
        if self.total_iterations < self.max_iterations * self.extra_args["phase_ratios"][0]:
            phase = 0
        elif self.total_iterations < self.max_iterations * self.extra_args["phase_ratios"][1]:
            phase = 1
        else:
            phase = 2

        self.optimizer.zero_grad()

        loss = 0
        for i in range(self.extra_args["minibatch_size"]):
            loss += self.step_novel_view(phase=phase)
        if self.dataset is not None:
            for data in iter(self.train_gt_data_loader):
                loss += self.step_gt(data, phase=phase)

        self.log_dict['total_loss'] += loss
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.scheduler is not None:
            self.scheduler.step(self.total_iterations / self.iterations_per_epoch)

    def begin_epoch(self):
        super().begin_epoch()

    def log_cli(self):
        log_text = 'EPOCH {}/{}'.format(self.epoch, self.max_epochs)
        log_text += ' | total loss: {:>.3E}'.format(self.log_dict['total_loss'] / len(self.train_data_loader))
        log_text += ' | rgb loss: {:>.3E}'.format(self.log_dict['rgb_loss'] / len(self.train_data_loader))

        log.info(log_text)

    def render_final_view(self, num_angles, camera_distance):
        angles = np.pi * 0.1 * np.array(list(range(num_angles + 1)))
        x = -camera_distance * np.sin(angles)
        y = self.extra_args["camera_origin"][1]
        z = -camera_distance * np.cos(angles)
        for d in range(self.extra_args["num_lods"]):
            out_rgb = []
            for idx in tqdm(range(num_angles + 1), desc=f"Generating 360 Degree of View for LOD {d}"):
                log_metric_to_wandb(f"LOD-{d}-360-Degree-Scene/step", idx, step=idx)
                out = self.renderer.shade_images(
                    self.pipeline,
                    f=[x[idx], y, z[idx]],
                    t=self.extra_args["camera_lookat"],
                    fov=self.extra_args["camera_fov"],
                    lod_idx=d,
                    camera_clamp=self.extra_args["camera_clamp"]
                )
                out = out.image().byte().numpy_dict()
                if out.get('rgb') is not None:
                    log_images_to_wandb(f"LOD-{d}-360-Degree-Scene/RGB", out['rgb'].T, idx)
                    out_rgb.append(Image.fromarray(np.moveaxis(out['rgb'].T, 0, -1)))
                if out.get('rgba') is not None:
                    log_images_to_wandb(f"LOD-{d}-360-Degree-Scene/RGBA", out['rgba'].T, idx)
                if out.get('depth') is not None:
                    log_images_to_wandb(f"LOD-{d}-360-Degree-Scene/Depth", out['depth'].T, idx)
                if out.get('normal') is not None:
                    log_images_to_wandb(f"LOD-{d}-360-Degree-Scene/Normal", out['normal'].T, idx)
                if out.get('alpha') is not None:
                    log_images_to_wandb(f"LOD-{d}-360-Degree-Scene/Alpha", out['alpha'].T, idx)
                wandb.log({})

            rgb_gif = out_rgb[0]
            gif_path = os.path.join(self.log_dir, "rgb.gif")
            rgb_gif.save(gif_path, save_all=True, append_images=out_rgb[1:], optimize=False, loop=0)
            wandb.log({f"360-Degree-Scene/RGB-Rendering/LOD-{d}": wandb.Video(gif_path)})

    def validate(self):
        return 
        self.pipeline.eval()

    def pre_training(self):
        self.writer = SummaryWriter(self.log_dir, purge_step=0)
        self.writer.add_text('Info', self.info)

        self.optimizer.zero_grad()

    def post_training(self):
        self.writer.close()
