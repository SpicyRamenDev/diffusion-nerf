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

from .utils import sample, spherical_to_cartesian, l2_normalize, sample_polar, sample_spherical_uniform, \
    get_rotation_matrix, generate_camera_rays
from .renderer import DiffuseNeuralRenderer
import math
from torchvision import transforms


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

        self.scene_state.nef_parameters = {
            "use_light": True,
            "shading": 'lambertian',
            "ambient_ratio": 0.1,
            "light": torch.tensor([1.0, 1.0, -1.0], device='cuda'),
            "light_azimuth": 45.0,
            "light_polar": 45.0,
        }

        bl_state = self.scene_state.graph.bl_renderers.get(self.exp_name)
        bl_state.setup_args["num_steps"] = 512
        bl_state.setup_args["raymarch_type"] = "ray"
        bl_state.setup_args["bg_color"] = self.extra_args["bg_color"]
        bl_state.setup_args["nef_parameters"] = self.scene_state.nef_parameters

    def init_renderer(self):
        nef_parameters = {
            "use_light": True,
            "shading": 'lambertian',
            "ambient_ratio": 0.1,
            "light": torch.tensor([1.0, 1.0, -1.0], device='cuda'),
        }
        self.renderer = DiffuseNeuralRenderer(self.pipeline.nef, self.pipeline.tracer,
                                              num_steps=self.extra_args["render_num_steps"],
                                              raymarch_type="ray",
                                              bg_color=self.extra_args["bg_color"],
                                              nef_parameters=nef_parameters)

    def init_optimizer(self):
        params_dict = {name: param for name, param in self.pipeline.nef.named_parameters()}

        params = []
        decoder_params = []
        decoder_bg_params = []
        decoder_normal_params = []
        grid_params = []
        rest_params = []

        for name in params_dict:
            if 'decoder_background' in name:
                decoder_bg_params.append(params_dict[name])
            elif 'decoder_normal' in name:
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

    def init_log_dict(self):
        super().init_log_dict()
        self.log_dict['rgb_loss'] = 0.0

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

    def pre_step(self):
        super().pre_step()

        if self.extra_args["prune_every"] > -1 and self.iteration > 0 and self.iteration % self.extra_args["prune_every"] == 0:
            self.pipeline.nef.prune()

    def step_gt(self, data, phase=0):
        rays = data['rays'].to(self.device).squeeze(0)
        img_gts = data['imgs'].to(self.device).squeeze(0)

        loss = 0

        with torch.cuda.amp.autocast():
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
        # loss.backward()

        return loss.item()

    def get_scene_parameters(self, phase=0):
        resolution = self.extra_args["resolutions"][phase]
        is_uniform = random.random() < 0.5
        camera_dir, azimuth, polar = sample_spherical_uniform(azimuth_range=self.extra_args["azimuth_range"],
                                                              polar_range=self.extra_args["polar_range"],
                                                              uniform=is_uniform)
        camera_offset = (2 * torch.rand(3) - 1) * self.extra_args["camera_offset"]
        camera_distance = sample(self.extra_args["camera_distance_range"])
        camera_coords = camera_dir * camera_distance + camera_offset
        focal_length_multiplier = sample(self.extra_args["focal_length_multiplier_range"])
        # if self.total_iterations < 1000:
        #     focal_length_multiplier = 1.0
        focal_length = 0.5 * resolution * camera_distance * focal_length_multiplier
        camera_up = l2_normalize(torch.tensor([0., 1., 0.]) + torch.randn(3) * self.extra_args["camera_up_std"])
        look_at = torch.randn(3) * self.extra_args["look_at_std"]
        look_at += torch.tensor(self.extra_args["camera_lookat"])

        camera = Camera.from_args(
            eye=camera_coords,
            at=look_at,
            up=camera_up,
            focal_x=focal_length,
            width=resolution, height=resolution,
            near=max(0, camera_distance-1.74),
            far=camera_distance+1.74,
            dtype=torch.float32,
            device='cuda'
        )

        rays = generate_camera_rays(camera)
        rays = rays.reshape(resolution ** 2, -1)

        light_dir, light_azimuth, light_polar = sample_spherical_uniform(polar_range=(0, 60), uniform=True)
        camera_rot = get_rotation_matrix(azimuth, polar)
        light_dir = camera_rot @ light_dir
        light_distance = sample(self.extra_args["light_distance_range"])
        light = light_dir * light_distance
        light = light.to(self.device)

        text_embeddings, view = self.get_text_embeddings(azimuth, polar)

        return dict(rays=rays, camera=camera, light=light,
                    azimuth=azimuth, polar=polar,
                    text_embeddings=text_embeddings, view=view)

    def get_shading_parameters(self, phase=0):
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
        if phase == 0 or True:
            weight_type = 'quadratic'
            min_ratio = 0.02
            max_ratio = 0.98
        elif phase == 1:
            weight_type = 'quadratic'
            min_ratio = 0.02
            max_ratio = 0.98
        else:
            weight_type = 'quadratic'
            min_ratio = 0.02
            max_ratio = 0.60
        guidance_scale = self.extra_args["guidance_scale"]

        return dict(weight_type=weight_type, min_ratio=min_ratio, max_ratio=max_ratio, guidance_scale=guidance_scale)

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

    def step_novel_view(self, phase=0):
        lod_idx = None

        shading_parameters = self.get_shading_parameters(phase=phase)
        diffusion_parameters = self.get_diffusion_parameters(phase=phase)
        scene = self.get_scene_parameters(phase=phase)
        rays = scene['rays']
        light = scene['light']
        camera = scene['camera']
        text_embeddings = scene['text_embeddings']
        resolution = self.extra_args["resolutions"][phase]
        if self.extra_args["aug_bg_color"] == "blur":
            bg_color_value = torch.randn([3, resolution, resolution], device='cuda')
            bg_color_value = transforms.GaussianBlur(7, sigma=5.0)(bg_color_value) * 0.5
            bg_color_value = 0.5 * (bg_color_value + 1.0).clamp(0.0, 1.0)
            bg_color_value = bg_color_value.permute(1, 2, 0).contiguous()
        else:
            bg_color_value = torch.rand(3, device='cuda').reshape(1, 1, 3).repeat(resolution, resolution, 1)
        if shading_parameters['shading'] == 'textureless':
            bg_color_value = bg_color_value.mean(dim=-1, keepdim=True).repeat(1, 1, 3)
        bg_color = self.aug_bg_color

        nef_parameters = dict(phase=phase,
                              use_light=True,
                              light=light,
                              **shading_parameters)
        render_trace_parameters = dict(lod_idx=lod_idx,
                                       channels=["rgb"],
                                       bg_color=bg_color,
                                       bg_color_value=bg_color_value,
                                       nef_parameters=nef_parameters)
        train_trace_parameters = dict(lod_idx=lod_idx,
                                      channels=["rgb", "orientation_loss"],
                                      bg_color=bg_color,
                                      bg_color_value=bg_color_value,
                                      nef_parameters=nef_parameters,
                                      total_variation_loss=self.extra_args["total_variation_loss"],
                                      orientation_loss=self.extra_args["orientation_loss"],
                                      entropy_loss=self.extra_args["entropy_loss"])

        diffusion_parameters = dict(text_embeddings=text_embeddings, **diffusion_parameters)

        total_loss_value = 0

        if self.total_iterations >= self.extra_args["reg_warmup_iterations"]:
            reg_factor = 1
        else:
            reg_factor = self.extra_args["reg_init_lr"] + (1 - self.extra_args["reg_init_lr"]) * self.total_iterations / self.extra_args["reg_warmup_iterations"]

        if 0 < self.extra_args["render_batch"] < rays.shape[0]:
            rb = RenderBuffer()
            rays_cache = []
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    for i, ray_pack in enumerate(rays.split(self.extra_args["render_batch"])):
                        start, end = i * self.extra_args["render_batch"], (i + 1) * self.extra_args["render_batch"]
                        pack_bg_color_value = bg_color_value.reshape(-1, 3)[start:end, ...]
                        render_trace_parameters["bg_color_value"] = pack_bg_color_value
                        rb += self.pipeline(rays=ray_pack, out_rays=rays_cache, **render_trace_parameters)

            image = rb.rgb[..., :3]
            image = image.reshape(1, camera.width, camera.height, 3)
            image = image.permute(0, 3, 1, 2).contiguous()

            image_f = image.float()
            diffusion_grad = self.diffusion.step(image=image_f, **diffusion_parameters)
            diffusion_grad = diffusion_grad.reshape(3, -1).contiguous()
            diffusion_grad = diffusion_grad / 4096.0

            for i, (ray_pack, ray_cache) in enumerate(zip(rays.split(self.extra_args["render_batch"]), rays_cache)):
                start, end = i * self.extra_args["render_batch"], (i + 1) * self.extra_args["render_batch"]
                pack_bg_color_value = bg_color_value.reshape(-1, 3)[start:end]
                train_trace_parameters["bg_color_value"] = pack_bg_color_value
                with torch.cuda.amp.autocast():
                    rb = self.pipeline(rays=ray_pack, raymarch_results=ray_cache, **train_trace_parameters)

                    pack_image = rb.rgb[..., :3]
                    pack_image = pack_image.reshape(self.extra_args["render_batch"], 3)
                    pack_image = pack_image.permute(1, 0).contiguous()
                    pack_diffusion_grad = diffusion_grad[..., start:end]
                    pack_diffusion_loss = (pack_diffusion_grad * pack_image).sum()
                    pack_diffusion_loss = self.extra_args["diffusion_loss"] * pack_diffusion_loss

                    pack_reg_loss = 0
                    if self.extra_args["orientation_loss"] > 0:
                        orientation_loss = rb.orientation_loss.sum()
                        pack_reg_loss += reg_factor * self.extra_args["orientation_loss"] * orientation_loss
                    if self.extra_args["opacity_loss"] > 0:
                        opacity_loss = torch.sqrt(rb.alpha ** 2 + 0.01).sum()
                        # alphas = rb.alpha.clamp(1e-5, 1 - 1e-5)
                        # opacity_loss = (-alphas * torch.log2(alphas) - (1 - alphas) * torch.log2(1 - alphas)).sum()
                        pack_reg_loss += self.extra_args["opacity_loss"] * opacity_loss
                    if self.extra_args["entropy_loss"] > 0:
                        entropy_loss = rb.entropy_loss.sum()
                        pack_reg_loss += self.extra_args["entropy_loss"] * entropy_loss
                    pack_reg_loss *= 1.0 / rays.shape[0]
                    pack_reg_loss *= 1.0 / self.extra_args["minibatch_size"]

                    pack_loss = pack_diffusion_loss + pack_reg_loss
                    pack_loss *= 1.0 / self.extra_args["rgb_loss"]

                self.scaler.scale(pack_loss).backward()
                # pack_loss.backward()
                total_loss_value += pack_loss.item()

                del rb
        else:
            with torch.cuda.amp.autocast():
                rb = self.pipeline(rays=rays, **train_trace_parameters)

            image = rb.rgb[..., :3]
            image = image.reshape(1, camera.width, camera.height, 3)
            image = image.permute(0, 3, 1, 2).contiguous()

            image_f = image.float()
            diffusion_grad = self.diffusion.step(image=image_f, **diffusion_parameters)
            diffusion_grad = diffusion_grad / 4096.0

            with torch.cuda.amp.autocast():
                diffusion_loss = (diffusion_grad * image).sum()
                diffusion_loss = self.extra_args["diffusion_loss"] * diffusion_loss
                diffusion_loss *= 1.0 / self.extra_args["minibatch_size"]

                reg_loss = 0
                if self.extra_args["orientation_loss"] > 0:
                    orientation_loss = rb.orientation_loss.sum()
                    reg_loss += reg_factor * self.extra_args["orientation_loss"] * orientation_loss
                if self.extra_args["opacity_loss"] > 0:
                    opacity_loss = torch.sqrt(rb.alpha ** 2 + 0.01).sum()
                    # alphas = rb.alpha.clamp(1e-5, 1 - 1e-5)
                    # opacity_loss = (-alphas * torch.log2(alphas) - (1 - alphas) * torch.log2(1 - alphas)).sum()
                    reg_loss += self.extra_args["opacity_loss"] * opacity_loss
                if self.extra_args["entropy_loss"] > 0:
                    entropy_loss = rb.entropy_loss.sum()
                    reg_loss += self.extra_args["entropy_loss"] * entropy_loss
                reg_loss *= 1.0 / rays.shape[0]
                reg_loss *= 1.0 / self.extra_args["minibatch_size"]

                loss = diffusion_loss + reg_loss
                loss *= 1.0 / self.extra_args["rgb_loss"]

            self.scaler.scale(loss).backward()
            # loss.backward()
            total_loss_value += loss.item()

        print("Iteration:", self.total_iterations,
              "phase:", phase,
              "Learning rate:", self.optimizer.param_groups[0]['lr'],
              "Diffusion loss: ", total_loss_value)

        if self.total_iterations % 50 == 0:
            out = self.render_camera(camera, channels=["rgb", "normal"])
            for channel, image in out.items():
                path = os.path.join(self.log_dir, f"it{self.total_iterations}_{channel}.png")
                image.save(path)

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
        # self.optimizer.step()
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

    def render_rays(self, rays):
        rb = RenderBuffer(hit=None)
        for ray_batch in rays.split(self.batch_size):
            rb += self.tracer(self.nef,
                              rays=ray_batch,
                              channels=self.channels,
                              lod_idx=None,  # TODO(ttakikawa): Add a way to control the LOD in the GUI
                              raymarch_type=self.raymarch_type,
                              num_steps=self.num_steps,
                              bg_color=self.bg_color,
                              nef_parameters=self.nef_parameters)

        rb = rb.reshape(self.render_res_y, self.render_res_x, -1)
        if self.render_res_x != self.output_width or self.render_res_y != self.output_height:
            rb = rb.scale(size=(self.output_height, self.output_width))
        return rb

    def render_camera(self, camera, channels):
        self.renderer.channels = channels
        self.renderer.render_res_x = camera.width
        self.renderer.render_res_y = camera.height
        self.renderer.output_width = camera.width
        self.renderer.output_height = camera.height
        rays = generate_camera_rays(camera)
        with torch.no_grad():
            rb = self.renderer.render(rays=rays)

        out = {}
        for channel, image in dict(iter(rb)).items():
            if channel not in channels:
                continue
            if channel == "normal":
                image = 0.5 * (image + 1.0)
            elif channel == "depth":
                image = (image - image.min()) / (image.max() - image.min())
            if image.shape[-1] == 1:
                image = torch.cat([image]*3, dim=-1)
            image = image.reshape(camera.height, camera.width, 3)
            np_img = image.cpu().numpy()
            np_img = (np_img * 255).astype(np.uint8)
            pil_image = Image.fromarray(np_img)
            out[channel] = pil_image

        return out

    def render_view(self, azimuth, polar, distance, width, height, fov, channels):
        coords = spherical_to_cartesian(torch.tensor(azimuth).float(),
                                        torch.tensor(polar).float(),
                                        torch.tensor(distance).float())
        camera = Camera.from_args(
            eye=coords,
            at=self.extra_args["camera_lookat"],
            up=torch.tensor([0.0, 1.0, 0.0]),
            fov=fov,
            width=width, height=height,
            near=max(0, distance-1.74),
            far=distance+1.74,
            dtype=torch.float32,
            device='cuda'
        )
        out = self.render_camera(camera, channels=channels)
        return out

    def render_views(self, num_angles, width, height, channels):
        azimuths = torch.arange(0, num_angles) * 360.0 / num_angles
        polar = self.extra_args["render_polar"]
        distance = self.extra_args["render_distance"]
        fov = self.extra_args["render_fov"]
        outs = {channel: [] for channel in channels}
        for idx, azimuth in tqdm(enumerate(azimuths), desc=f"Generating 360 Degree of View"):
            out = self.render_view(azimuth, polar, distance, width, height, fov, channels=channels)
            for channel, image in out.items():
                outs[channel].append(image)
        return outs

    def save_360_renders(self, num_angles, width, height, channels=("rgb",), prefix=""):
        outs = self.render_views(num_angles,
                                 width=width,
                                 height=height,
                                 channels=channels)
        for channel, images in outs.items():
            path = os.path.join(self.log_dir, f"{prefix}{self.epoch}_{channel}.webp")
            images[0].save(path,
                           append_images=images[1:],
                           save_all=True, lossless=True, loop=0)

    def validate(self):
        pass

    def end_epoch(self):
        super().end_epoch()

        if self.epoch % self.extra_args["render_every"] == 0:
            self.save_360_renders(prefix="",
                                  num_angles=self.extra_args["render_num_angles"] // 3,
                                  width=self.extra_args["render_res"][0] // 4,
                                  height=self.extra_args["render_res"][1] // 4,
                                  channels=["rgb", "alpha", "depth", "normal"])

    def pre_training(self):
        self.writer = SummaryWriter(self.log_dir, purge_step=0)
        self.writer.add_text('Info', self.info)

        self.optimizer.zero_grad()

    def post_training(self):
        self.save_360_renders(prefix=f"final_",
                              num_angles=self.extra_args["render_num_angles"],
                              width=self.extra_args["render_res"][0],
                              height=self.extra_args["render_res"][1],
                              channels=["rgb", "alpha", "depth", "normal"])

        self.writer.close()
