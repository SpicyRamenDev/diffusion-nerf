# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
from wisp.ops.raygen.raygen import generate_default_grid, generate_centered_pixel_coords, generate_pinhole_rays


class SampleNoisyRays:
    """ A dataset transform for sub-sampling a fixed amount of rays. """
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def get_noisy_pixel_coords(self, camera, image):
        img_width, img_height = camera.width, camera.height
        res_x, res_y = camera.width, camera.height
        pixel_y, pixel_x = generate_default_grid(res_x, res_y)

        pixel_x = pixel_x + 0.5
        pixel_y = pixel_y + 0.5
        pixel_x = pixel_x + torch.rand_like(pixel_x) - 1
        pixel_y = pixel_y + torch.rand_like(pixel_y) - 1
        pixel = torch.stack([pixel_y, pixel_x], dim=-1)

        ray_grid = pixel_y, pixel_x
        rays = generate_pinhole_rays(camera, ray_grid)

        pixel_values = torch.nn.functional.grid_sample(image.reshape([img_height, img_width, 3]).permute(2, 0, 1).unsqueeze(0),
                                                       pixel.unsqueeze(0),
                                                       align_corners=False)
        pixel_values = pixel_values.permute(0, 3, 1, 2).reshape(-1, 3)

        rays = rays.to(pixel_values.dtype)

        return rays, image

    def __call__(self, inputs):
        rays, pixel_values = self.get_noisy_pixel_coords(inputs['cameras'], inputs['imgs'])
        pixel_idx = torch.multinomial(torch.ones(pixel_values.shape[0]).float(), self.num_samples, replacement=False)
        rays = rays[pixel_idx]
        pixel_values = pixel_values[pixel_idx]

        out = {}
        out['rays'] = rays.contiguous()
        out['imgs'] = pixel_values.contiguous()
        return out
