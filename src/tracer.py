# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import torch.nn as nn
import kaolin.render.spc as spc_render
from wisp.core import RenderBuffer
from wisp.tracers import BaseTracer, PackedRFTracer
from .utils import rays_aabb_bounds

from wisp.accelstructs.base_as import BaseAS, ASQueryResults, ASRaytraceResults, ASRaymarchResults


class DiffuseTracer(PackedRFTracer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def stratified_sampling(self, nef, rays, num_samples=64) -> ASRaymarchResults:
        level = nef.grid.blas_level

        depth = torch.linspace(0, 1.0, num_samples, device=rays.origins.device)[None] + \
                (torch.rand(rays.origins.shape[0], num_samples, device=rays.origins.device) / num_samples)

        tmin, tmax, imask = rays_aabb_bounds(rays)
        tmin = tmin[imask]
        tmax = tmax[imask]
        depth = depth[imask]
        origins = rays.origins[imask]
        dirs = rays.dirs[imask]

        depth *= (tmax - tmin)
        depth += tmin

        num_rays = origins.shape[0]
        samples = torch.addcmul(origins[:, None], dirs[:, None], depth[:, :, None])
        deltas = depth.diff(dim=-1,
                            prepend=(torch.zeros(origins.shape[0], 1, device=depth.device) + tmin))
        query_results = nef.grid.query(samples.reshape(num_rays * num_samples, 3), level=level)
        pidx = query_results.pidx
        pidx = pidx.reshape(num_rays, num_samples)
        mask = pidx > -1

        depth_samples = depth[mask][:, None]
        num_hit_samples = depth_samples.shape[0]
        deltas = deltas[mask].reshape(num_hit_samples, 1)
        samples = samples[mask]
        ridx = torch.arange(0, rays.origins.shape[0], device=rays.origins.device)
        ridx = ridx[..., None].repeat(1, num_samples)[imask][mask]
        boundary = spc_render.mark_pack_boundaries(ridx)

        return ASRaymarchResults(
            ridx=ridx,
            samples=samples,
            depth_samples=depth_samples,
            deltas=deltas,
            boundary=boundary
        )

    @torch.no_grad()
    def importance_sampling(self, nef, rays, num_coarse_samples=64, num_fine_samples=0) -> ASRaymarchResults:
        level = nef.grid.blas_level
        lod_idx = nef.grid.num_lods - 1

        raymarch_results = self.stratified_sampling(nef, rays, num_coarse_samples)

        ridx = raymarch_results.ridx
        samples = raymarch_results.samples
        deltas = raymarch_results.deltas
        boundary = raymarch_results.boundary
        depth_samples = raymarch_results.depth_samples

        ridx_hit = ridx[spc_render.mark_pack_boundaries(ridx.int())]
        hit_ray_d = rays.dirs.index_select(0, ridx)

        density = nef.density(coords=samples, lod_idx=lod_idx)
        density = density.reshape(-1, 1)
        del ridx

        tau = density * deltas
        del density
        _, transmittance = spc_render.exponential_integration(torch.ones_like(tau), tau, boundary, exclusive=True)
        alpha = spc_render.sum_reduce(transmittance, boundary)

        hit = alpha > 0
        hit = hit.reshape(-1, 1)

        bins = torch.multinomial(transmittance[hit], num_fine_samples, replacement=True)
        print(bins.shape)
        noise = torch.rand_like(bins)
        new_depths = depth_samples[hit][bins] + noise * deltas[hit][bins]
        new_samples = torch.addcmul(rays.origins[ridx_hit][None], hit_ray_d[None], new_depths[..., None])

        return ASRaymarchResults(
            ridx=ridx,
            samples=samples,
            depth_samples=depth_samples,
            deltas=deltas,
            boundary=boundary
        )

    def trace(self, nef, rays, channels, extra_channels,
              lod_idx=None, raymarch_type='voxel', num_steps=64, step_size=1.0, bg_color='white',
              bg_color_value=None,
              raymarch_results=None,
              out_rays=None,
              scaler=None,
              entropy_loss=0, entropy_threshold=0.01,
              orientation_loss=0,
              total_variation_loss=0,
              nef_parameters=None):
        assert nef.grid is not None and "this tracer requires a grid"

        N = rays.origins.shape[0]

        if "depth" in channels:
            depth = torch.zeros(N, 1, device=rays.origins.device)
        else:
            depth = None

        if bg_color == 'white':
            rgb = torch.ones(N, 3, device=rays.origins.device)
        elif bg_color == 'noise' or bg_color == 'blur':
            rgb = bg_color_value.reshape(-1, 3).clone()
        elif bg_color == 'decoder':
            rgb = nef.background(rays.origins, rays.dirs)["background"]
        else:
            rgb = torch.zeros(N, 3, device=rays.origins.device)
        hit = torch.zeros(N, device=rays.origins.device, dtype=torch.bool)
        out_alpha = torch.zeros(N, 1, device=rays.origins.device)

        if lod_idx is None:
            lod_idx = nef.grid.num_lods - 1

        if raymarch_results is None:
            # raymarch_results = self.stratified_sampling(nef, rays,
            #                                             num_samples=num_steps)
            raymarch_results = nef.grid.raymarch(rays, num_samples=num_steps,
                                                 raymarch_type=raymarch_type)
        ridx = raymarch_results.ridx
        samples = raymarch_results.samples
        deltas = raymarch_results.deltas
        boundary = raymarch_results.boundary
        depths = raymarch_results.depth_samples
        if out_rays is not None:
            out_rays.append(raymarch_results)

        ridx_hit = ridx[spc_render.mark_pack_boundaries(ridx.int())]

        hit_ray_d = rays.dirs.index_select(0, ridx)

        queried_channels = {"rgb", "density"}.union(extra_channels)
        if nef_parameters is None:
            nef_parameters = self.nef_parameters
        queried_features = nef.features(coords=samples, ray_d=hit_ray_d, lod_idx=lod_idx, channels=queried_channels,
                                        **nef_parameters)
        color = queried_features["rgb"]
        density = queried_features["density"]
        density = density.reshape(-1, 1)
        del ridx

        tau = density * deltas
        # del density, deltas
        ray_colors, transmittance = spc_render.exponential_integration(color, tau, boundary, exclusive=True)

        if "depth" in channels:
            ray_depth = spc_render.sum_reduce(depths.reshape(-1, 1) * transmittance, boundary)
            depth[ridx_hit, :] = ray_depth

        alpha = spc_render.sum_reduce(transmittance, boundary)
        # alpha = 1.0 - torch.exp(spc_render.sum_reduce(-tau, boundary))
        out_alpha[ridx_hit] = alpha
        hit[ridx_hit] = alpha[..., 0] > 0.0

        rgb = rgb.type(ray_colors.dtype)
        if bg_color == 'white':
            color = (1.0 - alpha) + ray_colors
        else:
            color = (1.0 - alpha) * rgb[ridx_hit] + ray_colors
        rgb[ridx_hit] = color

        extra_outputs = {}
        for channel in extra_channels:
            feats = queried_features[channel]
            num_channels = feats.shape[-1]
            if "reg" in channel:
                ray_feats = spc_render.sum_reduce(feats.reshape(-1, num_channels).contiguous() * transmittance.detach(), boundary)
            else:
                ray_feats = spc_render.sum_reduce(feats.reshape(-1, num_channels).contiguous() * transmittance, boundary)
            out_feats = torch.zeros(N, num_channels, device=feats.device)
            out_feats[ridx_hit] = ray_feats
            extra_outputs[channel] = out_feats

        if False and orientation_loss > 0:
            samples_minus_delta = samples - hit_ray_d * deltas
            shifted_density = nef.density(coords=samples_minus_delta, lod_idx=lod_idx)
            density_gradient = (density - shifted_density) / deltas
            orientation_loss_term = (-density_gradient).clamp(min=0) ** 2
            orientation_loss = spc_render.sum_reduce(orientation_loss_term * transmittance.detach(), boundary)
            extra_outputs["orientation_loss"] = orientation_loss

        if total_variation_loss > 0:
            raise NotImplementedError("Total variation loss is not implemented")

        if entropy_loss > 0:
            opacity = 1 - torch.exp(-tau)
            opacity_sum = spc_render.sum_reduce(opacity, boundary)
            opacity_xlogx_sum = spc_render.sum_reduce(opacity * torch.log(opacity + 1e-10), boundary)
            entropy_ray = -opacity_xlogx_sum / (opacity_sum + 1e-10) + torch.log(opacity_sum + 1e-10)
            mask = (opacity_sum > entropy_threshold).detach()
            entropy_ray *= mask
            out_entropy = torch.zeros(N, 1, device=entropy_ray.device)
            out_entropy[ridx_hit] = entropy_ray
            extra_outputs["entropy_loss"] = out_entropy

        return RenderBuffer(depth=depth, hit=hit, rgb=rgb, alpha=out_alpha,
                            **extra_outputs)
