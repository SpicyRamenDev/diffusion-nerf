from __future__ import annotations
from wisp.core import RenderBuffer, Rays, PrimitivesPack
from wisp.renderer.core.api import RayTracedRenderer, FramePayload, field_renderer
from wisp.models.nefs.nerf import NeuralRadianceField, BaseNeuralField
from wisp.tracers import PackedRFTracer

from wisp.renderer.core.renderers.radiance_pipeline_renderer import NeuralRadianceFieldPackedRenderer

from .nerf import DiffuseNeuralField
from .tracer import DiffuseTracer
import torch


@field_renderer(DiffuseNeuralField, DiffuseTracer)
class DiffuseNeuralRenderer(NeuralRadianceFieldPackedRenderer):
    def __init__(self, *args, nef_parameters=None, bg_color='white', **kwargs):
        super().__init__(*args, **kwargs)

        self.nef_parameters = nef_parameters
        if self.nef_parameters is None:
            self.nef_parameters = {}

        self.bg_color = bg_color

    def render(self, rays: Rays) -> RenderBuffer:
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