# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from __future__ import annotations
from typing import Callable, Dict, List
from wisp.renderer.gui import WidgetImgui
from wisp.renderer.gui import WidgetInteractiveVisualizerProperties, WidgetGPUStats, WidgetSceneGraph, WidgetOptimization
from wisp.renderer.gizmos.gizmo import Gizmo
from wisp.renderer.app.wisp_app import WispApp
from wisp.renderer.core.api import request_redraw
from wisp.framework import WispState, watch
from wisp.datasets import MultiviewDataset, SDFDataset


# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from typing import List, Dict, Type, DefaultDict, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
from wisp.framework.event import watchedfields

import imgui
from wisp.renderer.gui.imgui.widget_imgui import WidgetImgui, widget
from wisp.core.colors import black, white, dark_gray
from wisp.framework import WispState, InteractiveRendererState
from wisp.renderer.gui.imgui.widget_property_editor import WidgetPropertyEditor
from .utils import spherical_to_cartesian
import torch


class WidgetShadingProperties(WidgetImgui):
    def __init__(self):
        super().__init__()

        self.properties_widget = WidgetPropertyEditor()

    def paint(self, state: WispState, *args, **kwargs):
        expanded, _ = imgui.collapsing_header("Shading", visible=True, flags=imgui.TREE_NODE_DEFAULT_OPEN)
        if expanded:
            def _shading_property():
                shading_modes = ["albedo", "lambertian", "textureless", "normal"]
                selected_shading_mode = state.nef_parameters["shading"]
                value = shading_modes.index(selected_shading_mode)
                clicked, new_value = imgui.combo("Shading mode", value, shading_modes)
                if new_value != value:
                    state.nef_parameters["shading"] = shading_modes[new_value]

            def _ambient_ratio_property():
                value = state.nef_parameters["ambient_ratio"]
                changed, value = imgui.core.slider_float(f"ambient_ratio", value=value,
                                                         min_value=0.0, max_value=1.0)
                if changed:
                    state.nef_parameters["ambient_ratio"] = value

            def _light_azimuth_property():
                value = state.nef_parameters["light_azimuth"]
                changed, value = imgui.core.slider_float(f"light_azimuth", value=value,
                                                         min_value=0.0, max_value=360.0)
                if changed:
                    state.nef_parameters["light_azimuth"] = value
                    state.nef_parameters["light"] = spherical_to_cartesian(
                        torch.tensor(state.nef_parameters["light_azimuth"]),
                        torch.tensor(state.nef_parameters["light_polar"])).cuda()

            def _light_polar_property():
                value = state.nef_parameters["light_polar"]
                changed, value = imgui.core.slider_float(f"light_polar", value=value,
                                                         min_value=0.0, max_value=180.0)
                if changed:
                    state.nef_parameters["light_polar"] = value
                    state.nef_parameters["light"] = spherical_to_cartesian(
                        torch.tensor(state.nef_parameters["light_azimuth"]),
                        torch.tensor(state.nef_parameters["light_polar"])).cuda()

            properties = {
                "Shading Mode": _shading_property,
                "Ambient Ratio": _ambient_ratio_property,
                "Light Azimuth": _light_azimuth_property,
                "Light Polar": _light_polar_property,
            }
            self.properties_widget.paint(state=state, properties=properties)


class App(WispApp):
    """ An app for running an optimization and visualizing it's progress interactively in real time. """

    def __init__(self, wisp_state: WispState, trainer_step_func: Callable[[], None], experiment_name: str):
        super().__init__(wisp_state, experiment_name)

        self.register_background_task(trainer_step_func)

    def init_wisp_state(self, wisp_state: WispState) -> None:
        """ A hook for applications to initialize specific fields inside the wisp state object.
        This function is called at the very beginning of WispApp initialization, hence the initialized fields can
        be customized to affect the behaviour of the renderer.
        """
        wisp_state.renderer.available_canvas_channels = ["rgb", "depth", "albedo", "normal"]
        wisp_state.renderer.selected_canvas_channel = "rgb"

        wisp_state.renderer.reference_grids = ['xz']

        wisp_state.renderer.clear_color_value = (1.0, 1.0, 1.0)
        train_sets = self.wisp_state.optimization.train_data
        if train_sets is not None and len(train_sets) > 0:
            train_set = train_sets[0]
            if isinstance(train_set, MultiviewDataset):
                if train_set.bg_color == 'white':
                    wisp_state.renderer.clear_color_value = (1.0, 1.0, 1.0)
                elif train_set.bg_color == 'black':
                    wisp_state.renderer.clear_color_value = (0.0, 0.0, 0.0)

        from wisp.core import channels_starter_kit, Channel, blend_normal, blend_alpha_slerp, normalize
        wisp_state.graph.channels = channels_starter_kit()

        wisp_state.graph.channels['normal'] = Channel(blend_fn=blend_normal,
                                                      normalize_fn=normalize,
                                                      min_val=-1.0, max_val=1.0)

    def create_widgets(self) -> List[WidgetImgui]:
        """ Returns the list of widgets the gui will display, in order. """
        return [WidgetGPUStats(),            # Current FPS, memory occupancy, GPU Model
                WidgetOptimization(),        # Live status of optimization, epochs / iterations count, loss curve
                WidgetShadingProperties(),
                WidgetInteractiveVisualizerProperties(),  # Canvas dims, user camera controller & definitions
                WidgetSceneGraph()]          # A scene graph tree of the entire hierarchy of objects in the scene and their properties

    def create_gizmos(self) -> Dict[str, Gizmo]:
        """ Override to control which gizmos appear on the canvas.
            For example:
            gizmos = dict(
                world_grid_xy=WorldGrid(squares_per_axis=20, grid_size=10,
                                        line_color=(128, 128, 128), line_size=1, plane="xy")
                grid_axis_painter=AxisPainter(axes_length=10, line_width=1, axes=('x', 'y'), is_bidirectional=True)
            )
        """
        return super().create_gizmos()

    def default_user_mode(self) -> str:
        """ Set the default camera controller mode.
            Possible choices: 'First Person View', 'Turntable', 'Trackball' """
        return "Turntable"

    def register_event_handlers(self) -> None:
        """ Register event handlers for various events that occur in a wisp app.
            For example, the renderer is able to listen on updates on fields of WispState objects.
            (note: events will not prompt when iterables like lists, dicts and tensors are internally updated!)
        """
        # Register default events, such as updating the renderer camera controller when the wisp state field changes
        super().register_event_handlers()

        # For this app, we define a custom event that prompts when an optimization epoch is done
        watch(watched_obj=self.wisp_state.optimization, field="epoch", status="changed", handler=self.on_epoch_ended)
        watch(watched_obj=self.wisp_state.optimization, field="running", status="changed",
              handler=self.on_optimization_running_changed)

    def update_renderer_state(self, wisp_state, dt) -> None:
        """
        Populate the scene state object with the most recent information about the interactive renderer.
        The scene state, for example, may be used by the GUI widgets to display up to date information.
        This function is invoked in the beginning of the render() function, before the gui and the canvas are drawn.
        :param wisp_state The WispState object holding shared information between components about the wisp app.
        :param dt Amount of time elapsed since the last update.
        """
        # Update the wisp state with new information about this frame.
        # i.e.: Current FPS, time elapsed.
        super().update_renderer_state(wisp_state, dt)

    def on_epoch_ended(self):
        """ A custom event used by the optimization renderer.
            When an epoch ends, this handler is invoked to force a redraw() and render() of the canvas if needed.
        """
        request_redraw(self.wisp_state)

        # Force render if target FPS is 0 (renderer only responds to specific events) or too much time have elapsed
        if self.is_time_to_render() or self.wisp_state.renderer.target_fps == 0:
            self.render()

    def on_optimization_running_changed(self, value: bool):
        # When training starts / resumes, invoke a redraw() to refresh the renderer core with newly
        # added objects to the scene graph (i.e. the optimized object, or some objects from the dataset).
        if value:
            self.redraw()
