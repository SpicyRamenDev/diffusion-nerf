# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


from typing import Callable
import torch
from torch.utils.data import Dataset
from wisp.datasets.formats import load_nerf_standard_data, load_rtmv_data
from wisp.core import Rays

from wisp.datasets.multiview_dataset import MultiviewDataset


class FullMultiviewDataset(MultiviewDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.basenames = list(self.data['cameras'])

    def __getitem__(self, idx: int):
        out = {}
        out['rays'] = self.data["rays"][idx]
        out['imgs'] = self.data["imgs"][idx]
        out['masks'] = self.data["masks"][idx]
        out['cameras'] = self.data["cameras"][self.basenames[idx]]

        if self.transform is not None:
            out = self.transform(out)

        return out
