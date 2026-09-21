# Copyright (c) OpenMMLab. All rights reserved.
# Copyright 2026 Hygon Information Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by Hygon.

"""MapTRv2 DataLoader replacement.

The validated reference implementation flips ``pin_memory`` from ``False`` to
``True`` in ``projects.mmdet3d_plugin/datasets/builder.build_dataloader`` so the
host-to-device copies can be issued from pinned memory and overlap with the
previous step's kernels.  Everything else is kept byte-for-byte identical to the
official builder.
"""

import os
import random
from functools import partial

import numpy as np
from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from mmdet.datasets.samplers import GroupSampler
from torch.utils.data import DataLoader

from projects.mmdet3d_plugin.datasets.samplers.distributed_sampler import (
    DistributedSampler,
)
from projects.mmdet3d_plugin.datasets.samplers.group_sampler import (
    DistributedGroupSampler,
)
from projects.mmdet3d_plugin.datasets.samplers.sampler import build_sampler


def _pin_memory_enabled():
    return os.getenv("TURBO_PHYSAI_PIN_MEMORY", "1") != "0"


def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     seed=None,
                     shuffler_sampler=None,
                     nonshuffler_sampler=None,
                     **kwargs):
    """Build a PyTorch DataLoader with pinned host memory."""

    rank, world_size = get_dist_info()
    if dist:
        if shuffle:
            sampler = build_sampler(
                shuffler_sampler if shuffler_sampler is not None
                else dict(type='DistributedGroupSampler'),
                dict(
                    dataset=dataset,
                    samples_per_gpu=samples_per_gpu,
                    num_replicas=world_size,
                    rank=rank,
                    seed=seed)
            )
        else:
            sampler = build_sampler(
                nonshuffler_sampler if nonshuffler_sampler is not None
                else dict(type='DistributedSampler'),
                dict(
                    dataset=dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=shuffle,
                    seed=seed)
            )
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        print('WARNING!!!!, Only can be used for obtain inference speed!!!!')
        sampler = GroupSampler(dataset, samples_per_gpu) if shuffle else None
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    kwargs.setdefault("pin_memory", _pin_memory_enabled())

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        worker_init_fn=init_fn,
        **kwargs)

    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
