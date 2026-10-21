# Copyright 2018-2019 OpenMMLab. All rights reserved.
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

"""Static-graph DDP settings for MapTRv2.

The reference implementation passes ``static_graph=True`` and
``find_unused_parameters=False`` to ``MMDistributedDataParallel`` from
``projects/mmdet3d_plugin/bevformer/apis/mmdet_train.py``, driven by a
``ddp_static_graph`` field added to the model config. Neither the field nor the
``static_graph`` argument exists in the official baseline, so the config alone
cannot express it.

This module wraps the constructor of ``mmcv.parallel.distributed.
MMDistributedDataParallel`` instead. That class does not define its own
``__init__``, so installing the wrapper adds an attribute on the *subclass* and
shadows the inherited one; ``torch.nn.parallel.distributed.
DistributedDataParallel`` is left untouched. Both DDP construction sites in
``mmdet_train.py`` (the training model and ``eval_model``) go through this
constructor.

``static_graph=True`` makes DDP infer the unused-parameter set from the first
iteration and reuse it afterwards. It therefore assumes the autograd graph does
not change between iterations, which is why it belongs with
``maptrv2.assigner``: ``pad_to_static_list`` pins the ground-truth side to a
fixed length and removes the main source of that variability. Keep the two
groups enabled together, or disable both.
"""

import functools
import os


def _env_flag(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no"}


def _option_flag(options, key, env_name, default):
    if key in options:
        return bool(options[key])
    return _env_flag(env_name, default)


def ddp_constructor_wrapper(original, options):
    """Inject static-graph settings into ``MMDistributedDataParallel``.

    ``mmdet_train.py`` passes ``device_ids``, ``broadcast_buffers`` and
    ``find_unused_parameters`` as keyword arguments, so the injected values go
    into ``kwargs`` as well. Set ``TURBO_PHYSAI_DDP_STATIC_GRAPH=0`` to keep the
    native static-graph behavior, or ``TURBO_PHYSAI_DDP_FIND_UNUSED_PARAMETERS=0``
    to leave ``find_unused_parameters`` at whatever the model config declares.
    """

    options = dict(options)

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        if _option_flag(
            options, "static_graph", "TURBO_PHYSAI_DDP_STATIC_GRAPH", True
        ):
            kwargs.setdefault("static_graph", True)
        if _option_flag(
            options,
            "find_unused_parameters",
            "TURBO_PHYSAI_DDP_FIND_UNUSED_PARAMETERS",
            True,
        ):
            kwargs["find_unused_parameters"] = False
        return original(self, *args, **kwargs)

    return wrapped


__all__ = ["ddp_constructor_wrapper"]
