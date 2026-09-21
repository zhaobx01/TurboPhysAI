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

"""MapTRv2 training-runtime recipe.

The validated reference implementation configures the channels-last layout, the
cuDNN algorithm selection and the DataLoader start method directly inside
``tools/train.py`` / ``custom_train_detector``.  TurboPhysAI owns the same
entrypoint through a wrapper so those changes apply without editing the model
repository.  Every element can be switched off individually through the Group
``options`` or the matching environment variable.
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


def _matmul_precision(options):
    value = options.get("matmul_precision")
    if value is None:
        value = os.getenv("TURBO_PHYSAI_MATMUL_PRECISION", "high")
    return str(value).strip().lower()


def _channels_last_scope(options):
    value = options.get("channels_last_scope")
    if value is None:
        value = os.getenv("TURBO_PHYSAI_CHANNELS_LAST_SCOPE", "model")
    scope = str(value).strip().lower()
    if scope not in {"model", "backbone"}:
        raise ValueError(
            "channels_last_scope must be 'model' or 'backbone', got "
            f"{value!r}")
    return scope


def _convert_to_channels_last(model, scope):
    """Move the reference channels-last tensor set to the NHWC layout.

    The reference converts ``model.img_backbone`` in ``tools/train.py`` and the
    complete model in ``custom_train_detector``; both are offered through
    ``scope``.  The conversion runs before the original entrypoint moves the
    model to the device, and ``Tensor.to`` preserves the memory format, so the
    device tensors keep the NHWC layout.
    """

    import torch

    if scope == "model":
        model = model.to(memory_format=torch.channels_last)
    else:
        if not hasattr(model, "img_backbone"):
            return model, 0
        model.img_backbone = model.img_backbone.to(
            memory_format=torch.channels_last)
    converted = sum(
        1 for module in model.modules()
        if isinstance(module, torch.nn.Conv2d)
    )
    return model, converted


def _install_input_layout_hook(backbone):
    """Feed the backbone NHWC tensors, mirroring the ``TransposeImage`` step."""

    import torch

    def _pre_hook(_module, inputs):
        if not inputs:
            return None
        first = inputs[0]
        if not torch.is_tensor(first) or first.dim() != 4:
            return None
        if first.is_contiguous(memory_format=torch.channels_last):
            return None
        return (first.contiguous(memory_format=torch.channels_last),
                ) + tuple(inputs[1:])

    return backbone.register_forward_pre_hook(_pre_hook)


def training_runtime_wrapper(original, options):
    """Apply the MapTRv2 training-runtime recipe around ``original``."""

    import torch

    options = dict(options)

    @functools.wraps(original)
    def wrapped(model, dataset, cfg, *args, **kwargs):
        if _option_flag(options, "fork_start_method",
                        "TURBO_PHYSAI_FORK_START_METHOD", True):
            start_method = str(
                options.get(
                    "start_method",
                    os.getenv("TURBO_PHYSAI_DATALOADER_START_METHOD", "fork"),
                )
            )
            if start_method:
                current = torch.multiprocessing.get_start_method(
                    allow_none=True)
                if current != start_method:
                    torch.multiprocessing.set_start_method(
                        start_method, force=current is not None)

        if _option_flag(options, "cudnn_benchmark",
                        "TURBO_PHYSAI_CUDNN_BENCHMARK", True):
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

        # The reference implementation calls this while ``encoder.py`` is
        # imported, which lets the LSSTransform matrix products use the faster
        # reduced-precision path.  Default to the reference value and let
        # ``TURBO_PHYSAI_MATMUL_PRECISION=off`` restore stock numerics.
        precision = _matmul_precision(options)
        if precision not in {"", "off", "none", "default"}:
            setter = getattr(torch, "set_float32_matmul_precision", None)
            if setter is not None:
                setter(precision)

        if _option_flag(options, "channels_last",
                        "TURBO_PHYSAI_CHANNELS_LAST", True):
            if hasattr(model, "img_backbone"):
                scope = _channels_last_scope(options)
                model, converted = _convert_to_channels_last(model, scope)
                _install_input_layout_hook(model.img_backbone)
                logger = getattr(cfg, "logger", None)
                if logger is not None:
                    logger.info(
                        "TurboPhysAI: channels-last %s enabled "
                        "(%d conv modules in channels-last)", scope, converted)

        return original(model, dataset, cfg, *args, **kwargs)

    return wrapped
