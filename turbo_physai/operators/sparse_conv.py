# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""spconv-compatible sparse convolution operators backed by native kernels."""

from __future__ import annotations

import torch


def _ops():
    from turbo_physai import ops

    return ops


def _dimensions(value, ndim):
    return list(value) if isinstance(value, (list, tuple)) else [value] * ndim


def _conv_output_size(input_size, kernel_size, stride, padding, dilation):
    return [
        1 if kernel == -1 else
        (size + 2 * pad - dilate * (kernel - 1) - 1) // step + 1
        for size, kernel, step, pad, dilate in zip(
            input_size, kernel_size, stride, padding, dilation
        )
    ]


def _deconv_output_size(input_size, kernel_size, stride, padding, dilation, output_padding):
    if any(kernel == -1 for kernel in kernel_size):
        raise ValueError("deconvolution does not support kernel_size < 0")
    return [
        (size - 1) * step - 2 * pad + kernel + out_pad
        for size, kernel, step, pad, dilate, out_pad in zip(
            input_size, kernel_size, stride, padding, dilation, output_padding
        )
    ]


def get_indice_pairs(
    indices, batch_size, spatial_shape, ksize=3, stride=1, padding=0,
    dilation=1, out_padding=0, subm=False, transpose=False, grid=None,
):
    """Match ``spconv.ops.get_indice_pairs`` for supported 2D/3D/4D inputs."""

    ndim = indices.shape[1] - 1
    ksize = _dimensions(ksize, ndim)
    stride = _dimensions(stride, ndim)
    padding = _dimensions(padding, ndim)
    dilation = _dimensions(dilation, ndim)
    out_padding = _dimensions(out_padding, ndim)
    if any(step != 1 and dilate != 1 for step, dilate in zip(stride, dilation)):
        raise ValueError("stride and dilation cannot both exceed one")
    if subm:
        output_shape = spatial_shape
    elif transpose:
        output_shape = _deconv_output_size(
            spatial_shape, ksize, stride, padding, dilation, out_padding
        )
    else:
        output_shape = _conv_output_size(
            spatial_shape, ksize, stride, padding, dilation
        )
    extension = _ops()
    if grid is None:
        function = getattr(extension, f"get_indice_pairs_{ndim}d", None)
        args = (indices, batch_size, output_shape, spatial_shape)
    else:
        function = getattr(extension, f"get_indice_pairs_grid_{ndim}d", None)
        args = (indices, grid, batch_size, output_shape, spatial_shape)
    if function is None:
        raise NotImplementedError(f"unsupported sparse convolution rank: {ndim}")
    return function(
        *(args + (
            ksize,
            stride,
            padding,
            dilation,
            out_padding,
            int(subm),
            int(transpose),
        ))
    )


def get_indice_pairs_2d(indices, batch_size, out_spatial_shape, spatial_shape,
                         kernel_size, stride, padding, dilation, out_padding,
                         subm, transpose):
    return _ops().get_indice_pairs_2d(
        indices, batch_size, out_spatial_shape, spatial_shape, kernel_size,
        stride, padding, dilation, out_padding, subm, transpose,
    )


def get_indice_pairs_3d(indices, batch_size, out_spatial_shape, spatial_shape,
                         kernel_size, stride, padding, dilation, out_padding,
                         subm, transpose):
    return _ops().get_indice_pairs_3d(
        indices, batch_size, out_spatial_shape, spatial_shape, kernel_size,
        stride, padding, dilation, out_padding, subm, transpose,
    )


def get_indice_pairs_4d(indices, batch_size, out_spatial_shape, spatial_shape,
                         kernel_size, stride, padding, dilation, out_padding,
                         subm, transpose):
    return _ops().get_indice_pairs_4d(
        indices, batch_size, out_spatial_shape, spatial_shape, kernel_size,
        stride, padding, dilation, out_padding, subm, transpose,
    )


def get_indice_pairs_grid_2d(indices, grid, batch_size, out_spatial_shape,
                             spatial_shape, kernel_size, stride, padding,
                             dilation, out_padding, subm, transpose):
    return _ops().get_indice_pairs_grid_2d(
        indices, grid, batch_size, out_spatial_shape, spatial_shape,
        kernel_size, stride, padding, dilation, out_padding, subm, transpose,
    )


def get_indice_pairs_grid_3d(indices, grid, batch_size, out_spatial_shape,
                             spatial_shape, kernel_size, stride, padding,
                             dilation, out_padding, subm, transpose):
    return _ops().get_indice_pairs_grid_3d(
        indices, grid, batch_size, out_spatial_shape, spatial_shape,
        kernel_size, stride, padding, dilation, out_padding, subm, transpose,
    )


def _check_dtype(features):
    if features.dtype == torch.float32:
        return "fp32"
    if features.dtype == torch.float16:
        return "half"
    raise TypeError(f"sparse operators do not support {features.dtype}")


def indice_conv_fp32(features, filters, indice_pairs, indice_num, num_act_out, inverse=0, sub_m=0):
    return _ops().indice_conv_fp32(
        features, filters, indice_pairs, indice_num, num_act_out, inverse, sub_m
    )


def indice_conv_backward_fp32(features, filters, out_grad, indice_pairs, indice_num, inverse=0, sub_m=0):
    return _ops().indice_conv_backward_fp32(
        features, filters, out_grad, indice_pairs, indice_num, inverse, sub_m
    )


def indice_conv_half(features, filters, indice_pairs, indice_num, num_act_out, inverse=0, sub_m=0):
    return _ops().indice_conv_half(
        features, filters, indice_pairs, indice_num, num_act_out, inverse, sub_m
    )


def indice_conv_backward_half(features, filters, out_grad, indice_pairs, indice_num, inverse=0, sub_m=0):
    return _ops().indice_conv_backward_half(
        features, filters, out_grad, indice_pairs, indice_num, inverse, sub_m
    )


def fused_indice_conv_fp32(features, filters, bias, indice_pairs, indice_num, num_act_out, inverse=0, sub_m=0):
    return _ops().fused_indice_conv_fp32(
        features, filters, bias, indice_pairs, indice_num, num_act_out, inverse, sub_m
    )


def fused_indice_conv_half(features, filters, bias, indice_pairs, indice_num, num_act_out, inverse=0, sub_m=0):
    return _ops().fused_indice_conv_half(
        features, filters, bias, indice_pairs, indice_num, num_act_out, inverse, sub_m
    )


def indice_maxpool_fp32(features, indice_pairs, indice_num, num_act):
    return _ops().indice_maxpool_fp32(features, indice_pairs, indice_num, num_act)


def indice_maxpool_backward_fp32(features, out_features, out_grad, indice_pairs, indice_num):
    return _ops().indice_maxpool_backward_fp32(
        features, out_features, out_grad, indice_pairs, indice_num
    )


def indice_maxpool_half(features, indice_pairs, indice_num, num_act):
    return _ops().indice_maxpool_half(features, indice_pairs, indice_num, num_act)


def indice_maxpool_backward_half(features, out_features, out_grad, indice_pairs, indice_num):
    return _ops().indice_maxpool_backward_half(
        features, out_features, out_grad, indice_pairs, indice_num
    )


class IndiceConvFunction(torch.autograd.Function):
    """Differentiable sparse convolution with automatic dtype dispatch."""

    @staticmethod
    def forward(ctx, features, filters, indice_pairs, indice_num, num_act_out, inverse=False, subm=False):
        suffix = _check_dtype(features)
        if filters.dtype != features.dtype:
            raise TypeError("features and filters must have the same dtype")
        function = globals()[f"indice_conv_{suffix}"]
        output = function(
            features, filters, indice_pairs, indice_num, int(num_act_out),
            int(inverse), int(subm),
        )
        ctx.save_for_backward(features, filters, indice_pairs, indice_num)
        ctx.inverse, ctx.subm, ctx.suffix = bool(inverse), bool(subm), suffix
        return output

    @staticmethod
    def backward(ctx, grad_output):
        features, filters, indice_pairs, indice_num = ctx.saved_tensors
        function = globals()[f"indice_conv_backward_{ctx.suffix}"]
        grad_features, grad_filters = function(
            features, filters, grad_output.contiguous(), indice_pairs, indice_num,
            int(ctx.inverse), int(ctx.subm),
        )
        return grad_features, grad_filters, None, None, None, None, None


def indice_conv(features, filters, indice_pairs, indice_num, num_act_out, inverse=False, subm=False, bias=None):
    """Apply an spconv-style sparse convolution using TurboPhysAI kernels."""

    output = IndiceConvFunction.apply(
        features, filters, indice_pairs, indice_num, num_act_out, inverse, subm
    )
    return output if bias is None else output + bias


class IndiceMaxPoolFunction(torch.autograd.Function):
    """Differentiable sparse max pooling with automatic dtype dispatch."""

    @staticmethod
    def forward(ctx, features, indice_pairs, indice_num, num_act):
        suffix = _check_dtype(features)
        output = globals()[f"indice_maxpool_{suffix}"](
            features, indice_pairs, indice_num, int(num_act)
        )
        ctx.save_for_backward(features, output, indice_pairs, indice_num)
        ctx.suffix = suffix
        return output

    @staticmethod
    def backward(ctx, grad_output):
        features, output, indice_pairs, indice_num = ctx.saved_tensors
        grad_features = globals()[f"indice_maxpool_backward_{ctx.suffix}"](
            features, output, grad_output.contiguous(), indice_pairs, indice_num
        )
        return grad_features, None, None, None


def indice_maxpool(features, indice_pairs, indice_num, num_act):
    """Apply spconv-style sparse max pooling using TurboPhysAI kernels."""

    return IndiceMaxPoolFunction.apply(features, indice_pairs, indice_num, num_act)


__all__ = [
    "get_indice_pairs",
    "get_indice_pairs_2d",
    "get_indice_pairs_3d",
    "get_indice_pairs_4d",
    "get_indice_pairs_grid_2d",
    "get_indice_pairs_grid_3d",
    "IndiceConvFunction",
    "indice_conv",
    "indice_conv_fp32",
    "indice_conv_backward_fp32",
    "indice_conv_half",
    "indice_conv_backward_half",
    "fused_indice_conv_fp32",
    "fused_indice_conv_half",
    "IndiceMaxPoolFunction",
    "indice_maxpool",
    "indice_maxpool_fp32",
    "indice_maxpool_backward_fp32",
    "indice_maxpool_half",
    "indice_maxpool_backward_half",
]
