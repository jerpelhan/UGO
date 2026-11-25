# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

import torch

from functions.ms_deform_attn_func import (
    MSDeformAttnFunction,
    ms_deform_attn_core_pytorch,
)


@torch.no_grad()
def check_forward(dtype, rtol, atol):
    batch_size, heads, channels = 1, 2, 2
    query_count, level_count, point_count = 2, 2, 2
    shapes = torch.as_tensor(
        [(6, 4), (3, 2)],
        dtype=torch.long,
        device="cuda",
    )
    level_start_index = torch.cat(
        (
            shapes.new_zeros((1,)),
            shapes.prod(1).cumsum(0)[:-1],
        )
    )
    spatial_size = sum(
        (height * width).item()
        for height, width in shapes
    )
    torch.manual_seed(3)
    value = torch.rand(
        batch_size,
        spatial_size,
        heads,
        channels,
        device="cuda",
    ) * 0.01
    sampling_locations = torch.rand(
        batch_size,
        query_count,
        heads,
        level_count,
        point_count,
        2,
        device="cuda",
    )
    attention_weights = torch.rand(
        batch_size,
        query_count,
        heads,
        level_count,
        point_count,
        device="cuda",
    ) + 1e-5
    attention_weights /= attention_weights.sum(
        -1,
        keepdim=True,
    ).sum(
        -2,
        keepdim=True,
    )
    value = value.to(dtype)
    sampling_locations = sampling_locations.to(dtype)
    attention_weights = attention_weights.to(dtype)
    expected = ms_deform_attn_core_pytorch(
        value,
        shapes,
        sampling_locations,
        attention_weights,
    )
    actual = MSDeformAttnFunction.apply(
        value,
        shapes,
        level_start_index,
        sampling_locations,
        attention_weights,
        2,
    )
    max_error = (actual - expected).abs().max().item()
    if not torch.allclose(actual, expected, rtol=rtol, atol=atol):
        raise AssertionError(
            f"{dtype} forward mismatch: max absolute error {max_error:.3e}"
        )
    print(f"{dtype}: max absolute error {max_error:.3e}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    check_forward(torch.double, rtol=1e-5, atol=1e-8)
    check_forward(torch.float, rtol=1e-2, atol=1e-3)
