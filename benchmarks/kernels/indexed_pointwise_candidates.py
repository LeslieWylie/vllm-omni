# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Experimental CUDA pointwise tiling; not integrated into any service.

Use the current indexed_modulation.py expression order as the baseline.
Keep the FP32 expression order; leave all RMSNorm reductions unchanged.
"""

from vllm.triton_utils import tl, triton


@triton.jit
def tiled_pointwise(
    output_ptr,
    x_ptr,
    bank1_ptr,
    bank2_ptr,
    other_ptr,
    indices_ptr,
    hidden_size,
    stride_output,
    stride_x,
    stride_bank1,
    stride_bank2,
    stride_other,
    stride_indices,
    gate: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.program_id(1) * block + tl.arange(0, block)
    mask = columns < hidden_size
    index = tl.load(indices_ptr + row * stride_indices)
    x = tl.load(x_ptr + row * stride_x + columns, mask=mask, other=0.0).to(tl.float32)
    bank1 = tl.load(bank1_ptr + index * stride_bank1 + columns, mask=mask, other=0.0).to(tl.float32)
    if gate:
        other = tl.load(other_ptr + row * stride_other + columns, mask=mask, other=0.0).to(tl.float32)
        value = x + bank1 * other
    else:
        bank2 = tl.load(bank2_ptr + index * stride_bank2 + columns, mask=mask, other=0.0).to(tl.float32)
        value = x * (1.0 + bank2) + bank1
    tl.store(output_ptr + row * stride_output + columns, value, mask=mask)
