# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from vllm_omni.diffusion.lora.layers.base_linear import DiffusionBaseLinearLayerWithLoRA
from vllm_omni.diffusion.lora.layers.column_parallel_linear import DiffusionMergedColumnParallelLinearWithLoRA

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclass
class _DummyLoRAConfig:
    fully_sharded_loras: bool = False


class _DummyQuantMethod:
    def __init__(self, weight: torch.Tensor):
        self._weight = weight

    def apply(self, _base_layer, x: torch.Tensor, bias: torch.Tensor | None):
        y = x @ self._weight.t()
        if bias is not None:
            y = y + bias
        return y


def test_diffusion_base_linear_apply_multi_slice(monkeypatch):
    # Build a fake diffusion LoRA layer with 2 slices and rank=2.
    layer = DiffusionBaseLinearLayerWithLoRA.__new__(DiffusionBaseLinearLayerWithLoRA)
    layer.tp_size = 1
    layer.lora_config = _DummyLoRAConfig()

    in_dim = 3
    out_slices = (2, 1)
    rank = 2

    # Base weight: identity-ish mapping to make base output easy to reason about.
    base_weight = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    layer.base_layer = type("Base", (), {})()
    layer.base_layer.quant_method = _DummyQuantMethod(base_weight)

    # Allocate stacked weights: (max_loras=1, 1, rank, in_dim) and (1, 1, out_slice, rank)
    a0 = torch.zeros((1, 1, rank, in_dim))
    b0 = torch.zeros((1, 1, out_slices[0], rank))
    a1 = torch.zeros((1, 1, rank, in_dim))
    b1 = torch.zeros((1, 1, out_slices[1], rank))

    # Slice 0: delta0 = (x @ A0.T) @ B0.T
    A0 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # (2, 3)
    B0 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # (2, 2)
    a0[0, 0, :, :] = A0
    b0[0, 0, :, :] = B0

    # Slice 1: delta1 = (x @ A1.T) @ B1.T
    A1 = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])  # (2, 3)
    B1 = torch.tensor([[2.0, 0.0]])  # (1, 2)
    a1[0, 0, :, :] = A1
    b1[0, 0, :, :] = B1

    layer.lora_a_stacked = (a0, a1)
    layer.lora_b_stacked = (b0, b1)
    layer.output_slices = out_slices

    addmm_calls = []
    original_addmm = torch.addmm

    def _record_addmm(input, mat1, mat2, *, out=None):
        addmm_calls.append((input, out))
        return original_addmm(input, mat1, mat2, out=out)

    monkeypatch.setattr(torch, "addmm", _record_addmm)

    x = torch.tensor([[1.0, 2.0, 3.0]])
    with torch.inference_mode():
        out = layer.apply(x)

    # Base output is identity: [1,2,3]
    base_out = x @ base_weight.t()
    # delta0:
    # (x @ A0.T) = [1,2]
    # [1,2] @ B0.T = [1,2]
    delta0 = torch.tensor([[1.0, 2.0]])
    # delta1:
    # (x @ A1.T) = [3,1]
    # [3,1] @ B1.T = [6]
    delta1 = torch.tensor([[6.0]])
    expected = torch.cat([base_out[:, :2] + delta0, base_out[:, 2:3] + delta1], dim=-1)
    assert torch.allclose(out, expected)
    assert len(addmm_calls) == 2
    assert all(input is out for input, out in addmm_calls)


def test_diffusion_base_linear_apply_preserves_autograd(monkeypatch):
    layer = DiffusionBaseLinearLayerWithLoRA.__new__(DiffusionBaseLinearLayerWithLoRA)
    layer.tp_size = 1
    layer.lora_config = _DummyLoRAConfig()

    layer.base_layer = type("Base", (), {})()
    layer.base_layer.quant_method = _DummyQuantMethod(torch.eye(2))

    a = torch.tensor([[[[1.0, 2.0]]]], requires_grad=True)
    b = torch.tensor([[[[3.0], [4.0]]]], requires_grad=True)
    layer.lora_a_stacked = (a,)
    layer.lora_b_stacked = (b,)
    layer.output_slices = (2,)
    layer._diffusion_lora_active_slices = (True,)

    def _unexpected_addmm(*args, **kwargs):
        raise AssertionError("the out= fast path must not run with autograd enabled")

    monkeypatch.setattr(torch, "addmm", _unexpected_addmm)

    x = torch.tensor([[2.0, 1.0]], requires_grad=True)
    out = layer.apply(x)
    assert torch.allclose(out, torch.tensor([[14.0, 17.0]]))

    out.sum().backward()
    assert torch.allclose(x.grad, torch.tensor([[8.0, 15.0]]))
    assert torch.allclose(a.grad, torch.tensor([[[[14.0, 7.0]]]]))
    assert torch.allclose(b.grad, torch.tensor([[[[4.0], [4.0]]]]))


def test_diffusion_base_linear_apply_falls_back_for_mixed_output_dtype(monkeypatch):
    layer = DiffusionBaseLinearLayerWithLoRA.__new__(DiffusionBaseLinearLayerWithLoRA)
    layer.tp_size = 1
    layer.lora_config = _DummyLoRAConfig()

    class _Float64OutputQuantMethod(_DummyQuantMethod):
        def apply(self, base_layer, x: torch.Tensor, bias: torch.Tensor | None):
            return super().apply(base_layer, x, bias).to(torch.float64)

    layer.base_layer = type("Base", (), {})()
    layer.base_layer.quant_method = _Float64OutputQuantMethod(torch.eye(2))
    layer.lora_a_stacked = (torch.ones((1, 1, 1, 2)),)
    layer.lora_b_stacked = (torch.tensor([[[[1.0], [2.0]]]]),)
    layer.output_slices = (2,)
    layer._diffusion_lora_active_slices = (True,)

    def _unexpected_addmm(*args, **kwargs):
        raise AssertionError("mixed output and LoRA dtypes must use the functional fallback")

    monkeypatch.setattr(torch, "addmm", _unexpected_addmm)

    with torch.inference_mode():
        out = layer.apply(torch.tensor([[1.0, 2.0]]))

    assert out.dtype == torch.float64
    assert torch.allclose(out, torch.tensor([[4.0, 8.0]], dtype=torch.float64))


@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float32])
def test_epilogue_scaling_preserves_native_bf16_order(output_dtype):
    layer = DiffusionBaseLinearLayerWithLoRA.__new__(DiffusionBaseLinearLayerWithLoRA)
    layer.tp_size = 1
    layer.lora_config = _DummyLoRAConfig()
    layer.base_layer = type("Base", (), {})()

    class _Output:
        def apply(self, _layer, x, _bias):
            return x.to(output_dtype).clone()

    generator = torch.Generator().manual_seed(12)
    a = torch.randn(3, 4, generator=generator).bfloat16()
    b = torch.randn(4, 3, generator=generator).bfloat16()
    x = torch.randn(5, 4, generator=generator)  # Native hook casts input to the adapter dtype.
    layer.base_layer.quant_method = _Output()
    layer.lora_a_stacked = (a[None, None],)
    layer.lora_b_stacked = (b[None, None],)
    layer.output_slices = (4,)
    layer._diffusion_lora_epilogue_scale = 1.37
    with torch.inference_mode():
        expected = x.to(output_dtype).clone()
        low_rank = x.bfloat16() @ a.T
        if output_dtype == torch.bfloat16:
            expected.addmm_(low_rank, b.T, alpha=1.37)
        else:
            expected.add_(((low_rank @ b.T) * 1.37).to(output_dtype))
        torch.testing.assert_close(layer.apply(x), expected, rtol=0, atol=0)


def test_diffusion_lora_buffer_device_persists_across_reallocation(monkeypatch):
    from vllm.lora.layers.base_linear import BaseLinearLayerWithLoRA

    layer = DiffusionBaseLinearLayerWithLoRA.__new__(DiffusionBaseLinearLayerWithLoRA)
    torch.nn.Module.__init__(layer)
    layer.base_layer = torch.nn.Module()
    layer.n_slices = 1
    layer.lora_a_stacked = (torch.zeros(2),)
    layer.lora_b_stacked = (torch.zeros(3),)

    layer._set_diffusion_lora_buffer_device(torch.device("meta"))
    assert all(tensor.is_meta for tensor in (*layer.lora_a_stacked, *layer.lora_b_stacked))

    def recreate_on_cpu(self, max_loras, lora_config, model_config):
        self.lora_a_stacked = (torch.zeros(4),)
        self.lora_b_stacked = (torch.zeros(5),)

    monkeypatch.setattr(BaseLinearLayerWithLoRA, "create_lora_weights", recreate_on_cpu)
    layer.create_lora_weights(max_loras=1, lora_config=object())

    assert all(tensor.is_meta for tensor in (*layer.lora_a_stacked, *layer.lora_b_stacked))


def test_diffusion_base_linear_reset_lora_disables_fast_path(monkeypatch):
    # Verify that after reset_lora(), apply() skips LoRA matmuls even if the
    # LoRA tensors are still allocated and non-empty.
    from vllm.lora.layers.base_linear import BaseLinearLayerWithLoRA

    layer = DiffusionBaseLinearLayerWithLoRA.__new__(DiffusionBaseLinearLayerWithLoRA)
    layer.tp_size = 1
    layer.lora_config = _DummyLoRAConfig()

    in_dim = 2
    out_dim = 2
    rank = 1

    base_weight = torch.eye(in_dim)
    layer.base_layer = type("Base", (), {})()
    layer.base_layer.quant_method = _DummyQuantMethod(base_weight)

    a = torch.ones((1, 1, rank, in_dim))
    b = torch.tensor([[[[1.0], [2.0]]]])  # (1,1,out_dim,rank)

    layer.lora_a_stacked = (a,)
    layer.lora_b_stacked = (b,)
    layer.output_slices = (out_dim,)
    layer._diffusion_lora_active_slices = (True,)

    x = torch.tensor([[1.0, 2.0]])
    out_active = layer.apply(x)
    assert torch.allclose(out_active, torch.tensor([[4.0, 8.0]]))

    monkeypatch.setattr(BaseLinearLayerWithLoRA, "reset_lora", lambda self, index: None)
    layer.reset_lora(0)

    assert layer._diffusion_lora_active_slices == (False,)
    out_inactive = layer.apply(x)
    assert torch.allclose(out_inactive, x)


def test_diffusion_base_linear_apply_respects_inactive_slices():
    # Build a fake diffusion LoRA layer with 2 slices and rank=2.
    layer = DiffusionBaseLinearLayerWithLoRA.__new__(DiffusionBaseLinearLayerWithLoRA)
    layer.tp_size = 1
    layer.lora_config = _DummyLoRAConfig()

    in_dim = 3
    out_slices = (2, 1)
    rank = 2

    base_weight = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    layer.base_layer = type("Base", (), {})()
    layer.base_layer.quant_method = _DummyQuantMethod(base_weight)

    a0 = torch.zeros((1, 1, rank, in_dim))
    b0 = torch.zeros((1, 1, out_slices[0], rank))
    a1 = torch.zeros((1, 1, rank, in_dim))
    b1 = torch.zeros((1, 1, out_slices[1], rank))

    A0 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # (2, 3)
    B0 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # (2, 2)
    a0[0, 0, :, :] = A0
    b0[0, 0, :, :] = B0

    A1 = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])  # (2, 3)
    B1 = torch.tensor([[2.0, 0.0]])  # (1, 2)
    a1[0, 0, :, :] = A1
    b1[0, 0, :, :] = B1

    layer.lora_a_stacked = (a0, a1)
    layer.lora_b_stacked = (b0, b1)
    layer.output_slices = out_slices
    layer._diffusion_lora_active_slices = (True, False)

    x = torch.tensor([[1.0, 2.0, 3.0]])
    out = layer.apply(x)

    # Only the first slice should be adapted.
    expected = torch.tensor([[2.0, 4.0, 3.0]])
    assert torch.allclose(out, expected)


def _shared_a_layer(tp_size: int = 1):
    """Exercise the real merged-column upload, including rank padding and TP B slicing."""
    layer = DiffusionMergedColumnParallelLinearWithLoRA.__new__(DiffusionMergedColumnParallelLinearWithLoRA)
    torch.nn.Module.__init__(layer)
    layer.tp_size = tp_size
    layer.n_slices = 2
    layer.lora_config = _DummyLoRAConfig()
    layer.output_slices = (3, 2)
    layer.output_sizes = tuple(size * tp_size for size in layer.output_slices)
    layer.output_ids = (tp_size - 1,) * 2
    layer.lora_a_stacked = tuple(torch.zeros(1, 1, 4, 3) for _ in range(2))
    layer.lora_b_stacked = tuple(torch.zeros(1, 1, size, 4) for size in layer.output_slices)
    layer.base_layer = torch.nn.Module()
    layer.base_layer.quant_method = _DummyQuantMethod(torch.arange(15).reshape(5, 3).float() / 8)
    a = torch.tensor([[1.0, 0.0, 0.5], [0.0, 1.0, -0.5]])
    b = [torch.arange(size * 2).reshape(size, 2).float() / 8 for size in layer.output_sizes]
    return layer, a, b


class _ShrinkCounter(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if str(func).startswith("aten.mm.") and args[1].shape == (3, 4):
            self.calls += 1
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("shared", [True, False])
def test_shared_a_shrink_preserves_padded_rank_and_tp_slices(tp_size, shared):
    layer, a, b = _shared_a_layer(tp_size)
    layer.set_lora(0, [a, a if shared else a.clone()], b)
    assert layer._diffusion_lora_a_sources == ((0, 0) if shared else ())
    x = torch.tensor([[1.0, 2.0, 3.0], [0.5, -1.0, 2.0]])
    # The old path remains the oracle; no tensor storage is shared after upload.
    assert layer.lora_a_stacked[0].data_ptr() != layer.lora_a_stacked[1].data_ptr()
    sources = layer._diffusion_lora_a_sources
    layer._diffusion_lora_a_sources = ()
    with torch.no_grad():
        expected = layer.apply(x)
        layer._diffusion_lora_a_sources = sources
        with _ShrinkCounter() as counter:
            actual = layer.apply(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert counter.calls == (1 if shared else 2)
    assert torch.count_nonzero(layer.lora_a_stacked[0][0, 0, 2:]) == 0
    for uploaded, source, size in zip(layer.lora_b_stacked, b, layer.output_slices):
        start = (tp_size - 1) * size
        torch.testing.assert_close(uploaded[0, 0, :, :2], source[start : start + size])


def test_shared_a_provenance_is_rebuilt_on_replace_and_reset():
    layer, a, b = _shared_a_layer()
    layer.set_lora(0, [a, a], b)
    assert layer._diffusion_lora_a_sources == (0, 0)
    layer.suspend_lora()
    with torch.no_grad(), _ShrinkCounter() as counter:
        layer.apply(torch.ones(1, 3))
    assert counter.calls == 0
    layer.resume_lora()
    assert layer._diffusion_lora_a_sources == (0, 0)
    layer.set_lora(0, [a, a.clone() + 1], b)
    assert layer._diffusion_lora_a_sources == ()
    with torch.no_grad(), _ShrinkCounter() as counter:
        layer.apply(torch.ones(1, 3))
    assert counter.calls == 2
    layer.set_lora(0, [a, a], b)
    layer.reset_lora(0)
    assert layer._diffusion_lora_a_sources == ()
    with torch.no_grad(), _ShrinkCounter() as counter:
        layer.apply(torch.ones(1, 3))
    assert counter.calls == 0


@pytest.mark.parametrize("compiling", [False, True])
@pytest.mark.parametrize("grad_enabled", [False, True])
def test_shared_a_shrink_keeps_compiled_and_grad_paths_unchanged(monkeypatch, compiling, grad_enabled):
    layer, a, b = _shared_a_layer()
    layer.set_lora(0, [a, a], b)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: compiling)
    with torch.set_grad_enabled(grad_enabled), _ShrinkCounter() as counter:
        layer.apply(torch.ones(1, 3))
    assert counter.calls == (2 if compiling or grad_enabled else 1)


def test_shared_a_failed_upload_discards_old_provenance(monkeypatch):
    from vllm.lora.layers.column_parallel_linear import MergedColumnParallelLinearWithLoRA

    layer, a, b = _shared_a_layer()
    layer.set_lora(0, [a, a], b)

    def fail_upload(*args, **kwargs):
        raise RuntimeError("upload failed")

    monkeypatch.setattr(MergedColumnParallelLinearWithLoRA, "set_lora", fail_upload)
    with pytest.raises(RuntimeError, match="upload failed"):
        layer.set_lora(0, [a, a.clone()], b)
    assert layer._diffusion_lora_a_sources == ()
