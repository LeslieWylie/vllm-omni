# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import vllm_omni.diffusion.offloader.distributed_layerwise_backend as backend_module
from vllm_omni.diffusion.hooks import HookRegistry
from vllm_omni.diffusion.offloader.base import OffloadConfig, OffloadStrategy
from vllm_omni.diffusion.offloader.distributed_layerwise_backend import (
    DistributedLayerwiseOffloadBackend,
)
from vllm_omni.diffusion.offloader.module_collector import PipelineModules
from vllm_omni.diffusion.offloader.module_residency import BoundedAllocatorCache
from vllm_omni.diffusion.offloader.offload_plan import OffloadPlan

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


class _DummyStream:
    def wait_stream(self, _stream) -> None:
        return None

    def wait_event(self, _event) -> None:
        return None


class _DummyEvent:
    def record(self, _stream) -> None:
        return None


@contextmanager
def _dummy_stream(_stream):
    yield


@pytest.fixture
def patched_runtime(monkeypatch):
    platform = backend_module.current_omni_platform
    monkeypatch.setattr(platform, "Stream", _DummyStream)
    monkeypatch.setattr(platform, "Event", _DummyEvent)
    monkeypatch.setattr(platform, "current_stream", _DummyStream)
    monkeypatch.setattr(platform, "stream", _dummy_stream)
    monkeypatch.setattr(platform, "synchronize", lambda: None)
    monkeypatch.setattr(platform, "empty_cache", lambda: None)


class _AddBlock(nn.Module):
    def __init__(self, value: float, *, width: int = 1) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.full((width,), value, dtype=torch.float32))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.weight.sum()


class _Encoder(nn.Module):
    def __init__(self, sizes: tuple[int, ...] = (3, 4)) -> None:
        super().__init__()
        self.vision = nn.Module()
        self.vision.blocks = nn.ModuleList(_AddBlock(index + 1) for index in range(sizes[0]))
        self.text = nn.Module()
        self.text.layers = nn.ModuleList(_AddBlock(index + 11) for index in range(sizes[1]))

    @property
    def is_loaded(self) -> bool:
        return True

    def load_to_device(self) -> None:
        return None

    def offload_to_cpu(self) -> None:
        return None


class _EncoderStub(nn.Module):
    @property
    def is_loaded(self) -> bool:
        return False

    def load_to_device(self) -> None:
        return None

    def offload_to_cpu(self) -> None:
        return None


def _backend(budget: int) -> DistributedLayerwiseOffloadBackend:
    return DistributedLayerwiseOffloadBackend(
        OffloadConfig(
            strategy=OffloadStrategy.DISTRIBUTED_LAYER_WISE,
            pin_cpu_memory=False,
            dlo_use_allgather=False,
            dlo_encoder_resident_layers=budget,
        ),
        torch.device("cpu"),
    )


def _od_config(**overrides):
    values = {
        "enable_cpu_offload": False,
        "enable_layerwise_offload": False,
        "enable_distributed_layerwise_offload": True,
        "dlo_use_allgather": False,
        "dlo_resident_layers": 0,
        "dlo_encoder_resident_layers": 0,
        "pin_cpu_memory": False,
        "parallel_config": SimpleNamespace(
            use_hsdp=False,
            data_parallel_size=1,
            sequence_parallel_size=1,
        ),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _plan(*, resident_paths: frozenset[str] | None = None) -> OffloadPlan:
    return OffloadPlan(
        encoder_block_attrs={"text_encoder": ("vision.blocks", "text.layers")},
        resident_encoder_block_paths=(
            frozenset({"text_encoder.text.layers"}) if resident_paths is None else resident_paths
        ),
        on_demand_component_paths=frozenset({"text_encoder"}),
    )


def _modules(**encoders: nn.Module) -> PipelineModules:
    return PipelineModules(
        dits=[],
        dit_names=[],
        encoder_names=list(encoders),
        encoders=list(encoders.values()),
        vaes=[],
    )


def _install(
    backend: DistributedLayerwiseOffloadBackend,
    plan: OffloadPlan,
    **encoders: nn.Module,
) -> None:
    modules = _modules(**encoders)
    backend._validate_encoder_residency_plan(modules, plan)
    for name, encoder in encoders.items():
        backend._try_layerwise_offload_encoder(encoder, name, plan)
    backend._load_resident_encoder_layers()


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"dlo_encoder_resident_layers": -1}, "must be >= 0"),
        (
            {
                "enable_distributed_layerwise_offload": False,
                "dlo_encoder_resident_layers": 1,
            },
            "requires distributed layerwise offload",
        ),
        (
            {"dlo_use_allgather": True, "dlo_encoder_resident_layers": 1},
            "requires --dlo-no-use-allgather",
        ),
    ],
)
def test_encoder_residency_config_rejects_invalid_modes(overrides, match):
    with pytest.raises(ValueError, match=match):
        OffloadConfig.from_od_config(_od_config(**overrides))


@pytest.mark.parametrize(
    ("budget", "resident_count", "streamed_count"),
    [(0, 0, 7), (2, 2, 5), (4, 4, 3)],
)
def test_zero_partial_and_all_residency(
    patched_runtime,
    budget,
    resident_count,
    streamed_count,
):
    encoder = _Encoder()
    backend = _backend(budget)

    _install(backend, _plan(), text_encoder=encoder)

    state = backend._encoder_offload_states[0]
    assert len(backend._resident_encoder_blocks) == resident_count
    assert sum(len(group) for group in state.streamed_groups) == streamed_count
    assert len(state.hooks) == streamed_count
    assert (backend._resident_encoder_stager is not None) is (budget > 0)
    assert all(block.weight.numel() for block in encoder.text.layers[:budget])
    assert all(
        HookRegistry.get_or_create(block).get_hook("layerwise_offload") is None
        for block in encoder.text.layers[:budget]
    )

    backend._cleanup_encoder_offload()


def test_partial_residency_wires_only_the_streamed_suffix(patched_runtime):
    encoder = _Encoder(sizes=(2, 5))
    backend = _backend(2)
    _install(backend, _plan(), text_encoder=encoder)
    state = backend._encoder_offload_states[0]
    text_hooks = state.hooks[-3:]

    assert state.resident_groups == [list(encoder.text.layers[:2])]
    assert state.streamed_groups[-1] == list(encoder.text.layers[2:])
    assert text_hooks[0].next_block is encoder.text.layers[2]
    assert text_hooks[1].next_block is encoder.text.layers[3]
    assert text_hooks[2].next_block is encoder.text.layers[4]
    assert text_hooks[0]._prev_hook is text_hooks[-1]
    assert text_hooks[1]._prev_hook is text_hooks[0]
    assert text_hooks[2]._prev_hook is text_hooks[1]

    backend._cleanup_encoder_offload()


def test_single_layer_suffix_streams_itself(patched_runtime):
    encoder = _Encoder(sizes=(2, 4))
    backend = _backend(3)
    _install(backend, _plan(), text_encoder=encoder)
    state = backend._encoder_offload_states[0]
    hook = state.hooks[-1]
    suffix = encoder.text.layers[-1]

    assert hook.next_block is suffix
    assert hook._prev_hook is hook
    assert suffix.weight.numel() == 0
    result = suffix(torch.tensor([1.0]))
    assert torch.equal(result, torch.tensor([15.0]))
    assert suffix.weight.numel() == 0

    backend._cleanup_encoder_offload()


def test_repeated_and_cache_skipped_forwards_keep_prefix_materialized(patched_runtime):
    encoder = _Encoder(sizes=(2, 5))
    backend = _backend(2)
    _install(backend, _plan(), text_encoder=encoder)
    resident_ptrs = [block.weight.untyped_storage().data_ptr() for block in encoder.text.layers[:2]]

    def run_all() -> torch.Tensor:
        value = torch.tensor([0.0])
        for block in encoder.text.layers:
            value = block(value)
        for hook in encoder._omni_layerwise_hooks:
            hook.offload_layer()
        return value

    assert torch.equal(run_all(), torch.tensor([65.0]))
    assert torch.equal(run_all(), torch.tensor([65.0]))

    # Simulate a cache path that skips the first streamed block. The next
    # hook must synchronously recover its own weights through the back-link.
    for hook in encoder._omni_layerwise_hooks:
        hook.offload_layer()
    assert torch.equal(encoder.text.layers[3](torch.tensor([0.0])), torch.tensor([14.0]))
    assert [block.weight.untyped_storage().data_ptr() for block in encoder.text.layers[:2]] == resident_ptrs
    assert backend._resident_encoder_stager is not None
    assert backend._resident_encoder_stager.loaded

    backend._cleanup_encoder_offload()


def test_disable_and_reenable_restores_complete_cpu_stacks(patched_runtime, mocker):
    encoder = _Encoder(sizes=(2, 4))
    backend = _backend(2)
    plan = _plan()
    _install(backend, plan, text_encoder=encoder)
    encoder._omni_non_block_stager = mocker.Mock(loaded=True)
    backend.enabled = True

    backend.disable()

    assert not encoder._omni_layerwise_enabled
    encoder._omni_non_block_stager.offload.assert_called_once_with()
    assert all(block.weight.numel() for block in encoder.vision.blocks)
    assert all(block.weight.numel() for block in encoder.text.layers)
    assert all(
        HookRegistry.get_or_create(block).get_hook("layerwise_offload") is None
        for block in (*encoder.vision.blocks, *encoder.text.layers)
    )

    _install(backend, plan, text_encoder=encoder)
    assert encoder._omni_layerwise_enabled
    assert len(backend._resident_encoder_blocks) == 2
    backend._cleanup_encoder_offload()


def test_multiple_encoders_and_stacks_share_per_stack_budget(patched_runtime):
    first = _Encoder(sizes=(3, 3))
    second = _Encoder(sizes=(2, 4))
    plan = OffloadPlan(
        encoder_block_attrs={
            "first": ("vision.blocks", "text.layers"),
            "second": ("vision.blocks", "text.layers"),
        },
        resident_encoder_block_paths=frozenset(
            {
                "first.vision.blocks",
                "first.text.layers",
                "second.text.layers",
            }
        ),
        on_demand_component_paths=frozenset({"first", "second"}),
    )
    backend = _backend(1)

    _install(backend, plan, first=first, second=second)

    assert len(backend._resident_encoder_blocks) == 3
    assert sum(len(state.resident_groups) for state in backend._encoder_offload_states) == 3
    assert all(len(group) == 1 for state in backend._encoder_offload_states for group in state.resident_groups)
    backend._cleanup_encoder_offload()


def test_parameter_free_encoder_rank_accepts_global_residency_declaration(patched_runtime):
    backend = _backend(2)
    plan = _plan()
    stub = _EncoderStub()

    backend._validate_encoder_residency_plan(
        _modules(text_encoder=stub),
        plan,
    )

    assert not backend._try_layerwise_offload_encoder(stub, "text_encoder", plan)
    assert not backend._resident_encoder_blocks


def test_multiple_encoder_cache_owners_are_rejected(patched_runtime, mocker):
    first = _Encoder()
    second = _Encoder()
    first._omni_component_cache = mocker.Mock()
    second._omni_component_cache = mocker.Mock()
    plan = OffloadPlan(
        encoder_block_attrs={
            "first": ("text.layers",),
            "second": ("text.layers",),
        },
        resident_encoder_block_paths=frozenset({"first.text.layers", "second.text.layers"}),
        on_demand_component_paths=frozenset({"first", "second"}),
    )
    backend = _backend(1)

    with pytest.raises(ValueError, match="different component-cache owners"):
        backend._validate_encoder_residency_plan(
            _modules(first=first, second=second),
            plan,
        )


def test_resident_encoder_requires_pipeline_managed_stage_ownership(patched_runtime):
    encoder = _Encoder()
    plan = OffloadPlan(
        encoder_block_attrs={"text_encoder": ("text.layers",)},
        resident_encoder_block_paths=frozenset({"text_encoder.text.layers"}),
    )
    backend = _backend(1)

    with pytest.raises(ValueError, match="pipeline-managed stage ownership"):
        backend._validate_encoder_residency_plan(
            _modules(text_encoder=encoder),
            plan,
        )


def test_resident_encoder_requires_component_lifecycle_methods(patched_runtime):
    encoder = nn.Module()
    encoder.text = nn.Module()
    encoder.text.layers = nn.ModuleList([_AddBlock(1), _AddBlock(2)])
    plan = OffloadPlan(
        encoder_block_attrs={"text_encoder": ("text.layers",)},
        resident_encoder_block_paths=frozenset({"text_encoder.text.layers"}),
        on_demand_component_paths=frozenset({"text_encoder"}),
    )
    backend = _backend(1)

    with pytest.raises(ValueError, match="must implement load_to_device"):
        backend._validate_encoder_residency_plan(
            _modules(text_encoder=encoder),
            plan,
        )


@pytest.mark.parametrize(
    ("plan", "encoders", "match"),
    [
        (OffloadPlan(), {"text_encoder": _Encoder()}, "declares no resident_encoder_block_paths"),
        (
            OffloadPlan(
                encoder_block_attrs={"text_encoder": ("text.layers",)},
                resident_encoder_block_paths=frozenset({"text_encoder.vision.blocks"}),
                on_demand_component_paths=frozenset({"text_encoder"}),
            ),
            {"text_encoder": _Encoder()},
            "must also be declared",
        ),
        (
            OffloadPlan(
                encoder_block_attrs={"missing": ("text.layers",)},
                resident_encoder_block_paths=frozenset({"missing.text.layers"}),
                on_demand_component_paths=frozenset({"missing"}),
            ),
            {"text_encoder": _Encoder()},
            "were not discovered",
        ),
    ],
)
def test_unsupported_residency_plans_fail_before_hook_installation(
    patched_runtime,
    plan,
    encoders,
    match,
):
    backend = _backend(1)
    with pytest.raises(ValueError, match=match):
        backend._validate_encoder_residency_plan(_modules(**encoders), plan)
    assert not backend._encoder_offload_states


def test_duplicate_aliases_for_one_stack_are_rejected(patched_runtime):
    encoder = _Encoder()
    encoder.alias = encoder.text.layers
    plan = OffloadPlan(
        encoder_block_attrs={"text_encoder": ("text.layers", "alias")},
        resident_encoder_block_paths=frozenset({"text_encoder.text.layers", "text_encoder.alias"}),
        on_demand_component_paths=frozenset({"text_encoder"}),
    )
    backend = _backend(1)

    with pytest.raises(ValueError, match="same block stack"):
        backend._validate_encoder_residency_plan(
            _modules(text_encoder=encoder),
            plan,
        )


def test_overlapping_block_stacks_are_rejected(patched_runtime):
    encoder = _Encoder()
    encoder.overlap = nn.ModuleList(list(encoder.text.layers))
    plan = OffloadPlan(
        encoder_block_attrs={"text_encoder": ("text.layers", "overlap")},
        resident_encoder_block_paths=frozenset({"text_encoder.text.layers", "text_encoder.overlap"}),
        on_demand_component_paths=frozenset({"text_encoder"}),
    )
    backend = _backend(1)

    with pytest.raises(ValueError, match="block stacks overlap"):
        backend._validate_encoder_residency_plan(
            _modules(text_encoder=encoder),
            plan,
        )


def test_budget_larger_than_declared_stack_is_rejected(patched_runtime):
    encoder = _Encoder(sizes=(2, 3))
    backend = _backend(4)

    with pytest.raises(ValueError, match="exceeds the 3 blocks"):
        backend._validate_encoder_residency_plan(
            _modules(text_encoder=encoder),
            _plan(),
        )


def test_resident_group_preserves_packed_stride_aliases_and_quantization_owner(patched_runtime):
    encoder = _Encoder(sizes=(2, 2))
    storage = torch.arange(16, dtype=torch.uint8)
    first = encoder.text.layers[0]
    second = encoder.text.layers[1]
    first.weight = nn.Parameter(
        torch.empty(0, dtype=torch.uint8).set_(storage.untyped_storage(), 0, (2, 2), (4, 1)),
        requires_grad=False,
    )
    second.weight = nn.Parameter(
        torch.empty(0, dtype=torch.uint8).set_(storage.untyped_storage(), 2, (2, 2), (4, 1)),
        requires_grad=False,
    )
    quant_method = object()
    first.quant_method = quant_method
    second.quant_method = quant_method
    backend = _backend(2)

    _install(backend, _plan(), text_encoder=encoder)

    assert first.weight.untyped_storage().data_ptr() == second.weight.untyped_storage().data_ptr()
    assert first.weight.stride() == second.weight.stride() == (4, 1)
    assert (first.weight.storage_offset(), second.weight.storage_offset()) == (0, 2)
    assert first.quant_method is second.quant_method is quant_method
    backend._cleanup_encoder_offload()


def test_storage_alias_crossing_residency_boundary_is_rejected(patched_runtime):
    encoder = _Encoder(sizes=(2, 4))
    storage = torch.arange(8, dtype=torch.float32)
    encoder.text.layers[0].weight = nn.Parameter(storage[:4])
    encoder.text.layers[2].weight = nn.Parameter(storage[2:6])
    backend = _backend(2)

    with pytest.raises(ValueError, match="storage aliases cross"):
        backend._validate_encoder_residency_plan(
            _modules(text_encoder=encoder),
            _plan(),
        )


def test_shared_submodule_crossing_residency_boundary_is_rejected(patched_runtime):
    encoder = _Encoder(sizes=(2, 4))
    shared = nn.Linear(2, 2)
    encoder.text.layers[0].shared = shared
    encoder.text.layers[2].shared = shared
    backend = _backend(2)

    with pytest.raises(ValueError, match="module is declared in both"):
        backend._validate_encoder_residency_plan(
            _modules(text_encoder=encoder),
            _plan(),
        )


def test_rank_local_tp_shapes_are_not_gathered(patched_runtime):
    rank0 = _Encoder(sizes=(2, 2))
    rank1 = _Encoder(sizes=(2, 2))
    rank0.text.layers[0] = _AddBlock(1, width=3)
    rank1.text.layers[0] = _AddBlock(2, width=5)
    plan = OffloadPlan(
        encoder_block_attrs={
            "rank0": ("text.layers",),
            "rank1": ("text.layers",),
        },
        resident_encoder_block_paths=frozenset({"rank0.text.layers", "rank1.text.layers"}),
        on_demand_component_paths=frozenset({"rank0", "rank1"}),
    )
    backend = _backend(1)

    _install(backend, plan, rank0=rank0, rank1=rank1)

    assert rank0.text.layers[0].weight.shape == (3,)
    assert rank1.text.layers[0].weight.shape == (5,)
    assert torch.equal(rank0.text.layers[0].weight, torch.ones(3))
    assert torch.equal(rank1.text.layers[0].weight, torch.full((5,), 2.0))
    backend._cleanup_encoder_offload()


def test_startup_failure_cleans_resident_and_streamed_encoder_state(
    monkeypatch,
    patched_runtime,
    mocker,
):
    class _Transformer(nn.Module):
        _layerwise_offload_blocks_attrs = ["blocks"]

        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([_AddBlock(1), _AddBlock(2)])

    class _Pipeline(nn.Module):
        _offload_plan = _plan()

        def __init__(self) -> None:
            super().__init__()
            self.transformer = _Transformer()
            self.text_encoder = _Encoder(sizes=(2, 4))

    pipeline = _Pipeline()
    cache = mocker.Mock(spec=BoundedAllocatorCache)
    pipeline.text_encoder._omni_component_cache = cache
    backend = _backend(2)
    monkeypatch.setattr(backend, "_register_on_demand_hook", lambda *args, **kwargs: None)

    def fail_after_encoder_residency(*args, **kwargs):
        raise RuntimeError("DiT setup failed")

    monkeypatch.setattr(backend, "_prepare_dit_non_block_modules", fail_after_encoder_residency)

    with pytest.raises(RuntimeError, match="DiT setup failed"):
        backend.enable(pipeline)

    assert not pipeline.text_encoder._omni_layerwise_enabled
    assert not backend._encoder_offload_states
    assert backend._resident_encoder_stager is None
    assert all(block.weight.numel() for block in pipeline.text_encoder.text.layers)
    cache.release_if_needed.assert_any_call(force=True)
