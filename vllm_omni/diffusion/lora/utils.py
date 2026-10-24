# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping
from typing import Any

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from vllm.config.lora import LoRAConfig
from vllm.logger import init_logger
from vllm.lora.layers.fused_moe import FusedMoEWithLoRA
from vllm.model_executor.layers.fused_moe import MoERunner
from vllm.platforms import current_platform

from vllm_omni.diffusion.lora.layers import (
    DiffusionColumnParallelLinearWithLoRA,
    DiffusionMergedColumnParallelLinearWithLoRA,
    DiffusionMergedQKVParallelLinearWithLoRA,
    DiffusionQKVParallelLinearWithLoRA,
    DiffusionReplicatedLinearWithLoRA,
    DiffusionRowParallelLinearWithLoRA,
)

logger = init_logger(__name__)


def _select_moe_lora_wrapper_cls() -> type:
    """Select the vLLM or Ascend MoE LoRA wrapper for the current platform."""
    if current_platform.device_type == "npu":
        from vllm_ascend.lora.fused_moe import AscendFusedMoEWithLoRA

        return AscendFusedMoEWithLoRA
    return FusedMoEWithLoRA


def _move_moe_lora_buffers(instance: FusedMoEWithLoRA, device: torch.device) -> None:
    """Keep MoE LoRA sidecars resident while DLO moves base weights.

    Rebuild expert views after moving their source W13/W2 stacks.
    """
    for attr in (
        "w13_lora_a_stacked",
        "w13_lora_b_stacked",
        "w2_lora_a_stacked",
        "w2_lora_b_stacked",
    ):
        buffers = getattr(instance, attr)
        setattr(instance, attr, tuple(buffer.to(device=device) for buffer in buffers))
    instance.adapter_enabled = instance.adapter_enabled.to(device=device)

    lora_a_stacked: list[torch.Tensor] = []
    lora_b_stacked: list[torch.Tensor] = []
    for lora_id in range(instance.max_loras):
        for expert_id in range(instance.local_num_experts):
            w13_a_expert_id = 0 if instance.enable_moe_shared_loras else expert_id
            w2_b_expert_id = 0 if instance.enable_moe_shared_loras else expert_id
            lora_a_stacked.extend(
                (
                    instance.w13_lora_a_stacked[0][lora_id][w13_a_expert_id],
                    instance.w2_lora_a_stacked[0][lora_id][expert_id],
                )
            )
            lora_b_stacked.extend(
                (
                    instance.w13_lora_b_stacked[0][lora_id][expert_id],
                    instance.w2_lora_b_stacked[0][lora_id][w2_b_expert_id],
                )
            )
            if instance._w13_slices == 2:
                lora_a_stacked.append(instance.w13_lora_a_stacked[1][lora_id][w13_a_expert_id])
                lora_b_stacked.append(instance.w13_lora_b_stacked[1][lora_id][expert_id])
    instance.lora_a_stacked = lora_a_stacked
    instance.lora_b_stacked = lora_b_stacked


def _validate_moe_lora_token_capacity(hidden_states: torch.Tensor, max_tokens: int) -> None:
    num_tokens = hidden_states.shape[0]
    if num_tokens > max_tokens:
        raise ValueError(
            f"MoE LoRA token count {num_tokens} exceeds the configured capacity {max_tokens}. "
            "Increase max_num_batched_tokens before model initialization."
        )


def _match_target_modules(module_name: str, target_modules: list[str]) -> bool:
    """from vllm/lora/model_manager.py _match_target_modules, helper function"""
    import regex as re

    return any(
        re.match(rf".*\.{target_module}$", module_name) or target_module == module_name
        for target_module in target_modules
    )


def _expand_expected_modules_for_packed_layers(
    supported_modules: set[str],
    packed_modules_mapping: dict[str, list[str]] | None,
) -> set[str]:
    """Expand expected LoRA module suffixes for packed (fused) projections.

    Some diffusion models use packed projections like `to_qkv` or `w13`, while
    LoRA checkpoints are typically saved against the logical sub-projections
    (e.g. `to_q`/`to_k`/`to_v`, `w1`/`w3`). The packed layer name is present in
    `supported_modules`, but the sublayer names are not. Expanding the set
    ensures these sublayer keys are not dropped when loading a LoRA checkpoint.

    The packed→sublayer mapping is model-specific and is derived from each
    diffusion model's `stacked_params_mapping` (used by `load_weights()`), so
    new packed layers are added alongside the model implementation rather than
    hard-coded in the LoRA framework.
    """
    expanded = set(supported_modules)
    if not packed_modules_mapping:
        return expanded

    for packed_name, sub_names in packed_modules_mapping.items():
        if packed_name in supported_modules:
            expanded.update(sub_names)

    return expanded


def from_layer_diffusion(
    layer: nn.Module,
    max_loras: int,
    lora_config: LoRAConfig,
    packed_modules_list: list[str],
    model_config: PretrainedConfig | None = None,
    max_num_batched_tokens: int | None = None,
) -> nn.Module:
    """
    Diffusion-specific layer replacement. similar to vLLM's `from_layer`
    """
    # MoE wrappers own the whole runner and do not use dense packed-layer mapping.
    if isinstance(layer, MoERunner):
        if max_num_batched_tokens is None:
            raise RuntimeError("MoE LoRA token capacity was not configured")
        wrapper_cls = _select_moe_lora_wrapper_cls()
        instance = wrapper_cls(layer)
        buffer_device: list[torch.device | None] = [None]
        _suspended: list[int | None] = [None]
        original_create_lora_weights = instance.create_lora_weights
        original_set_lora = instance.set_lora
        original_reset_lora = instance.reset_lora

        def _set_lora(*args, **kwargs) -> None:
            _suspended[0] = None
            original_set_lora(*args, **kwargs)

        def _reset_lora(*args, **kwargs) -> None:
            _suspended[0] = None
            original_reset_lora(*args, **kwargs)

        # Saved enablement belongs to the current upload only. Rebinding,
        # resetting or reallocating invalidates it, just like dense wrappers.
        instance.set_lora = _set_lora  # type: ignore[method-assign]
        instance.reset_lora = _reset_lora  # type: ignore[method-assign]

        def _set_diffusion_lora_buffer_device(device: torch.device) -> None:
            buffer_device[0] = device
            _move_moe_lora_buffers(instance, device)

        def _create_lora_weights(max_loras, lora_config, model_config=None) -> None:
            _suspended[0] = None
            original_create_lora_weights(max_loras, lora_config, model_config)
            if buffer_device[0] is not None:
                _move_moe_lora_buffers(instance, buffer_device[0])

        instance._set_diffusion_lora_buffer_device = _set_diffusion_lora_buffer_device  # type: ignore[method-assign]
        instance.create_lora_weights = _create_lora_weights  # type: ignore[method-assign]
        instance.create_lora_weights(max_loras, lora_config, model_config)
        # Runner pre-hooks initialize token counts and NPU communication context.
        assert len(layer._forward_pre_hooks) > 0, (
            "MoERunner lost its forward pre-hooks after LoRA wrapping; "
            "ForwardContext.num_tokens / NPU moe_comm_method would be "
            "uninitialized."
        )

        # Call through nn.Module.__call__ to preserve those hooks.
        # Instance-assigned functions capture the wrapper instead of taking self.
        def _forward_via_base_call(*args, **kwargs):
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None and args:
                hidden_states = args[0]
            if hidden_states is not None:
                _validate_moe_lora_token_capacity(hidden_states, max_num_batched_tokens)
            return instance.base_layer(*args, **kwargs)

        instance.forward = _forward_via_base_call  # type: ignore[method-assign]

        # Toggle the context-referenced enablement tensor in place, retaining weights.
        def _suspend_lora() -> None:
            if _suspended[0] is not None:
                return
            _suspended[0] = int(instance.adapter_enabled[0].item())
            instance.adapter_enabled[0] = 0

        def _resume_lora() -> None:
            prev = _suspended[0]
            if prev is None:
                return
            instance.adapter_enabled[0] = prev
            _suspended[0] = None

        instance.suspend_lora = _suspend_lora  # type: ignore[method-assign]
        instance.resume_lora = _resume_lora  # type: ignore[method-assign]
        return instance

    diffusion_lora_classes = [
        DiffusionMergedQKVParallelLinearWithLoRA,
        DiffusionQKVParallelLinearWithLoRA,
        DiffusionMergedColumnParallelLinearWithLoRA,
        DiffusionColumnParallelLinearWithLoRA,
        DiffusionRowParallelLinearWithLoRA,
        DiffusionReplicatedLinearWithLoRA,
    ]

    for lora_cls in diffusion_lora_classes:
        if lora_cls.can_replace_layer(
            source_layer=layer,
            lora_config=lora_config,
            packed_modules_list=packed_modules_list,
            model_config=model_config,
        ):
            instance = lora_cls(layer)  # type: ignore[arg-type]
            instance.create_lora_weights(max_loras, lora_config, model_config)
            return instance

    return layer


_MODEL_LOADER_HINT = (
    "Provide a model-specific loader via the pipeline's "
    "`_load_diffusion_lora_adapter` hook, or supply a canonical "
    "single-rank checkpoint."
)


def read_peft_rank_alpha_patterns(lora_path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Best-effort read of ``rank_pattern``/``alpha_pattern`` from a PEFT config.

    ``vllm.lora.peft_helper.PEFTHelper`` has no ``rank_pattern``/``alpha_pattern``
    fields and silently drops them (see ``PEFTHelper.from_dict``), so a checkpoint
    that declares per-module ranks/alphas loses that information before the
    generic diffusion loader ever sees it. We re-read the raw ``adapter_config.json``
    only to detect (not honor) those declarations, so the loader can fail loudly
    instead of applying one global scale.

    Returns empty dicts when the config is absent or unreadable (e.g. tensorizer
    or remote paths); the tensor-shape based check below still runs in that case.
    """
    config_path = os.path.join(lora_path, "adapter_config.json")
    if not os.path.isfile(config_path):
        return {}, {}
    try:
        with open(config_path) as f:
            config = json.load(f)
    except (OSError, ValueError) as exc:  # unreadable / not JSON
        logger.debug("Could not read %s for rank_pattern detection: %s", config_path, exc)
        return {}, {}
    rank_pattern = config.get("rank_pattern") or {}
    alpha_pattern = config.get("alpha_pattern") or {}
    rank_pattern = rank_pattern if isinstance(rank_pattern, Mapping) else {}
    alpha_pattern = alpha_pattern if isinstance(alpha_pattern, Mapping) else {}
    return dict(rank_pattern), dict(alpha_pattern)


def _iter_module_ranks(loras: Mapping[str, Any]) -> Iterator[tuple[str, int, int]]:
    """Yield ``(module_name, a_rank, b_rank)`` for every loaded LoRA slice.

    ``a_rank`` is the row count of ``lora_A`` and ``b_rank`` the column count of
    ``lora_B`` (LoRA stores ``A`` as ``[rank, in]`` and ``B`` as ``[out, rank]``).
    ``PackedLoRALayerWeights`` exposes ``lora_a``/``lora_b`` as lists of slices,
    so each slice is reported separately.
    """
    for name, weights in loras.items():
        lora_a = getattr(weights, "lora_a", None)
        lora_b = getattr(weights, "lora_b", None)
        a_slices = list(lora_a) if isinstance(lora_a, (list, tuple)) else [lora_a]
        b_slices = list(lora_b) if isinstance(lora_b, (list, tuple)) else [lora_b]
        for a, b in zip(a_slices, b_slices):
            if a is None or b is None:
                continue
            yield name, int(a.shape[0]), int(b.shape[1])


def validate_generic_peft_ranks(
    loras: Mapping[str, Any],
    global_rank: int,
    *,
    rank_pattern: Mapping[str, Any] | None = None,
    alpha_pattern: Mapping[str, Any] | None = None,
) -> None:
    """Guard the generic PEFT fallback against incorrect per-module rank scaling.

    The generic path in ``DiffusionLoRAManager._load_adapter`` builds every module
    from the single global ``r``/``lora_alpha`` in ``adapter_config.json`` (via
    ``vllm.lora.peft_helper.PEFTHelper`` and ``LoRALayerWeights.from_config``) and
    applies one uniform ``scaling = lora_alpha / r``. It never re-derives the
    per-module rank from the actual ``A``/``B`` tensors. As a result a checkpoint
    whose physical modules do not all share ``r`` — e.g. a fused-QKV adapter whose
    packed projection has rank ``3 * r`` while the remaining modules keep ``r`` —
    is loaded at the wrong rank and silently scaled incorrectly.

    This function keeps the generic path honest: it accepts only uniform-rank
    adapters with no per-module ``rank_pattern``/``alpha_pattern``. Anything else
    raises an actionable ``ValueError`` pointing at a model-specific loader.

    It deliberately does **not** attempt to recover a per-module scale (e.g.
    ``lora_alpha / actual_rank``): some distilled checkpoints bake the effective
    scale into the ``A``/``B`` tensors themselves, so silently reinterpreting the
    scale here could double-apply or drop it. Rejecting is the safe generic
    behavior; a model that understands its own contract can supply a loader.

    Args:
        loras: Mapping of module name to loaded LoRA weights (``lora_model.loras``).
        global_rank: The single ``r`` from the PEFT config (``peft_helper.r``).
        rank_pattern: Raw ``rank_pattern`` declared in ``adapter_config.json``, if any.
        alpha_pattern: Raw ``alpha_pattern`` declared in ``adapter_config.json``, if any.

    Raises:
        ValueError: If the adapter declares a per-module pattern the generic loader
            cannot honor, or if any module's tensor rank disagrees with ``global_rank``,
            or if a module's ``lora_A``/``lora_B`` ranks are mutually inconsistent.
    """
    if rank_pattern:
        raise ValueError(
            f"adapter_config.json declares a non-empty 'rank_pattern' "
            f"({sorted(rank_pattern)}), but the generic diffusion PEFT loader applies "
            f"a single global rank/scale and does not honor per-module rank_pattern "
            f"(vllm.lora.peft_helper.PEFTHelper drops it). " + _MODEL_LOADER_HINT
        )
    if alpha_pattern:
        raise ValueError(
            f"adapter_config.json declares a non-empty 'alpha_pattern' "
            f"({sorted(alpha_pattern)}), but the generic diffusion PEFT loader applies "
            f"a single global lora_alpha and does not honor per-module alpha_pattern. " + _MODEL_LOADER_HINT
        )

    inconsistent: list[tuple[str, int, int]] = []
    mismatched: list[tuple[str, int]] = []
    for name, a_rank, b_rank in _iter_module_ranks(loras):
        if a_rank != b_rank:
            inconsistent.append((name, a_rank, b_rank))
        elif a_rank != global_rank:
            mismatched.append((name, a_rank))

    if inconsistent:
        detail = ", ".join(f"{n} (lora_A rank={a}, lora_B rank={b})" for n, a, b in inconsistent[:8])
        raise ValueError(
            f"LoRA tensors have inconsistent internal rank between lora_A and lora_B "
            f"for {len(inconsistent)} module(s): {detail}. The checkpoint may be corrupt "
            f"or use a packed/fused layout the generic diffusion loader cannot interpret. " + _MODEL_LOADER_HINT
        )

    if mismatched:
        observed = sorted({rank for _, rank in mismatched})
        detail = ", ".join(f"{n} (rank={r})" for n, r in mismatched[:8])
        more = "" if len(mismatched) <= 8 else f" (+{len(mismatched) - 8} more)"
        raise ValueError(
            f"LoRA tensor rank does not match the global PEFT config r={global_rank} for "
            f"{len(mismatched)} module(s): {detail}{more}. Observed ranks: {observed}. "
            f"The checkpoint likely contains packed/fused or mixed-rank modules; the generic "
            f"diffusion loader applies a single global rank/scale and would silently scale "
            f"these modules. " + _MODEL_LOADER_HINT
        )
