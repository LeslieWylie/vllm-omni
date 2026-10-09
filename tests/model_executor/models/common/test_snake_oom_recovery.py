# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.model_executor.models.common.snake_activation import Snake, SnakeBeta

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("activation_cls", [SnakeBeta, Snake])
@pytest.mark.parametrize("error_cls", [torch.OutOfMemoryError, MemoryError])
def test_oom_propagates_without_disabling_shared_kernel(activation_cls, error_cls, mocker):
    kernel = object()
    mocker.patch.object(SnakeBeta, "_triton_kernel", kernel)
    activation = activation_cls(2)
    sibling = activation_cls(2)
    hidden_states = mocker.Mock(spec=torch.Tensor, is_cuda=True)
    result = torch.zeros(1, 2, 3)
    error = error_cls("injected transient OOM")
    fused = mocker.patch.object(activation, "_triton_forward", side_effect=error)
    eager = mocker.patch.object(activation, "_eager_forward")
    sibling_fused = mocker.patch.object(sibling, "_triton_forward", return_value=result)

    with torch.no_grad():
        with pytest.raises(error_cls) as caught:
            activation(hidden_states)
        assert caught.value is error
        assert SnakeBeta._triton_kernel is kernel
        # Subsequent requests, including other decoder instances, keep the kernel.
        assert sibling(hidden_states) is result

    fused.assert_called_once_with(hidden_states)
    eager.assert_not_called()
    sibling_fused.assert_called_once_with(hidden_states)


def test_non_oom_failure_retains_eager_fallback(mocker):
    mocker.patch.object(SnakeBeta, "_triton_kernel", object())
    activation = SnakeBeta(2)
    hidden_states = mocker.Mock(spec=torch.Tensor, is_cuda=True)
    fused = mocker.patch.object(activation, "_triton_forward", side_effect=RuntimeError("kernel compilation failed"))
    result = torch.zeros(1, 2, 3)
    eager = mocker.patch.object(activation, "_eager_forward", return_value=result)

    with torch.no_grad():
        assert activation(hidden_states) is result
        assert SnakeBeta._triton_kernel is False
        assert activation(hidden_states) is result

    fused.assert_called_once_with(hidden_states)
    assert eager.call_count == 2


@pytest.mark.parametrize("activation_cls", [SnakeBeta, Snake])
def test_partial_exp_cache_is_rebuilt_after_oom(activation_cls, monkeypatch):
    activation = activation_cls(2, alpha_logscale=True)
    original_exp = torch.exp
    calls = 0

    def exp_with_transient_oom(tensor):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise torch.OutOfMemoryError("injected OOM while building inverse beta")
        return original_exp(tensor)

    monkeypatch.setattr(torch, "exp", exp_with_transient_oom)
    hidden_states = torch.randn(1, 2, 3)
    with torch.no_grad():
        with pytest.raises(torch.OutOfMemoryError):
            activation(hidden_states)
        assert activation._exp_alpha is not None
        assert not activation._cached

        actual = activation(hidden_states)
        expected = hidden_states + torch.sin(hidden_states).square() / (1.0 + activation.no_div_by_zero)

    assert activation._cached
    torch.testing.assert_close(actual, expected)
