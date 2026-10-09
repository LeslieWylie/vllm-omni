# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compare indexed pointwise launch geometry without changing runtime dispatch.

No GPU is used by default. GPU timings concern two pointwise kernels only;
they do not establish full-model speed or media quality.

From the repository root:

    python benchmarks/kernels/bench_indexed_pointwise.py --plan
    CUDA_VISIBLE_DEVICES='' TRITON_INTERPRET=1 python \
        benchmarks/kernels/bench_indexed_pointwise.py --check-indexing
    CUDA_VISIBLE_DEVICES=0 python benchmarks/kernels/bench_indexed_pointwise.py \
        --run --device 0 --isolation-note 'dedicated GPU allocation' \
        --output /tmp/indexed-pointwise.json

GPU mode requires an isolated allocation. --strided selects non-contiguous
row/index strides for timing; default timings use contiguous input layouts.
BF16 correctness checks always exercise both layouts before timing.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import subprocess
from pathlib import Path
from types import ModuleType

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "vllm_omni/diffusion/layers/indexed_modulation.py"


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def plan(args: argparse.Namespace) -> dict[str, object]:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        revision = None
    return {
        "status": "prepared_not_gpu_measured",
        "checkout_revision": revision,
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "scope": ["indexed_gate", "indexed_scale_shift_"],
        "baseline": {"num_warps": 8, "block_n": 1 << (args.hidden_size - 1).bit_length()},
        "rows": args.rows,
        "hidden_size": args.hidden_size,
        "shape_provenance": "Synthetic rows; the default hidden size 5376 matches H3",
        "variants": ["row_warps_2", "row_warps_4", "tile_1024_warps_4", "tile_2048_warps_4"],
        "validation": "BF16 bitwise equality vs current source at seeds 7, 42, 3100; reject any differing output",
        "timing": (
            "CUDA-event graph replay time per call, 5 paired alternating rounds, "
            "preallocated buffers; no allocation/E2E claim"
        ),
        "unchanged": "RMSNorm reduction tree, checkpoint, sampling, API and running services",
        "required_next": "Dedicated GPU, then matched full-checkpoint request/video/audio validation",
    }


def fixture(torch, rows, hidden, device, dtype, seed, exact=False, strided=True):
    torch.manual_seed(seed)

    def tensor(shape, salt):
        if exact:
            count = 1
            for dimension in shape:
                count *= dimension
            return (((torch.arange(count, device=device).reshape(shape) * (salt + 1) + salt) % 17 - 8) / 8).to(dtype)
        return (torch.randn(shape, device=device, dtype=torch.float32) * 0.1).to(dtype)

    # Columns stay contiguous; different row strides expose accidental stride/pointer reuse.
    x_factor, other_factor = (2, 3) if strided else (1, 1)
    x = tensor((rows * x_factor, hidden), 1)[::x_factor]
    other = tensor((rows * other_factor, hidden), 3)[::other_factor]
    bank1 = tensor((6 * x_factor, hidden), 5)[::x_factor]
    bank2 = tensor((6 * other_factor, hidden), 7)[::other_factor]
    indices_buffer = torch.zeros(rows * x_factor, device=device, dtype=torch.int64)
    indices = indices_buffer[::x_factor]
    indices.copy_((torch.arange(rows, device=device, dtype=torch.int64) * 5 + 1) % 6)
    return x, bank1, bank2, other, indices


def candidate_launch(candidate, tensors, out, op, block=1024, warps=4):
    x, bank1, bank2, other, indices = tensors
    rows, hidden = x.shape
    if not rows:
        return
    candidate.tiled_pointwise[(rows, (hidden + block - 1) // block)](
        out,
        x,
        bank1,
        bank2,
        other,
        indices,
        hidden,
        out.stride(0),
        x.stride(0),
        bank1.stride(0),
        bank2.stride(0),
        other.stride(0),
        indices.stride(0),
        gate=op == "gate",
        block=block,
        num_warps=warps,
    )


def row_launch(base, tensors, out, op, warps):
    x, bank1, bank2, other, indices = tensors
    rows, hidden = x.shape
    if not rows:
        return
    meta = {"block_n": 1 << (hidden - 1).bit_length(), "num_warps": warps}
    if op == "gate":
        base._indexed_gate_kernel[(rows,)](
            out,
            x,
            bank1,
            other,
            indices,
            hidden,
            out.stride(0),
            x.stride(0),
            bank1.stride(0),
            other.stride(0),
            indices.stride(0),
            0,
            **meta,
        )
    else:
        if out.stride(0) != x.stride(0):
            raise ValueError("The baseline affine kernel shares input/output row stride")
        base._indexed_scale_shift_kernel[(rows,)](
            out,
            x,
            bank1,
            bank2,
            indices,
            hidden,
            x.stride(0),
            bank1.stride(0),
            bank2.stride(0),
            indices.stride(0),
            0,
            **meta,
        )


def check_indexing():
    os.environ["TRITON_INTERPRET"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch

    candidate = load_module("indexed_candidates", HERE / "indexed_pointwise_candidates.py")
    checks = []
    for rows, hidden in ((0, 7), (1, 7), (5, 129), (3, 5376)):
        tensors = fixture(torch, rows, hidden, "cpu", torch.float32, 7, exact=True)
        x, bank1, bank2, other, indices = tensors
        for op in ("gate", "affine"):
            if op == "gate":
                expected = x + bank1.index_select(0, indices) * other
            else:
                expected = x * (1 + bank2.index_select(0, indices)) + bank1.index_select(0, indices)
            for block in (1024, 2048):
                out = torch.empty_strided(x.shape, x.stride(), dtype=x.dtype)
                candidate_launch(candidate, tensors, out, op, block)
                assert torch.equal(out, expected), (rows, hidden, op, block)
                # Affine's public path aliases its input; check this explicitly.
                if op == "affine":
                    alias = torch.empty_strided(x.shape, x.stride(), dtype=x.dtype)
                    alias.copy_(x)
                    candidate_launch(candidate, (alias, *tensors[1:]), alias, op, block)
                    assert torch.equal(alias, expected), ("alias", rows, hidden, block)
            checks.append({"rows": rows, "hidden": hidden, "operator": op, "indexing": "pass"})
    from vllm.triton_utils import triton

    return {
        "status": "cpu_interpreter_indexing_pass",
        "torch": torch.__version__,
        "triton": triton.__version__,
        "checks": checks,
        "limit": "FP32 exact fixtures only; BF16 compiled CUDA numerics and speed remain unverified",
    }


def gpu_run(args):
    if not args.isolation_note:
        raise SystemExit("--run requires --isolation-note identifying the dedicated GPU allocation")
    if os.environ.get("TRITON_INTERPRET") == "1":
        raise SystemExit("Disable TRITON_INTERPRET before measuring compiled CUDA kernels")
    import torch
    from vllm.triton_utils import triton

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; no benchmark was run")
    torch.accelerator.set_device_index(args.device)
    device = torch.device("cuda", args.device)
    base = load_module("indexed_main_snapshot", SOURCE)
    candidate = load_module("indexed_candidates", HERE / "indexed_pointwise_candidates.py")
    result = plan(args)
    result.update(
        {
            "status": "kernel_experiment_only",
            "isolation_note": args.isolation_note,
            "gpu": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": triton.__version__,
            "warmup_launches": 10,
            "launches_per_graph": args.iterations,
            "paired_rounds": 5,
            "timing_layout": "strided" if args.strided else "contiguous",
            "dtype": "bfloat16",
            "samples": [],
        }
    )
    variants = (
        ("row_warps_2", None, 2),
        ("row_warps_4", None, 4),
        ("tile_1024_warps_4", 1024, 4),
        ("tile_2048_warps_4", 2048, 4),
    )
    for rows in args.rows:
        for op in ("gate", "affine"):
            for name, block, warps in variants:
                sample = {"rows": rows, "hidden": args.hidden_size, "operator": op, "variant": name}
                try:
                    for seed, strided in ((7, False), (42, False), (3100, False), (7, True), (42, True), (3100, True)):
                        tensors = fixture(torch, rows, args.hidden_size, device, torch.bfloat16, seed, strided=strided)
                        x, bank1, bank2, other, indices = tensors
                        if op == "gate":
                            expected = base.indexed_gate(x, bank1, other, indices)
                            out = torch.empty_like(x)
                        else:
                            clone = torch.empty_strided(x.shape, x.stride(), device=device, dtype=x.dtype)
                            clone.copy_(x)
                            expected = base.indexed_scale_shift_(clone, bank1, bank2, indices)
                            out = x  # Candidate must preserve in-place affine behavior.
                        reference_out = torch.empty_strided(x.shape, x.stride(), device=device, dtype=x.dtype)
                        row_launch(base, tensors, reference_out, op, 8)
                        if not torch.equal(reference_out.view(torch.int16), expected.view(torch.int16)):
                            raise ValueError("Preallocated baseline does not match the public wrapper bitwise")
                        if block:
                            candidate_launch(candidate, tensors, out, op, block, warps)
                        else:
                            row_launch(base, tensors, out, op, warps)
                        unequal = int((out.view(torch.int16) != expected.view(torch.int16)).sum().item())
                        if unequal:
                            raise ValueError(
                                f"BF16 mismatch: seed={seed}, strided={strided}, differing_elements={unequal}"
                            )
                    sample["bitwise_equal"] = True
                    tensors = fixture(torch, rows, args.hidden_size, device, torch.bfloat16, 42, strided=args.strided)
                    x, bank1, bank2, other, indices = tensors
                    original = x.clone()
                    out = torch.empty_like(x) if op == "gate" else x

                    def baseline():
                        row_launch(base, tensors, out, op, 8)

                    def trial():
                        if block:
                            candidate_launch(candidate, tensors, out, op, block, warps)
                        else:
                            row_launch(base, tensors, out, op, warps)

                    def capture(call):
                        x.copy_(original)
                        for _ in range(10):
                            call()
                        torch.accelerator.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            for _ in range(args.iterations):
                                call()
                        return graph

                    baseline_graph, trial_graph = capture(baseline), capture(trial)
                    x.copy_(original)
                    baseline_graph.replay()
                    replay_reference = out.clone()
                    x.copy_(original)
                    trial_graph.replay()
                    if not torch.equal(out.view(torch.int16), replay_reference.view(torch.int16)):
                        raise ValueError("CUDA Graph replay differs bitwise from the baseline graph")
                    sample["graph_replay_equal"] = True

                    def measure(graph):
                        # Reset outside the event interval, especially for in-place affine.
                        x.copy_(original)
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        graph.replay()
                        end.record()
                        end.synchronize()
                        return start.elapsed_time(end) / args.iterations

                    before, after = [], []
                    for round_index in range(5):
                        order = (
                            ((baseline_graph, before), (trial_graph, after))
                            if round_index % 2 == 0
                            else ((trial_graph, after), (baseline_graph, before))
                        )
                        for graph, bucket in order:
                            bucket.append(measure(graph))
                    before_ms, after_ms = statistics.median(before), statistics.median(after)
                    sample.update(
                        {
                            "baseline_ms": before,
                            "candidate_ms": after,
                            "median_speedup": before_ms / after_ms,
                            "saved_ms_per_call": before_ms - after_ms,
                        }
                    )
                except Exception as error:
                    # Persist rejected candidates, then fail the CLI after saving all results.
                    sample.update({"eligible": False, "error_type": type(error).__name__, "error": str(error)})
                result["samples"].append(sample)
                print(json.dumps(sample), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--check-indexing", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rows", type=int, nargs="+", default=[512, 4096, 32768])
    parser.add_argument("--hidden-size", type=int, default=5376)
    parser.add_argument("--strided", action="store_true")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--isolation-note")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.rows) <= 0 or args.hidden_size <= 0 or args.iterations <= 0:
        parser.error("rows, hidden size and iterations must be positive")
    result = check_indexing() if args.check_indexing else gpu_run(args) if args.run else plan(args)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.run and any("error" in sample for sample in result["samples"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
