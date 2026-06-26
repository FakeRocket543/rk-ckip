#!/usr/bin/env python3
"""
Convert CKIP BERT ONNX models to RKNN and benchmark on RK3588S NPU.

Prerequisites:
    1. ONNX models exported via ckip_onnx_bench.py
    2. IsNaN nodes removed via remove_isnan.py
    3. rknn-toolkit2 + rknn-toolkit-lite2 installed (Python 3.12)
    4. librknnrt.so in /usr/lib/
    5. User in 'video' group (for /dev/dri/card1 access)

Usage:
    python convert_rknn.py --onnx-dir /path/to/clean/onnx --output-dir ./rknn
    python convert_rknn.py --onnx-dir /path/to/clean/onnx --output-dir ./rknn --bench
"""
import argparse
import os
import time
import json

import numpy as np
import onnx


# ── Step 1: Fix dynamic shapes to static ────────────────────

def fix_static_shape(model: onnx.ModelProto, seq_len: int = 64) -> onnx.ModelProto:
    """Replace dynamic 'batch'/'seq' dims with fixed values [1, seq_len]."""
    dim_map = {"batch": 1, "seq": seq_len}

    for inp in model.graph.input:
        for dim in inp.type.tensor_type.shape.dim:
            if dim.dim_param in dim_map:
                dim.dim_value = dim_map[dim.dim_param]
                dim.ClearField("dim_param")

    for out in model.graph.output:
        for dim in out.type.tensor_type.shape.dim:
            if dim.dim_param in dim_map:
                dim.dim_value = dim_map[dim.dim_param]
                dim.ClearField("dim_param")

    return model


# ── Step 2: Convert ONNX → RKNN ─────────────────────────────

def convert_to_rknn(onnx_path: str, rknn_path: str, seq_len: int = 64) -> bool:
    """Convert an ONNX model to RKNN format for RK3588."""
    from rknn.api import RKNN

    # Load and fix shapes
    model = onnx.load(onnx_path)

    # Check for IsNaN (must be removed first)
    isnan_count = sum(1 for n in model.graph.node if n.op_type == "IsNaN")
    if isnan_count > 0:
        print(f"    ERROR: {isnan_count} IsNaN nodes found. Run remove_isnan.py first!")
        return False

    model = fix_static_shape(model, seq_len)

    # Save temp static model
    static_path = onnx_path.replace(".onnx", f"_static{seq_len}.onnx")
    onnx.save(model, static_path)

    # RKNN conversion
    rknn = RKNN(verbose=False)
    rknn.config(target_platform="rk3588")

    ret = rknn.load_onnx(model=static_path)
    if ret != 0:
        print(f"    load_onnx failed: {ret}")
        rknn.release()
        os.remove(static_path)
        return False

    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print(f"    build failed: {ret}")
        rknn.release()
        os.remove(static_path)
        return False

    ret = rknn.export_rknn(rknn_path)
    rknn.release()
    os.remove(static_path)

    if ret != 0:
        print(f"    export failed: {ret}")
        return False

    size_mb = os.path.getsize(rknn_path) / 1024 / 1024
    print(f"    Exported: {rknn_path} ({size_mb:.1f} MB)")
    return True


# ── Step 3: NPU Benchmark ───────────────────────────────────

def bench_npu(rknn_path: str, seq_len: int = 64, core_mask: int = None,
              warmup: int = 10, runs: int = 30) -> dict:
    """Benchmark RKNN model on NPU."""
    from rknnlite.api import RKNNLite

    if core_mask is None:
        core_mask = RKNNLite.NPU_CORE_0_1_2

    rknn = RKNNLite()
    ret = rknn.load_rknn(rknn_path)
    if ret != 0:
        return {"status": "fail", "step": "load", "ret": ret}

    ret = rknn.init_runtime(core_mask=core_mask)
    if ret != 0:
        rknn.release()
        return {"status": "fail", "step": "init_runtime", "ret": ret}

    # Prepare inputs
    inp_ids = np.ones((1, seq_len), dtype=np.int64) * 101  # [CLS] padding
    att_mask = np.ones((1, seq_len), dtype=np.int64)
    token_type = np.zeros((1, seq_len), dtype=np.int64)

    # Warmup
    for _ in range(warmup):
        rknn.inference(inputs=[inp_ids, att_mask, token_type])

    # Benchmark
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        out = rknn.inference(inputs=[inp_ids, att_mask, token_type])
        times.append((time.perf_counter() - t0) * 1000)

    rknn.release()

    return {
        "status": "ok",
        "avg_ms": round(float(np.mean(times)), 1),
        "min_ms": round(float(np.min(times)), 1),
        "max_ms": round(float(np.max(times)), 1),
        "std_ms": round(float(np.std(times)), 1),
        "output_shape": list(out[0].shape),
        "core_mask": hex(core_mask),
        "seq_len": seq_len,
    }


def bench_cpu_ort(onnx_path: str, seq_len: int = 64, warmup: int = 10, runs: int = 30) -> dict:
    """Benchmark ONNX model on CPU with ORT for comparison."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 4

    session = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])

    inp_ids = np.ones((1, seq_len), dtype=np.int64) * 101
    att_mask = np.ones((1, seq_len), dtype=np.int64)
    token_type = np.zeros((1, seq_len), dtype=np.int64)

    feed = {"input_ids": inp_ids, "attention_mask": att_mask, "token_type_ids": token_type}

    for _ in range(warmup):
        session.run(None, feed)

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run(None, feed)
        times.append((time.perf_counter() - t0) * 1000)

    return {
        "avg_ms": round(float(np.mean(times)), 1),
        "min_ms": round(float(np.min(times)), 1),
        "std_ms": round(float(np.std(times)), 1),
    }


# ── Main ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert CKIP BERT ONNX to RKNN + NPU benchmark")
    parser.add_argument("--onnx-dir", required=True, help="Directory with cleaned ONNX models")
    parser.add_argument("--output-dir", default="./rknn", help="Output directory for RKNN files")
    parser.add_argument("--seq-len", type=int, default=64, help="Fixed sequence length (default: 64)")
    parser.add_argument("--bench", action="store_true", help="Run NPU benchmark after conversion")
    parser.add_argument("--results-file", default=None, help="Save benchmark results to JSON")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tasks = ["ckip_bert_ws", "ckip_bert_pos", "ckip_bert_ner"]
    results = {}

    # Convert
    print("=" * 60)
    print(f"CKIP BERT → RKNN (seq_len={args.seq_len})")
    print("=" * 60)

    for task in tasks:
        onnx_path = os.path.join(args.onnx_dir, f"{task}.onnx")
        rknn_path = os.path.join(args.output_dir, f"{task}.rknn")

        if not os.path.exists(onnx_path):
            print(f"\n  {task}: ONNX not found, skipping")
            continue

        print(f"\n  Converting {task}...")
        ok = convert_to_rknn(onnx_path, rknn_path, seq_len=args.seq_len)
        if not ok:
            continue

        if args.bench:
            # CPU baseline
            print(f"    Benchmarking CPU ORT...")
            cpu_result = bench_cpu_ort(onnx_path, seq_len=args.seq_len)

            # NPU single core
            print(f"    Benchmarking NPU (1 core)...")
            from rknnlite.api import RKNNLite
            npu1_result = bench_npu(rknn_path, seq_len=args.seq_len,
                                    core_mask=RKNNLite.NPU_CORE_0)

            # NPU all cores
            print(f"    Benchmarking NPU (3 cores)...")
            npu3_result = bench_npu(rknn_path, seq_len=args.seq_len,
                                    core_mask=RKNNLite.NPU_CORE_0_1_2)

            label = task.replace("ckip_bert_", "").upper()
            results[label] = {
                "cpu_ort": cpu_result,
                "npu_1core": npu1_result,
                "npu_3core": npu3_result,
            }

            if npu3_result["status"] == "ok":
                speedup = cpu_result["avg_ms"] / npu3_result["avg_ms"]
                print(f"    {label}: CPU={cpu_result['avg_ms']:.1f}ms | "
                      f"NPU×1={npu1_result['avg_ms']:.1f}ms | "
                      f"NPU×3={npu3_result['avg_ms']:.1f}ms | "
                      f"speedup={speedup:.2f}x")

    # Summary
    if results:
        print(f"\n{'=' * 60}")
        print("NPU BENCHMARK SUMMARY")
        print(f"{'=' * 60}")
        print(f"  {'Task':<6} {'CPU ORT':<12} {'NPU×1':<12} {'NPU×3':<12} {'Speedup'}")
        for label, r in results.items():
            cpu = r["cpu_ort"]["avg_ms"]
            n1 = r["npu_1core"].get("avg_ms", "N/A")
            n3 = r["npu_3core"].get("avg_ms", "N/A")
            sp = f"{cpu/n3:.2f}x" if isinstance(n3, (int, float)) else "N/A"
            print(f"  {label:<6} {cpu:<12.1f} {n1:<12} {n3:<12} {sp}")

        # Save results
        out_file = args.results_file or os.path.join(args.output_dir, "npu_benchmark.json")
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved: {out_file}")


if __name__ == "__main__":
    main()
