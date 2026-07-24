#!/usr/bin/env python3
"""
Quantize CKIP ONNX models to INT8 (dynamic) and FP16, benchmark all variants.

INT8 dynamic quantization: weights → int8, activations stay float (dynamic scale).
FP16: all weights/activations → float16.

On ARM CPU, INT8 dynamic is typically the only one that might help (FP16 on CPU
usually has no hardware acceleration and is often SLOWER due to cast overhead).
"""
import os, sys, time, json
import numpy as np
import torch
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType, QuantFormat

ONNX_DIR = "/mnt/nvme/rk-ckip/onnx"
Q8_DIR = "/mnt/nvme/rk-ckip/onnx/q8"
os.makedirs(Q8_DIR, exist_ok=True)

test_sentences = [
    "台積電宣布將在高雄設立先進製程晶圓廠。",
    "蘋果公司執行長庫克表示將持續投資台灣供應鏈。",
    "聯發科的天璣9000處理器採用台積電4奈米製程。",
]


def bench(fn, warmup=3, runs=10):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.time()
        fn()
        times.append((time.time() - t0) * 1000)
    return float(np.mean(times)), float(np.std(times))


def make_ort(path, threads=4):
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = threads
    return ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])


def quantize_int8(onnx_path, out_path):
    """Dynamic INT8 quantization (weight-only)."""
    quantize_dynamic(
        onnx_path,
        out_path,
        weight_type=QuantType.QInt8,
    )
    return out_path


def main():
    labels = ["ws", "pos", "ner"]

    # ─── Step 1: Quantize ────────────────────────────────────────────────
    print("=" * 60)
    print("Quantizing models...")
    print("=" * 60)

    for label in labels:
        src = os.path.join(ONNX_DIR, f"ckip_bert_{label}.onnx")
        dst = os.path.join(Q8_DIR, f"ckip_bert_{label}_int8.onnx")
        src_sz = os.path.getsize(src) / 1024 / 1024
        print(f"\n  {label.upper()}: FP32={src_sz:.0f}MB → quantizing...", flush=True)
        t0 = time.time()
        quantize_int8(src, dst)
        dst_sz = os.path.getsize(dst) / 1024 / 1024
        print(f"  INT8={dst_sz:.0f}MB ({time.time()-t0:.1f}s) — {src_sz/dst_sz:.1f}x smaller")

    # ─── Step 2: Benchmark all variants ──────────────────────────────────
    from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger, CkipNerChunker
    import types
    from transformers.modeling_outputs import TokenClassifierOutput

    print("\n" + "=" * 60)
    print("Loading tokenizers (no model forward needed)...")
    print("=" * 60)

    drivers = {
        "ws": CkipWordSegmenter(model="bert-base", device=-1),
        "pos": CkipPosTagger(model="bert-base", device=-1),
        "ner": CkipNerChunker(model="bert-base", device=-1),
    }

    # Prepare tokenized inputs for raw ORT benchmark (no ckip overhead)
    encoded_inputs = {}
    for label, driver in drivers.items():
        enc = driver.tokenizer(
            test_sentences,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        np_inputs = {
            "input_ids": enc["input_ids"].numpy(),
            "attention_mask": enc["attention_mask"].numpy(),
            "token_type_ids": enc.get("token_type_ids", torch.zeros_like(enc["input_ids"])).numpy(),
        }
        encoded_inputs[label] = np_inputs

    results = {}

    for variant_name, get_path in [
        ("FP32", lambda l: os.path.join(ONNX_DIR, f"ckip_bert_{l}.onnx")),
        ("INT8", lambda l: os.path.join(Q8_DIR, f"ckip_bert_{l}_int8.onnx")),
    ]:
        print(f"\n{'='*60}")
        print(f"Benchmarking: {variant_name}")
        print(f"{'='*60}")

        results[variant_name] = {}
        for label in labels:
            path = get_path(label)
            if not os.path.exists(path):
                print(f"  {label.upper()}: SKIP (not found)")
                continue
            sess = make_ort(path)
            np_in = encoded_inputs[label]

            mean, std = bench(lambda: sess.run(None, np_in))
            per_sent = mean / len(test_sentences)
            sz = os.path.getsize(path) / 1024 / 1024
            print(f"  {label.upper()}: {per_sent:.1f} ms/sent  (±{std/len(test_sentences):.1f})  [{sz:.0f}MB]")
            results[variant_name][label] = {"ms": round(per_sent, 1), "size_mb": round(sz, 0)}

    # ─── Step 3: Full pipeline benchmark (end-to-end via ckip driver) ────
    print(f"\n{'='*60}")
    print("Full pipeline (ckip driver + ORT patch)")
    print(f"{'='*60}")

    for variant_name, get_path in [
        ("FP32", lambda l: os.path.join(ONNX_DIR, f"ckip_bert_{l}.onnx")),
        ("INT8", lambda l: os.path.join(Q8_DIR, f"ckip_bert_{l}_int8.onnx")),
    ]:
        print(f"\n  --- {variant_name} ---")
        # Re-load fresh drivers and patch
        ws = CkipWordSegmenter(model="bert-base", device=-1)
        pos = CkipPosTagger(model="bert-base", device=-1)
        ner = CkipNerChunker(model="bert-base", device=-1)

        for label, drv in [("ws", ws), ("pos", pos), ("ner", ner)]:
            path = get_path(label)
            sess = make_ort(path)

            def make_patch(s):
                def patched(self, input_ids, attention_mask=None, token_type_ids=None, return_dict=True, **kw):
                    np_in = {
                        "input_ids": input_ids.cpu().numpy(),
                        "attention_mask": attention_mask.cpu().numpy() if attention_mask is not None else np.ones_like(input_ids.cpu().numpy(), dtype=np.int64),
                        "token_type_ids": token_type_ids.cpu().numpy() if token_type_ids is not None else np.zeros_like(input_ids.cpu().numpy(), dtype=np.int64),
                    }
                    out = s.run(None, np_in)
                    lg = torch.from_numpy(out[0]).to(input_ids.device)
                    return TokenClassifierOutput(logits=lg) if return_dict else (lg,)
                return patched

            drv.model.forward = types.MethodType(make_patch(sess), drv.model)

        # Benchmark
        ws_mean, _ = bench(lambda: ws(test_sentences))
        ws_out = ws(test_sentences)
        pos_mean, _ = bench(lambda: pos(ws_out, use_delim=True))
        ner_mean, _ = bench(lambda: ner(test_sentences))

        total = (ws_mean + pos_mean + ner_mean) / len(test_sentences)
        print(f"    WS:  {ws_mean/len(test_sentences):.1f} ms/sent")
        print(f"    POS: {pos_mean/len(test_sentences):.1f} ms/sent")
        print(f"    NER: {ner_mean/len(test_sentences):.1f} ms/sent")
        print(f"    Total: {total:.1f} ms/sent")

        # Correctness
        ner_out = ner(test_sentences[:1])
        print(f"    NER check: {ner_out[0][:3]}")

    # ─── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY (raw ORT forward, 3 sentences)")
    print(f"{'='*60}")
    print(f"  {'':6s} {'FP32':>12s} {'INT8':>12s} {'Speedup':>10s}")
    for label in labels:
        fp32 = results.get("FP32", {}).get(label, {}).get("ms", 0)
        int8 = results.get("INT8", {}).get(label, {}).get("ms", 0)
        sp = f"{fp32/max(int8,0.1):.2f}x" if int8 else "-"
        print(f"  {label.upper():6s} {fp32:10.1f}ms {int8:10.1f}ms {sp:>10s}")

    # Size comparison
    print(f"\n  Sizes:")
    print(f"  {'':6s} {'FP32':>8s} {'INT8':>8s} {'Ratio':>8s}")
    for label in labels:
        fp32_sz = results.get("FP32", {}).get(label, {}).get("size_mb", 0)
        int8_sz = results.get("INT8", {}).get(label, {}).get("size_mb", 0)
        print(f"  {label.upper():6s} {fp32_sz:6.0f}MB {int8_sz:6.0f}MB {fp32_sz/max(int8_sz,1):.1f}x")

    # Save
    out_path = "/mnt/nvme/rk-ckip/benchmarks/quant_bench_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
