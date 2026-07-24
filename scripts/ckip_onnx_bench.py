#!/usr/bin/env python3
"""
CKIP Transformers ONNX acceleration baseline.
Three BERT-base models: WS, POS, NER — all BertForTokenClassification.
"""
import os, time, json
import numpy as np
import torch
import onnxruntime as ort
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger, CkipNerChunker
from transformers import BertForTokenClassification

EXPORT_DIR = "/mnt/nvme/rk-ckip/onnx"
BENCH_DIR = "/mnt/nvme/rk-ckip/benchmarks"
os.makedirs(EXPORT_DIR, exist_ok=True)
os.makedirs(BENCH_DIR, exist_ok=True)

test_sentences = [
    "台積電宣布將在高雄設立先進製程晶圓廠。",
    "蘋果公司執行長庫克表示將持續投資台灣供應鏈。",
    "聯發科的天璣9000處理器採用台積電4奈米製程。",
]


def bench(fn, warmup=3, runs=10):
    for _ in range(warmup): fn()
    times = []
    for _ in range(runs):
        t0 = time.time(); fn(); times.append((time.time()-t0)*1000)
    return {"mean": float(np.mean(times)), "std": float(np.std(times))}


def make_ort(path, threads=4):
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = threads
    return ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])


def export_bert_onnx(model, name, tokenizer):
    """Export BERT to ONNX with dynamic axes."""
    model.eval()
    path = os.path.join(EXPORT_DIR, f"{name}.onnx")

    # Dummy input
    dummy = tokenizer("測試", return_tensors="pt")
    input_ids = dummy["input_ids"]
    attention_mask = dummy["attention_mask"]
    token_type_ids = dummy.get("token_type_ids", torch.zeros_like(input_ids))

    torch.onnx.export(
        model,
        (input_ids, attention_mask, token_type_ids),
        path,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "token_type_ids": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    sz = os.path.getsize(path) / 1024 / 1024
    print(f"  {name}: {sz:.1f} MB ✓")
    return path


def main():
    print("=" * 60)
    print("CKIP Transformers ONNX Acceleration")
    print("=" * 60)

    # Load models
    print("\nLoading models...")
    t0 = time.time()
    ws = CkipWordSegmenter(model="bert-base", device=-1)
    pos = CkipPosTagger(model="bert-base", device=-1)
    ner = CkipNerChunker(model="bert-base", device=-1)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # =============================================================
    # STEP 1: Baseline benchmark (PyTorch)
    # =============================================================
    print(f"\n{'='*60}")
    print("STEP 1: PyTorch Baseline")
    print(f"{'='*60}")

    # WS
    r_ws = bench(lambda: ws(test_sentences))
    ws_per = r_ws["mean"] / len(test_sentences)
    print(f"  WS:  {ws_per:.0f} ms/sentence")

    # POS (needs WS output)
    ws_out = ws(test_sentences)
    r_pos = bench(lambda: pos(ws_out))
    pos_per = r_pos["mean"] / len(test_sentences)
    print(f"  POS: {pos_per:.0f} ms/sentence")

    # NER
    r_ner = bench(lambda: ner(test_sentences))
    ner_per = r_ner["mean"] / len(test_sentences)
    print(f"  NER: {ner_per:.0f} ms/sentence")

    total_base = ws_per + pos_per + ner_per
    print(f"\n  Total: {total_base:.0f} ms/sentence")

    # Verify correctness
    print("\n  Correctness check:")
    ws_out = ws(test_sentences)
    for i, s in enumerate(ws_out):
        print(f"    WS[{i}]: {' '.join(s[:6])}...")
    ner_out = ner(test_sentences)
    for i, s in enumerate(ner_out):
        for e in s:
            print(f"    NER[{i}]: {e}")

    # =============================================================
    # STEP 2: Export ONNX
    # =============================================================
    print(f"\n{'='*60}")
    print("STEP 2: Export BERT → ONNX")
    print(f"{'='*60}")

    for label, driver in [("ws", ws), ("pos", pos), ("ner", ner)]:
        export_bert_onnx(driver.model, f"ckip_bert_{label}", driver.tokenizer)

    # =============================================================
    # STEP 3: ONNX Runtime benchmark (model forward only)
    # =============================================================
    print(f"\n{'='*60}")
    print("STEP 3: ONNX Runtime Model Forward")
    print(f"{'='*60}")

    for label, driver in [("ws", ws), ("pos", pos), ("ner", ner)]:
        onnx_path = os.path.join(EXPORT_DIR, f"ckip_bert_{label}.onnx")
        sess = make_ort(onnx_path)

        # Tokenize test sentences
        encoded = driver.tokenizer(
            test_sentences,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        np_inputs = {
            "input_ids": encoded["input_ids"].numpy(),
            "attention_mask": encoded["attention_mask"].numpy(),
            "token_type_ids": encoded.get("token_type_ids", torch.zeros_like(encoded["input_ids"])).numpy(),
        }

        # Warmup
        for _ in range(3):
            sess.run(None, np_inputs)

        # Benchmark
        r = bench(lambda: sess.run(None, np_inputs))
        per_sent = r["mean"] / len(test_sentences)
        print(f"  {label.upper()} ORT: {per_sent:.0f} ms/sentence")

        # Verify correctness vs PyTorch
        with torch.no_grad():
            pt_out = driver.model(**encoded)
        ort_out = sess.run(None, np_inputs)

        pt_logits = pt_out.logits.numpy()
        ort_logits = ort_out[0]

        max_diff = np.max(np.abs(pt_logits - ort_logits))
        pt_pred = np.argmax(pt_logits, axis=-1)
        ort_pred = np.argmax(ort_logits, axis=-1)
        match = np.all(pt_pred == ort_pred)
        print(f"    max_diff={max_diff:.6f}, predictions_match={match}")

    # =============================================================
    # STEP 4: Full pipeline ONNX (monkey-patch)
    # =============================================================
    print(f"\n{'='*60}")
    print("STEP 4: Full Pipeline ONNX (monkey-patch)")
    print(f"{'='*60}")

    # Patch each driver's model forward to use ORT
    for label, driver in [("ws", ws), ("pos", pos), ("ner", ner)]:
        onnx_path = os.path.join(EXPORT_DIR, f"ckip_bert_{label}.onnx")
        ort_sess = make_ort(onnx_path)
        orig_forward = type(driver.model).forward

        def make_patched_forward(sess):
            def patched(self, input_ids, attention_mask=None, token_type_ids=None, return_dict=True, **kwargs):
                np_in = {
                    "input_ids": input_ids.cpu().numpy(),
                    "attention_mask": attention_mask.cpu().numpy() if attention_mask is not None else np.ones_like(input_ids, dtype=np.int64),
                    "token_type_ids": token_type_ids.cpu().numpy() if token_type_ids is not None else np.zeros_like(input_ids, dtype=np.int64),
                }
                out = sess.run(None, np_in)
                logits = torch.from_numpy(out[0]).to(input_ids.device)
                if return_dict:
                    from transformers.modeling_outputs import TokenClassifierOutput
                    return TokenClassifierOutput(logits=logits)
                return (logits,)
            return patched

        import types
        driver.model.forward = types.MethodType(make_patched_forward(ort_sess), driver.model)
        print(f"  {label.upper()}: patched ✓")

    # Benchmark patched
    r_ws_p = bench(lambda: ws(test_sentences))
    ws_p = r_ws_p["mean"] / len(test_sentences)
    print(f"  WS ONNX: {ws_p:.0f} ms/sentence")

    ws_out_p = ws(test_sentences)
    r_pos_p = bench(lambda: pos(ws_out_p))
    pos_p = r_pos_p["mean"] / len(test_sentences)
    print(f"  POS ONNX: {pos_p:.0f} ms/sentence")

    r_ner_p = bench(lambda: ner(test_sentences))
    ner_p = r_ner_p["mean"] / len(test_sentences)
    print(f"  NER ONNX: {ner_p:.0f} ms/sentence")

    total_patched = ws_p + pos_p + ner_p
    print(f"\n  Total: {total_patched:.0f} ms/sentence")

    # Verify
    print("\n  Correctness check (patched):")
    ws_out_p = ws(test_sentences)
    for i, s in enumerate(ws_out_p):
        print(f"    WS[{i}]: {' '.join(s[:6])}...")
    ner_out_p = ner(test_sentences)
    for i, s in enumerate(ner_out_p):
        for e in s:
            print(f"    NER[{i}]: {e}")

    # =============================================================
    # Summary
    # =============================================================
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"  {'':6s} {'PyTorch':>10s} {'ONNX':>10s} {'Speedup':>10s}")
    print(f"  {'WS':6s} {ws_per:10.0f} {ws_p:10.0f} {ws_per/max(ws_p,1):10.2f}x")
    print(f"  {'POS':6s} {pos_per:10.0f} {pos_p:10.0f} {pos_per/max(pos_p,1):10.2f}x")
    print(f"  {'NER':6s} {ner_per:10.0f} {ner_p:10.0f} {ner_per/max(ner_p,1):10.2f}x")
    print(f"  {'Total':6s} {total_base:10.0f} {total_patched:10.0f} {total_base/max(total_patched,1):10.2f}x")

    results = {
        "baseline": {"ws": round(ws_per,1), "pos": round(pos_per,1), "ner": round(ner_per,1), "total": round(total_base,1)},
        "onnx": {"ws": round(ws_p,1), "pos": round(pos_p,1), "ner": round(ner_p,1), "total": round(total_patched,1)},
        "speedup": {
            "ws": round(ws_per/max(ws_p,1), 2),
            "pos": round(pos_per/max(pos_p,1), 2),
            "ner": round(ner_per/max(ner_p,1), 2),
            "total": round(total_base/max(total_patched,1), 2),
        },
    }
    out_path = os.path.join(BENCH_DIR, "ckip_onnx_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
