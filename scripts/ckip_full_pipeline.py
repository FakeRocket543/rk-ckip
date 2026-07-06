#!/usr/bin/env python3
"""
CKIP Full Pipeline: WS + POS + NER with ONNX/NPU acceleration.

Reads text from JSONL (field "text") or plain text (one doc per line),
runs Word Segmentation → POS Tagging (aligned via use_delim) → NER,
outputs JSONL with aligned results.

Backends:
  - pytorch: vanilla ckip-transformers (CPU/GPU)
  - onnx:    monkey-patched ORT inference (CPU, ~3.3x faster)
  - npu:     RKNN on RK3588S NPU (requires rknn-toolkit-lite2)

Usage:
    # PyTorch baseline
    cat input.jsonl | python ckip_full_pipeline.py --backend pytorch > output.jsonl

    # ONNX accelerated
    python ckip_full_pipeline.py --backend onnx --onnx-dir ./onnx -i input.jsonl -o output.jsonl

    # NPU accelerated (RK3588S)
    python ckip_full_pipeline.py --backend npu --rknn-dir ./rknn --seq-len 256 -i input.jsonl

    # Plain text input (one document per line)
    python ckip_full_pipeline.py --backend onnx --onnx-dir ./onnx --input-format text -i docs.txt

Environment variables (override CLI):
    BACKEND=onnx|npu|pytorch
    ONNX_DIR=/path/to/onnx
    RKNN_DIR=/path/to/rknn
    SEQ_LEN=256
    BATCH_SIZE=32
"""
import argparse
import json
import os
import sys
import time
import types
from typing import TextIO

import numpy as np
import torch
from ckip_transformers.nlp import CkipNerChunker, CkipPosTagger, CkipWordSegmenter


# ─── Backend patching ────────────────────────────────────────────────────────


def patch_onnx(driver, onnx_path: str, threads: int = 4):
    """Monkey-patch a CKIP driver to use ONNX Runtime for inference."""
    import onnxruntime as ort
    from transformers.modeling_outputs import TokenClassifierOutput

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = threads
    sess = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])

    def patched_forward(
        self, input_ids, attention_mask=None, token_type_ids=None, return_dict=True, **kw
    ):
        np_in = {
            "input_ids": input_ids.cpu().numpy(),
            "attention_mask": (
                attention_mask.cpu().numpy()
                if attention_mask is not None
                else np.ones_like(input_ids.cpu().numpy(), dtype=np.int64)
            ),
            "token_type_ids": (
                token_type_ids.cpu().numpy()
                if token_type_ids is not None
                else np.zeros_like(input_ids.cpu().numpy(), dtype=np.int64)
            ),
        }
        out = sess.run(None, np_in)
        logits = torch.from_numpy(out[0]).to(input_ids.device)
        if return_dict:
            return TokenClassifierOutput(logits=logits)
        return (logits,)

    driver.model.forward = types.MethodType(patched_forward, driver.model)


def patch_npu(driver, rknn_path: str, seq_len: int = 256):
    """Monkey-patch a CKIP driver to use RKNN NPU for inference."""
    from rknnlite.api import RKNNLite
    from transformers.modeling_outputs import TokenClassifierOutput

    rkl = RKNNLite()
    ret = rkl.load_rknn(rknn_path)
    assert ret == 0, f"Failed to load RKNN: {rknn_path}"
    ret = rkl.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    assert ret == 0, "Failed to init NPU runtime"

    def patched_forward(
        self, input_ids, attention_mask=None, token_type_ids=None, return_dict=True, **kw
    ):
        ii = input_ids
        B, L = ii.shape
        am = attention_mask if attention_mask is not None else torch.ones_like(ii)
        tt = token_type_ids if token_type_ids is not None else torch.zeros_like(ii)

        # Truncate if longer than seq_len
        if L > seq_len:
            ii, am, tt = ii[:, :seq_len], am[:, :seq_len], tt[:, :seq_len]
            L = seq_len

        # Pad to fixed seq_len
        pad = seq_len - L
        if pad > 0:
            ii = torch.cat([ii, torch.zeros(B, pad, dtype=torch.long)], 1)
            am = torch.cat([am, torch.zeros(B, pad, dtype=torch.long)], 1)
            tt = torch.cat([tt, torch.zeros(B, pad, dtype=torch.long)], 1)

        out = rkl.inference(
            inputs=[
                ii.numpy().astype(np.int64),
                am.numpy().astype(np.int64),
                tt.numpy().astype(np.int64),
            ]
        )
        logits = torch.from_numpy(out[0])[:, :L, :]
        if return_dict:
            return TokenClassifierOutput(logits=logits)
        return (logits,)

    driver.model.forward = types.MethodType(patched_forward, driver.model)


# ─── Pipeline ────────────────────────────────────────────────────────────────


def load_models(backend: str, onnx_dir: str | None, rknn_dir: str | None, seq_len: int):
    """Load WS/POS/NER models and apply backend patches."""
    print("Loading CKIP models...", file=sys.stderr, flush=True)
    t0 = time.time()

    ws = CkipWordSegmenter(model="bert-base", device=-1)
    pos = CkipPosTagger(model="bert-base", device=-1)
    ner = CkipNerChunker(model="bert-base", device=-1)

    if backend == "onnx":
        assert onnx_dir, "--onnx-dir required for onnx backend"
        for label, driver in [("ws", ws), ("pos", pos), ("ner", ner)]:
            path = os.path.join(onnx_dir, f"ckip_bert_{label}.onnx")
            assert os.path.exists(path), f"ONNX not found: {path}"
            patch_onnx(driver, path)
        print(f"  Backend: ONNX Runtime (CPU)", file=sys.stderr)

    elif backend == "npu":
        assert rknn_dir, "--rknn-dir required for npu backend"
        for label, driver in [("ws", ws), ("pos", pos), ("ner", ner)]:
            path = os.path.join(rknn_dir, f"ckip_bert_{label}.rknn")
            assert os.path.exists(path), f"RKNN not found: {path}"
            patch_npu(driver, path, seq_len=seq_len)
        print(f"  Backend: RKNN NPU (seq_len={seq_len})", file=sys.stderr)

    else:
        print(f"  Backend: PyTorch (CPU)", file=sys.stderr)

    print(f"  Loaded in {time.time() - t0:.1f}s", file=sys.stderr, flush=True)
    return ws, pos, ner


def process_batch(
    ws, pos, ner, texts: list[str], batch_size: int
) -> list[dict]:
    """Run WS → POS (aligned) → NER on a batch of texts.

    Returns list of dicts with keys: text, ws, pos, ner.
    POS is guaranteed to align with WS via use_delim=True.
    """
    # Step 1: Word Segmentation
    ws_results = ws(texts, batch_size=batch_size)

    # Step 2: POS Tagging — feed WS results with use_delim=True for alignment
    pos_results = pos(ws_results, use_delim=True, batch_size=batch_size)

    # Step 3: NER — independent, operates on raw text
    ner_results = ner(texts, batch_size=batch_size)

    # Assemble output
    output = []
    for i, text in enumerate(texts):
        ws_tokens = ws_results[i]
        pos_tags = pos_results[i]

        # Sanity check alignment
        assert len(ws_tokens) == len(pos_tags), (
            f"WS/POS mismatch: {len(ws_tokens)} vs {len(pos_tags)} for: {text[:50]}"
        )

        # Format NER as serializable list
        ner_entities = [
            {"word": e.word, "type": e.ner, "start": e.idx[0], "end": e.idx[1]}
            for e in ner_results[i]
        ]

        output.append(
            {
                "text": text,
                "ws": ws_tokens,
                "pos": pos_tags,
                "ner": ner_entities,
            }
        )

    return output


def read_input(f: TextIO, input_format: str, text_field: str) -> list[tuple[str, dict]]:
    """Read input documents. Returns list of (text, extra_fields)."""
    docs = []
    for line in f:
        line = line.strip()
        if not line:
            continue
        if input_format == "jsonl":
            obj = json.loads(line)
            text = obj.pop(text_field, "")
            if text:
                docs.append((text, obj))
        else:
            docs.append((line, {}))
    return docs


def run(args):
    """Main pipeline execution."""
    backend = os.environ.get("BACKEND", args.backend)
    onnx_dir = os.environ.get("ONNX_DIR", args.onnx_dir)
    rknn_dir = os.environ.get("RKNN_DIR", args.rknn_dir)
    seq_len = int(os.environ.get("SEQ_LEN", args.seq_len))
    batch_size = int(os.environ.get("BATCH_SIZE", args.batch_size))

    # Load models
    ws, pos, ner = load_models(backend, onnx_dir, rknn_dir, seq_len)

    # Read input
    if args.input and args.input != "-":
        f_in = open(args.input, "r", encoding="utf-8")
    else:
        f_in = sys.stdin

    docs = read_input(f_in, args.input_format, args.text_field)
    if f_in is not sys.stdin:
        f_in.close()

    print(f"Processing {len(docs):,} documents (batch_size={batch_size})...",
          file=sys.stderr, flush=True)

    # Open output
    if args.output and args.output != "-":
        f_out = open(args.output, "w", encoding="utf-8")
    else:
        f_out = sys.stdout

    # Process in chunks
    total = len(docs)
    processed = 0
    t0 = time.time()

    for chunk_start in range(0, total, batch_size):
        chunk = docs[chunk_start : chunk_start + batch_size]
        texts = [t for t, _ in chunk]
        extras = [e for _, e in chunk]

        try:
            results = process_batch(ws, pos, ner, texts, batch_size=batch_size)
        except Exception as e:
            print(f"Error at chunk {chunk_start}: {e}", file=sys.stderr)
            # Fall back to one-by-one
            results = []
            for text in texts:
                try:
                    r = process_batch(ws, pos, ner, [text], batch_size=1)
                    results.extend(r)
                except Exception as e2:
                    print(f"  Skip: {text[:50]}... ({e2})", file=sys.stderr)
                    results.append({"text": text, "ws": [], "pos": [], "ner": []})

        for result, extra in zip(results, extras):
            # Merge extra fields back
            output_row = {**extra, **result}
            f_out.write(json.dumps(output_row, ensure_ascii=False) + "\n")

        processed += len(chunk)
        elapsed = time.time() - t0
        rate = processed / elapsed if elapsed > 0 else 0
        eta = (total - processed) / rate if rate > 0 else 0
        print(
            f"\r  {processed:,}/{total:,} ({rate:.1f} doc/s, ETA {eta:.0f}s)",
            end="", file=sys.stderr, flush=True,
        )

    if f_out is not sys.stdout:
        f_out.close()

    elapsed = time.time() - t0
    print(f"\nDone: {processed:,} docs in {elapsed:.1f}s ({processed/elapsed:.1f} doc/s)",
          file=sys.stderr, flush=True)


# ─── CLI ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="CKIP Full Pipeline: WS + POS (aligned) + NER",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # PyTorch, JSONL input
  python ckip_full_pipeline.py -i input.jsonl -o output.jsonl

  # ONNX accelerated, plain text input
  python ckip_full_pipeline.py --backend onnx --onnx-dir ./onnx \\
      --input-format text -i docs.txt -o results.jsonl

  # NPU on RK3588S
  python ckip_full_pipeline.py --backend npu --rknn-dir ./rknn \\
      --seq-len 256 --batch-size 1 -i input.jsonl

  # Pipe from stdin
  cat texts.jsonl | python ckip_full_pipeline.py --backend onnx --onnx-dir ./onnx
""",
    )
    parser.add_argument("-i", "--input", default="-", help="Input file (default: stdin)")
    parser.add_argument("-o", "--output", default="-", help="Output file (default: stdout)")
    parser.add_argument(
        "--input-format",
        choices=["jsonl", "text"],
        default="jsonl",
        help="Input format: jsonl (default) or text (one doc per line)",
    )
    parser.add_argument(
        "--text-field", default="text", help="JSON field name for text (default: 'text')"
    )
    parser.add_argument(
        "--backend",
        choices=["pytorch", "onnx", "npu"],
        default="pytorch",
        help="Inference backend (default: pytorch)",
    )
    parser.add_argument("--onnx-dir", help="Directory with ONNX models (for onnx backend)")
    parser.add_argument("--rknn-dir", help="Directory with RKNN models (for npu backend)")
    parser.add_argument("--seq-len", type=int, default=256, help="Fixed seq length for NPU (default: 256)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32, use 1 for NPU)")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
