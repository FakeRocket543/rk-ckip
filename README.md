# RK-CKIP — CKIP BERT ONNX on RK3588S

CKIP (繁體中文 NLP) ONNX Runtime acceleration benchmark on NanoPi M6 (RK3588S).

## Results

### CPU: PyTorch vs ONNX FP32 vs ONNX INT8

| Task | PyTorch (ms) | ONNX FP32 (ms) | ONNX INT8 (ms) | INT8 vs PyTorch |
|------|-------------|----------------|----------------|-----------------|
| **WS** (斷詞) | 205 | 47 | **17** | **12.1x** |
| **POS** (詞性) | 205 | 48 | **17** | **12.1x** |
| **NER** (實體) | 205 | 49 | **17** | **12.1x** |
| **Total** | 615 | 144 | **50** | **12.3x** |

| Model size | FP32 | INT8 | Ratio |
|------------|------|------|-------|
| per model | 388 MB | 98 MB | 4.0x |
| ×3 models | 1.16 GB | 0.29 GB | 4.0x |

> **Note**: FP16 is not benchmarked — ARM Cortex-A76/A55 CPU has no FP16 SIMD
> hardware path, so FP16 inference adds cast overhead with no benefit. INT8
> dynamic quantization leverages NEON dot-product instructions for real speedup.

### NPU: RKNN on RK3588S (seq_len=64)

| Task | CPU ORT (ms) | NPU ×1 (ms) | NPU ×3 (ms) | Speedup (×3 vs ORT) |
|------|-------------|-------------|-------------|---------------------|
| **WS** | 57 | 58 | 42 | **1.37x** |
| **POS** | 57 | 58 | 42 | **1.37x** |
| **NER** | 58 | 58 | 42 | **1.38x** |

**Total acceleration vs PyTorch**: ~4.5x (ORT 3.3x × NPU 1.37x)

### NPU Conversion Notes

CKIP BERT (bert-base-chinese) requires two fixes before RKNN conversion:

1. **Remove `IsNaN` ops** — PyTorch opset-17 export inserts safe_softmax checks (`IsNaN` + `Where`) that librknnrt does not support. These are safely removable since softmax never produces NaN in normal inference.

2. **Fix to static input shape** — RKNN requires fixed tensor dimensions. Dynamic `batch`/`seq` dims must be replaced with concrete values (e.g. `[1, 64]`).

## Hardware

- NanoPi M6 (RK3588S, 4×A76 + 4×A55, 6 TOPS NPU)
- OS: Debian 13 (Trixie), kernel 6.1.141
- NPU driver: rknpu 0.9.8 (DRM mode, `/dev/dri/card1`)
- Runtime: librknnrt 2.3.2
- rknn-toolkit2 / rknn-toolkit-lite2: 2.3.2
- Python 3.12 (rknn max supported), ONNX Runtime 1.27

## Usage

```bash
# 1. Export CKIP BERT to ONNX (requires ckip-transformers)
python scripts/ckip_onnx_bench.py

# 2. Remove IsNaN ops for RKNN compatibility
python scripts/remove_isnan.py --input-dir /path/to/onnx --output-dir /path/to/clean

# 3. Convert to RKNN + benchmark on NPU
python scripts/convert_rknn.py --onnx-dir /path/to/clean --output-dir ./rknn --bench
```

## Full Pipeline: WS + POS + NER

`scripts/ckip_full_pipeline.py` runs the complete CKIP NLP pipeline with correct WS→POS alignment.

### WS→POS Alignment

CKIP's POS tagger has its own internal word segmenter. If you feed raw text directly to POS,
it will produce a **different** segmentation than WS — lengths won't match and tokens can't be paired.

**Correct approach**: run WS first, then feed WS output to POS with `use_delim=True`:

```python
ws_results = ws(sentences)                          # Step 1: segment
pos_results = pos(ws_results, use_delim=True)       # Step 2: tag (aligned!)
# len(ws_results[i]) == len(pos_results[i])  ← guaranteed
```

NER is independent — it operates on raw text and returns span-based entities.

### Usage

```bash
# Prepare input (JSONL with "text" field, or plain text one-doc-per-line)
echo '{"id": 1, "text": "台北市長柯文哲出席記者會。"}' > input.jsonl

# PyTorch (no acceleration)
python scripts/ckip_full_pipeline.py -i input.jsonl -o output.jsonl

# ONNX accelerated (~3.3x faster)
python scripts/ckip_full_pipeline.py --backend onnx --onnx-dir ./onnx -i input.jsonl

# NPU on RK3588S (~4.5x faster, batch_size must be 1)
python scripts/ckip_full_pipeline.py --backend npu --rknn-dir ./rknn \
    --seq-len 256 --batch-size 1 -i input.jsonl

# Plain text input
python scripts/ckip_full_pipeline.py --input-format text -i docs.txt -o results.jsonl

# Pipe from stdin
cat input.jsonl | python scripts/ckip_full_pipeline.py --backend onnx --onnx-dir ./onnx
```

### Output Format

```json
{
  "id": 1,
  "text": "台北市長柯文哲出席記者會。",
  "ws": ["台北市長", "柯文哲", "出席", "記者會", "。"],
  "pos": ["Nc", "Nb", "VC", "Na", "PERIODCATEGORY"],
  "ner": [{"word": "台北", "type": "GPE", "start": 0, "end": 2},
          {"word": "柯文哲", "type": "PERSON", "start": 4, "end": 7}]
}
```

- `ws` and `pos` are **always the same length** (aligned by design)
- `ner` uses character-level `start`/`end` indices into the original text
- Extra fields from input JSONL (e.g. `id`) are preserved in output

### Backend Comparison

| Backend | Speed (sent/s) | Memory | Requirements |
|---------|---------------|--------|--------------|
| pytorch | ~5 | 1.8 GB | ckip-transformers, torch |
| onnx (FP32) | ~21 | 1.8 GB | + onnxruntime |
| onnx (INT8) | ~60 | 0.5 GB | + onnxruntime (dynamic quantization) |
| npu | ~18* | 0.6 GB | + rknn-toolkit-lite2, .rknn, RK3588S |

\* NPU requires batch_size=1 and fixed seq_len.

### INT8 Quantization

```bash
# Quantize FP32 → INT8 (dynamic, weight-only)
python scripts/bench_quant.py
# Output: onnx/q8/ckip_bert_{ws,pos,ner}_int8.onnx
```

## Repo Structure

```
rk-ckip/
├── scripts/
│   ├── ckip_api.py           # FastAPI server: /ws /pos /ner /pipeline (ONNX)
│   ├── ckip_full_pipeline.py # WS+POS+NER CLI pipeline (pytorch/onnx/npu)
│   ├── ckip_npu_ner_pipeline.py  # NER-only NPU pipeline (wowsight.tw)
│   ├── ckip_onnx_bench.py    # PyTorch baseline + ONNX export + benchmark
│   ├── bench_quant.py        # INT8 quantization + FP32 vs INT8 benchmark
│   ├── remove_isnan.py       # Remove IsNaN+Where for RKNN compatibility
│   └── convert_rknn.py       # ONNX → RKNN conversion + NPU benchmark
├── benchmarks/                # Benchmark result JSONs
└── README.md
```
