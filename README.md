# RK-CKIP — CKIP BERT ONNX on RK3588S

CKIP (繁體中文 NLP) ONNX Runtime acceleration benchmark on NanoPi M6 (RK3588S).

## Results

### CPU: PyTorch vs ONNX Runtime

| Task | PyTorch (ms) | ONNX (ms) | Speedup |
|------|-------------|-----------|---------|
| **WS** (斷詞) | 158.3 | 44.8 | **3.53x** |
| **POS** (詞性) | 155.4 | 48.2 | **3.22x** |
| **NER** (實體) | 157.2 | 48.7 | **3.23x** |
| **Total** | 470.9 | 141.7 | **3.32x** |

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

| Backend | Speed (sent/s) | Requirements |
|---------|---------------|--------------|
| pytorch | ~4 | ckip-transformers, torch |
| onnx | ~13 | + onnxruntime, exported .onnx |
| npu | ~18* | + rknn-toolkit-lite2, .rknn, RK3588S |

\* NPU requires batch_size=1 and fixed seq_len.

## Repo Structure

```
rk-ckip/
├── scripts/
│   ├── ckip_full_pipeline.py # WS+POS+NER full pipeline (generic input)
│   ├── ckip_npu_ner_pipeline.py  # NER-only NPU pipeline (wowsight.tw specific)
│   ├── ckip_onnx_bench.py    # PyTorch baseline + ONNX export + CPU ORT benchmark
│   ├── remove_isnan.py       # Remove IsNaN+Where pattern for RKNN compatibility
│   └── convert_rknn.py       # ONNX → RKNN conversion + NPU benchmark
├── benchmarks/                # Benchmark result JSONs
└── README.md
```
