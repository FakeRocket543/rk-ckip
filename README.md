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

## Repo Structure

```
rk-ckip/
├── scripts/
│   ├── ckip_onnx_bench.py    # PyTorch baseline + ONNX export + CPU ORT benchmark
│   ├── remove_isnan.py       # Remove IsNaN+Where pattern for RKNN compatibility
│   └── convert_rknn.py       # ONNX → RKNN conversion + NPU benchmark
├── benchmarks/                # Benchmark result JSONs
└── README.md
```
