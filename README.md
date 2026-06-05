# RK-CLIP — CKIP BERT ONNX on RK3588S

CKIP (繁體中文 NLP) ONNX Runtime acceleration benchmark on NanoPi M6 (RK3588S).

## Results

| Task | Baseline (ms) | ONNX (ms) | Speedup |
|------|--------------|-----------|---------|
| **WS** (斷詞) | 158.3 | 44.8 | **3.53x** |
| **POS** (詞性) | 155.4 | 48.2 | **3.22x** |
| **NER** (實體) | 157.2 | 48.7 | **3.23x** |
| **Total** | 470.9 | 141.7 | **3.32x** |

## Hardware

- NanoPi M6 (RK3588S, 4×A76 + 4×A55)
- CPU-only ONNX Runtime, no NPU

## Repo

```
https://github.com/FakeRocket543/rk-clip
```
