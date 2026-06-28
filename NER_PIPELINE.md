# Full NPU NER Pipeline (text → entities on RK3588S NPU)

`scripts/ckip_npu_ner_pipeline.py` — runs CKIP BERT NER over text using the
converted `.rknn` on the **NPU** (cool + fast), via `rknn-toolkit-lite2` +
`ckip_transformers` (tokenize + BIO decode reused through a forward-patch).

## How it works
- `RKNNLite.load_rknn(ckip_bert_ner.rknn)` + `init_runtime(NPU_CORE_0_1_2)`.
- Monkey-patch `CkipNerChunker.model.forward` to: take variable-len input →
  truncate/pad to fixed `seq_len` (256) → `rkl.inference()` → slice logits back.
- `CkipNerChunker` does tokenization + BIO decoding as usual → reuse its decode.

## Findings / gotchas
1. **.rknn is fixed `batch=1`** (convert_rknn `fix_static_shape` sets batch=1).
   → MUST call `CkipNerChunker(..., batch_size=1)`. batch>1 hangs (shape mismatch).
2. **seq_len boundary**: input must tokenize WELL under `seq_len` (e.g. content[:150]
   for seq_len=256). Inputs near the boundary (~255 tokens) hit an off-by-one in
   ckip decode ("index 255 out of bounds for axis 1 with size 255").
3. **Conversion deps** (rknn-toolkit2 2.3.2): needs `setuptools<81`
   (pkg_resources removed in 81+) and `onnx==1.15` (`onnx.mapping` removed later).
4. **Cool**: NPU run ≈ 49°C vs CPU ORT ≈ 73°C (8-core sustained) on fanless NanoPi-M6.
   7,670 posts in ~18 min on NPU (batch=1, seq_len 256).

## Real-world result (wowsight.tw sports news, 7,670 posts)
Union of LLM-NER (Qwen-2.5-7B) + this CKIP-NPU NER:
- 100% post coverage (7,670/7,670), 36,879 entities, 102,825 links.
- Recall vs grep: 中華隊 33%→57%, adidas 38%→53%, Nike 19%→29%
  (CKIP sees title+lede only at seq_len 256; deeper mentions need longer context).

## Usage
```bash
# after convert_rknn.py produced rknn/ckip_bert_ner.rknn (seq_len 256)
pip install ckip_transformers rknn-toolkit-lite2 onnxruntime torch transformers requests
SEQ=256 LIMIT=0 python scripts/ckip_npu_ner_pipeline.py   # dumps ckip_npu_entities.jsonl
```
