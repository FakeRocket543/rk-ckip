#!/usr/bin/env python3
"""
CKIP NLP API Server (no auth, localhost only)
WS + POS + NER via ONNX Runtime on RK3588S CPU.

Run:
    ~/npu-test/venv/bin/python3 ckip_api.py
    # or with options:
    PORT=8900 ~/npu-test/venv/bin/python3 ckip_api.py
"""
import os
import sys
import time
import types
import logging

import numpy as np
import torch
import onnxruntime as ort
from transformers.modeling_outputs import TokenClassifierOutput
from ckip_transformers.nlp import (
    CkipWordSegmenter,
    CkipPosTagger,
    CkipNerChunker,
)
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ckip-api")

ONNX_DIR = os.environ.get("ONNX_DIR", "/mnt/nvme/rk-ckip/onnx/q8")
ONNX_SUFFIX = os.environ.get("ONNX_SUFFIX", "_int8.onnx")
ORT_THREADS = int(os.environ.get("ORT_THREADS", "4"))
PORT = int(os.environ.get("PORT", "8900"))

# ─── Models ──────────────────────────────────────────────────────────────────

app = FastAPI(title="CKIP NLP API", version="1.0")
_models: dict = {}


class Models:
    ws: CkipWordSegmenter
    pos: CkipPosTagger
    ner: CkipNerChunker


def _patch_onnx(driver, onnx_path: str, threads: int = ORT_THREADS):
    """Monkey-patch a CKIP driver's model.forward to use ONNX Runtime."""
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = threads
    sess = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])
    log.info(f"  ORT session loaded: {onnx_path} ({threads} threads)")

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


@app.on_event("startup")
def load():
    log.info("Loading CKIP models (ONNX backend)...")
    t0 = time.time()

    m = Models()
    m.ws = CkipWordSegmenter(model="bert-base", device=-1)
    m.pos = CkipPosTagger(model="bert-base", device=-1)
    m.ner = CkipNerChunker(model="bert-base", device=-1)

    for label, driver in [("ws", m.ws), ("pos", m.pos), ("ner", m.ner)]:
        path = os.path.join(ONNX_DIR, f"ckip_bert_{label}{ONNX_SUFFIX}")
        assert os.path.exists(path), f"ONNX not found: {path}"
        _patch_onnx(driver, path)

    _models["m"] = m
    log.info(f"Ready in {time.time()-t0:.1f}s — listening on :{PORT}")


# ─── Request / Response schemas ──────────────────────────────────────────────

class TextRequest(BaseModel):
    text: str | list[str]
    batch_size: int = Field(default=32, ge=1)


class WSRequest(TextRequest):
    pass


class POSRequest(BaseModel):
    ws_result: list[list[str]]
    batch_size: int = Field(default=32, ge=1)


class PipelineRequest(TextRequest):
    pass


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    if "m" not in _models:
        return {"status": "loading"}
    return {"status": "ok", "backend": "onnx-int8", "threads": ORT_THREADS}


@app.post("/ws")
def word_segmentation(req: WSRequest):
    m = _models["m"]
    texts = [req.text] if isinstance(req.text, str) else req.text
    t0 = time.time()
    results = m.ws(texts, batch_size=req.batch_size)
    elapsed = (time.time() - t0) * 1000
    return {
        "ws": results,
        "count": len(results),
        "ms": round(elapsed, 1),
    }


@app.post("/pos")
def pos_tagging(req: POSRequest):
    m = _models["m"]
    t0 = time.time()
    results = m.pos(req.ws_result, use_delim=True, batch_size=req.batch_size)
    elapsed = (time.time() - t0) * 1000
    return {
        "pos": results,
        "count": len(results),
        "ms": round(elapsed, 1),
    }


@app.post("/ner")
def named_entity_recognition(req: TextRequest):
    m = _models["m"]
    texts = [req.text] if isinstance(req.text, str) else req.text
    t0 = time.time()
    raw = m.ner(texts, batch_size=req.batch_size)
    elapsed = (time.time() - t0) * 1000
    # Serialize NerToken objects
    results = []
    for entities in raw:
        results.append([
            {"word": e.word, "type": e.ner, "start": e.idx[0], "end": e.idx[1]}
            for e in entities
        ])
    return {
        "ner": results,
        "count": len(results),
        "ms": round(elapsed, 1),
    }


@app.post("/pipeline")
def full_pipeline(req: PipelineRequest):
    """WS → POS (aligned) → NER in one call."""
    m = _models["m"]
    texts = [req.text] if isinstance(req.text, str) else req.text
    t0 = time.time()

    ws_results = m.ws(texts, batch_size=req.batch_size)
    pos_results = m.pos(ws_results, use_delim=True, batch_size=req.batch_size)
    ner_raw = m.ner(texts, batch_size=req.batch_size)

    elapsed = (time.time() - t0) * 1000

    results = []
    for i, text in enumerate(texts):
        ner_entities = [
            {"word": e.word, "type": e.ner, "start": e.idx[0], "end": e.idx[1]}
            for e in ner_raw[i]
        ]
        results.append({
            "text": text,
            "ws": ws_results[i],
            "pos": pos_results[i],
            "ner": ner_entities,
        })

    return {
        "results": results,
        "count": len(results),
        "ms": round(elapsed, 1),
    }


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")
