#!/usr/bin/env python3
"""
Generate the synthetic tiny-vindex used by the isolation-harness CI tests.

Reproducible: fixed seed=42, no external dependencies beyond numpy.
Run from the repo root:
    python3 testdata/tiny-vindex/generate.py

Architecture parameters match what larql-server expects for family="llama":
  num_layers=8, hidden=128, intermediate=256, vocab=512,
  num_q_heads=4, num_kv_heads=2, head_dim=32

Weight tensor names follow the HuggingFace / larql extraction convention:
  model.embed_tokens.weight
  layers.{N}.self_attn.q_proj.weight  (and k/v/o)
  layers.{N}.mlp.gate_proj.weight     (and up/down)
  layers.{N}.input_layernorm.weight
  layers.{N}.post_attention_layernorm.weight
  model.norm.weight
  lm_head.weight
"""
import json
import os
import struct
import numpy as np

SEED       = 42
NUM_LAYERS = 8
HIDDEN     = 128
INTER      = 256
VOCAB      = 512
Q_HEADS    = 4
KV_HEADS   = 2
HEAD_DIM   = 32
# Features per layer in the vindex. This MUST equal INTER: a vindex feature
# *is* an FFN intermediate channel, so the gate-vector table and the MLP's
# gate/up/down projections are indexed by the same axis. The fixture used to
# declare 32 here against intermediate_size=256, which loads and serves
# DESCRIBE/WALK fine but blows up in the dense FFN as
# "ShapeError/IncompatibleShape" — silu_gate_up multiplying a 32-wide gate by
# a 256-wide up. That is what made INFER unusable on this fixture.
FEATURES   = INTER
DOWN_TOP_K = 10

rng = np.random.default_rng(SEED)
OUT = os.path.dirname(__file__)


def write_f32(path, arr):
    arr = np.array(arr, dtype=np.float32).flatten()
    with open(path, "wb") as f:
        f.write(arr.tobytes())


def norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


# ── embeddings.bin ──────────────────────────────────────────────────────────
embeds = rng.standard_normal((VOCAB, HIDDEN)).astype(np.float32)
write_f32(os.path.join(OUT, "embeddings.bin"), embeds)

# ── gate_vectors.bin  (NUM_LAYERS × FEATURES × HIDDEN) ──────────────────────
gate = rng.standard_normal((NUM_LAYERS, FEATURES, HIDDEN)).astype(np.float32)
for li in range(NUM_LAYERS):
    for fi in range(FEATURES):
        gate[li, fi] = norm(gate[li, fi])
write_f32(os.path.join(OUT, "gate_vectors.bin"), gate)

# ── down_meta.bin  (binary "DMET" format) ──────────────────────────────────
# Mirrors larql_vindex::format::down_meta::write_binary. This used to emit a
# bare f32 array with no header, which the reader has rejected since the
# format grew one — the committed fixture was unloadable ("invalid
# down_meta.bin magic"). Keep this in step with that writer.
#
#   Header (16 bytes): magic u32, version u32, num_layers u32, top_k u32
#   Per layer:  num_features u32
#   Per feature: top_token_id u32, c_score f32,
#                then top_k × (token_id u32, logit f32)
DOWN_META_MAGIC   = 0x444D4554  # "DMET"
DOWN_META_VERSION = 1

with open(os.path.join(OUT, "down_meta.bin"), "wb") as fh:
    fh.write(struct.pack("<IIII", DOWN_META_MAGIC, DOWN_META_VERSION,
                         NUM_LAYERS, DOWN_TOP_K))
    for li in range(NUM_LAYERS):
        fh.write(struct.pack("<I", FEATURES))
        for fi in range(FEATURES):
            ids    = rng.choice(VOCAB, DOWN_TOP_K, replace=False)
            logits = rng.standard_normal(DOWN_TOP_K).astype(np.float32)
            logits = np.exp(logits - logits.max())
            logits /= logits.sum()
            # top_token_id is the highest-scoring entry, and c_score its logit.
            best = int(np.argmax(logits))
            fh.write(struct.pack("<If", int(ids[best]), float(logits[best])))
            for k in range(DOWN_TOP_K):
                fh.write(struct.pack("<If", int(ids[k]), float(logits[k])))

# ── model_weights.bin  (all attention + FFN + norm tensors) ─────────────────
tensors = {}

def t(name, shape):
    tensors[name] = rng.standard_normal(shape).astype(np.float32)

t("model.embed_tokens.weight", (VOCAB, HIDDEN))
for n in range(NUM_LAYERS):
    t(f"layers.{n}.self_attn.q_proj.weight",              (Q_HEADS  * HEAD_DIM, HIDDEN))
    t(f"layers.{n}.self_attn.k_proj.weight",              (KV_HEADS * HEAD_DIM, HIDDEN))
    t(f"layers.{n}.self_attn.v_proj.weight",              (KV_HEADS * HEAD_DIM, HIDDEN))
    t(f"layers.{n}.self_attn.o_proj.weight",              (HIDDEN, Q_HEADS * HEAD_DIM))
    t(f"layers.{n}.mlp.gate_proj.weight",                 (INTER, HIDDEN))
    t(f"layers.{n}.mlp.up_proj.weight",                   (INTER, HIDDEN))
    t(f"layers.{n}.mlp.down_proj.weight",                 (HIDDEN, INTER))
    t(f"layers.{n}.input_layernorm.weight",               (HIDDEN,))
    t(f"layers.{n}.post_attention_layernorm.weight",       (HIDDEN,))
t("model.norm.weight", (HIDDEN,))
t("lm_head.weight",    (VOCAB, HIDDEN))

# Pack all tensors into one blob; weight_manifest.json is a LIST of entries
# matching larql_vindex::format::weights::WeightEntry:
#   {key, kind, shape, offset, length, file}
#
# It used to be a {name: {...}} map, which the loader rejects outright
# ("invalid type: map, expected a sequence"). `kind` distinguishes a 2D
# tensor from a 1D vector (norms) — the loader routes on it, and a norm
# mislabelled "tensor" would fail the Array2 reshape.
manifest = []
blob_parts = []
offset = 0
for name, arr in tensors.items():
    raw = arr.tobytes()
    manifest.append({
        "key":    name,
        "kind":   "tensor" if arr.ndim == 2 else "vector",
        "shape":  list(arr.shape),
        "offset": offset,
        "length": len(raw),
        "file":   "model_weights.bin",
    })
    blob_parts.append(raw)
    offset += len(raw)

with open(os.path.join(OUT, "model_weights.bin"), "wb") as f:
    for part in blob_parts:
        f.write(part)

with open(os.path.join(OUT, "weight_manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

# ── tokenizer.json  (minimal HF tokenizer schema) ───────────────────────────
vocab_map = {f"[{i}]": i for i in range(VOCAB)}
vocab_map["[UNK]"] = 0
tok = {
    "version": "1.0",
    "truncation": None,
    "padding": None,
    "added_tokens": [],
    "normalizer": None,
    "pre_tokenizer": None,
    "post_processor": None,
    "decoder": None,
    "model": {
        "type": "BPE",
        "dropout": None,
        "unk_token": "[UNK]",
        "continuing_subword_prefix": None,
        "end_of_word_suffix": None,
        "fuse_unk": False,
        "byte_fallback": False,
        "vocab": vocab_map,
        "merges": [],
    },
}
with open(os.path.join(OUT, "tokenizer.json"), "w") as f:
    json.dump(tok, f, indent=2)

# ── index.json ───────────────────────────────────────────────────────────────
layers_meta = []
for li in range(NUM_LAYERS):
    byte_offset = li * FEATURES * HIDDEN * 4
    layers_meta.append({
        "layer":        li,
        "num_features": FEATURES,
        "offset":       byte_offset,
        "length":       FEATURES * HIDDEN * 4,
    })

index = {
    "version":           2,
    "model":             "test/tiny-vindex",
    "family":            "llama",
    "source": {
        "huggingface_repo":     "test/tiny-vindex",
        "huggingface_revision": None,
        "safetensors_sha256":   None,
        "extracted_at":         "2026-04-19T00:00:00Z",
        "larql_version":        "0.1.0",
    },
    "checksums":         {},
    "num_layers":        NUM_LAYERS,
    "hidden_size":       HIDDEN,
    "intermediate_size": INTER,
    "vocab_size":        VOCAB,
    "embed_scale":       1.0,
    "extract_level":     "all",
    "dtype":             "f32",
    "layer_bands": {
        "syntax":    [0, 1],
        "knowledge": [2, 5],
        "output":    [6, 7],
    },
    "layers":            layers_meta,
    "down_top_k":        DOWN_TOP_K,
    "has_model_weights": True,
    "model_config": {
        "model_type":      "llama",
        "head_dim":        HEAD_DIM,
        "num_q_heads":     Q_HEADS,
        "num_kv_heads":    KV_HEADS,
        "rope_base":       10000.0,
        "sliding_window":  None,
        "moe":             None,
        "attention_k_eq_v": False,
    },
}
with open(os.path.join(OUT, "index.json"), "w") as f:
    json.dump(index, f, indent=2)

total = sum(os.path.getsize(os.path.join(OUT, fn))
            for fn in os.listdir(OUT) if not fn.endswith(".py"))
print(f"Generated tiny-vindex: {total/1024/1024:.1f} MB  ({NUM_LAYERS} layers, hidden={HIDDEN})")
