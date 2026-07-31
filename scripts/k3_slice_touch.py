#!/usr/bin/env python3
"""What does the K3 demo slice touch per token, and what tok/s does that allow?

The disk-resident arm is dead (see k3_fetch_budget.py / k3_batch_union.py), so
20 tok/s now belongs to the slice arm — a ~55-65 GB carve that lives in RAM.
Residency removes the link, but it does not remove the *weight-touch* ceiling:
every token still reads whatever the forward pass touches, at memory bandwidth.

The question this answers: how much does a token touch, split into the part
expert-side levers (row sparsity, branch-dropping, precision) can reduce and the
part they cannot, and what tok/s does each imply.

The split is the whole point. K3's always-resident weights are dominated by five
[12288, 7168] attention projections per KDA layer, and *no expert-side lever
touches them*. That puts a hard ceiling on what DEC-8.1 can deliver, which is
worth knowing before any model compute is spent on it.

Reads three real safetensors headers (one KDA layer, one MLA layer, cached by
k3_fetch_budget.py) plus config.json. No inference, no download.
"""

import argparse
import json
import math
from pathlib import Path

from k3_fetch_budget import DEFAULT_CACHE, RESOLVE, REPO, _get, load_config

# ── Bandwidth premises (project_memory_bandwidth_roofline, measured) ───────
BW_CPU_GB_S = 127.0
BW_GPU_GB_S = 367.0
TARGET_TOK_S = 20.0

# Demo-slice tier as specified in vindex-factory.md §15.3 / k3-funnel.md:
# "hot experts only, down-row payloads, up folded to per-feature scalars,
#  client slice included" — ~55-65 GB, the tier people actually download.
SLICE_GB = (55.0, 60.0, 65.0)

# Bytes per weight once carved. MXFP4 routed experts are already 4-bit+scales;
# the dense half is assumed re-quantised to the same nominal width.
BYTES_PER_WEIGHT_4BIT = 0.53125

# A KDA layer's shard was cached as hdr.json (tensor layer 49); an MLA layer
# needs its own. NOTE the index base: config's full_attn_layers/kda_layers are
# 1-indexed against 0-indexed tensor names, so config layer 4 == tensor layer 3.
MLA_SHARD = "model-00004-of-000096.safetensors"
SAFETENSORS_LEN_PREFIX = 8


def load_mla_header(cache: Path):
    path = cache / "hdr_mla.json"
    if not path.exists():
        import struct
        url = RESOLVE.format(repo=REPO, path=MLA_SHARD)
        n = struct.unpack("<Q", _get(url, (0, SAFETENSORS_LEN_PREFIX - 1)))[0]
        lo = SAFETENSORS_LEN_PREFIX
        path.write_bytes(_get(url, (lo, lo + n - 1)))
    h = json.loads(path.read_text())
    h.pop("__metadata__", None)
    return h


def load_kda_header(cache: Path):
    h = json.loads((cache / "hdr.json").read_text())
    h.pop("__metadata__", None)
    return h


def params(entry):
    n = 1
    for d in entry["shape"]:
        n *= d
    return n


def split_layer(hdr):
    """(non-expert params, one expert's params) for a layer's tensor table."""
    non_expert = sum(params(v) for k, v in hdr.items() if ".experts." not in k)
    # MXFP4: weight_packed holds 2 weights per byte, weight_scale is metadata.
    expert = sum(params(v) * 2 for k, v in hdr.items()
                 if ".experts.0." in k and k.endswith("weight_packed"))
    return non_expert, expert


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.cache)
    kda_hdr, mla_hdr = load_kda_header(args.cache), load_mla_header(args.cache)

    n_layers = cfg["num_hidden_layers"]
    n_moe = n_layers - cfg["first_k_dense_replace"]
    E, k = cfg["num_experts"], cfg["num_experts_per_token"]
    n_full = len(cfg["linear_attn_config"]["full_attn_layers"])
    n_kda = len(cfg["linear_attn_config"]["kda_layers"])

    kda_dense, expert_params = split_layer(kda_hdr)
    mla_dense, _ = split_layer(mla_hdr)
    embed = cfg["vocab_size"] * cfg["hidden_size"] * 2      # untied: embed + lm_head

    # ── Activated parameters per token ─────────────────────────────────────
    dense_params = n_kda * kda_dense + n_full * mla_dense + embed
    routed_params = k * n_moe * expert_params
    activated = dense_params + routed_params

    print("=" * 78)
    print("K3 slice arm — per-token weight touch and the tok/s it allows")
    print("=" * 78)
    print(f"layers {n_layers} = {n_kda} KDA + {n_full} MLA, {k}-of-{E}, "
          f"{n_moe} MoE layers")
    print()
    print("--- per-layer dense (non-expert) parameters ---")
    print(f"  KDA layer  {kda_dense/1e6:8.1f} M  "
          f"(5 x [12288,7168] attention projections dominate)")
    print(f"  MLA layer  {mla_dense/1e6:8.1f} M  "
          f"(q_a/q_b/kv_a/kv_b compress the same role into {(mla_dense-kda_dense)/1e6:+.0f} M)")
    print()
    print("--- activated parameters per token ---")
    print(f"  dense (always on)   {dense_params/1e9:7.2f} B   "
          f"{100*dense_params/activated:5.1f}%")
    print(f"  routed ({k} experts)  {routed_params/1e9:7.2f} B   "
          f"{100*routed_params/activated:5.1f}%")
    print(f"  TOTAL               {activated/1e9:7.2f} B")
    print()

    dense_b = dense_params * BYTES_PER_WEIGHT_4BIT
    routed_b = routed_params * BYTES_PER_WEIGHT_4BIT
    touch = dense_b + routed_b

    print("--- per-token touch at 4-bit, and the bandwidth ceiling ---")
    print(f"{'':<26}{'GB/token':>10}{'@127 GB/s':>12}{'@367 GB/s':>12}")
    for label, b in (("dense (irreducible)", dense_b),
                     ("routed experts", routed_b),
                     ("TOTAL, dense execution", touch)):
        print(f"  {label:<24}{b/1e9:>10.2f}"
              f"{BW_CPU_GB_S*1e9/b:>12.1f}{BW_GPU_GB_S*1e9/b:>12.1f}")
    print()

    budget = BW_GPU_GB_S * 1e9 / TARGET_TOK_S
    print(f"--- the {TARGET_TOK_S:.0f} tok/s budget ---")
    print(f"  allowed          {budget/1e9:.2f} GB/token at {BW_GPU_GB_S:.0f} GB/s")
    print(f"  actual           {touch/1e9:.2f} GB/token  -> need "
          f"{touch/budget:.2f}x reduction")
    print(f"  dense alone      {dense_b/1e9:.2f} GB/token = "
          f"{dense_b/budget:.2f}x the ENTIRE budget, with zero expert bytes")
    print()
    print(f"  >> expert-side levers cap out at {touch/dense_b:.2f}x "
          f"(experts are only {100*routed_b/touch:.0f}% of the bytes).")
    print(f"  >> even a PERFECT expert lever leaves "
          f"{BW_GPU_GB_S*1e9/dense_b:.1f} tok/s. "
          f"{TARGET_TOK_S:.0f} is unreachable expert-side.")
    print()

    # ── What actually fits in the slice, and how much of routing it covers ──
    print("--- slice composition (demo tier: down-row payloads, up folded) ---")
    down_only = expert_params / 3 * BYTES_PER_WEIGHT_4BIT   # w2 of w1/w2/w3
    print(f"  per-expert demo payload (w2 only) {down_only/1e6:.2f} MB "
          f"of {expert_params*BYTES_PER_WEIGHT_4BIT/1e6:.2f} MB full")
    rows = []
    for gb in SLICE_GB:
        expert_budget = gb * 1e9 - dense_b
        n_res = max(0.0, expert_budget / down_only)
        frac = n_res / (E * n_moe)
        hits = k * frac                       # uniform routing, per layer
        row_touch = dense_b + hits * n_moe * down_only
        rows.append({"slice_gb": gb, "resident_experts": n_res,
                     "resident_frac": frac, "uniform_hits_per_layer": hits,
                     "touch_gb": row_touch / 1e9,
                     "tok_s_gpu": BW_GPU_GB_S * 1e9 / row_touch})
        print(f"  {gb:.0f} GB slice: dense {dense_b/1e9:.1f} GB + "
              f"{n_res:,.0f} experts ({100*frac:.1f}% of {E*n_moe:,}) -> "
              f"{hits:.2f} of {k} hit per layer under uniform routing")
    print()
    print("  Under uniform routing a demo slice hits ~1 of 16 experts per layer.")
    print("  Everything the slice arm claims rests on routing skew making that")
    print("  number much larger — which is exactly what gate GC measures, and")
    print("  what V2/V3's 'no cacheable hot subset' result argues against.")
    print("=" * 78)

    out = {
        "premises": {"bw_cpu_gb_s": BW_CPU_GB_S, "bw_gpu_gb_s": BW_GPU_GB_S,
                     "target_tok_s": TARGET_TOK_S,
                     "bytes_per_weight": BYTES_PER_WEIGHT_4BIT},
        "layer_counts": {"kda": n_kda, "mla": n_full, "moe": n_moe},
        "params": {"kda_layer_dense": kda_dense, "mla_layer_dense": mla_dense,
                   "expert": expert_params, "embeddings": embed,
                   "dense_total": dense_params, "routed_activated": routed_params,
                   "activated_total": activated},
        "touch_bytes": {"dense": dense_b, "routed": routed_b, "total": touch},
        "tok_s": {"dense_execution_gpu": BW_GPU_GB_S * 1e9 / touch,
                  "dense_execution_cpu": BW_CPU_GB_S * 1e9 / touch,
                  "perfect_expert_lever_gpu": BW_GPU_GB_S * 1e9 / dense_b},
        "reduction_needed_x": touch / budget,
        "expert_side_lever_ceiling_x": touch / dense_b,
        "routed_share_of_touch": routed_b / touch,
        "slice_rows": rows,
    }
    (args.out or (args.cache / "k3_slice_touch.json")).write_text(
        json.dumps(out, indent=2))
    print(f"wrote {args.out or (args.cache / 'k3_slice_touch.json')}")


if __name__ == "__main__":
    main()
