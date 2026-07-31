#!/usr/bin/env python3
"""Price K3's per-token expert fetch against a link budget, from the real
checkpoint header — no download, no inference.

The question: to serve full Kimi K3 at T tok/s with experts on an external SSD,
whatever isn't resident must cross the link. That sets a hard bytes/token miss
budget. Everything else (row sparsity, dropping a GLU branch, precision tiering,
striping ports, cache locality) is a divisor on the required fetch. This script
computes the required fetch exactly, so the divisors are applied to a real
number rather than an assumed one.

Inputs are the safetensors header of one middle shard (fetched via HTTP
range-GET) plus config.json. Both are already on disk in this scratchpad.
"""

import argparse
import json
import re
import struct
import urllib.request
from pathlib import Path

# ── Where the inputs come from ─────────────────────────────────────────────
# The header of one middle shard is enough: every layer has identical shape.
REPO = "moonshotai/Kimi-K3"
SHARD = "model-00050-of-000096.safetensors"
RESOLVE = "https://huggingface.co/{repo}/resolve/main/{path}"
SAFETENSORS_LEN_PREFIX = 8  # little-endian u64 header length

DEFAULT_CACHE = Path(__file__).parent / ".k3_budget_cache"

# ── Link / target premises (the user's stated operating point) ──────────────
LINK_GB_S = 3.5          # measured real USB4 throughput to one NVMe enclosure
TARGET_TOK_S = 20.0      # the serving target under test
TB_PORTS = 3             # Thunderbolt ports on the M3 Max (the "stripe" lever)

# ── Lever divisors, as claimed in the memo (unproven ones flagged) ──────────
LEVERS = {
    # name:            (divisor, proven?)
    "locality":        (1.5,  "measured (obs/shuffled, Qwen3-30B-A3B, K=8)"),
    "stripe_3_ports":  (float(TB_PORTS), "hardware, ~GBP700"),
    "precision_tail":  (1.4,  "UNPROVEN — cold-tail bounded-KL sweep"),
    "drop_up_branch":  (None, "UNPROVEN — computed below from real shapes"),
    "row_sparsity":    (None, "UNPROVEN — solved for below"),
}


def _get(url, byte_range=None):
    req = urllib.request.Request(url)
    if byte_range:
        req.add_header("Range", f"bytes={byte_range[0]}-{byte_range[1]}")
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def load_header(cache: Path):
    """Fetch just the safetensors header of one shard — ~824 KB of a 17 GB file."""
    path = cache / "hdr.json"
    if not path.exists():
        url = RESOLVE.format(repo=REPO, path=SHARD)
        n = struct.unpack("<Q", _get(url, (0, SAFETENSORS_LEN_PREFIX - 1)))[0]
        lo = SAFETENSORS_LEN_PREFIX
        path.write_bytes(_get(url, (lo, lo + n - 1)))
    hdr = json.loads(path.read_text())
    hdr.pop("__metadata__", None)
    return hdr


def load_config(cache: Path):
    path = cache / "config.json"
    if not path.exists():
        path.write_bytes(_get(RESOLVE.format(repo=REPO, path="config.json")))
    return json.loads(path.read_text())["text_config"]


def tensor_bytes(entry):
    lo, hi = entry["data_offsets"]
    return hi - lo


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE,
                    help="where to keep the fetched config/header")
    ap.add_argument("--out", type=Path, default=None,
                    help="write the result JSON here (default: <cache>/k3_fetch_budget.json)")
    args = ap.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    out_path = args.out or (args.cache / "k3_fetch_budget.json")

    hdr, cfg = load_header(args.cache), load_config(args.cache)

    n_layers = cfg["num_hidden_layers"]
    n_dense = cfg["first_k_dense_replace"]
    n_moe_layers = n_layers - n_dense
    n_experts = cfg["num_experts"]
    top_k = cfg["num_experts_per_token"]

    # ── Split the shard into routed-expert bytes vs everything else ─────────
    expert_pat = re.compile(r"\.experts\.(\d+)\.")
    per_expert = {}   # expert_id -> bytes
    other = {}        # tensor name -> bytes
    for name, entry in hdr.items():
        b = tensor_bytes(entry)
        m = expert_pat.search(name)
        if m:
            per_expert[m.group(1)] = per_expert.get(m.group(1), 0) + b
        else:
            other[name] = b

    sizes = set(per_expert.values())
    assert len(sizes) == 1, f"experts differ in size: {sorted(sizes)[:5]}"
    expert_bytes = sizes.pop()
    assert len(per_expert) == n_experts, f"{len(per_expert)} experts != {n_experts}"

    # Per-expert breakdown by GLU branch (w1=gate, w3=up, w2=down)
    branch = {}
    for name, entry in hdr.items():
        m = expert_pat.search(name)
        if m and m.group(1) == "0":
            w = re.search(r"\.(w[123])\.", name).group(1)
            branch[w] = branch.get(w, 0) + tensor_bytes(entry)

    always_on_per_layer = sum(other.values())

    # ── The binding equation ───────────────────────────────────────────────
    miss_budget = LINK_GB_S * 1e9 / TARGET_TOK_S            # bytes/token
    fetch_per_layer = top_k * expert_bytes
    fetch_per_token = fetch_per_layer * n_moe_layers
    expert_visits = top_k * n_moe_layers
    gap = fetch_per_token / miss_budget

    # What the memo's "~50x" actually was: down-branch (w2) only.
    down_only_token = branch["w2"] * top_k * n_moe_layers

    print("=" * 78)
    print("K3 fetch budget — from moonshotai/Kimi-K3 shard 50 header + config")
    print("=" * 78)
    print(f"layers {n_layers} ({n_dense} dense + {n_moe_layers} MoE), "
          f"{top_k}-of-{n_experts}, latent {cfg['routed_expert_hidden_size']}, "
          f"moe_intermediate {cfg['moe_intermediate_size']}")
    print()
    print("--- per routed expert (MXFP4 packed + scales) ---")
    for w, label in (("w1", "gate branch"), ("w3", "up branch"), ("w2", "down branch")):
        print(f"  {w} ({label:<11}) {branch[w]/1e6:8.2f} MB   "
              f"{100*branch[w]/expert_bytes:5.1f}% of expert")
    print(f"  TOTAL             {expert_bytes/1e6:8.2f} MB")
    print()
    print("--- always-resident (per layer, BF16 on disk) ---")
    print(f"  {always_on_per_layer/1e9:.3f} GB/layer  ->  "
          f"{always_on_per_layer*n_layers/1e9:.1f} GB over {n_layers} layers (BF16)")
    print(f"  at 4-bit:                   "
          f"~{always_on_per_layer*n_layers/1e9*4/16:.1f} GB")
    print()
    print("--- the binding equation ---")
    print(f"  miss budget   = {LINK_GB_S} GB/s / {TARGET_TOK_S} tok/s "
          f"= {miss_budget/1e6:.1f} MB/token")
    print(f"  required fetch= {top_k} experts x {n_moe_layers} MoE layers "
          f"x {expert_bytes/1e6:.2f} MB = {fetch_per_token/1e9:.2f} GB/token")
    print(f"  GAP           = {gap:.1f}x        "
          f"({expert_visits} expert-visits/token)")
    print()
    print(f"  per expert-visit you may fetch {miss_budget/expert_visits/1e3:.1f} KB "
          f"of {expert_bytes/1e6:.2f} MB = "
          f"{100*miss_budget/expert_visits/expert_bytes:.3f}% of the expert")
    print()
    print(f"  [memo's '~50x' reconstructed: down-branch (w2) only = "
          f"{down_only_token/1e9:.2f} GB/token = {down_only_token/miss_budget:.0f}x"
          f"  <- baseline already had the branch-drop lever banked]")
    print()

    # ── Lever stack ────────────────────────────────────────────────────────
    drop_up = expert_bytes / (expert_bytes - branch["w3"])
    banked = LEVERS["locality"][0] * LEVERS["stripe_3_ports"][0] \
        * LEVERS["precision_tail"][0] * drop_up
    need_rows = gap / banked

    print("--- lever stack ---")
    print(f"  locality              1.50x   {LEVERS['locality'][1]}")
    print(f"  stripe 3 ports        {TB_PORTS:.2f}x   {LEVERS['stripe_3_ports'][1]}")
    print(f"  precision tail        1.40x   {LEVERS['precision_tail'][1]}")
    print(f"  drop up branch (w3)   {drop_up:.2f}x   UNPROVEN — and it is 1.5x, "
          f"not 2-3x: w1/w2/w3 are equal thirds")
    print(f"  ---------------------------")
    print(f"  everything but rows   {banked:.2f}x")
    print(f"  => row sparsity must supply {need_rows:.1f}x on its own")
    print(f"  => fidelity must survive at {100/need_rows:.2f}% of features "
          f"(memo assumed 20-25%)")
    print()
    no_stripe = gap / (banked / TB_PORTS)
    print(f"  without the 3-port stripe: rows must supply {no_stripe:.1f}x "
          f"= {100/no_stripe:.2f}% of features")
    print()
    memo_stack = 4.0 * 1.5 * 1.4 * TB_PORTS * 2.5  # memo's own numbers
    corrected = 4.0 * 1.5 * 1.4 * TB_PORTS * drop_up
    print(f"  memo's stack as written  {memo_stack:.0f}x vs its 50x baseline "
          f"-> closes")
    print(f"  same stack, real numbers {corrected:.0f}x vs {gap:.0f}x baseline "
          f"-> short by {gap/corrected:.1f}x")
    print("=" * 78)

    # ── Second, independent constraint: read granularity, not bandwidth ────
    # Row-granular fetch converts one large sequential read per expert into
    # many small scattered ones. Two floors bite before bandwidth does:
    # (a) a 4 KiB page is the smallest thing the stack actually moves, and
    # (b) MXFP4 stores weight_packed and weight_scale as SEPARATE tensors, so
    #     a naive feature-major layout pays two extents per feature per matrix.
    PAGE = 4096
    n_features = cfg["moe_intermediate_size"]
    frac = 1.0 / need_rows
    n_sel = frac * n_features
    # one feature's slice of w1 / w2 (packed row + its scale row)
    w1_row = branch["w1"] / n_features
    w2_row = branch["w2"] / n_features
    ideal_visit = n_sel * (w1_row + w2_row)             # w3 dropped
    # expected pages spanned by an unaligned extent of length L: 1 + L/PAGE
    pages_per_extent = 1.0 + w1_row / PAGE
    paged_visit = n_sel * 2 * pages_per_extent * PAGE
    amp = paged_visit / ideal_visit
    # IOPS: 2 matrices x n_sel extents per visit, discounted by cache locality
    iops_token = n_sel * 2 * expert_visits / LEVERS["locality"][0]
    iops_needed = iops_token * TARGET_TOK_S
    # what the link could do at 4 KiB if it were purely bandwidth-limited
    iops_link_equiv = LINK_GB_S * 1e9 * TB_PORTS / PAGE

    print("--- second constraint: read granularity (memo does not price this) ---")
    print(f"  at {100*frac:.2f}% of {n_features} features -> {n_sel:.0f} feature "
          f"rows per expert-visit, {w1_row:.0f} B each")
    print(f"  ideal bytes/visit  {ideal_visit/1e3:7.1f} KB")
    print(f"  4 KiB-paged        {paged_visit/1e6:7.2f} MB   "
          f"read amplification {amp:.2f}x")
    print(f"  => byte budget misses by {amp:.1f}x on granularity alone; "
          f"rows must reach {100*frac/amp:.2f}% of features")
    print()
    print(f"  IOPS required      {iops_needed/1e6:7.2f} M/s "
          f"({iops_needed/TB_PORTS/1e6:.2f} M/s per drive across {TB_PORTS})")
    print(f"  link at 4 KiB      {iops_link_equiv/1e6:7.2f} M/s "
          f"(bandwidth-equivalent ceiling, {TB_PORTS} ports, ignores protocol)")
    print(f"  => short by {iops_needed/iops_link_equiv:.1f}x before any USB4 "
          f"per-command overhead is counted")
    print("  NB: MXFP4 keeps weight_packed and weight_scale in separate tensors —")
    print("      a feature-major serving layout must interleave them or double this.")
    print("=" * 78)

    out = {
        "link_gb_s": LINK_GB_S,
        "target_tok_s": TARGET_TOK_S,
        "miss_budget_bytes_per_token": miss_budget,
        "expert_bytes": expert_bytes,
        "branch_bytes": branch,
        "n_moe_layers": n_moe_layers,
        "top_k": top_k,
        "n_experts": n_experts,
        "expert_visits_per_token": expert_visits,
        "required_fetch_bytes_per_token": fetch_per_token,
        "gap_x": gap,
        "bytes_allowed_per_expert_visit": miss_budget / expert_visits,
        "frac_of_expert_allowed": miss_budget / expert_visits / expert_bytes,
        "down_only_gap_x": down_only_token / miss_budget,
        "drop_up_branch_divisor": drop_up,
        "banked_divisor_excl_rows": banked,
        "row_sparsity_required_x": need_rows,
        "row_frac_required": 1.0 / need_rows,
        "row_frac_required_no_stripe": 1.0 / no_stripe,
        "always_resident_bytes_per_layer": always_on_per_layer,
        "always_resident_bf16_gb": always_on_per_layer * n_layers / 1e9,
        "corrected_stack_x": corrected,
        "shortfall_x": gap / corrected,
        "granularity": {
            "page_bytes": PAGE,
            "features_selected_per_visit": n_sel,
            "bytes_per_feature_row": w1_row,
            "ideal_bytes_per_visit": ideal_visit,
            "paged_bytes_per_visit": paged_visit,
            "read_amplification_x": amp,
            "row_frac_required_after_amp": frac / amp,
            "iops_required_per_s": iops_needed,
            "iops_link_equivalent_per_s": iops_link_equiv,
            "iops_shortfall_x": iops_needed / iops_link_equiv,
        },
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
