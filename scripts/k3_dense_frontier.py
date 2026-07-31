#!/usr/bin/env python3
"""DEC-8.7a — the dense-path throughput frontier for K3 single-token decode.

The target generator. Every later fidelity bar should cite a row of this table
rather than embed one assumed compression result in a headline.

The physics is fixed: on a bandwidth-bound decode step, tok/s = BW / (bytes a
token reads). Splitting that read into the part expert-side levers can reduce
and the part they cannot turns "what tok/s is possible" into "what effective
precision must the dense path average, given whatever the routed side earns."

Two conventions for "effective bits", because they differ by up to 0.6 bits at
the aggressive end and the distinction matters when comparing against a real
codec:

  payload      — mantissa/codeword width only, excluding block-scale metadata.
                 Compare against "4-bit" as a codec name.
  all-in       — total bytes on the wire per weight, including scales.
                 MXFP4's all-in width is 4.25 bits, not 4.

Reads the ledger DEC-8.6 computed from real safetensors headers. No inference.
"""

import argparse
import json
from pathlib import Path

from k3_fetch_budget import DEFAULT_CACHE

# ── Premises ───────────────────────────────────────────────────────────────
BW_GPU_GB_S = 367.0          # measured attainable, project_memory_bandwidth_roofline
BW_CPU_GB_S = 127.0
SCALE_BITS_PER_WEIGHT = 8 / 32     # MXFP4: one 8-bit block scale per 32 weights
MXFP4_PAYLOAD_BITS = 4.0

# The routed side's stated maximum, from DEC-8.6: experts are 46.4% of the read,
# so driving them to zero is a 1.87x reduction on the total and no more.
ROUTED_MAX_REDUCTION_X = 1.87

TARGETS = (6.6, 8.0, 10.0, 12.0, 14.0, 20.0)


def bits(byte_budget, n_params, convention):
    """Effective bits per weight implied by a byte budget."""
    if n_params <= 0:
        return float("nan")
    all_in = byte_budget * 8.0 / n_params
    return all_in if convention == "all-in" else all_in - SCALE_BITS_PER_WEIGHT


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--bandwidth-gb-s", type=float, default=BW_GPU_GB_S)
    ap.add_argument("--dequant-efficiency", type=float, default=1.0,
                    help="fraction of attainable bandwidth actually realised once "
                         "dequantisation cost is counted (1.0 = free)")
    ap.add_argument("--routed-reduction", type=float, default=ROUTED_MAX_REDUCTION_X,
                    help="reduction the routed side earns (1.0 = none, "
                         f"{ROUTED_MAX_REDUCTION_X} = its stated maximum)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    led = json.loads((args.cache / "k3_slice_touch.json").read_text())
    dense_b = led["touch_bytes"]["dense"]
    routed_b = led["touch_bytes"]["routed"]
    dense_p = led["params"]["dense_total"]
    routed_p = led["params"]["routed_activated"]

    bw = args.bandwidth_gb_s * 1e9 * args.dequant_efficiency
    routed_floor = routed_b / args.routed_reduction

    print("=" * 82)
    print("DEC-8.7a — dense-path throughput frontier, K3 single-token decode")
    print("=" * 82)
    print(f"bandwidth {args.bandwidth_gb_s:.0f} GB/s x {args.dequant_efficiency:.2f} "
          f"dequant efficiency = {bw/1e9:.1f} GB/s effective")
    print(f"dense   {dense_p/1e9:6.2f} B params, {dense_b/1e9:6.2f} GB/token at "
          f"{bits(dense_b, dense_p, 'all-in'):.2f} bits all-in")
    print(f"routed  {routed_p/1e9:6.2f} B params, {routed_b/1e9:6.2f} GB/token; "
          f"at {args.routed_reduction:.2f}x reduction -> {routed_floor/1e9:.2f} GB/token")
    print()
    print(f"{'target':>8} {'budget':>9} {'dense budget':>13} "
          f"{'payload bits':>13} {'all-in bits':>12}  verdict")
    rows = []
    for t in TARGETS:
        budget = bw / t
        dense_budget = budget - routed_floor
        pay = bits(dense_budget, dense_p, "payload")
        allin = bits(dense_budget, dense_p, "all-in")
        if dense_budget <= 0:
            verdict = "IMPOSSIBLE — routed floor alone exceeds the budget"
        elif pay >= MXFP4_PAYLOAD_BITS:
            verdict = "existing 4-bit fits"
        elif pay >= 3.0:
            verdict = "~3-bit dense — credible research target"
        elif pay >= 2.0:
            verdict = "~2-bit dense — serious stretch"
        elif pay >= 1.0:
            verdict = "sub-2-bit — retrain or architecture change"
        else:
            verdict = "NOT a quantisation problem — new-model programme"
        rows.append({"target_tok_s": t, "budget_bytes": budget,
                     "dense_budget_bytes": dense_budget,
                     "payload_bits": pay, "all_in_bits": allin,
                     "verdict": verdict})
        print(f"{t:>7.1f} {budget/1e9:>8.2f}G {dense_budget/1e9:>12.2f}G "
              f"{pay:>13.2f} {allin:>12.2f}  {verdict}")
    print()

    # The R4 check, run explicitly: zero the routed bytes entirely.
    print("--- R4 zero-out check: routed traffic ELIMINATED ---")
    for t in (12.0, 14.0, 20.0):
        b = bw / t
        print(f"  {t:>4.0f} tok/s: dense alone must reach "
              f"{bits(b, dense_p, 'payload'):.2f} payload bits "
              f"({bits(b, dense_p, 'all-in'):.2f} all-in)")
    ceiling = bw / dense_b
    print(f"  >> with dense at its current 4-bit, the ceiling is "
          f"{ceiling:.1f} tok/s no matter what the routed side does.")
    print(f"  >> therefore no expert-side experiment can decide any target "
          f"above {ceiling:.1f} tok/s.")
    print("=" * 82)

    out = {"premises": {"bandwidth_gb_s": args.bandwidth_gb_s,
                        "dequant_efficiency": args.dequant_efficiency,
                        "effective_bw_gb_s": bw / 1e9,
                        "routed_reduction_x": args.routed_reduction,
                        "scale_bits_per_weight": SCALE_BITS_PER_WEIGHT},
           "dense_params": dense_p, "routed_params": routed_p,
           "dense_bytes_4bit": dense_b, "routed_bytes_4bit": routed_b,
           "routed_floor_bytes": routed_floor,
           "expert_side_ceiling_tok_s": ceiling,
           "frontier": rows}
    (args.out or (args.cache / "k3_dense_frontier.json")).write_text(
        json.dumps(out, indent=2))
    print(f"wrote {args.out or (args.cache / 'k3_dense_frontier.json')}")


if __name__ == "__main__":
    main()
