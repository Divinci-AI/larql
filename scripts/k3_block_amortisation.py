#!/usr/bin/env python3
"""DEC-9.2 — speculative block economics for K3, with work-width separated from
commit-width.

WITHDRAWN TWICE, both before any compute:

  v1: "L ~= 1.6" scaled DEC-8.7a's 12.3 tok/s constant by block length. That
      constant is the ROUTED-ZEROED ceiling (its whole purpose under R4 is to
      set routed bytes to zero), so carrying it here deleted the routed term
      from the one setting where it returns AND grows.

  v2: the composed budget (D + rho*R*u(L))/L still used ONE variable for both
      the union and the amortisation, which silently assumes perfect
      acceptance. A drafter proposing T positions of which A commit makes the
      target evaluate T positions and bank A tokens. Costs key on T; throughput
      keys on A. Standing rule R5.

The honest condition is therefore

    D + rho_r * R * u(T)  <=  eta * B20 * A(T)

with three independent quantities: T (proposal width, a DESIGN KNOB), A(T)
(expected committed tokens, a drafter property), u(T) (routed union across all
T evaluated positions, a model property).

The useful derived statistic is the routed retention each drafter can tolerate:

    rho_max(T) = (eta * B20 * A(T) - D) / (R * u(T))

Because u(T) grows near-linearly at K3's 1.79% activation while A(T) saturates
with drafter quality, rho_max(T) generally has an INTERIOR OPTIMUM. Nobody
should evaluate a drafter only at its shipped proposal width.

ROUTED RETENTION IS NOT 1.87. That number was DEC-8.6's R4 zero-out ratio
(total/dense), never a routed-side result. The banked routed-side floor is
DEC-8.0's equal-thirds result: w3 folds to per-feature scalars, w1 and w2 must
still be read, so retention <= 0.667 (a 1.5x routed reduction). Anything below
that is row sparsity, owned by DEC-8.1 and unset until it reports.
"""

import argparse
import json
import math
from pathlib import Path

from k3_fetch_budget import DEFAULT_CACHE

BW_GPU_GB_S = 367.0
TARGET_TOK_S = 20.0

# Banked routed-side retention ceiling: DEC-8.0 measured w1/w2/w3 at 5,849,088
# bytes each, so folding w3 away leaves two thirds. Everything below this is
# row sparsity and belongs to DEC-8.1.
ROUTED_RETENTION_BANKED = 2.0 / 3.0

# Across-batch locality from the DEC-0 routed arm. Used ONLY as an optimistic
# bound on the ALONG-SEQUENCE axis, which is a different quantity (R2).
ACROSS_BATCH_LOCALITY = 0.56

# Vision is EXCLUDED from D: the DEC-8.6 ledger sums language_model layers plus
# text embeddings only. vision_tower + mm_projector are ~2 GB BF16 by residual
# against the 1.561 TB repo, i.e. under 2% of D at 4-bit. Named, not assumed.
VISION_INCLUDED_IN_DENSE = False


def union(n_experts, top_k, T):
    """Expected distinct experts across T evaluated positions."""
    return n_experts * (1.0 - (1.0 - top_k / n_experts) ** T)


def accepted_from_alpha(alpha, T, bonus=True):
    """Standard speculative-decoding expectation over T drafted positions."""
    if abs(alpha - 1.0) < 1e-12:
        return T + (1 if bonus else 0)
    total = (1.0 - alpha ** (T + 1)) / (1.0 - alpha)
    return total if bonus else total - 1.0


def fit_alpha(T, A, bonus=True):
    """Recover the per-token acceptance probability from one observed (T, A)."""
    lo, hi = 1e-6, 1.0 - 1e-9
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if accepted_from_alpha(mid, T, bonus) < A:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--proposal-width", type=int, default=7,
                    help="T: positions the target evaluates per pass")
    ap.add_argument("--mean-accepted", type=float, default=3.85,
                    help="A(T): committed tokens per pass, at the given width")
    ap.add_argument("--include-bonus-token", action="store_true", default=True)
    ap.add_argument("--union-equivalents", type=float, default=None,
                    help="override u(T) with a measured value (single-token "
                         "equivalents); omit to use the uniform/locality band")
    ap.add_argument("--routed-retention", type=float, default=ROUTED_RETENTION_BANKED,
                    help=f"rho_r, fraction of routed bytes still read "
                         f"(banked floor {ROUTED_RETENTION_BANKED:.3f} = up-fold only)")
    ap.add_argument("--dequant-efficiency", type=float, default=1.0)
    ap.add_argument("--draft-time-ms", type=float, default=0.0)
    ap.add_argument("--state-time-ms", type=float, default=0.0)
    ap.add_argument("--bandwidth-gb-s", type=float, default=BW_GPU_GB_S)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    T, A = args.proposal_width, args.mean_accepted
    if abs(T - A) < 1e-9:
        print("!! WARNING: proposal width == mean accepted. That is the R5 "
              "error (perfect acceptance assumed). Costs key on T, throughput "
              "on A. Supply a real A(T) or state the assumption explicitly.\n")

    front = json.loads((args.cache / "k3_dense_frontier.json").read_text())
    cfg = json.loads((args.cache / "config.json").read_text())["text_config"]
    E, k = cfg["num_experts"], cfg["num_experts_per_token"]

    D = front["dense_bytes_4bit"]                 # per forward pass
    R = front["routed_bytes_4bit"]                # per position, RAW (no lever)
    eta = args.dequant_efficiency
    bw = args.bandwidth_gb_s * 1e9
    B20 = bw / TARGET_TOK_S
    rho = args.routed_retention

    print("=" * 80)
    print("DEC-9.2 — speculative block economics, work-width separated from commit")
    print("=" * 80)
    print(f"D (dense, per PASS)      {D/1e9:6.2f} GB   "
          f"[vision {'included' if VISION_INCLUDED_IN_DENSE else 'EXCLUDED'}]")
    print(f"R (routed, per POSITION) {R/1e9:6.2f} GB   raw, before any lever")
    print(f"rho_r (routed retention) {rho:6.3f}      "
          f"banked floor {ROUTED_RETENTION_BANKED:.3f} = up-fold only; "
          f"below that is DEC-8.1's row_factor (UNSET)")
    print(f"eta {eta:.2f}, B20 {B20/1e9:.2f} GB/token, "
          f"effective budget {eta*B20/1e9:.2f} GB/token")
    print()

    # R4 in speculative clothing: zero the routed term and see if dense refuses.
    dense_only_ok = eta * B20 * A > D
    print("--- R4 check: zero the routed term entirely ---")
    print(f"  eta*B20*A(T) = {eta*B20*A/1e9:.2f} GB  vs  D = {D/1e9:.2f} GB  "
          f"-> {'survives' if dense_only_ok else 'DIES ON DENSE ALONE'}")
    if not dense_only_ok:
        print("  >> no routed optimisation can rescue this arm. Stop here.")
    print()

    alpha = fit_alpha(T, A, args.include_bonus_token)
    print(f"--- proposal-width sweep (alpha={alpha:.4f} fitted from T={T}, A={A}) ---")
    print(f"{'T':>3} {'A(T)':>7} {'u_unif':>8} {'rho_max':>9} {'G_r':>7}   "
          f"{'u_loc':>7} {'rho_max':>9} {'G_r':>7}")
    rows, best = [], None
    for t in range(1, 13):
        a = accepted_from_alpha(alpha, t, args.include_bonus_token)
        uu = union(E, k, t) / k
        ul = max(1.0, ACROSS_BATCH_LOCALITY * uu)
        if args.union_equivalents is not None and t == T:
            uu = ul = args.union_equivalents
        head = eta * B20 * a - D
        ru = head / (R * uu) if head > 0 else 0.0
        rl = head / (R * ul) if head > 0 else 0.0
        rows.append({"T": t, "A": a, "u_uniform": uu, "u_locality": ul,
                     "rho_max_uniform": ru, "rho_max_locality": rl,
                     "G_r_min_uniform": (1 / ru) if ru > 0 else None,
                     "G_r_min_locality": (1 / rl) if rl > 0 else None})
        gu = f"{1/ru:7.2f}" if ru > 0 else "    inf"
        gl = f"{1/rl:7.2f}" if rl > 0 else "    inf"
        print(f"{t:>3} {a:>7.2f} {uu:>8.2f} {ru:>9.4f} {gu}   "
              f"{ul:>7.2f} {rl:>9.4f} {gl}")
        if best is None or rl > best["rho_max_locality"]:
            best = rows[-1]
    print()
    print(f"  interior optimum on the locality arm: T = {best['T']} "
          f"(rho_max {best['rho_max_locality']:.4f}, "
          f"needs G_r >= {1/best['rho_max_locality']:.2f}x)")
    print(f"  >> a drafter's SHIPPED width is not its best width. Sweep it.")
    print(f"  !! PROVISIONAL: alpha={alpha:.4f} is fitted from ONE "
          f"card-conditioned point (T={T}, A={A}) under a constant-alpha "
          f"geometric model.")
    print(f"     Early-token acceptance may be much stronger or weaker than "
          f"geometric. These optima are DESIGN POINTS, not performance bars — "
          f"they harden only when")
    print(f"     DEC-9.0's swept acceptance DISTRIBUTION replaces the fit. "
          f"Nothing downstream may cite T={best['T']} as measured.")
    print()

    # What the banked retention leaves for rows.
    print(f"--- what DEC-8.1 must supply, given the banked floor "
          f"rho_r <= {ROUTED_RETENTION_BANKED:.3f} ---")
    for label, key in (("uniform", "rho_max_uniform"), ("locality", "rho_max_locality")):
        r = next(x for x in rows if x["T"] == T)[key]
        if r <= 0:
            print(f"  {label:<9} T={T}: dense alone refuses")
            continue
        need = ROUTED_RETENTION_BANKED / r if r > 0 else float("inf")
        print(f"  {label:<9} T={T}: rho_max {r:.4f} -> total G_r {1/r:.2f}x; "
              f"up-fold gives 1.50x, so rows must supply {need:.2f}x "
              f"(retain {100*r/ROUTED_RETENTION_BANKED:.0f}% of remaining rows)")
    print("=" * 80)

    out = {"premises": {"D_bytes": D, "R_bytes_per_position": R,
                        "routed_retention": rho,
                        "routed_retention_banked_floor": ROUTED_RETENTION_BANKED,
                        "dequant_efficiency": eta, "B20_bytes": B20,
                        "vision_included_in_dense": VISION_INCLUDED_IN_DENSE,
                        "alpha_fitted": alpha,
                        "observed_pair": {"T": T, "A": A}},
           "r4_dense_only_survives": dense_only_ok,
           "sweep": rows, "interior_optimum_locality": best,
           "PROVISIONAL": ("alpha fitted from ONE card-conditioned point under a "
                           "constant-alpha geometric model; optima are design "
                           "points, not performance bars, until DEC-9.0's swept "
                           "acceptance distribution replaces the fit"),
           "assumes_physical_reuse_unmeasured": (
               "this model banks d(T)=1 and r(T)=u(T): every dense tensor read "
               "once per block, every unique expert loaded once. Neither is "
               "established — see R6 and DEC-9.1. A token loop gives d=r=T."),
           "withdrawn": ["v1: L~=1.6, scaled the routed-zeroed constant",
                         "v2: single L for both union and amortisation (R5)"]}
    (args.out or (args.cache / "k3_block_amortisation.json")).write_text(
        json.dumps(out, indent=2))
    print(f"wrote {args.out or (args.cache / 'k3_block_amortisation.json')}")


if __name__ == "__main__":
    main()
