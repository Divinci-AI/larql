#!/usr/bin/env python3
"""Rank precision-map candidates by fidelity recovered per extra byte.

    python3 sensitivity.py <bank-dir> <baseline-label> <candidate-label>...

Every candidate is compared against the same frozen BF16 reference and
against the R0 baseline, so the question answered is not "is this good"
but "what did these extra bytes buy".
"""
import json, os, sys
import numpy as np

def load(d, label):
    p = os.path.join(d, f"compare-{label}.json")
    return json.load(open(p))

def stats(rows):
    kl = np.array([r["kl"] for r in rows])
    dn = np.array([r["dnll"] for r in rows if r["dnll"] is not None])
    flip = np.array([r["flip"] for r in rows])
    marg = np.array([r["margin"] for r in rows])
    hi = int((flip & (marg >= 0.01)).sum())
    return {
        "kl_mean": kl.mean(), "kl_p95": np.percentile(kl, 95),
        "kl_p99": np.percentile(kl, 99),
        "dnll_mean": dn.mean(), "flips": int(flip.sum()), "hi_flips": hi,
    }

def by_category(rows):
    out = {}
    for c in sorted({r["category"] for r in rows}):
        sel = [r for r in rows if r["category"] == c]
        k = np.array([r["kl"] for r in sel])
        out[c] = (k.mean(), int(sum(r["flip"] for r in sel)))
    return out

def main(d, base_label, labels):
    base = load(d, base_label)
    b = stats(base["rows"])
    bbytes = base["container"].get("compiled_bytes")
    print(f"\nQ-BANK-1 sensitivity — baseline {base_label}")
    print(f"  {base['positions'] if 'positions' in base else len(base['rows']):,} positions")
    print(f"  KL mean {b['kl_mean']:.4f}  p95 {b['kl_p95']:.4f}  p99 {b['kl_p99']:.4f}"
          f"   flips {b['flips']} (high-margin {b['hi_flips']})")
    print()
    hdr = (f"  {'candidate':<18}{'+MiB':>9}{'KL mean':>10}{'KL p99':>9}"
           f"{'flips':>7}{'hi-flip':>9}{'hi recovered':>14}{'per +MiB':>10}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    rows_out = []
    for lab in labels:
        try:
            c = load(d, lab)
        except FileNotFoundError:
            print(f"  {lab:<18}  (not run yet)")
            continue
        s = stats(c["rows"])
        extra = max(0, c.get("payload_bytes", 0) - base.get("payload_bytes", 0)) / 2**20
        rec = b["hi_flips"] - s["hi_flips"]
        per = (rec / extra) if extra > 0 else float("nan")
        rows_out.append((lab, extra, s, rec, per))
        print(f"  {lab:<18}{extra:>9.0f}{s['kl_mean']:>10.4f}{s['kl_p99']:>9.4f}"
              f"{s['flips']:>7}{s['hi_flips']:>9}{rec:>14}{per:>10.3f}")
    if rows_out:
        print("\n  ranked by high-margin flips recovered per extra MiB:")
        for lab, extra, s, rec, per in sorted(rows_out, key=lambda r: -(r[4] if r[4] == r[4] else -1)):
            print(f"    {lab:<18} {per:>8.3f}   ({rec} flips for {extra:.0f} MiB)")
    print("\n  per category (KL mean / flips)")
    cats = by_category(base["rows"])
    names = list(cats)
    print("    " + "category".ljust(13) + base_label.rjust(16)
          + "".join(l.rjust(16) for l in labels if os.path.exists(os.path.join(d, f"compare-{l}.json"))))
    for c in names:
        line = f"    {c:<13}" + f"{cats[c][0]:>9.4f}/{cats[c][1]:<6}"
        for lab in labels:
            p = os.path.join(d, f"compare-{lab}.json")
            if not os.path.exists(p):
                continue
            cc = by_category(json.load(open(p))["rows"])[c]
            line += f"{cc[0]:>9.4f}/{cc[1]:<6}"
        print(line)
    print("\n  A candidate that improves aggregate KL by helping one category")
    print("  while another gets worse is not obviously preferable — which is")
    print("  why the category split stays in the report.")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
