# SENSITIVITY-1 — can a cheap local signal predict the expensive verdict?

Q-BANK is the promotion gate and is too expensive to be the search. Screening
every projection × depth candidate through 1,622 teacher-forced positions
does not scale to R2's combinatorics, let alone K3's.

So the question is narrow and falsifiable:

> Can a cheap **local** score rank precision-map candidates the way
> Q-BANK's **global** verdict ranks them?

## The bar, fixed before scoring

Both halves, or it is not a predictor:

1. **Identifies late-FFN as the highest-return region.**
2. **Rejects `v_proj`, `k_proj` and `down_proj` as low-value.**

An aggregate correlation passing while (2) fails is a *failure*. It would
mean the proxy has learned "protecting more bytes helps", which is true,
useless, and exactly what the frozen negatives exist to catch:

| candidate | +MiB | Q-BANK verdict |
|---|---|---|
| `v_proj` | 72 | KL 0.278 → 0.264. Near-zero benefit, and the canonical ecosystem move. |
| `k_proj` | 72 | Flip difference +4, bootstrap 95% CI [−8, +16]. Indistinguishable from zero. |
| `down_proj` | 1,150 | p95 got **worse** (1.525 → 1.592) for over a gigabyte. |

A ranking driven by byte cost alone would place `down_proj` third of
fifteen. Q-BANK places it near the bottom.

## Validation set

Fifteen Granite candidates with frozen end-to-end outcomes, in
`granite-4.1-3b-sweep.json` — R0, four single-projection probes, two
class probes, two depth probes, five role×depth intersections, and a
determinism control. The knee (`late5-ffn`, +7.93 p99/GiB, with
`late10-ffn` at +0.05) is the shape the screen has to reproduce.

## Method

The screen scores a *region* by what quantising it does locally, with no
forward pass over the bank:

```text
for each candidate region R:
    e(R) = sum over tensors t in R of
             ||W_t - dequant(quant(W_t))||^2 / ||W_t||^2   (relative error)
           weighted by t's share of the region's bytes
```

This is the oQ-style normalised local error, computed from weights alone.
It is deliberately *not* a forward-pass metric: if a local signal suffices,
screening costs one pass over the weights rather than one per candidate.

## Order

1. Glimmer coarse surface — is its low R0 damage flat or concentrated?
2. Compute local scores over Granite's fifteen candidate regions.
3. Rank by the cheap score alone.
4. Compare against the frozen Q-BANK ranking, and check both halves of the bar.
5. Only if it passes, let it propose a map nobody has tested.
6. Q-BANK remains the promotion gate. The screen proposes; it never promotes.

No optimizer and no budget solver until step 4 passes. A search primitive
that cannot predict the verdict is not a search primitive.
