# SENSITIVITY-1B′ — absolute activation-weighted consequence (pre-registered)

Written before the disjoint calibration set is captured or any score
computed. One shot; no revision afterwards.

## Why this rung rather than 1C

1B-a did not fail for lack of information. It failed because dividing by
`‖XW‖²` reintroduced 1A's bias — both rewarded operands for being small.
Escalating to curvature now would skip the cheaper hypothesis the failure
itself motivated: that activation weighting is sufficient once the
normalisation stops cancelling it.

## Primary score — pre-registered, one form only

For region `R` with `ΔW = W − Q(W)` and per-feature second moments
`d_j = E[x_j²]` from the **disjoint** calibration set:

```text
raw(R)  = Σ_{t∈R} Σ_j  d_j · ‖ΔW_t[:,j]‖²        absolute consequence
S(R)    = raw(R) / raw(model)                     fraction of the model's total
return(R) = S(R) / extra_MiB(R)                   fraction removed per MiB
```

The only normalisation is by a **model-level total**, never by the
operand's own magnitude. That is the whole correction: it makes scores
comparable across models without re-rewarding small operands.

No other variant is computed. There is nothing to choose between
afterwards.

## Calibration — disjoint

`calibration-disjoint.jsonl`: prompts written for this test and **absent
from Q-BANK-1**. The 1B-a capture is discarded, not reused. A proxy that
scored well on bank-derived activations could be echoing the distribution
that produced the verdicts; this removes that route entirely.

## The bar — all conditions, no rank correlation rescue

Spearman may be reported. **It cannot rescue a failed condition.**

**Granite (a model with a sharp knee)**

1. `late5-ffn` ranks above all three negatives on `return`.
2. `v_proj`, `k_proj`, `down_proj` all fall in the bottom half.
3. The knee shape survives: `return(late5-ffn) > return(late10-ffn)` and
   `> return(late15-ffn)`.

**Glimmer (a model with no useful knee)**

4. No coarse region has material predicted return:
   `max return(Glimmer) < 0.10 × max return(Granite)`.

Condition 4 is the one that matters most, and it is why Glimmer is here.
A predictor that learns "late FFN matters" from Granite and then
recommends late-FFN on Glimmer is **wrong**, however good its Granite
ranking looks — Granite's late5-FFN removes 72% of p99 for +431 MiB while
Glimmer's equivalent buys 2% for +2,920 MiB. The truth ratio is roughly
240×, so a 10× bar is generous and still discriminating.

## If 1B′ passes

It may propose **one** unseen precision map, which Q-BANK then judges. The
screen proposes; Q-BANK promotes. Nothing else changes.

## If 1B′ fails

1C is then earned, and the conclusion is stronger than "try something
fancier": absolute local consequence is insufficient, so downstream
curvature or context is required.

## The STOP decision this makes possible

Glimmer's flat surface means an automatic compiler must be able to
conclude:

```text
R0 already on the Pareto knee
additional protection not worth its bytes
emit R0
```

Optimisers that assume every model benefits from a precision map will
spend bytes on Glimmer for nothing. Condition 4 is the test of whether a
screen can support that decision.
