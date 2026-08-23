# SENSITIVITY-1B — activation-weighted local error (pre-registered)

Written before any activation is captured or any score computed. 1A failed
because relative weight error is flat across projections (0.00893–0.00904),
so a per-byte ranking degenerated into "protect the smallest tensor". The
missing factor is not how far the weights move but how much the model's
activations amplify that movement.

## Primary score — pre-registered

For tensor `W` with input activation `X`:

```text
E_local(W) = || XW - X·Q(W) ||²  /  || XW ||²
```

**This is the primary. The decision is recorded here so it cannot be
chosen after seeing which normalisation fits Granite.**

A secondary variant is computed from the same captured activations and
reported alongside, but it is *not* eligible to be promoted to primary
after the fact:

```text
E_residual(W) = || XW - X·Q(W) ||²  /  || residual ||²
```

The difference matters: the primary asks "how wrong is this operand's own
output", the secondary asks "how wrong is it relative to the stream it
joins". A tensor whose output is small compared to the residual can be
badly wrong in its own terms and barely move the model.

## Method

1. **Calibration subset**: a fixed 12 prompts drawn from Q-BANK-1, two per
   category (code, prose, arithmetic, structured, factual, longform), named
   in `calibration.json`. Deliberately not the whole bank — a screen that
   needed the whole bank would not be a screen.
2. **Capture once, from BF16.** Per tensor input site, accumulate the
   per-feature second moment `d_j = E[x_j²]` over the calibration
   positions. That is a vector, not a matrix, and there are three distinct
   sites per layer (attention input, attention output, FFN intermediate).
3. **Reuse those exact moments for every candidate.** Nothing is
   re-captured per candidate; that is what keeps the screen cheap.
4. **Score each tensor** using the diagonal approximation
   `||XΔW||² ≈ Σ_j d_j · ||ΔW_{j,:}||²`, with `ΔW = W − Q(W)`, normalised
   as above.
5. **Aggregate a candidate region** as the sum over the tensors it
   protects, and a per-MiB variant, exactly as 1A did.

## The bar — unchanged

Both halves, judged on the primary score:

1. identifies late-FFN as highest-return;
2. rejects `v_proj`, `k_proj`, `down_proj` as low-value.

No relaxation if Spearman looks encouraging. An aggregate correlation with
the negatives still ranked highly is a failure, as it was for 1A.

## If 1B fails

Escalate to 1C (curvature / Hessian-aware), do not add heuristics. A local
output-error score that still ranks V/K/down highly would say local
consequence is insufficient and second-order structure is required — which
is a result, and the earned justification for 1C's cost.

## Note for K3

1A's failure already carries a scaling consequence: precision cannot be
assigned intelligently from weight statistics alone, so any 2.8T-scale
compiler needs representative activation traffic through the model. 1B is
the cheapest form of that, and its per-feature moments are a vector per
site rather than anything that scales with expert count.
