# ENCODER-R4 — is the damage the format, or the rounding? (pre-registered)

Written before any calibration-aware encoder exists. One shot; no revision
afterwards.

## Question

> **Can calibration-aware encoding improve Granite's fidelity substantially
> at exactly the same NVFP4 storage cost — and if it does, does the
> previously measured late-FFN knee survive?**

Every result in the SENSITIVITY programme measured the damage that
`nvfp4-nearest-v1` does. None of them established that the damage is a
property of the *format*. R4 is therefore upstream of the whole sensitivity
question: it asks whether we have been measuring the model's tolerance, or
one encoder's choices.

## Why this is upstream

Q-BANK's Granite verdicts are true statements about
`codec nvfp4/rev1 + encoder nvfp4-nearest-v1`. The late-FFN knee, the
unsafe R0 default, and the four sensitivity falsifications all inherit that
qualifier.

If a better encoder recovers most of R0's damage at identical bytes, then
part of what looked like *late-layer sensitivity* may have been *nearest
rounding making particularly bad choices in those layers*. That would not
invalidate any measurement — it would change what kind of phenomenon they
measured, and it would change the baseline every later rung is judged
against.

## The one variable

```text
                   CONTROL                  CANDIDATE
codec ABI          nvfp4/rev1               nvfp4/rev1        identical
encoder recipe     nvfp4-nearest-v1         nvfp4-gptq-v1     <-- THE VARIABLE
precision map      R0 (uniform)             R0 (uniform)      identical
runtime / kernels  unchanged                unchanged         identical
physical bytes     2,283,690,080            must be EQUAL     identical
evaluation         Q-BANK-1                 Q-BANK-1          identical
```

**Byte equality is asserted, not assumed.** The candidate pack must report
exactly `2,283,690,080` payload bytes. If it does not, the arm is void —
there must be no "but it used more bits" explanation available for any
result, in either direction.

This is the experiment the codec/recipe separation was built for. The
loader neither knows nor cares which recipe produced the bytes; a new
recipe emits the same ABI and different numbers, and a recipe mismatch is
**not** a refusal.

## Frozen control — the banked `nvfp4-nearest-v1` numbers

1,622 teacher-forced positions, `granite-4.1-3b.vindex3` (model
`c0650403…`, BF16 payload `374562b3…`, `head.output_multiplier 0.1`).

```text
R0-recheck      2,283,690,080 B   2.127 GiB
  KL   mean 0.2778   p50 0.0607   p95 1.5247   p99 4.6224   max 6.7675
  dNLL 0.1184        flips 267    high-margin 189

late5-ffn       2,735,888,420 B   2.548 GiB
  KL   mean 0.1283   p50 0.0510   p95 0.4468   p99 1.2826   max 4.2759
  dNLL -0.0058       flips 234    high-margin 167

late10-ffn      3,188,086,760 B   2.969 GiB
  KL   mean 0.1193   p50 0.0461   p95 0.4017   p99 1.2628   max 4.5325
  dNLL -0.0076       flips 233    high-margin 165
```

## Calibration — reuse the frozen disjoint bank

R4 is the first **calibrated encoder** in this programme, which introduces
a leakage route that did not previously exist: an encoder fitted on the
evaluation prompts would look excellent for a reason that has nothing to do
with the format.

```text
encoder calibration   the existing disjoint 12-prompt bank
  prompt text digest  f628ce4739f81e9e5b8dd42b8d83d907112718e61ce711443846d379fa5fb34f
  token bank digest   df0e3644aba068c4687baefe178399d1f0a62cb7509377211499927a55da96c3
  disjointness        verified by content against prompts.json:
                      0 id overlap, 0 text overlap, 0 near-duplicates

evaluation            Q-BANK-1 only, all 69 prompts, 1,622 positions
```

No new calibration set is minted. The pack's provenance must record which
calibration produced it — as metadata beside the recipe, not crammed into
the recipe string — so any pack can answer *which activations chose these
values*.

## Primary arm: uniform R0, and only R0

The first arm is `R0-GPTQ` against `R0-nearest`. Nothing mixed, nothing
protected.

That isolates the question R4 exists to answer: **how much of Granite R0's
damage is avoidable at exactly the same representation size?** Starting
with a mixed profile would confound the encoder change with a precision-map
change, which is R5's question and not this one.

## Metrics — the whole response, pre-committed

Reported together, for both arms, whatever the direction of each:

```text
KL mean, p50, p95, p99, max
dNLL mean
top-1 flips, and flips at BF16 margin >= 0.01
per-category breakdown (all seven Q-BANK regimes)
payload bytes (equality assertion)
```

**No quality threshold is invented here.** There is no pre-set "GPTQ must
beat X". The reason is that any threshold would be either arbitrary or
fitted, and the interesting result is the shape of the change, not whether
it clears a line someone drew.

**The full response is reported, not the metric GPTQ improves most.** If
GPTQ improves the mean and worsens p99, that is the finding and it is
stated as such — a tail that gets worse while the body improves is a
different deployment story from a uniform gain, and the PR body reports
both or neither.

## Negative control — per-category response

GPTQ is fitted to twelve prompts. The main control is already structural:
those twelve are disjoint from the 69 it is judged on, verified by content.

Additionally, R4 reports Q-BANK **by category**. If the improvement
concentrates in regimes resembling the calibration prompts and degrades
elsewhere, the per-category table shows it. No further held-out set is
minted unless that table looks suspicious — inventing one now would be
guarding against a failure mode nothing has yet suggested.

## The second question — only if the first arm wins

If `R0-GPTQ` materially improves on `R0-nearest`, re-run a *small* part of
the surface under the new encoder:

```text
R0-GPTQ           late5-ffn-GPTQ           late10-ffn-GPTQ
```

and ask: **does the knee survive a better encoder?**

Two outcomes, both informative:

```text
A.  R0-GPTQ improves, and late5-ffn-GPTQ is still sharply better than it
    -> the sensitivity structure is real and encoder-independent

B.  R0-GPTQ lands near nearest's late5-ffn, and protecting late FFN then
    buys little
    -> much of the "late-layer sensitivity" was nearest rounding making
       bad choices in those layers
```

Outcome B would reinterpret the sensitivity programme without invalidating
it: every banked measurement remains a true statement about
`nvfp4-nearest-v1`. What would change is the *kind* of phenomenon they
measured, and the baseline R5 and any future selection rung are judged
against.

This second arm is **not** run if the first shows no material improvement.

## Explicitly out of scope

- **R5 — richer precision vocabulary.** `Q8`, `Q4`, `Mxfp4` and `Bf16` are
  all already executable (`WeightFormat`), so mixing an intermediate
  precision in here would be cheap and would ruin the experiment. R4 asks
  *better quality at zero additional cost*; R5 asks *is extra precision
  worth its bytes*. Different questions, and R5 becomes more interesting
  afterwards because its baseline stops being an unnecessarily weak
  nearest-rounded one.
- **Automatic selection.** No sensitivity screen, no optimizer, no map
  proposal. The SENSITIVITY ladder stays paused.
- **Other models.** Granite only. Glimmer and an MoE are R6.

## Implementation note, established before committing

GPTQ's objective needs the **full input second-moment matrix**
`H = E[x xᵀ]` per site, not the per-feature diagonal `d_j = E[x_j²]` the
SENSITIVITY capture banked. Those are different objects and the existing
moments cannot be reused for this.

The memory cost is not uniform across sites, and `down_proj` is the
expensive one:

```text
attention / FFN input sites   2560 x 2560   ~26 MB per site
down_proj input site          8192 x 8192   ~268 MB per layer
```

Across 40 layers that is roughly 10.7 GB for the `down_proj` Hessians
alone if all are held at once. Any implementation must stream or
block-process them rather than materialise the set — and the fact that the
most expensive Hessian belongs to the operand the sensitivity programme
found most interesting is worth noticing before it becomes a surprise.

## Sequence

1. Pre-register (this document).
2. Implement `nvfp4-gptq-v1` as a recipe against the unchanged
   `nvfp4/rev1` ABI. Record the calibration identity in pack provenance.
3. Assert byte equality with the nearest-encoded R0 pack.
4. Compile `R0-GPTQ`; verify it executes and reproduces its own reference.
5. Q-BANK it once, against the frozen control above.
6. Report the whole response, per-category included.
7. Only if it wins materially: `late5-ffn-GPTQ`, `late10-ffn-GPTQ`, and the
   knee-survival question.

Steps 5 and 7 happen exactly once each.
