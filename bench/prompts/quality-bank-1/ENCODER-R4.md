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

## Calibration — three partitions, and a sufficiency gate

R4 is the first **calibrated encoder** in this programme, which introduces
a leakage route that did not previously exist: an encoder fitted on the
evaluation prompts would look excellent for a reason that has nothing to do
with the format.

**Design correction, made before any encoder or result exists.** An earlier
draft reused the frozen 12-prompt SENSITIVITY bank (458 positions). That
bank was sized for a *per-feature diagonal* `d_j = E[x_j²]`, where 458
samples are ample for 2,560 independent scalars. GPTQ estimates a full
`d×d` correlation — a different object with different sample requirements,
and one this programme has no evidence has converged at 458. Reusing the
bank is right for **disjointness** and unjustified for **sufficiency**.

The correction is *not* "make `n ≥ d`" — see the rank note below. It is to
determine the calibration size from evidence, using no Q-BANK observation.

```text
CALIBRATION POOL     newly written, disjoint from Q-BANK by content check.
                     Deterministic nested prefixes so larger strictly
                     contains smaller:
                        N0 =    458   (the existing SENSITIVITY bank)
                        N1 =  2,048
                        N2 =  8,192
                        N3 = 32,768
                        N4 = 65,536

CALIBRATION-VALIDATION   a separate frozen partition, disjoint from BOTH
                     the calibration pool and Q-BANK. Used only to measure
                     whether GPTQ's fit has converged.

EVALUATION           Q-BANK-1, 69 prompts, 1,622 positions. NOT OBSERVED
                     until N is frozen.
```

`N0` stays in the ladder deliberately: it is the set already known adequate
for the diagonal statistic, so the ladder directly tests whether a
covariance-aware encoder needs more than a diagonal-sufficient sample.

Every partition carries the digests `freeze_calibration.py` already emits
(prompt text, token ids, tokeniser, container) and the same content-verified
zero-overlap gate. Pack provenance records **which calibration and which N**
chose the values — as metadata beside the recipe, not crammed into the
recipe string.

## Calibration-sufficiency gate — decided without Q-BANK

Run the *fixed* `nvfp4-gptq-v1` at each `N`, and measure on the held-out
calibration-validation partition the quantity congruent with GPTQ's own
objective:

```text
                ‖ X_val W  −  X_val Q_N(W) ‖²
  recon(N)  =   ─────────────────────────────       per site class
                       ‖ X_val W ‖²
```

reported for q/k/v, gate/up and down separately. Secondary, reported but
not decisive: pack stability between `N` and `4N` — how many E2M1 codes and
group scales change.

### The selection rule — mechanical, frozen here

"Materially stopped improving" would leave the decision to be made after
seeing the numbers, which is the thing this gate exists to prevent. The
rule is therefore a **one-standard-error rule**, fixed now:

```text
for each site class c in {qkv, gate_up, down}:
    r_c(N)  =  held-out reconstruction error at calibration size N
    SE_c    =  standard error of the BEST observed r_c,
               by resampling VALIDATION PROMPTS (not token positions)

choose the smallest N such that, for EVERY site class c:

    r_c(N)  <=  min_N' r_c(N')  +  SE_c(at the argmin)
```

Resampling is over **prompts**, because positions inside one prompt are not
independent — 458 positions from 12 prompts are not 458 independent
observations, and a position-level bootstrap would understate the error by
a large factor.

This is the standard "smallest model statistically indistinguishable from
the best" rule. It invents no quality threshold, and it can be evaluated by
a script with no judgement call. **No Q-BANK number is looked at before
that freeze.**

Two informative outcomes, both worth having:

```text
recon improves markedly from 458 upward
  -> demonstrates directly why the SENSITIVITY bank was adequate for
     E[x_j²] and inadequate for a covariance-aware encoder

recon at 458 ~ 2,048 ~ 8,192
  -> the rank deficiency did not matter for the encoding decisions
```

### Why not simply run at 458 and attribute a null to sample size

Because that is unfalsifiable. A loss would be read as "calibration was
insufficient" and a win as "458 was evidently sufficient" — an escape hatch
available only after the result. The gate above spends the decision before
the evaluation instead.

### Rank note — stated correctly

`rank(XᵀX) ≤ n`, so at `N0` only 458 eigen-directions are data-informed,
whatever `d` is. That fact is true and it is **not** an argument about
quality:

```text
H_λ = XᵀX + λI          is FULL RANK after damping
H_λ⁻¹ = λ⁻¹I − (dense low-rank correction)
```

The correction is dense, so `H⁻¹` has non-zero off-diagonal structure
everywhere and GPTQ's sequential update couples essentially every column
pair. A coordinate with components outside `span(X)` does **not** become
round-to-nearest.

Checked numerically rather than argued: at `n = 20, d = 100, λ = 1e-2`,
`H⁻¹` is 100% dense off-diagonal and every row of `Cholesky(H⁻¹)` reaches
every later column. Any claim of the form "458/8192 = 5.6% of directions,
therefore nearest elsewhere" is **false** and must not appear in the R4
record.

Whether 458 samples give a *converged* correlation estimate is a separate,
empirical question — which is what the sufficiency gate measures.

## Hessian strategy — settled

**Dense `H`, one site at a time. No Woodbury, no block approximation.**

Peak memory is bounded by a single site, not the stack:

```text
attention / FFN input   2560 x 2560     25 MB
down_proj input         8192 x 8192    256 MB   (128 MB triangular)
```

The ~10.7 GB figure quoted earlier applies only to holding all 40 layers
at once, which the streaming order below never does.

**Woodbury is rejected on the merits, not on memory.** GPTQ needs the
Cholesky factor of `H⁻¹`; the Cholesky of `λI + low-rank` has no compact
low-rank form — it is dense and inherently sequential. Woodbury yields
cheap *entries* of `H⁻¹`, not its factorisation, so the low-rank route
would require reformulating GPTQ rather than implementing it.

A block-diagonal `H` would make it cheap by **changing the algorithm**. If
that approximation is ever wanted it gets its own recipe
(`nvfp4-gptq-block-v1`) and its own test; it must not be smuggled into
`nvfp4-gptq-v1`.

### Sequential candidate-path calibration — part of the algorithm

Frozen here as algorithm, not implementation detail:

> Layer `L+1` is calibrated from activations produced after layers `0..L`
> have already been encoded by the candidate recipe.

```text
calibration states entering layer L
  ├─ X_attn   -> calibrate q / k / v   -> release
  ├─ execute candidate attention
  ├─ X_ffn    -> calibrate gate / up   -> release
  ├─ execute candidate gate / up
  ├─ X_down = act(gate_q(x)) ⊙ up_q(x) -> calibrate down -> release
  └─ execute completed candidate layer
        -> calibration states for L+1
```

So the encoder fits against activations the candidate will actually
produce, not against BF16 activations from a model it never executes as.

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

## R4.1 — `nvfp4-gptq-v1` is FIXED-GRID GPTQ

**Only the E2M1 code nibbles may differ from `nvfp4-nearest-v1`.** Every
scale byte is byte-identical, by construction rather than by assertion.

That is a stronger contract than "same codec, same size, different
encoder", and it makes R4 answer one question exactly:

> Is nearest **rounding** the problem?

not the compound question "is nearest rounding, or the tensor scale, or the
group scales, the problem?" — which no single arm could separate.

### The grid, taken from `larql_models::quant::nvfp4` as the authority

Transcribed from the local encoder, not from generic NVFP4 descriptions:

```text
orientation     W is [rows = output, k = input], row-major
grouping        contiguous runs of 16 along the INPUT axis; groups = k/16
packing         [rows, groups, 8] bytes, LO nibble = even element
scales          [rows, groups] E4M3 bytes — one per (row, group)

tensor_scale    amax(|W|) over the WHOLE matrix / (E4M3_MAX * E2M1_MAX)
                = amax / (448 * 6) = amax / 2688
                (1.0 for an all-zero or non-finite matrix)

group scale     wanted   = amax(|group|) / E2M1_MAX / tensor_scale
                byte     = f32_to_e4m3(wanted)

step            tensor_scale * e4m3_to_f32(byte)
inv             1/step, or 0 when step is non-positive or non-finite

code            f32_to_e2m1(value * inv) & 0x0F
                grid |m| in {0, .5, 1, 1.5, 2, 3, 4, 6}
                ties -> even index; SATURATES at ±6
```

### What is frozen, and what GPTQ may choose

```text
FROZEN, from the ORIGINAL W, before any compensation:
    tensor_scale        byte-identical to nvfp4-nearest-v1
    every group scale   byte-identical to nvfp4-nearest-v1
    grouping, layout, nibble order, serialization

GPTQ MAY CHOOSE:
    the E2M1 code for each element, after error compensation,
    against that element's already-frozen step
```

Scales are computed **once from the original weights and never recomputed**.
This is GPTQ's "static groups" mode, and it closes the ambiguity that would
otherwise be fatal: **compensation cannot trigger a rescale, so column
order cannot mutate the grid** — only the rounding decisions on it.

### Parameters

```text
groupsize     16, static (the codec's own grouping)
blocksize     128
damping       0.01 x mean(diag(H))
act-order     false
column order  original K order
error propagation   canonical GPTQ update
```

### Saturation must be instrumented, not assumed away

`nvfp4-nearest-v1` **cannot** saturate: the group scale is chosen so the
group's amax lands exactly at `E2M1_MAX`. Fixed-grid GPTQ **can**, because
compensation moves values against a scale chosen for the originals, and
`f32_to_e2m1` saturates to ±6 rather than erroring.

This is a real cost of byte-identical scales, not a defect. It is therefore
**counted and reported**: saturation events per site class, as a fraction of
elements. If a fixed-grid arm underperforms, that number distinguishes
"error compensation does not help here" from "compensation was clipped
away", which are different findings.

### Determinism

Same weights, same calibration pool, same `N` → **byte-identical pack**.
Any ordering-dependent accumulation must be fixed, not left to a thread
pool.

### Scale optimisation is a LATER, SEPARATE recipe

Calibration-aware scale selection is a real idea and it is explicitly
**not** in R4. It would get its own recipe (`nvfp4-gptq-scales-v1`) and its
own arm, because folding it in here would make a win unattributable between
better rounding, better group scales and a better tensor scale. Fixed-grid
answers the causal question first.

## R4.2 — cost benchmark before the N ladder is frozen

Memory is settled; **compute is not**. Dense `H` formation plus Cholesky is
not free at the 8192-wide site, and cost grows linearly in `N` for `XᵀX`:

```text
down_proj, per layer, order-of-magnitude
    N =    458    XᵀX ~0.06 TFLOP   chol ~0.18 TFLOP
    N =  8,192    XᵀX ~1.10 TFLOP   chol ~0.18 TFLOP
    N = 65,536    XᵀX ~8.80 TFLOP   chol ~0.18 TFLOP
```

times 40 layers. So `N = 65,536` must not enter the ladder merely because
it is statistically comforting.

Before any GPTQ *values* are produced, benchmark on one real Granite
`down_proj` site: `H` accumulation at `N0`/`N1`/`N2`, plus one `8192²`
Cholesky and inverse-Cholesky. That is a pure cost measurement — it reveals
nothing about R4 quality and therefore cannot contaminate the experiment —
and it decides which rungs of the ladder are feasible at all.

## Sequence

```text
1.  Correct the rank interpretation in the record.            [DONE]
2.  Read quantize_nvfp4 and pin the exact grid.               [DONE]
3.  Specify nvfp4-gptq-v1 as FIXED-GRID GPTQ.                 [DONE, R4.1]
4.  Freeze the objective N-selection rule (one-SE, prompt
    resampling).                                              [DONE]
5.  Benchmark H construction + factorisation on one real
    8192-wide down_proj site.                                 [R4.2]
6.  Freeze the FEASIBLE N ladder from that cost.
7.  Write the larger disjoint calibration + validation pools,
    with digests and the content-verified zero-overlap gate.
8.  Implement: dense H, one site at a time, sequential
    candidate-path calibration, static scales.
9.  Choose N mechanically from held-out reconstruction.
    NO Q-BANK OBSERVATION.
10. Freeze N.
11. Compile R0-GPTQ; assert scale bytes are byte-identical to
    R0-nearest and total payload equals 2,283,690,080;
    verify it executes and reproduces its own reference.
12. Q-BANK exactly ONCE against the frozen control.
13. Report the whole response, per-category and saturation included.
14. Only if it wins materially: late5-ffn-GPTQ, late10-ffn-GPTQ,
    and the knee-survival question.
```

Steps 12 and 14 happen exactly once each. Steps 5–10 involve no Q-BANK
observation whatsoever; that is what keeps step 12 genuinely one-shot.

Note the ordering of 5 and 6 before 7: there is no point writing 65,536
positions of calibration corpus if the cost benchmark shows that rung
cannot be run.

## What a win would mean

If `R0-GPTQ` improves substantially on `R0-nearest` while **every scale
byte is identical**, the claim is unusually sharp:

> Granite's four-bit representation was not intrinsically that bad. The
> naive decisions about *which* four-bit values to store were.

That would reinterpret the sensitivity programme's baseline without
invalidating any of its measurements — each remains a true statement about
`nvfp4-nearest-v1` — and it would make R5's comparison meaningful, since a
richer precision vocabulary would then be judged against a competent 4-bit
encoder rather than an unnecessarily weak one.
