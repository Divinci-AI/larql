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

### R4.0-CAL-A CLOSED, 2026-08-26 — pools frozen, no encoder touched

`bench/prompts/quality-bank-1/r4_cal_a_build.py` (`build` / `verify`),
outputs `r4-cal-a-calibration-pool.json` / `r4-cal-a-validation-pool.json`.

**Calibration extends the existing frozen `N0` bank, unchanged, rather
than replacing it.** `N0 = 458` is literally
`calibration-disjoint.json`'s 12 prompts — not reproduced or
re-verified here, just taken as a fixed prefix. Every larger rung
appends more prompts, in one fixed deterministic order, so every
smaller `N`'s prompt list is a strict prefix of every larger `N`'s:

```text
N =    458    12 prompts    (unchanged existing bank)
N =  2,048    16 prompts    achieved 2,134 positions
N =  8,192    27 prompts    achieved 8,893 positions
N = 32,768    68 prompts    achieved 32,994 positions
N = 65,536   122 prompts    achieved 65,972 positions
```

**Source: real public-domain text, decided explicitly, not defaulted
into.** The existing Q-BANK-1 / SENSITIVITY-1B' banks are hand-authored
by design, at a scale (69 / 12 prompts) that stays practical to write
by hand — R4's ladder needs ~40x that many positions, where genuine
prose gives more realistic activation statistics than the same volume
of synthetic writing would. Calibration draws from *Walden* (Thoreau)
and *On the Origin of Species* (Darwin), interleaved prompt-by-prompt
so every rung — not just the largest — gets a mix of both, not all of
one source before any of the other. Validation draws from *Pride and
Prejudice* (Austen) and *Twenty Thousand Leagues Under the Sea*
(Verne) — **different books entirely from calibration**, so cross-pool
content overlap is impossible by construction, not merely checked for
after the fact (it is also checked, exactly: 0 exact-text overlap, 0
substring overlap, both directions, against both Q-BANK-1 and each
other). All four sources are Project Gutenberg, sha256-pinned in the
build script so `verify` refuses rather than silently rebuilding a
different pool if the hosted text ever changes.

**Validation is deliberately over-provisioned for prompt-level power,
not for position count.** 683 independent prompts, 176,879 positions —
about 10x Q-BANK-1's own prompt count. The one-SE rule resamples
*prompts*, not positions (positions inside one prompt aren't
independent), so what makes the eventual uncertainty estimate
meaningful is prompt count, not token count — a validation set with a
handful of very long prompts would give a confident-looking but
statistically hollow standard error. Not every one of the 683 has to
be used in every R4.0-CAL-B evaluation; having the full frozen pool to
draw from is the safe direction to over-provision in.

**A real data-quality catch, worth recording so it doesn't recur.**
The first extraction pass let two pathologies through: a table-of-
contents block (Walden ships its chapter list as one long paragraph
with no blank lines, so it read as legitimate prose by a length-only
filter) and, more subtly, *On the Origin of Species*'s per-chapter
opening "argument" lists — dense runs of short topic-sentence
fragments, each properly period-terminated, so a naive
periods-per-length check missed them too. Caught by inspection, not by
either filter — fixed by requiring both the average *and the median*
sentence length in a candidate paragraph to look like real prose (a
few genuinely long sentences can drag an average up while the block is
still mostly fragments; the median doesn't move the same way). Re-ran
end to end after the fix; every sampled chunk checked clean afterward.
Worth naming because it's exactly the kind of contamination a
"looks fine, has periods, is long enough" check would let straight
into a supposedly-frozen calibration corpus.

**Disjointness, mechanically verified, not asserted:**

```text
calibration vs Q-BANK-1     exact_overlap=0  substring_hits=0
validation  vs Q-BANK-1     exact_overlap=0  substring_hits=0
calibration vs validation   exact_overlap=0  substring_hits=0
```

**What R4.0-CAL-A does NOT do**: no `nvfp4-gptq-v1` pack, no `Q_N(W)`
for any `N`, no reconstruction error, no Q-BANK observation. There is
nothing here for the encoder to have contaminated, because the encoder
was never invoked — this step froze inputs and rules, nothing that
depends on the thing it will later evaluate.

**Next is R4.1/R4.2's own implementation** (Sequence step 8) — dense
`H`, one site at a time, sequential candidate-path calibration, static
scales, drawing on these now-frozen pools. R4.0-CAL-B (choosing `N`
from held-out reconstruction) cannot run before that exists.

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
attention / FFN input   2560 x 2560     25 MB   (f32)
down_proj input         8192 x 8192    256 MB   (128 MB triangular)   (f32)
```

These are **f32** figures. Numerical precision for this machinery is not
decided yet — see "Numerical precision is part of the recipe" under
R4.1 — and f64 roughly doubles both. R4.2 restates this table rather
than the assumption going stale.

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

### Dead coordinates — an exact rule, not a threshold

GPTQ's textbook behaviour for a zero-variance input coordinate — after
damping folds a small constant onto every diagonal entry — is silently
well-defined but semantically wrong here: damping would manufacture a
correlation for a coordinate the calibration data never informed at all.
Freezing an epsilon threshold ("small enough to count as dead") would
smuggle a tuned parameter into what is supposed to be a fixed-grid,
one-variable experiment, so the rule is exact and checked on the RAW
Hessian, before damping:

```text
dead coordinate j:
    detected from raw H[j,j] == 0 before damping
    Wwork[:,j] remains W0[:,j]
    encode using ordinary fixed-grid nearest E2M1
    no error propagated from or into j
```

No epsilon. `H[j,j] == 0` in exact arithmetic means that input channel
carried zero energy across the entire calibration pass — its row and
column in the raw Hessian are exactly zero — so there is nothing for GPTQ
to compensate with or against. Column `j` is encoded with ordinary
nearest rounding against the already-frozen grid, exactly as it would be
under `nvfp4-nearest-v1`, and it neither contributes to nor receives
error propagation. This preserves the one-variable contract: a dead
coordinate falls back to the control's own behaviour rather than
introducing a new one. Tested on a tiny synthetic matrix with a
manufactured dead column before the encoder ships.

### The zero-compensation oracle

A weaker version of this already appears in the Sequence below (step 11:
scale bytes byte-identical). This oracle is stronger, and it runs first,
before any real calibration data touches the encoder:

```text
nvfp4-gptq-v1 with compensation/update disabled
==
nvfp4-nearest-v1

    tensor scale bytes      identical
    all E4M3 group scales   identical
    all E2M1 nibbles        identical
    entire payload          identical
```

Not "the scales match" — the entire byte stream matches, including every
E2M1 nibble. If the GPTQ code path produces so much as one nibble
different from nearest when its own compensation term is forced to zero,
the infrastructure has changed something other than the one variable it
is meant to isolate — a code-selection order dependency, a rounding mode
picked up from a different code path, a layout bug — and that has to be
found and fixed before real calibration data is ever used. This runs on
every build of the encoder, not once.

### Numerical precision is part of the recipe, chosen by R4.2

H accumulation, damping, Cholesky factorisation, inverse-factor
construction, and the compensation arithmetic are all part of
`nvfp4-gptq-v1`, not a free implementation choice — two implementations
differing only in precision are not the same recipe, because they can
select different E2M1 codes at the numerical margins. One precision is
frozen, and it is frozen by R4.2's cost/stability benchmark, before any
GPTQ pack exists — **not** decided here as "f64, because it's the boring
safe choice." That framing is tempting and wrong: it treats a question
with a real, cheaply measurable answer (does f32 factorise this site
reliably? does f64 fit the memory envelope at every feasible `N`? does
the existing hand-rolled Cholesky even finish in reasonable time at
`d = 8192`?) as a matter of taste rather than a measurement.

That last question is not rhetorical. The only Cholesky/inverse
machinery in this workspace today (`larql_compute::cpu::ops::linalg`,
built for MEMIT's covariance solve — same H-accumulate → damp →
Cholesky → solve shape, at far smaller `d`) is a hand-rolled, f64-only,
unblocked triple-nested loop — **not** LAPACK-accelerated. The `X^T X`
accumulation is BLAS/Accelerate-backed (AMX-fast, via `ndarray`'s
`.dot()`); the factorisation is not. At `d = 8192` that's on the order of
`8192³/6 ≈ 9.2×10¹⁰` scalar FLOPs for Cholesky alone, before the second
factorisation GPTQ needs for the inverse-Cholesky form. Whether that
finishes in a benchmarkable time, or whether R4.2 needs to add a real
LAPACK binding (`ndarray-linalg` or `lapack-src`, both currently absent
from the dependency graph) before a `d = 8192` site is even attemptable,
is exactly what R4.2 measures — not assumed in either direction here.

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

### The protocol — small, deliberately non-scientific

```text
one real Granite down_proj site, d = 8192

measure separately:
A. H accumulation
   N = 458 / 2,048 / 8,192 / 32,768 / 65,536
   (shape/cost only — larger N may use deterministic synthetic/tiled X;
    this step is about H's cost as a function of N, not about the
    encoder's quality, so a real calibration corpus buys nothing here)
B. damping
C. Cholesky(H)
D. the inverse / inverse-Cholesky form GPTQ actually needs

for f32 and f64, each of A-D:
   wall time
   peak memory
   success / failure
   NaN / Inf
   factorisation residual  (‖H_λ − L Lᵀ‖ / ‖H_λ‖, not just "it returned")
```

The separation matters because the two costs scale differently:
`H` construction is `O(N·d²)`; factorisation is `O(d³)` and does not
depend on `N` at all. So step A is measured across the whole `N` ladder,
but steps C and D only need to run a handful of times against the same
`d = 8192` matrix — **the future calibration corpus does not need to
exist yet to learn whether the numerical path is affordable.** Building
32,768 or 65,536 positions of real calibration text before this benchmark
runs would be spending real effort on a question the benchmark answers
for free.

**Use the path the encoder will actually run, not a NumPy proxy** — a
Python timing tells you little if the eventual Rust implementation takes
a different BLAS/LAPACK route entirely. Concretely, that means:

```text
H = X^T X   ->  TWO existing paths, not one, and R4.2 measures both
                rather than asserting either is "the" path:

                (a) estimate_ffn_covariance (trace.rs:289) — the
                    function MEMIT already uses for exactly this
                    quantity (down_proj's input covariance) at
                    EXTRACT time. Streams K^T K in a scalar double
                    loop, bounded memory regardless of N (never
                    holds the full [N, 8192] activation matrix at
                    once) — the more directly relevant precedent,
                    but not GEMM-accelerated.

                (b) capture_ffn_activation_matrix (trace.rs:237) per
                    prompt, concatenated into one [N, 8192] buffer,
                    then ndarray's BLAS-backed .dot() (Accelerate/AMX
                    on this machine, same GEMM idiom already proven
                    in ridge_decomposition_solve, linalg.rs:124) —
                    faster per FLOP, but holds all N activations in
                    memory at once (~2.1 GB at N=65,536, d=8192, f32).

                Which one the encoder should actually use is a real
                design question this benchmark answers, not a premise
                it starts from: streaming-but-scalar vs
                batched-but-BLAS is exactly the kind of tradeoff R4.2
                exists to measure rather than assume.

Cholesky, solve, inverse
            ->  larql_compute::cpu::ops::linalg::{cholesky,
                cholesky_solve, cholesky_inverse} — the existing,
                f64-only, hand-rolled (NOT LAPACK-accelerated) machinery
                MEMIT already ships and tests, same H-accumulate ->
                damp -> Cholesky -> solve shape at far smaller d.
                GPTQ's inverse-Cholesky needs a SECOND cholesky() call
                on the H^-1 that cholesky_inverse() returns — pin that
                two-call composition now so R4.2's benchmark and the
                eventual encoder implementation cannot silently disagree
                about which primitive combination produces H^-1's
                Cholesky factor.

f32 variant ->  no f32 path exists today (linalg.rs is f64-only by
                design, "the MEMIT covariance inverse is ill-conditioned
                at f32 for ffn_dim > 2048"). Benchmarking f32 means
                writing a parallel f32 cholesky/cholesky_solve, mirroring
                ridge_decomposition_solve's f32-in/f64-Gram/f32-out cast
                pattern (linalg.rs:121-128) but keeping the factorisation
                itself in f32 — not casting the whole pipeline and
                calling that "the f32 arm."
```

Reusing the existing `linalg.rs` functions is deliberate, not a
convenience: it is proven code (MEMIT depends on it today) rather than a
new, unvalidated implementation whose own bugs could be mistaken for a
GPTQ finding. What it is **not** is fast — it has no blocking, no SIMD,
no LAPACK call, so its wall-clock at `d = 8192` is a real open question,
not an assumption either direction. If it does not finish in reasonable
time, the honest conclusion is "R4.2 needs a LAPACK binding first" —
`ndarray-linalg` or `lapack-src` with an Accelerate backend, both
currently absent from the dependency graph — not "reduce `d`" or "skip
the benchmark."

### Feasibility rule — frozen before any timing happens

```text
R4.2 passes if at least one precision:
    - factorises the real 8192-wide site reliably
    - fits comfortably in the intended offline-compiler memory envelope
    - makes at least N0/N1/N2 practically runnable

The feasible N ladder is truncated based only on measured cost.
No reconstruction, GPTQ codes, or Q-BANK are observed at this stage.
```

`N4 = 65,536` does not survive merely because it was written down above.
If it takes absurdly long or does not fit in memory, dropping it now —
before the encoder exists, on a pure cost measurement — is experimental
design, not tuning after seeing a result. The same applies to any other
rung the benchmark shows is impractical.

### R4.2 partial result, 2026-08-26 — an implementation-backend finding, not an R4.2 verdict

First real measurement, on `granite-4.1-3b`, layer 20, `down_proj`
(`d = 8192`), `N0 = 458` real calibration positions
(`larql-probes/examples/encoder_r4/r4_2_cholesky_cost.rs`, path-dep
build against `worktree-encoder-r4`, not yet committed/pushed):

```text
real covariance accumulation (estimate_ffn_covariance, path (a)):
    21.811 s  (458 samples, ~21.0 samples/s)

existing hand-written Cholesky (larql_compute::cpu::ops::linalg::cholesky):
    > 6 minutes CPU, N0 rung, did not complete
    manually terminated — no output row was ever produced
```

**What this establishes:** the existing scalar/hand-rolled `linalg.rs`
path is not a viable implementation for an 8192-wide GPTQ factorisation.
That is an *implementation-backend* finding.

**What this does NOT establish** — explicitly, so it cannot be
misremembered as more than it is:

```text
NOT established:
    dense-H GPTQ itself is too expensive
    f32 vs f64
    feasible calibration N
    inverse / inverse-Cholesky cost
    whether an accelerated backend changes any of the above
```

This is an **incomplete R4.2 measurement**, not the R4.2 verdict. Per its
own feasibility rule above ("R4.2 passes if *at least one* precision..."),
a hand-rolled scalar loop failing to finish in six minutes says nothing
about whether an accelerated implementation can — that is a different,
still-open question, answered below.

**A second warning came along with the first**: 21.8 seconds to
accumulate `H` from only 458 samples is itself expensive, naively
extrapolated per site —

```text
N =    458    ~22 sec
N =  2,048    ~98 sec
N =  8,192    ~6.5 min
N = 32,768    ~26 min
N = 65,536    ~52 min
```

— across 40 `down_proj` sites, once factorisation is accelerated this
becomes the dominant cost, not a rounding error next to it. `H = XᵀX` is
exactly a GEMM, so the next probe accelerates **both** steps together —
accelerating only the factorisation and leaving the scalar covariance
loop in place would just trade one bottleneck for the other and still
misreport the real economics.

**What the sequential update actually needs, so the next probe targets
the right quantity.** The canonical GPTQ update (Frantar et al.) reads
row `i` of the upper-triangular Cholesky factor of `H⁻¹` to propagate
column `i`'s quantisation error onto every remaining column in one shot
— it needs that factor materialised, not merely triangular solves against
`L`, which is why every reference implementation computes exactly
`Cholesky(H) → H⁻¹ → Cholesky(H⁻¹)`. The two-Cholesky shape this probe
already targets is the right one. What should change is the *middle*
step: LAPACK's fused Cholesky-inverse routine (`dpotri`/`spotri` —
"invert a matrix given its own Cholesky factor," what `torch.cholesky_
inverse` calls) replaces a generic dense solve-against-identity, cheaper
and numerically cleaner, and it is what every real GPTQ implementation
actually calls. Confirm this is available (`ndarray-linalg` or a direct
`lapack`/`lapack-src` binding, both currently absent from the dependency
graph) before assuming the accelerated probe's shape.

**Next probe, not yet run:** same real `H` (or freshly accumulated via
an accelerated GEMM), same `d = 8192` site, `Cholesky(H) → dpotri-style
inverse → Cholesky(H⁻¹)`, via Accelerate/LAPACK rather than `linalg.rs`,
for both f32 and f64 — wall time, peak memory, factorisation residual,
NaN/Inf, deterministic repeatability. No E2M1 codes, no reconstruction,
no Q-BANK — same "pure cost, cannot contaminate the experiment" rule as
the rest of R4.2.

### R4.2 result, 2026-08-26 — dense-H GPTQ is tractable; `linalg.rs` was the bottleneck

Ran the accelerated probe (`larql-probes/examples/encoder_r4/
r4_2_accelerated_cholesky_cost.rs`) against the same real
`granite-4.1-3b` layer-20 `down_proj` site (`d = 8192`, `N0 = 458`),
via raw `dpotrf`/`dpotri`/`spotrf` LAPACK calls against Accelerate
(`lapack-sys`, no `ndarray-linalg` — it has no Accelerate feature).
**Validated against `linalg.rs`'s own reference Cholesky on synthetic
SPD matrices first** (`n = 4, 8, 33`; rel. error `2.8e-16` to
`3.3e-15` — the row-major/column-major transpose trick is correct, not
assumed) before trusting it at `d = 8192`.

```text
H-accumulation:
  scalar streaming (estimate_ffn_covariance, path a)   22.099 s
  capture + BLAS GEMM (path b)                          2.663 s   (8.3x)

f64:
  Cholesky(H)                                           5.227 s
  dpotri (fused inverse) + Cholesky(H^-1)               1.063 s
  factorisation residual                                1.6e-15
  NaN/Inf                                               none

f32:
  Cholesky(H)                                           0.328 s
  factorisation residual                                2.8e-6
  NaN/Inf                                               none
  (inverse not yet wired — spotri exists, not called in this run)
```

**This resolves the open question the partial result above left
hanging.** Dense-H GPTQ at `d = 8192` is computationally tractable — the
entire accelerated pipeline (load, both H-accumulation paths, f64
Cholesky+inverse+inverse-Cholesky, f32 Cholesky) finished in well under a
minute of wall time, against a hand-rolled implementation that hadn't
produced one result after 6+ minutes. The negative was real and worth
finding, but it was about `linalg.rs`, never about the algorithm or the
matrix size.

**What's now established:**
```text
dense-H GPTQ at d=8192            tractable, seconds not minutes
f64 Cholesky + fused inverse       both fast, exact to 1e-15
GEMM-based H-accumulation          8.3x over the scalar streaming path
```

**What's still open** — deliberately not decided by this result:
```text
f32 vs f64, chosen for production   f64 is proven correct+fast; f32's
                                     Cholesky alone is faster and still
                                     accurate to 2.8e-6, but its inverse
                                     path isn't measured yet (spotri is
                                     declared, not called, in this run)
feasible calibration N               re-extrapolate below with the GEMM
                                     rate, not the scalar one
peak memory                          not instrumented in this run —
                                     wall time and residual only
```

**The N-ladder extrapolation changes with the accelerated H-accumulation
rate** (2.663 s at `N0 = 458`, not 22.1 s) — factorisation cost is
`O(d³)` and independent of `N` either way (confirmed: both the earlier
partial result and this one used the same `d = 8192`, and the
factorisation time here doesn't depend on which `N` produced `H`).

**SUPERSEDED by actual measurement, not extrapolation — see the "R4.2
CLOSED" section below.** The table originally here extrapolated
`N0`'s 2.663 s linearly across the ladder (~381 s at `N4`, ~4.2 hr
across 40 layers) — exactly the mistake a later note in this same
section warns against ("an optimised GEMM may scale differently
enough that the extrapolation is misleading"). It did. The real,
directly-benchmarked GEMM cost is dramatically cheaper and sublinear in
`N` — `N4 = 65,536` costs ~79 s across all 40 layers, not ~4.2 hours.
Left here struck through in spirit rather than deleted, per this
programme's own "don't delete, mark superseded" convention: whether a
number is worth spending remains separate from whether it's possible —
this section's *shape* of reasoning was right, its *numbers* were an
extrapolation this document itself later corrected.

**Remaining before R4.2 can be called complete:** wire `spotri` for the
f32 inverse arm (declared, unused — flagged by the compiler, not
silently skipped), and add peak-memory instrumentation. Neither blocks
moving forward — f64's full pipeline already answers the tractability
question this rung exists to settle.

### The f32/f64 production-precision decision, frozen before the f32 inverse arm runs

Not "2.8e-6 looks probably good enough" — a mechanical acceptance
condition, written before the missing measurement exists, so the
measurement can only pass or fail it, not retroactively justify a
threshold picked after seeing the number:

```text
f32 is production precision iff, ALL of:

1. POTRF and POTRI both complete with no NaN/Inf.

2. Factorisation residual ‖H_λ − L Lᵀ‖ / ‖H_λ‖  <=  1e-4.

3. Deterministic update-vector probes agree with the f64 reference
   within max relative difference <= 1e-3 (see below for what a
   "probe" is).

4. Repeat runs on the same H are BYTE-IDENTICAL, not merely close —
   LAPACK's blocked algorithms have no data-dependent branching for a
   fixed matrix and fixed thread count, so this is an equality check,
   not a tolerance.

If f32 fails any of these, f64 is production precision. f64 is always
the numerical oracle/reference regardless — these criteria decide
production, not correctness.
```

Where do `1e-4` and `1e-3` come from, so they don't read as arbitrary:
the standard Cholesky backward-error bound (Higham) is on the order of
`n * u` for an `n`-wide well-conditioned SPD matrix at unit roundoff
`u`. At `d = 8192`, f32 (`u ≈ 1.19e-7`) gives a textbook worst case of
`8192 * 1.19e-7 ≈ 9.8e-4`. `1e-4` is inside that bound by about an
order of magnitude — a real margin, not the day's observed 2.8e-6
worked backwards into a threshold. `1e-3` for the update-vector probe
is the same order, loosened slightly because that quantity compounds
through *two* sequential factorisations (`H → H⁻¹ → Cholesky(H⁻¹)`)
plus a division by a diagonal entry, each of which can add its own
rounding rather than cancel it.

**What a "deterministic update-vector probe" is.** No quantised weight,
no E2M1 code, no reconstruction — this checks the numerical machinery
alone, not GPTQ's output, so it cannot contaminate the experiment any
more than the residual check above does. Fix a small set of probe
columns spanning the matrix (e.g. `q ∈ {0, 100, 4096, 8000, 8191}`) and
a fixed synthetic per-column error scalar (not derived from any real
weight or quantisation — just a constant, e.g. `1.0`). For each `q`,
compute the propagated update vector the real GPTQ step would compute
— `(err_q / Hinv_chol[q,q]) * Hinv_chol[q, q:]` — once from the f32
pipeline's `Hinv_chol` and once from f64's (upcast to f64 for the
comparison), and compare. This is closer to what GPTQ actually
*consumes* than the residual check alone: the residual asks "does
`L Lᵀ` reconstruct `H`", this asks "does the row of the factor GPTQ
would actually read and scale agree across precisions."

### R4.2 CLOSED, 2026-08-26 — gate run, memory measured, GEMM ladder actually benchmarked

**Precision gate result** (`r4_2_precision_and_memory.rs --mode compare`,
same real `d = 8192` site, `spotri` now wired):

```text
precision   chol+inv(s)   inv-chol(s)   residual     nan/inf
f64         4.252         1.189         1.611e-15    false
f32         1.537         0.330         2.794e-6     false

update-vector probe (5 probe columns, err=1):
  max relative difference: 6.050e1
  RMS difference:          6.704e-6

f64 determinism (repeat run, byte-identical): true

frozen gate:
  1. no NaN/Inf (f32)                    pass
  2. residual <= 1e-4                    pass  (2.794e-6)
  3. update-vector max-rel <= 1e-3       FAIL  (6.050e1)
  4. deterministic repeat                pass

VERDICT: production precision = f64
```

**f32 fails the frozen gate on criterion 3 — production precision is
f64.** Worth recording honestly, not smoothed over: `max-rel` (60.5) and
`RMS` (6.7e-6) tell very different stories, which means the failure is
concentrated in at least one entry where the true (f64) value is close
to zero — a small absolute difference divided by a near-zero
denominator explodes relatively while the bulk of the vector agrees to
~6.7e-6. That is a real, disclosed property of *this metric* on a
quantity (a Cholesky factor's off-diagonal entries, which decay in
magnitude away from the pivot) that spans many orders of magnitude —
not a retroactive excuse to overturn the verdict. The gate was frozen
before this run specifically so an inconvenient number doesn't get
argued around; f64 stands as production precision. A *future*,
separately pre-registered gate revision (e.g. an absolute+relative
hybrid tolerance) could reasonably be proposed before its own run, not
after this one's.

**Peak memory — measured, and it doesn't discriminate the way it might
seem to.** Two isolated process invocations (`--mode memory-f64` /
`--mode memory-f32`, each wrapped externally with `/usr/bin/time -l`,
never both precisions in one process — RSS high-water-marks don't go
down, so running both together would let whichever runs first inflate
the other's reading):

```text
f64: maximum resident set size  15,430,139,904 B  (14.370 GiB)
f32: maximum resident set size  15,430,057,984 B  (14.370 GiB)
```

Essentially identical. **This is dominated by loading the whole 3B
model plus one layer's activation capture (~14 GiB), not by the
d=8192 linalg** — `H`/`L`/`H⁻¹`/inverse-Cholesky at f64 are ~512 MiB
each (four such buffers ≈ 2 GiB, f32 about half that), invisible
against a 14 GiB baseline. This is a property of *this probe's* design
(load one whole model to capture one layer), not evidence that a real
per-site compiler pass needs 14 GiB — a production encoder that
streams one layer's weights at a time rather than holding the whole
model resident would have a much smaller, more informative peak to
measure. Recorded honestly rather than allowed to imply "GPTQ needs 14
GiB/site," which this measurement does not show.

**GEMM ladder — actually benchmarked, not extrapolated, and the
extrapolation earlier in this doc was wrong.** Deterministic synthetic
`X` at each candidate `N` (`--mode gemm-ladder`, no model load, pure
GEMM cost):

```text
N          GEMM(s)/site   x40 layers
458        0.214           8.6 s
2,048      0.236           9.4 s
8,192      0.397          15.9 s
32,768     1.063          42.5 s
65,536     1.969          78.8 s
```

Sublinear in `N` (143x more data, N0→N4, only 9.2x more GEMM time) —
larger batches amortise better on this hardware, exactly the "an
optimised GEMM may scale differently enough that the extrapolation is
misleading" the earlier scalar-rate-based table (elsewhere in this
document) fell into. **`N4 = 65,536` costs ~79 seconds of GEMM across
all 40 layers, not the ~4.2 hours the earlier extrapolation implied.**
Nothing about the GEMM step disqualifies any ladder rung on cost
grounds.

**What this does NOT cover, and is explicitly out of scope for R4.2
closing**: `X` here is synthetic because GEMM cost only depends on
shape, not content — but *real* `X` requires actually running the
forward pass to capture calibration activations, which is a separate
cost this rung doesn't benchmark. The one real measurement available
(458 real positions captured in ~2.4s at one layer) suggests capture
dominates the real end-to-end per-site cost far more than GEMM does,
and — per R4.1's already-frozen "sequential candidate-path calibration"
— activations have to be captured layer-by-layer as encoding proceeds,
not as 40 independent whole-model passes, which is a genuinely
different cost shape than either the GEMM ladder or a naive
per-layer-independent estimate would suggest. That characterisation
belongs to step 8 ("Implement: dense H, one site at a time, sequential
candidate-path calibration"), not to R4.2's feasibility gate.

**R4.2 RESULT**

```text
scalar linalg (larql_compute::cpu::ops::linalg):
    REJECTED as implementation backend — did not complete one
    d=8192 factorisation in 6+ minutes.

accelerated dense H (Accelerate/LAPACK via lapack-sys, validated
against the scalar reference on synthetic matrices first):
    PASS.

real Granite d=8192 site (granite-4.1-3b, layer 20, down_proj):
    H accumulation (GEMM)        2.66 s   (8.3x over scalar streaming)
    f64 Cholesky + fused inverse  5.44 s  residual 1.6e-15
    f64 inverse-Cholesky          1.19 s
    f32 Cholesky + fused inverse  1.87 s  residual 2.8e-6
    peak memory                  14.37 GiB (dominated by model
                                  residency, not d=8192 linalg —
                                  see above)

numeric precision:
    production = f64  (f32 failed the frozen update-vector-probe gate,
                        criterion 3, by the rule written before this
                        measurement ran)
    reference  = f64

feasible N ladder (GEMM-cost basis only):
    all five rungs (458 / 2,048 / 8,192 / 32,768 / 65,536) survive on
    GEMM cost alone — nothing here truncates the ladder. Real
    calibration-capture cost is a separate, still-open question for
    the implementation step, not this gate.

No:
    GPTQ codes computed
    reconstruction inspected
    Q-BANK observed
```

R4.2 is closed. Full-Hessian, fixed-grid GPTQ at Granite's scale has
gone from "possibly too expensive to belong in VINDEX3" to "a
practical offline-compiler operation" — no low-rank approximation, no
block-diagonal `H`, no weakened GPTQ was needed to make this runnable.
The scientifically clean version is the one being tested. The
remaining unknowns move from "can we afford this" to "does it help":
calibration sufficiency (the N-selection rule, already frozen, not yet
run), and ultimately whether Hessian-aware code selection recovers
meaningful Q-BANK quality at identical bytes — which R4 still has a
genuine chance to answer either way.

## Sequence

```text
1.  Correct the rank interpretation in the record.            [DONE]
2.  Read quantize_nvfp4 and pin the exact grid.               [DONE]
3.  Specify nvfp4-gptq-v1 as FIXED-GRID GPTQ.                 [DONE, R4.1]
4.  Freeze the objective N-selection rule (one-SE, prompt
    resampling).                                              [DONE]
5.  Benchmark H construction + factorisation on one real
    8192-wide down_proj site.                                 [DONE, R4.2
    CLOSED — linalg.rs rejected as implementation backend (6+ min,
    incomplete); Accelerate/LAPACK PASS (seconds, f64 residual
    1.6e-15); production precision = f64 (f32 fails the frozen
    update-vector gate); peak memory measured (dominated by model
    residency in this probe, not d=8192 linalg); GEMM ladder actually
    benchmarked (not extrapolated) — all 5 rungs cheap.]
6.  Freeze the FEASIBLE N ladder from that cost.                [DONE,
    R4.2 — GEMM cost doesn't disqualify any rung (458/2,048/8,192/
    32,768/65,536 all survive). Real calibration-CAPTURE cost (running
    the forward pass to get real activations, as opposed to GEMM's
    synthetic-X cost) is a separate, not-yet-measured question for
    step 8's sequential candidate-path calibration — not resolved by
    this step, not blocking it either.]
7.  Write the larger disjoint calibration + validation pools,
    with digests and the content-verified zero-overlap gate.        [R4.0-CAL-A]
8.  Implement: dense H, one site at a time, sequential
    candidate-path calibration, static scales.            [R4.1/R4.2 IMPLEMENTATION]
9.  Choose N mechanically from held-out reconstruction.
    NO Q-BANK OBSERVATION.                                          [R4.0-CAL-B]
10. Freeze N.                                                        [R4.0-CAL-B]
11. Compile R0-GPTQ; assert scale bytes are byte-identical to
    R0-nearest and total payload equals 2,283,690,080;
    verify it executes and reproduces its own reference.                  [R4.3]
12. Q-BANK exactly ONCE against the frozen control.                        [R4.3]
13. Report the whole response, per-category and saturation included.      [R4.3]
14. Only if it wins materially: late5-ffn-GPTQ, late10-ffn-GPTQ,
    and the knee-survival question.
```

Steps 12 and 14 happen exactly once each. Steps 5–10 involve no Q-BANK
observation whatsoever; that is what keeps step 12 genuinely one-shot.

Note the ordering of 5 and 6 before 7: there is no point writing 65,536
positions of calibration corpus if the cost benchmark shows that rung
cannot be run.

**"R4.0-CAL" is not one step — it is two, on opposite sides of
implementation, and that split is load-bearing, not cosmetic.** Step 7
(**R4.0-CAL-A**) freezes everything that can be decided *without*
`nvfp4-gptq-v1` existing: the pools, the disjointness gate, the nested
`N` prefixes, the one-SE prompt-bootstrap rule, the site-family
aggregation, all digests. Step 9-10 (**R4.0-CAL-B**) is a different
kind of step — it runs the *now-implemented* encoder independently at
each frozen `N` and measures held-out reconstruction of `Q_N(W)`, which
does not exist as a quantity until step 8 has produced it. **The
sufficiency statistic is reconstruction error of the encoder's own
output — there is nothing to compute a ladder over before the encoder
that produces `Q_N(W)` exists.** Writing "R4.0-CAL" as a single label
anywhere (a summary, a memory note, a status line) invites exactly the
error of treating it as one step that could run before step 8; it
cannot. R4.2's own closing result (cost is not a constraint on any
ladder rung) answers a different question than R4.0-CAL-B answers
(which rung, if any, is *statistically* enough) — R4.2 closes the
economic question, R4.0-CAL-B is the still-open statistical one, and
the encoder has to exist first for either half of R4.0-CAL-B to run.

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
