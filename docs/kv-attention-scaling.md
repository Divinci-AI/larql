# KV attention scaling — measurement schema

Schema and pre-registered predictions for the KV ladder's context-scaling
runs (KV-B1 clean e2e, KV-B2, KV-C, and the eventual `larql-kv` engine
tournament).

The purpose of fixing this before the runs is narrow: **a "tok/s vs context
length" curve is not interpretable on an architecture with mixed attention
layer classes**, and every model on the ladder has them. Recording the
decomposition at capture time costs nothing; reconstructing it afterwards is
impossible.

## Four things that must not be conflated

The ladder moves along four independent axes. Collapsing any two of them
produces a graph that looks like a result and isn't.

| Axis | What it is | Who changes it |
| --- | --- | --- |
| **Context** | Prompt + generated tokens the model is holding | The workload |
| **Effective span** | Rows the attention kernel actually reduces over, per layer | The architecture, via the sliding window |
| **Execution geometry** | How that reduction is scheduled — serial, seqpar, multi-TG | KV-B1, KV-B2 |
| **Representation** | Width and encoding of the KV rows themselves | KV-C, the `larql-kv` engines |

B1 and B2 move execution geometry only and are exactness-preserving. KV-C
moves representation and is not. The scoring rules differ accordingly —
see [Gate classes](#gate-classes).

## Why context ≠ span

`ops::kv_cache::attention_span(t, window_size)` bounds the span at the
layer's sliding window; only layers with `window_size == 0` grow with
context. So within one forward pass, at a given context depth, different
layers sit in different span tiers — and `ops::kv_seqpar::slices_for`
selects a *different threadgroup width per layer class* as a result.

A single "tok/s at 8K" number is therefore a weighted blend over layer
classes, with weights set by the architecture. Two consequences:

1. **The B1/B2 gain asymptotes.** As depth grows, sliding layers stop
   contributing new span; the incremental benefit comes only from the full
   layers. The curve tends to a ceiling set by the full-attention share of
   attention cost, not to an unbounded gain.
2. **Cross-model curves are not comparable** unless the layer-class mix is
   reported alongside, which is exactly what a cross-engine tournament
   needs them to be.

## Row schema

One row per (config, depth, layer class). Emit the class rows; the
per-token aggregate is derived, not measured separately.

| Field | Notes |
| --- | --- |
| `model` | Registry id |
| `context_length` | Prompt + generated, tokens |
| `generated_depth` | Tokens generated so far — span driver during decode |
| `layer_class` | `sliding` \| `full` |
| `layers_in_class` | Count, from the loaded config — not assumed |
| `window_size` | 0 for full layers |
| `head_dim` | Per class; drives the seqpar policy |
| `effective_span` | `attention_span(t, window_size)` for the class |
| `attention_kernel` | `serial` \| `seqpar` \| `seqpar_long` \| B2 variant |
| `seqpar_slices` | 0 when refused; records what the policy actually chose |
| `latency_per_token_ms` | |
| `tok_s` | |
| `gpu_occupancy_pre` | Mean and max, from the exclusivity check |

`seqpar_slices` is worth carrying explicitly: a run where the policy
refused (0) and a run where it chose 1 slice-equivalent look identical in
the aggregate and mean different things.

## Pre-registered prediction

Written before the deep-context blocks are re-run, so the shape is not
fitted afterwards.

For a model with `S` sliding layers at window `W` and `F` full layers:

```text
sliding layer span → min(depth, W)      saturates at W
full layer span    → depth              grows without bound
```

Therefore:

- Below `depth = W`, every layer is in the same regime and the B1 gain
  tracks the short-context number.
- Past `depth = W`, sliding layers pin to their tier while full layers
  climb through the tiers. The measured gain rises, then flattens toward
  the full-layer share.
- **The gain must not keep climbing linearly with context.** If it does,
  the layer-class mix or the window is not what the config says, and the
  decomposition is wrong before the performance claim is.

For gpt-oss-20b the expectation is an even split — 12 sliding at window
128, 12 full — which predicts a rise between depth 128 and roughly the
long tier, then a plateau. **Confirm this against the loaded config before
the run rather than assuming it**: dump `attn_spec.sliding_window` per
layer and record `layers_in_class` from what comes back. The prediction is
only falsifiable if the mix is read, not assumed.

## Gate classes

Two kinds of change move along this ladder and they take different
authorities. Do not borrow one for the other.

**Execution-order changes (KV-B1, KV-B2).** KV stays f32, slice and tile
partials accumulate in f32, the merge runs in f32. The only permitted
difference from the serial kernel is reassociation of the weighted-V sum.
Gated by `max_rel < 1e-4` against the serial f32 kernel, with negative
controls calibrated at ~1e-1 and bitwise determinism across repeats — see
`crates/larql-compute-metal/tests/test_kernel_kv_attention_seqpar.rs` and
its two siblings. That tolerance separates reassociation from *defect*. It
cannot separate reassociation from *approximation*, which is why no
representation-width change is allowed inside B2.

**Representation changes (KV-C, `larql-kv` engines).** An f16 KV cache
exceeds 1e-4 by construction, so the gates above cannot be reused and must
not be loosened to accommodate it. KV-C needs its own oracle: an f32-KV
reference scored in predictive units on the deployment path, with a
quality budget fixed **before** any latency is measured. Otherwise an
approximation win folds silently into the B2 number.

## Gate step: dump the layer specs first

Before any timed block, dump per layer and confirm the class mix against
the prediction above:

```text
layer   window_size   head_dim   q_heads   kv_heads
```

This is a gate, not a convenience. The predicted curve is only falsifiable
if the mix is read from the checkpoint; if it is assumed, a matching curve
confirms nothing and a mismatched one is unattributable.

## Blocks and arms

Measure the **default**, not `LARQL_KV_SEQPAR=auto`. The explicit-auto path
already has evidence; the shipping question is whether an unset env fires
the policy at head_dim 64, which is a different code path through
`kv_seqpar_from_env` → `SeqparRequest::Unset` → `default_is_auto`.

**This requires the enablement change applied first.**
`SEQPAR_DEFAULT_ON_HEAD_DIMS` ships empty, so on a clean checkout `Unset`
resolves exactly like `Off` and an `off / default` A/B would compare the
serial kernel against itself — a null result indistinguishable from a
negative one.

### Candidate tree

Build the candidate as "the enablement commit minus its evidence", all
gates green — not as a tree with two tests deliberately red:

1. `SEQPAR_DEFAULT_ON_HEAD_DIMS = &[64]`
2. Flip the two expectation tests
   (`nothing_defaults_on_until_the_gate_closes`,
   `the_default_list_is_empty_pending_the_gate`) to assert the enabled
   state.
3. `cargo test -p larql-compute-metal --lib ops::kv_seqpar` — green.
4. **Rebuild `--release`.** A stale `target/release/larql` is exactly how
   this experiment yields a beautifully reproducible null.

The candidate marker is then the shape of the diff, not a red suite:

```text
git diff:
  exactly 1 policy constant
  exactly 2 expectation changes
```

`unset_resolves_through_the_default_list` and
`explicit_off_stays_off_on_every_geometry` must stay unchanged and green
across both commits. If either needs editing, the change is to policy
*mechanics*, not to policy *evidence*, and does not belong in the
enablement commit.

### Invoking

Make the absence of the env var explicit rather than trusting shell state:

```bash
env -u LARQL_KV_SEQPAR LARQL_GPU_ROUTE=1 ./target/release/larql bench ...
```

The `off` arm is `LARQL_KV_SEQPAR=off` on the same binary.

```text
PRE  GPU idle check
A    short context          off / default / off / default
B    pinned ~550-token      off / default / off / default
C    ~2K regime             off / default / off / default
POST GPU check per block
```

## Pre-registered decision rules

Fixed before the run so the outcome cannot be reinterpreted after it.

| Outcome | Action |
| --- | --- |
| Short + medium + deep all positive | Default head_dim 64 ON; close B1 |
| Short positive, deep flat | Still default ON, provided no meaningful regression and the short win is robust |
| Short positive, deep negative | Add a span restriction to the default; explicit `auto` stays available outside it |
| Any correctness or integration failure | Fix before defaulting |

A deep-context negative does not invalidate B1 — `slices_for` already takes
`span`, so the result would simply say the *default region* is narrower
than the capability.

## KV-B2 contract

Recorded here so the B2 kernel is specified before it is written, and so
the f32 invariant above has something concrete to attach to.

```text
grid.x = query head
grid.y = sequence tile
```

Each tile computes local `m` (max), `l` (exp sum), and `o[head_dim]`
(weighted-V partial). The structural difference from B1: sequence tiles no
longer share `tg_scores`, so **the softmax state must travel with the
partial output** rather than being resolved before accumulation. Merge is
online-softmax:

```text
tile state = { m, l, o[head_dim] }

merge(A, B):
    m = max(A.m, B.m)
    α = exp(A.m - m)
    β = exp(B.m - m)
    l = α*A.l + β*B.l
    o = α*A.o + β*B.o

final = o / l
```

All f32, per the invariant above. `examples/bench_attention_span` can then
answer whether B2 buys anything past B1's 1024-thread ceiling before any
e2e run is spent on it.

## Run hygiene

- Exclusive GPU, verified immediately before each block, not only after.
  A flat baseline arm does not establish that the candidate arm was
  uncontended — the arms have different threadgroup widths and a competing
  workload can hit them asymmetrically.
- Interleave arms within a block (`off/auto/off/auto`), and report the raw
  per-arm readings, not just the derived delta.
- Steady state: warmup 16, n 256. Short runs read slow.
- Run the control triple (`unset` / `off` / `auto`) once per session.
  `unset ≈ auto ≪ off` is what distinguishes "the default fired and helped"
  from "the default never fired"; an `off`-vs-`default` pair alone cannot
  tell those apart, and reads the second as a negative result.

### The instrument is not stable by default

Measured 2026-08-15 on the M3 Max, gpt-oss-20b. Both effects below are
larger than the ~11% being measured, and neither announces itself:
`pmset -g therm` stays silent, the GPU reads idle between runs, and memory
stays healthy throughout.

**Cold working set.** `larql bench` is one process per arm, and this model
is ~27.5 GB across the two containers (18 GB q4k spine + 9.5 GB routed
MXFP4). With the page cache evicted, each arm re-faults tens of GB and
consecutive runs warm progressively — an `off` arm read 17.40, 16.58,
16.05, 15.76, 15.51 ms across five identical runs. Warm explicitly before
timing anything:

```bash
find <spine> <routed> -type f -size +10M -exec cat {} + > /dev/null
```

**Sustained-load degradation, recoverable.** Past roughly 5–10 consecutive
runs the machine collapses — the same `off` arm walked 14.17 → 36.22 ms
over ten runs, a 2.5× loss, monotonic. A 5-minute rest fully restores it:
13.70, 13.72, 13.75, 13.75, 13.75, a 0.36% spread reproducing the
pre-degradation value. So this is pacing, not a leak, and the fix is rest
between blocks rather than a faster harness.

Root cause on this machine: a **67W adapter negotiating 65W**, on hardware
specced for 96W/140W, with the battery charging concurrently. Sustained
20B MoE decode exceeds that envelope by itself. Check before benching —
`pmset -g batt` says "AC Power" and is blind to wattage:

```bash
ioreg -rn AppleSmartBattery | grep -o '"Watts"=[0-9]*'
system_profiler SPPowerDataType | grep -iE "wattage|Name:"
```

**The power cap clips the faster arm.** This is the part that matters for
A/B design: `default` does more work per unit time than `off`, so it draws
more instantaneous power and is the arm the cap truncates — variably, and
only downward. Same-arm spread therefore tracks how fast the arm is, not
how noisy the machine is, and it grows with work per token (0.18% at short
context, 1.91% at ~574, 5.43% at ~2024 while every `off` arm stayed under
0.4%). A power-limited A/B is biased *against* the faster arm, so it
understates rather than inflates — but it cannot produce a point estimate.

### Validity preconditions

A block counts only if all four hold. Check them **before** computing any
delta:

1. Adapter delivers the machine's rated wattage. On the 65W adapter this
   gate cannot pass at depth — see above.
2. Warm to plateau, then confirm it: repeat one arm until consecutive
   readings agree within ~1%. A still-improving or still-degrading series
   means the block is not yet runnable.
3. Rest ~5 minutes before each block, and keep a block to its 4 arms.
4. The two same-arm readings inside the block agree within ~1%. If they do
   not, the block is void — do **not** average across the disagreement.

Precondition 4 is the load-bearing one. An arm-mean computed over a
disagreeing pair will happily produce a plausible number, and interleaving
does not protect against it: interleaving controls for *monotonic* drift,
and this failure is not monotonic.

## Provisional readings — 2026-08-15, VOID

Recorded because the prediction check pre-dates the clean run, which makes
the re-run a genuine replication rather than a first look. **These are not
results.** Blocks B and C fail precondition 4 on the `default` arm, and
precondition 1 failed for the whole session (65W adapter).

```
                     off (ms/tok)     default (ms/tok)   same-arm spread
A  ~36 + 256 tok     12.14  12.15     10.88  10.90       0.08% / 0.18%  PASS
B  ~574 + 256        13.73  13.69     10.98  11.19       0.29% / 1.91%  VOID
C  ~2024 + 256       18.19  18.26     13.67  12.93       0.38% / 5.43%  VOID
```

What survives without the arm means, as a description of the data rather
than an inference: the arms do not overlap in any block, and the *slowest*
`default` reading beats the *fastest* `off` reading by 10.2%, 18.3% and
24.9% respectively.

The prediction holds directionally — the gain grows with depth, as 12
sliding layers pinning at window 128 while 12 full layers keep climbing
predicts. The asymptote toward the full-attention share is **not** observed;
it is still rising at ~2K, so locating the plateau needs deeper context.

The token-by-token prefill phase is a second instrument on the same kernel
(`encode_attention_block` has one caller, `decode/token.rs`, and prefill
runs at decode rate — 19706 ms for ~2024 tokens). It averages over the
whole span ramp rather than sitting at final depth, so it should show a
*smaller* delta than decode at the same block, and it does: −10.8% at B and
−18.2% at C, with arms tight to 0.01–1.7%. Two instruments, same shape.
