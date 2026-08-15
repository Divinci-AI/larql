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
