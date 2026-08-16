# Pinned bench prompts

The steady-state decode protocol pins **four** things, not two. Warmup and
step count were already written down; the prompt and the power state were
not, and both move the number by more than the code changes being measured.

| knob | value | why it is pinned |
|---|---|---|
| `--warmup` | 16 | first-call allocation + Metal pipeline JIT land in the discarded steps |
| `-n` | 256 | a 49-step run reads ~22% slow; 256 reaches steady state |
| `--prompt` | see below | context length sets the KV work per step — the single largest term |
| power | AC, idle | battery and charging state both move decode by ~1-2 tok/s |

## The mechanism: decode cost is linear in KV depth

The prompt and `n` both matter for one reason — they set how deep the KV
cache is while the timed steps run. Per-step timings from a single
`-n 256` run, same prompt throughout:

| steps | mean ms |
|---:|---:|
| 0-32 | 11.28 |
| 32-64 | 11.56 |
| 64-128 | 11.82 |
| 128-192 | 11.98 |
| 192-271 | 12.43 |

Monotonic, ~**4.7 µs per KV position**. So the reported mean is the cost at
the run's *mean* depth:

```text
mean KV depth  ≈  prompt_tokens + n/2
ms/token       ≈  11.4 + 0.0047 × mean_depth
```

Measured on whole-run means, which are far more stable than within-run
per-step fits (those have ~2x run-to-run spread and cannot resolve this):

| `-n` | mean depth | ms/token | tok/s | marginal slope |
|---:|---:|---:|---:|---:|
| 256 | ~134 | 11.99 | 83.4 | — |
| 512 | ~262 | 12.59 | 79.4 | 4.7 µs/pos |
| 1024 | ~518 | 13.60 | 73.5 | 3.9 µs/pos |

The slope falls with depth, which is what `sliding_window: 128` predicts:
gpt-oss alternates `sliding_attention` / `full_attention`, so past depth 128
only the 12 full layers keep growing. Attempting to see that as a sharp
"knee" in per-step timings did NOT work — the within-run instrument is too
noisy — but the marginal cost between whole-run means shows it.

**The engineering point:** at 3.9 µs/position the 12 growing layers move
12 x 8 kv_heads x 64 head_dim x 4 B x 2 (K+V) = 48 KB per position, i.e.
~**12.6 GB/s** against a machine that does ~300 GB/s on these kernels. The
KV-scaling term is ~25x off the bandwidth floor, so it is latency/kernel
bound, not bandwidth bound — and the cache is f32, which is 2x what f16
would cost on the same term.

Two consequences:

- **`n` is part of the measurement, not a precision knob.** Doubling it
  costs 0.6 ms/token — bigger than most kernel work being compared. A
  number quoted without `n` is not reproducible.
- **`lm_head` is flat across all of this** (1.489 vs 1.490 ms at a 90x
  prompt-length change), because the head does not read the KV cache. Head
  optimisations must be reported as absolute ms, never as a share of a
  total that moves with depth.

## Why the prompt has to be pinned

Measured on gpt-oss-20b, M3 Max, AC, idle, same binary, same session:

| prompt | ms/token | tok/s | GPU fwd | lm_head |
|---|---:|---:|---:|---:|
| `The capital of France is` (~6 tokens) | 12.28 | 81.4 | 10.74 | 1.489 |
| `gpt-oss-steady-state.txt` (~550 tokens) | 14.34 | 69.8 | 12.81 | 1.490 |

**2.0 ms/token — 11.6 tok/s — from prompt length alone.** That is four
times the size of the TOKEN-B1 rung-2 win it would otherwise be compared
against. Note also that `lm_head` is flat to within 0.001 ms across both:
head cost is context-independent, so a head optimisation must be reported
as an absolute millisecond delta, never as a percentage of a total that
moves with the prompt.

The historical `77.2 tok/s` ladder figure was recorded as "long prompt"
without the text, and on battery. It is therefore **not reproducible** and
must not be used as a control arm. Numbers from it are kept in
`ROADMAP.md` as history, not as a baseline.

## Files

- `gpt-oss-steady-state.txt` — the long-prompt arm. Prose, no code fences,
  ends on an open-ended instruction so decode does not hit EOS early.

The KV attention ladder pins three depths, so that its A/B blocks measure
a context *slope* rather than one context. All three end on an open-ended
instruction, for the same EOS reason as above:

| file | tokens | chars |
|---|---:|---:|
| `gpt-oss-kv-ladder-a.txt` | ~36 | 147 |
| `gpt-oss-kv-ladder-b.txt` | ~574 | 2296 |
| `gpt-oss-kv-ladder-c.txt` | ~2024 | 8096 |

B and C reach their depth by repeating one paragraph — 2× and 7×. That is
deliberate and harmless for a KV measurement, because decode cost depends
on cache *depth*, not on content. Do not reuse them for anything that is
sensitive to what the tokens say.

## Invocation

```sh
# short arm (the documented ladder prompt)
LARQL_GPU_ROUTE=1 larql bench <spine>.vindex \
  --warmup 16 -n 256 --routed-from <container>.v3

# long arm
LARQL_GPU_ROUTE=1 larql bench <spine>.vindex \
  --warmup 16 -n 256 --prompt "$(cat bench/prompts/gpt-oss-steady-state.txt)" \
  --routed-from <container>.v3
```

Report both arms, or say which one you ran. A single number without its
prompt is not a measurement anyone can repeat.

## Pairing: bracket, do not interleave

Machine state drifts over a session — the same binary read 83.2 and 85.4
tok/s twenty minutes apart while the battery charged from 69% to 95%. So
the claim is the delta, never either absolute.

This section used to prescribe interleaving (`F, U, F, U`). **That is
superseded and should not be used.** Interleaving fixes the candidate at
positions 2 and 4, so under any position-dependent drift the candidate is
systematically later in the run than the baseline — a measured ~2.5%
bias per position on this machine — and it only cancels drift that is
monotonic, which this machine's is not.

The unit is a **bracket**: `baseline / candidate / baseline`, warmed to
plateau first, with the candidate counting only if the two baselines agree
within ~1%. If they disagree the machine changed underneath the block and
the block is void — do not average across the disagreement. One recorded
block read `19.75 / 18.04 / 40.49`; without the closing baseline it looks
like a clean ~9% win.

Two things the bracket does **not** establish, each needing its own check:

- It validates the machine, not the arm. An arm that stops early still
  prints a plausible row, and if the *candidate* is the truncated one both
  baselines still agree. Gate on the step count (`n_steps`), which is
  column 6 of the bench row and is easy to filter away by accident.
- It catches drift, not steady contention. A peer holding the GPU across
  the whole block moves all three arms together. Only a handshake with
  every peer session catches that; a utilisation probe between runs does
  not.

`scripts/kv-ladder-bracket.sh` implements the ordering and both arithmetic
checks. The full derivation, and the measured cases behind each rule, are
in [`docs/kv-attention-scaling.md`](../../docs/kv-attention-scaling.md)
§Run hygiene.

`LARQL_FUSED_DECODE_HEAD=0` restores the pre-rung-2 head for exactly this
purpose.
