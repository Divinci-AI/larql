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

## Pairing

Machine state drifts over a session — the same binary read 83.2 and 85.4
tok/s twenty minutes apart while the battery charged from 69% to 95%. So
an A/B must **interleave** its arms (`F, U, F, U`), not run them in
blocks, and the claim is the delta, not either absolute.

`LARQL_FUSED_DECODE_HEAD=0` restores the pre-rung-2 head for exactly this
purpose.
