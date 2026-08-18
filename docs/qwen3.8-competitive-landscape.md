# Qwen3.8-27B competitive landscape — a research note for R2

**Programme:** informs [K3 Funnel](k3-funnel.md) R2 (Kimi Linear 48B-A3B, the KDA
stack) — Qwen3.8-27B is a second real-world instance of the same problem class
(hybrid linear/full attention + MTP), so it's evidence for how the R2
abstraction should be shaped, not a separate work item.
**Scope:** external-ecosystem survey (MLX-family speculative-decoding and
quantization projects) plus the ground-truth architecture facts for
`Qwen/Qwen3.8-27B`, synthesized into what VINDEX3 should generalize versus
what's out of scope until R2 lands for real.
**Status:** research note, 2026-08-19. Nothing here is implemented; the
concurrent [`worktree-qwen35-linear-cfg`](../.claude/worktrees/qwen35-linear-cfg)
branch is the actual (config-plumbing-only) code change this cycle.

---

## 1. Ground truth: what Qwen3.8-27B actually declares

Pulled directly from the checkpoint's own `config.json`
(`Qwen/Qwen3.8-27B`, `model_type: qwen3_5`) — not from secondhand
descriptions. `larql vindex3 plan` against this checkpoint is
**inadmissible** (30 blocking findings: 1 mismatched, 29 unrepresented); see
`worktree-qwen35-linear-cfg` for the plumbing pass addressing the text-side
subset.

| Fact | Declared value |
|---|---|
| Layers | 64 total |
| Attention layout | `full_attention_interval: 4` → **48 `linear_attention` layers + 16 `full_attention` layers** (every 4th layer is full). This independently confirms the community's "48 GDN + 16 attention" claim (§2) against the primary source. |
| Linear-attention (GDN) dims | `linear_conv_kernel_dim: 4`, `linear_key_head_dim: 128`, `linear_value_head_dim: 128`, `linear_num_key_heads: 16`, `linear_num_value_heads: 48` |
| Linear-attention state dtype | `mamba_ssm_dtype: "float32"` — **the SSM/recurrent state is kept in fp32 even though the model's own default dtype is bf16.** This is Qwen's own precision choice, made independently of anything the MLX community has found by quantizing after the fact — it corroborates §3.2's "state deserves higher precision than bulk weights" argument from a primary source, not just from third-party quantization experiments. |
| Attention output gate | `attn_output_gate: true`, `output_gate_type: "swish"` |
| MTP head | `mtp_num_hidden_layers: 1`, `mtp_use_dedicated_embeddings: false` (drafts share the main embedding table) |
| RoPE | `partial_rotary_factor: 0.25` (declared identically at `text_config` and `text_config.rope_parameters`), `rope_theta: 1e7`, plus M-RoPE (`mrope_interleaved: true`, `mrope_section: [11, 11, 10]`) for the multimodal position scheme |
| Vision tower | Qwen3-VL-style ViT: `depth: 27`, `hidden_size: 1152`, `num_heads: 16`, `patch_size: 16`, `spatial_merge_size: 2`, `temporal_patch_size: 2`, `deepstack_visual_indexes: []`. Execution-surface build currently fails outright (`hidden 1152 not divisible by 0 heads`) — **out of scope for this note and for R2**; tracked separately, no ETA. |

Takeaway: the hybrid-attention layout claim from the community writeups (§2)
checks out exactly against the primary source (48/16 split falls straight out
of `full_attention_interval: 4` over 64 layers). Treat that as the one
figure in this note that's fully verified, independent of any third party.

---

## 2. External landscape (verified to exist, 2026-08-19 — not independently re-derived)

Three MLX-ecosystem projects, checked live via `api.github.com` /
`huggingface.co/api` on 2026-08-19 (all return 200 — they're real, not
hallucinated citations, though individual benchmark figures below are
**quoted, not reproduced on our own harness**):

- **`mlx-dspark`** (`ARahim3/mlx-dspark`) — native MLX port of DeepSeek's
  DSpark / z-lab's DFlash speculative decoding, covering Qwen3.8, Gemma-4,
  Muse-Glimmer, Nemotron, and others. Claims token-identical output to plain
  greedy decoding (i.e. draft-and-verify, not a lossy approximation), and
  reports Qwen3.8 8-bit + speculative decoding *outrunning* plain 4-bit
  decoding on an M4 Pro (~20–27 tok/s vs ~14.6 tok/s, workload-dependent).
- **`MTPLX`** (`youssofal/MTPLX`) — exact rejection-sampling MTP with
  residual correction, targeting the *same output distribution* as ordinary
  decoding at temperature/top-p (not just greedy-token match). Notable
  practice: custom kernels self-validate against stock MLX at load time and
  fall back if they don't match closely enough on real weight shapes —
  worth stealing as a *development method* independent of the kernels
  themselves (§3.3).
- **`oMLX`** (`jundot/omlx`) — mixed-precision quantizer plus
  ANE/GPU auto-split tuning; ~66–71 tok/s Qwen3.8 numbers are on an M5 Max
  with an `oQ4e-fp16-mtp` checkpoint (i.e. an already-quantized model, not
  BF16 — the fast absolute numbers and the "no accuracy trade" numbers are
  answering different questions; see §3.5).

Confirmed real but **not re-verified for numeric accuracy**: `ml-explore/mlx`
issues [#3839](https://github.com/ml-explore/mlx/issues/3839) and
[#3852](https://github.com/ml-explore/mlx/issues/3852) (small-M quantized-matmul
cost cliffs — MLX 0.32 added Metal split-K specifically for small-M
workloads, and there's reportedly still ~1.47–2.69× residual cost at M=4–8
vs M=1 on some shapes), and two community MLX quant conversions —
`EigenLabs/Qwen3.6-27B-MLX-mixed-4bit` (reports ~5% perplexity penalty
overall from protecting GDN-recurrence and K/V-projection tensors at higher
precision) and `tngtech/Qwen3.6-27B-NVFP4-GGUF` (quantizes the MTP drafter
aggressively on the grounds that it's draft-only and can't alter the
accepted output).

**Caveat carried forward, on the record:** "lossless"/parity claims above are
the *source projects'* own test results. Given this repo's own history —
GPT-OSS's six independent forward-pass divergences (`k3-funnel.md` §4.7), a
sinks bug that sat undetected for months despite a comment already naming
the hazard (§4.6.1) — these claims should be re-derived on LARQL's own
harness before they inform a design decision, not accepted as ground truth.

---

## 3. What generalizes into VINDEX3 (and what doesn't)

### 3.1 Decode / verify / prefill are different *operations*, not one matmul at different M

Ordinary decode is `W × vector` (M=1). MTP verification is `W × [tok₁ tok₂
tok₃ tok₄...]` — small-M matmul, not matrix-vector. The MLX issues above
suggest this regime is still being discovered, not a solved problem
underneath everyone: **the smallest representation isn't necessarily the
fastest representation for a given M.** That's a real claim about the
physical cost surface, and it's exactly the shape of thing an
`OperationPlan` is supposed to capture — `DecodeMatMul(M=1)` /
`VerifyMatMul(M=2..8)` / `PrefillMatMul(M≫1)` as distinct lowerings of the
same logical operand, each free to pick its own representation and kernel.
Whether this becomes real VINDEX3 surface is an R2/P1 design question, not
decided here.

### 3.2 Precision should be able to follow semantic authority, not just a global `--q4`

`mamba_ssm_dtype: "float32"` (§1) is Qwen's own admission that GDN state is
precision-sensitive; the community's independent finding (GDN-recurrence and
attention K/V tensors held at higher precision than bulk MLP weights, MTP
drafter quantized aggressively because it's draft-only) says the same thing
from the other direction. This is a natural extension of the
`representable`/`unrepresented`/`mismatched` vocabulary `vindex3 plan`
already has — from *representability* to *precision selection* — via
properties like "authoritative", "state-amplifying", "verification-only",
"reconstructible" on an operand. Not proposed as a concrete schema change
here; flagged as the natural next question once R2 actually has a hybrid
architecture to design against.

### 3.3 Build exactness into the optimization loop as a habit, not a one-off gate

MTPLX's load-time self-validation of custom kernels against stock MLX (on
real weight shapes, checking greedy-token agreement, falling back on
mismatch) is the same instinct as this repo's G3/G4/G5b gates
(`vindex3 plan/encode/verify/ops/exec`). Nothing to add here except: it's
independent confirmation the instinct is right, from outside this codebase.

### 3.4 Autotuning the execution plan is workload/hardware-specific, not a constant

`mlx-dspark` reportedly measures acceptance behaviour and verify-cost curve
per machine/architecture/quantization/MLX-version rather than hardcoding a
draft cap. Consistent with §3.1: if M=1 vs M=4 vs M=8 costs are genuinely
non-linear and hardware-dependent, a fixed draft-count constant is the wrong
shape of answer regardless of what LARQL's own numbers turn out to be.

### 3.5 Keep the two benchmarking questions separate

"Fastest absolute tok/s" (often already-quantized, e.g. `oQ4e-fp16-mtp`) and
"speedup from speculative/MTP execution at fixed precision" are different
claims, and the second is the one with essentially no quality trade
(mlx-dspark's greedy-identity + MTPLX's distributional-match claims, both
unverified independently per §2 caveat, but structurally a different kind of
claim than a quantization win). Recommend, when R2 produces real numbers:
report a same-precision AR-vs-MTP comparison and a separate
representation/precision sweep (PPL delta, KL, top-1 agreement, bytes,
tok/s) rather than one blended tok/s headline.

---

## 4. Relationship to current work

- **This cycle's actual code change** is `worktree-qwen35-linear-cfg`:
  parse the declared `linear_*`/`attn_output_gate`/`output_gate_type`/
  `mtp_*`/`partial_rotary_factor` fields into `ModelConfig` (§1's table is
  the full field list) and fix the `layer_types` mismatch so `plan` reports
  the real 48/16 interleave instead of collapsing it — no compute, no
  vision, config-plumbing only. That's a prerequisite for any of §3 being
  buildable, not an alternative to it.
- **R2/P1's own first task** — "verify a reference implementation runs on
  Apple Silicon before any adapter work" — now has two candidate
  architectures instead of one (Kimi Linear, Qwen3.8), which is useful:
  designing the linear-attention/MTP abstraction against a single model
  risks overfitting the abstraction to that model's specific quirks.
- **Recommended experiment sequence once R2 actually starts** (unchanged
  from the source analysis, restated here for the record): exact AR parity
  → per-shape M=1..8 roofline for the dominant linear layers → native MTP →
  acceptance-depth measurement → theoretical vs. actual bytes/token → decompose
  any remaining gap into "catch-up engineering" (kernel/MTP/representation/
  scheduling multipliers) versus "we're moving more bytes than semantically
  required" (the actual LARQL research question). Don't set the initial
  success criterion at a tok/s number; set it at whether the gap decomposes
  cleanly.

## 5. Explicitly not claimed here

Not claimed: that LARQL has a durable architectural advantage over these
projects — they're building real runtime infrastructure too (oMLX's
ANE/GPU auto-split tuner, mlx-dspark's cross-architecture generalization).
Not claimed: that any quoted tok/s or PPL figure in §2 is accurate beyond
"the source page exists". Not claimed: that vision-tower support, MTP
compute, or GDN kernels are scheduled — all remain explicitly deferred to
real R2 work with a reference implementation to verify against.
