# The TTS funnel — audio-token output as a forcing function

Status: steps 0-1 green. Branch `worktree-tts-audio-tokens`, started 2026-08-09.

Gate log:

- **Step 0 PASS** (2026-08-09) — reference dump harness at
  `jarvis-voice/.engines/moss_parity_dump.py`; fixture long-23 with the
  aru-12 voice-clone splice; cpu / fp32 / eager / greedy / repetition
  penalty off. 138 frames (11.0 s audio) in ~34 s, EOS reached, prefill
  343 rows. Two independent process runs **bit-for-bit identical** across
  all eleven arrays (prompt matrix, prefill embeds/hiddens, per-step ids /
  summed embeds / backbone hiddens, 138×16×1027 local logits, frames,
  emitted tokens). Dumps in
  `jarvis-voice/renders/moss-realtime/parity-dump/` (run manifest records
  versions and hashes). Steps 2–4 compare against `run1.npz`.

- **Step 1 PASS** (2026-08-09) — MOSS loads in LARQL with full tensor
  accounting. `MossTtsRealtimeArch` (detection via nested
  `language_config`, `language_model.` prefix strip, `has_lm_head=false` —
  a new arch capability, since the backbone genuinely has no output
  projection and the untied-but-missing error is a text-LM invariant);
  auxiliary weights (per-codebook embedding tables + depth transformer +
  16 heads) side-load via `larql_models::speech::moss_tts_realtime`,
  every count config-derived. The coverage audit is bidirectional and
  derives its expected sets from the arch's own key methods: on the real
  403-tensor inventory it reports 310 backbone / 92 aux / 1 justified
  skip (the byte-identical text-table duplicate, asserted at load) /
  0 unexplained / 0 missing. Real-checkpoint aux load passes in ~6 s with
  the pad-row-zero and duplication assertions running on real bytes
  (`real_checkpoint_aux_load`, ignored in CI).

This document plays the role `k3-funnel.md` plays for sparse execution: it
names one model that cannot run, inventories exactly why, and commits to
removing blockers one at a time. No abstraction gets built speculatively;
every change must move the target model closer to parity.

**Target:** LARQL executes the generative portion of
`OpenMOSS-Team/MOSS-TTS-Realtime` — text tokens in, RVQ audio tokens out —
with per-step logit/token parity against the reference implementation.

**Non-goals (initially):** audio playback, vocoding/codec decode inside
LARQL, STT, persona/orchestration, "designing multimodal LARQL". The codec
(`MOSS-Audio-Tokenizer`) stays external: LARQL emits audio tokens, an
external decoder turns them into PCM. The codec is itself a causal
transformer, so pulling it in later is a plausible second act — after
parity, not before.

Why this matters beyond Jarvis: K3 forced routing, residency, containers.
TTS forces the orthogonal axis — typed token domains, multimodal *output*,
multi-codebook generation, persistent conversational state, and eventually
a continuous realtime deadline (§5). `docs/multi-modal.md` declares audio
output a non-goal ("No diffusion head, no audio decoder"); this funnel
deletes that line, one blocker at a time.

Provenance: everything below was read out of the locally cached checkpoint
(`models--OpenMOSS-Team--MOSS-TTS-Realtime`, rev `7568278`, 4.3 GB bf16
safetensors) and the reference source at
`project-jarvis/jarvis-voice/.engines/moss-src` (OpenMOSS/MOSS-TTS
@ `58b20a0`), 2026-08-09. Vendor performance figures (180 ms TTFB,
RTF 0.51 on an L20) are claims, not measurements.

---

## 1. The model

### 1.1 Backbone: stock Qwen3

The backbone is literally `transformers.models.qwen3.Qwen3Model`
(`modeling_mossttsrealtime.py:97`) — not a fork. hidden 2048, 28 layers
(all full attention), 16 heads / 8 KV heads (GQA 2:1), head_dim 128
(explicit), intermediate 6144, SwiGLU, RMSNorm eps 1e-6 with Qwen3
q_norm/k_norm, rope theta 1e6, text vocab 151936.

Two load-time facts:

- **No text lm_head exists.** The backbone never predicts text; its only
  output is `last_hidden_state`. Text is always supplied externally.
- **`language_model.embed_tokens.weight` is dead weight** — a byte-identical
  duplicate of `embed_tokens.0.weight` (the model always runs on
  `inputs_embeds`). 311 MB skippable at load.

Param split: 2.332 B total; backbone 1.721 B (incl. dead embed), depth
transformer 266.5 M, top-level embedding tables 344.8 M.

### 1.2 Audio token scheme

- 16 RVQ codebooks (of the codec's 32; only the first 16 used), codebook
  vocab 1024, model-side `audio_vocab_size` 1027 (+3 specials).
- Frame rate 12.5 Hz: 1 frame = 80 ms = 1920 samples at 24 kHz.
- **No codebook delay pattern** — all 16 codebooks of a frame come from one
  backbone step. (The sibling `moss_tts_delay` package *does* delay; do not
  cross-reference it.)
- Audio specials: 1024 pad (all channels), 1025 BOS / 1026 EOS
  (**channel 1 / codebook 0 only**). Text-side specials include
  `<|audio_pad|>` 151654 (reference splice placeholder) and
  `<|text_pad|>` 151655 (text-exhausted filler).
- Checkpoint property, assert at load: backbone codebook embedding row 1024
  is exactly zero in all 16 tables, rows 1025/1026 zero except table 1 —
  so summing pad channels is a no-op and 16 lookups can be skipped on
  text-only positions. A property of this checkpoint, not the architecture.

### 1.3 One generation step

Input layout is a `[T, 17]` integer matrix: column 0 text, columns 1–16
the codebooks.

```text
step_ids = [ next_text_token | 16 codebook tokens sampled last step ]   [B,1,17]
        │
        ▼
h = Σ over 17 embedding tables (plain sum, no projection, no scaling)
        │
        ▼
backbone forward, 1 position, persistent KV  →  hidden [B,1,2048]
        │
        ▼
depth transformer (4 Qwen3-shaped layers, hidden 2048, fresh 16-slot
static cache each frame — no cross-frame state):
  micro-step 0 :  input = raw backbone hidden, pos 0 → head 0 → c0
  micro-step i :  input = embed_tokens[i-1](c_{i-1}), pos i → head i → c_i
        │
        ▼
frame [B,16] emitted; c0 == 1026 marks stop; frame feeds next step's input
```

There is no backbone↔depth projection (hidden sizes match by design). The
16 output heads (`local_lm_heads.0..15`, each [1027, 2048], no bias) map
depth position `i` to codebook `i`. Hidden state at position t−1 predicts
the frame at position t.

### 1.4 The streaming contract

The entire incremental protocol is: **the text channel leads the audio
channels by 12 positions** (`delay_tokens_len = 12`), and generation
consumes exactly one text token per 80 ms frame. Prefill waits for 12
pending text tokens; audio BOS (1025) sits on channel 1 of the *last*
prefill text token; when text is exhausted the model is fed
`<|text_pad|>` until EOS. The lead is enforced in three separate places in
the reference; an off-by-one desynchronises text and audio silently —
fluent but wrong-timed speech, not a crash.

Voice cloning is a **multi-channel splice, no speaker encoder**: the
system prompt contains N `<|audio_pad|>` text tokens, and channels 1–16 of
exactly those rows are overwritten with codec-encoded reference audio.

Multi-turn is **KV continuation, not re-prefill**: turn 0 prefls the
system prompt; later turns prefill only the new user turn on top of the
retained cache (32 K max context ≈ 40 min).

### 1.5 Sampling

Identical across all 16 heads: temperature 0.8, top_k 30, top_p 0.6,
repetition penalty 1.1 over a 50-frame per-codebook window. No CFG.
Greedy mode (`do_sample=False`) takes an argmax shortcut that bypasses
temperature/top-k/top-p entirely.

Parity traps, verified in source:

1. `apply_top_p` is non-standard: ascending sort, remove where
   `cumsum(softmax) <= 1 - top_p`. Replicate literally; HF's warper
   tie-breaks differently.
2. `apply_repetition_penalty` mutates the logit tensor **in place through a
   view** and changes the return shape `[B,1,V] → [B,V]`. Disable it while
   establishing baseline parity (the reference itself forces it off for the
   prefill frame).
3. Reference runs bf16 with `torch.compile` on the depth loop and
   attention-backend-dependent cache types. For reference dumps: eager,
   fp32, no compile.

### 1.6 Parity hooks in the reference

Per-step dump points (file:line in `moss-src/mossttsrealtime/`):
summed input embedding `modeling_mossttsrealtime.py:101-109`; backbone
hidden `modeling_mossttsrealtime.py:134`; per-codebook logits
`modeling_mossttsrealtime_local.py:437`; the 16-micro-step loop
`streaming_mossttsrealtime.py:378-411`; sampled frame
`streaming_mossttsrealtime.py:285-301`; prefill frame 0
`streaming_mossttsrealtime.py:221-238`; non-streaming reference loop
`inferencer.py:195-303`.

Compare logits, not sampled tokens (`torch.multinomial` RNG will never
match across engines). Greedy end-to-end, logit-level for the samplers.

---

## 2. What LARQL lacks — blocker inventory

From a full workspace survey (2026-08-09). The input side is genuinely
ready; the output side has no seam.

**Ready and reusable as-is:**

- `QwenArch` loads Qwen3-shaped checkpoints (QK-norms included) — the
  backbone is the easy case.
- `Sampler` (`layer_graph/generate/sampling.rs`) is pure
  `&[f32] → Option<u32>`; callable 16× per step with zero modification.
- The multimodal input seam: `EmbeddingPlan` / `EmbeddingChunk`
  (`larql-compute/src/forward/embedding_plan.rs`), `ModalEncoder` /
  `Connector` / `MultiModalProtocol` (`larql-models/src/multimodal.rs` —
  the `Audio` variants already exist, unimplemented), and
  `KvEngine::prefill_from_hidden` (ADR-0023).
- `KvCache.next_position` is already decoupled from row index (ADR-0023
  pinned this) — correct for prefills whose rows aren't text tokens.
- Side-loading non-backbone weights from safetensors is proven by the
  vision tower (`--mm-weights`); the PLE sidecar
  (`format/weights/ple_sidecar.rs`) is the pattern for a vindex-carried
  auxiliary tensor group later.
- `shannon layer-dump` / `layer-diff` — the parity tool. It closed OLMoE
  and GPT-OSS; it is the instrument for step 2 below.

**Tier 1 — structural, no seam exists:**

1. Every decode loop carries a scalar `current_token_id: u32`; ~25
   `generate*` entry points assume one token from one vocabulary. MOSS
   needs `[u32; 16]` + text per step.
2. Detokenization is unconditional inside the sampling step
   (`gpu/sampling_step.rs:37`); `run_decode_loop` takes non-optional
   `&Tokenizer` + `&mut Detokenizer`. `GenerateResult.tokens` is
   `Vec<(String, f64)>` — ids are discarded entirely.
3. One `lm_head` / one `vocab_size` in five places (`ModelWeights`,
   `VectorIndex`, `ModelConfig`, `WeightSource`, `VindexConfig`). MOSS has
   zero text heads and 16 audio heads.
4. `embed_plan` **concatenates** rows; MOSS **sums** 17 embeddings into one
   row, per decode step — exactly what ADR-0023 scoped out ("decode is
   text-out by definition").
5. No `decode_step_from_hidden` peer to `prefill_from_hidden` on
   `KvEngine`; decode never passes through the multimodal seam.
6. EOS is string-shaped (`EosConfig.stop_strings`, literal matching in
   `larql-kv`). Audio EOS is "codebook-0 id == 1026" — no representation,
   and numeric collisions with text special ids would spuriously halt.

**Tier 2 — format & policy:**

7. The vindex extractor actively drops audio towers
   (`extract/coverage/rules.rs::NON_TEXT_TOWER`). Route around it first:
   side-load depth transformer + embedding tables the way SigLIP does;
   revisit the vindex format (sidecar pattern) once parity exists.

**Tier 3 — surface & state:**

8. Both streaming surfaces discard the token id and send text-only frames;
   no binary/typed frame exists.
9. The server holds no engine or KV across requests; `ChatSession` is
   text-token-only. MOSS's persistent acoustic history (backbone KV +
   frame history across turns) has no home. The `WS_CMD_CANCEL` path is a
   useful barge-in precedent.

---

## 3. The funnel — steps and gates

Discipline: each step has a falsifiable gate. No step starts before the
previous gate is green. Abstractions (a `TokenStep` enum, a multi-head
output type) are introduced at the step that needs them, shaped by what
that step needs, not before.

**Step 0 — reference dump harness (Python, in the jarvis-voice venv).**
Instrument the reference at the §1.6 hooks: fixed prompt, greedy, eager,
fp32, repetition penalty off. Dump per-step summed embeddings, backbone
hiddens, per-codebook logits, sampled frames to disk.
*Gate: the dump is reproducible bit-for-bit across two runs.*

**Step 1 — weights load.** MOSS detection arm + config fields
(`num_codebooks`, `audio_vocab_size`, depth dims); backbone through the
existing Qwen3 path; embedding tables 0–16 + depth transformer + 16 heads
side-loaded from safetensors (vision-tower mechanism, not the vindex).
Assert the pad-row-zero checkpoint property at load; skip the dead
311 MB embed.
*Gate: every tensor accounted for against the checkpoint inventory —
the GPT-OSS silent-drop lesson applies here with 34 auxiliary tables.*

**Step 2 — backbone step parity.** Summed-embedding primitive
(`embed_step` peer to `embed_plan`); one backbone forward from a `[T,17]`
prompt matrix.
*Gate: `layer-diff` vs the step-0 dump — first-drifting-capture empty,
final hidden within fp32 tolerance.*

**Step 3 — depth transformer parity.** The 16-micro-step inner loop +
16 heads, greedy.
*Gate: per-codebook logits match the dump within tolerance; greedy frames
identical.*

**Step 4 — full decode loop.** The architectural invariant this step
installs is more fundamental than an audio token type: **core generation
produces model-domain ids/state; interpretation — detokenisation included —
belongs to an output adapter.** The present defect is not "LARQL lacks
`AudioToken`"; it is that generated ids are destroyed by unconditional
text detokenisation inside the sampling step and `GenerateResult` retains
only strings. Fix that seam first (ids preserved end-to-end, the text
detokeniser demoted to one adapter among possible others); then a decode
path carrying `(text_token, [u32;16])` per step — with audio EOS on
codebook 0 — is merely the first non-text consumer of the seam, not the
definition of it.
*Gate: full-utterance greedy audio-token sequence identical to the
reference for the canonical fixture — long enough to exercise the KV
cache well past attention-window degeneracies (the short-fixture lesson).*

**Step 5 — streaming.** The 12-token-lead protocol: incremental text in,
frames out, `<|text_pad|>` drain, voice-clone splice in the prefill.
External codec decodes the emitted tokens for listening checks; measure
real TTFA (first frame emitted after 12th text token) — the first genuine
streaming TTFA figure in the whole Jarvis evaluation.
*Gate: token stream from incremental feeding is identical to batch
generation on the same text.* The gate is token-based deliberately: if
audio sounds wrong while the RVQ sequence matches, LARQL has done its job
and the fault is downstream. A listening check (external codec decode of
the emitted tokens; is the aru-12 clone audibly the same speaker as the
reference implementation's output?) is recorded as a separate integration
observation, never as parity evidence.

**Later, in rough order, each gated on the above:** sampled mode (replicate
the non-standard top-p; validate against raw logits, not tokens);
session/turn state (KV continuation across turns — server-side home for a
persistent engine); vindex format extension for auxiliary tensor groups
(PLE-sidecar pattern) so extraction stops refusing the model; realtime
axis (§5); codec-in-LARQL (§6).

---

## 4. The voice-as-data ladder (EXP-V, runs in parallel)

Lives in `chris-experiments/voice/`, not here — it probes the *reference*
implementations, needs no LARQL changes, and can start immediately. Full
plan there; summary of what scoping established (from the installed
`qwen-tts` 0.1.1 package and cached checkpoints):

- Qwen3-TTS Base carries a jointly-checkpointed ECAPA-TDNN speaker encoder
  (mel in, no BatchNorm/Dropout — deterministic forward), whose output
  dimension **is** the talker hidden size: 1024 (0.6B) / 2048 (1.7B), no
  adapter. Per-model encoders; VoiceDesign/CustomVoice ship none.
- `x_vector_only_mode` is real but is an **ICL ablation**, not
  "embedding-only vs something else": the embedding conditions both modes;
  the flag removes reference codes + transcript. Cleaner for EXP-V1 than
  assumed.
- The embedding enters as **one summed sequence position** in the prefill
  (index 7 under the harness defaults) — layer-0 input only, nothing
  re-injects downstream. Layer-k insertion experiments are forward-hook
  work on `talker.model.layers[k]`.
- Extraction is one call (`extract_speaker_embedding`,
  `modeling_qwen3_tts.py:1941`); injection needs no monkeypatching
  (`VoiceClonePromptItem` + the shipped save/load prompt precedent in
  `cli/demo.py:501/526`).
- Consequence for the ladder: same-family transplant is confined to
  same-size checkpoints (1024 vs 2048 are dimensionally incompatible, and
  it's architectural). Cross-model portability starts at zero shared
  space — which is the research question, not a disappointment.
- MOSS contrast worth exploiting: MOSS has **no speaker encoder at all** —
  it conditions on spliced audio tokens, where Qwen3-TTS conditions on a
  pooled vector. Two unrelated serialisations of the same identity
  (aru-12) that both cause a transformer to instantiate the same perceived
  speaker. The endpoint of the ladder is therefore not "which embedding
  format wins" but whether both conditioning mechanisms converge onto an
  equivalent downstream residual state — an `I_voice` that is neither the
  x-vector nor the acoustic prompt, just induced by both. If that object
  exists, *it* is the portable representation, and "voice as data" stops
  being a metaphor.

---

## 5. The realtime axis (design note, gated on §3 step 5)

What K3 doesn't have: a continuous external deadline. The DAC needs a
frame every 80 ms forever; a page fault at the wrong moment is audible.

The objective is not tokens/sec:

```text
minimise    time_to_first_playable_frame
subject to  sustained_generation_rate >= playback_consumption_rate
            no_buffer_underrun
            bounded_memory
```

Runtime shape (all hot-path, no Python, no subprocesses, no WAV files):
inference thread → audio-token queue → codec worker → PCM ring buffer →
audio callback. The callback never waits on the model; it reads
already-prepared PCM from a bounded ring or plays silence. Barge-in
(RFC-0001 §7's 100 ms cancellation budget) maps to: stop reading the ring,
cancel in-flight synthesis — the `WS_CMD_CANCEL` select-loop is the
precedent.

The interesting eventual coupling: the residency scheduler working
backwards from the audio clock ("buffer below 120 ms — promote the next
block now"). That turns predictive residency from a throughput
optimisation into deadline satisfaction. Design only until step 5 emits
real frames; then the jarvis-voice t0–t4 timeline (token commit → commit
latency → TTS TTFA → first playable frame → audible) becomes the
measurement surface, and backlog-vs-drain becomes measurable instead of
inferred from batch RTF.

## 6. Codec-in-LARQL (explicitly deferred)

MOSS-Audio-Tokenizer is a fully causal transformer codec (no conv stacks
in the transform path; strided patching + RoPE transformer stages,
RLFQ quantiser, streaming via ring KV with 10 s context, fp32, ~6.7 GB).
That makes "the whole path in one engine" genuinely plausible — text →
speech LM → RVQ tokens → causal-transformer decode → PCM. It stays out of
scope until §3 is green end-to-end; the external decoder is the oracle the
in-engine one would be judged against.
