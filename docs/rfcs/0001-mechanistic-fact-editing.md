# RFC-0001: Mechanistic Fact Editing Commands (`crown`, `edit`, `memit`)

**Status:** Draft
**Author:** Divinci AI team
**Created:** 2026-04-17
**Related research:** Divinci-AI/server notebooks/CHAPTER_15_GHOST_TRANSPLANT.md through CHAPTER_23_PER_EDIT_CROWN.md

## Summary

Extend LarQL from a **weight-analysis tool** (vindex extraction, QK-rank, OV-gate) into a **weight-analysis-and-editing tool** by adding three new subcommands: `crown`, `edit`, and `memit`. These commands bring mechanistic-interpretability-native fact editing (ROME / MEMIT family algorithms) to LarQL's existing Rust CLI surface, using the model loaders and inference infrastructure already in `larql-inference/`.

## Motivation

Nine chapters of mechanistic experiments on Gemma 4 (4B and 26B variants) in 2026 established:

1. **Entity zone** lives at L20-29 (~67-97% depth). Factual recall commits via an identifiable "crown" MLP whose ablation breaks the retrieval (Chapter 17, Phase 125c).
2. **L27 MLP** on Gemma 4 4B is the load-bearing country→capital lookup module — ablating it causes "Paris" to flip to "France" (the country).
3. **Single-fact editing works**: a rank-1 update with `ΔW = d ⊗ k / (k·k)` and `d = 2·(o_target − o_source)` produces 11/11 specificity at ~0.9% relative weight perturbation (Chapter 20, Phase 140).
4. **Multi-fact editing needs MEMIT**: naive stacking of rank-1 updates fails catastrophically (Chapter 21); joint least-squares solves the constraint cleanly (Phase 141c).
5. **Per-edit crown discovery**: different facts have different crown layers (L23 for UK/Poland, L27 for France/Germany/Russia) — editing requires a layer-selection audit per fact (Chapter 23, Phase 143).

These capabilities are the technical kernel of a **fact-editing / unlearning product**. LarQL is the natural home: it already has GGUF loading, the inference forward pass, and vindex-based static analysis. Adding editing is a natural next layer.

## User experience

```bash
# 1. Find the crown layer for a given edit
larql crown \
  --model /path/to/gemma4 \
  --prompt "Capital of France? A:" \
  --expect " Paris"
# Output: { "crown_layer": 27, "layer_type": "mlp", "delta": -14.19, "top_after_ablation": "France" }

# 2. Apply a single rank-1 fact edit (writes a patch file)
larql edit \
  --model /path/to/gemma4 \
  --prompt "Capital of France? A:" \
  --old " Paris" \
  --new " Tokyo" \
  --auto-scale \
  --output france_to_tokyo.patch
# Output: patch file with (layer, d, k/(k·k)) — applied via larql apply-patch

# 3. Batch-edit via MEMIT (grouped by per-edit crown)
cat edits.json
# [
#   {"prompt": "Capital of France? A:", "old": " Paris", "new": " Tokyo"},
#   {"prompt": "Capital of Germany? A:", "old": " Berlin", "new": " Rome"},
#   ...
# ]
larql memit \
  --model /path/to/gemma4 \
  --edits edits.json \
  --output patches/ \
  --validate-specificity 50
# Output: per-layer patches + validation report on 50 held-out facts

# 4. Apply patches at inference time (non-destructive)
larql apply-patch \
  --model /path/to/gemma4 \
  --patch france_to_tokyo.patch \
  --prompt "Capital of France? A:"
# Output: " Tokyo"
```

## Design

### `larql crown`

**Input:** model dir, prompt, expected-token string.

**Algorithm (Phase 125c):**
1. Run baseline forward pass, record `baseline_logit[expect]`.
2. For each layer L ∈ `crown_scan_range` (default L18-L{N-2}):
   - Zero out L's MLP output at the last-token position via a forward hook
   - Re-run forward, record `ablated_logit[expect]`
   - Compute `delta = ablated - baseline` and `top_token_after`
3. Select layer with minimum delta that also flips top_token (first tier); if no flip, select layer with minimum delta (second tier).

**Output (JSON):**
```json
{
  "crown_layer": 27,
  "layer_type": "mlp",
  "delta_expect": -14.19,
  "top_after_ablation": "France",
  "scan": [{"layer": L, "delta": d, "top": t}, ...]
}
```

**Rust integration:** Uses `larql-inference/capture.rs` hook mechanism (already present) + `larql-inference/forward/` for the ablation forward pass.

### `larql edit`

**Input:** model dir, source prompt, old-token, new-token, optional scale / auto-scale.

**Algorithm (Phase 140 + 130):**
1. Run `crown` internally → `L` (or use `--layer` override).
2. Run a second prompt with the NEW token substituted (synthesizing the target-output by swapping source-capital → target-capital in the same template) → capture `o_target` at L.
3. Capture `k` = L's MLP intermediate on source prompt; `o_source` = L's MLP output on source prompt.
4. If `--auto-scale`: binary-search scale `s ∈ [0.5, 5.0]` such that `W_edited @ k = W @ k + s(o_target - o_source)` flips the source prompt's top token to new-token. Cache on first hit.
5. Compute `ΔW = s(o_target - o_source) ⊗ k / (k·k)`.
6. Write patch file: layer index, `d = s(o_target - o_source)`, `k/(k·k)`.

**Output:** LarQL patch file format (new — see "Patch format" below).

### `larql memit`

**Input:** model dir, edits.json (list of {prompt, old, new} entries), output dir.

**Algorithm (Phase 141c + 143b):**
1. For each edit, run `crown` internally → group edits by crown layer.
2. Per group at layer L:
   - Stack keys K = [k₁, k₂, ...] (from each edit's prompt)
   - Stack deltas D = [d₁, d₂, ...] (each with auto-scale search)
   - Solve `ΔW_L = D (K^T K)^{-1} K^T` with Tikhonov regularization ε=1e-4
3. Write one patch per layer; write a manifest.json listing all patches.
4. If `--validate-specificity N` given, run N random held-out facts and report preserve rate.

**Output:** `patches/<layer>.patch` files + `manifest.json` + `validation.json`.

### Patch file format (new)

Binary + JSON pair:
- `patches/<name>.meta.json`:
  ```json
  {
    "version": 1,
    "layer": 27,
    "module": "down_proj",
    "rank": 1,
    "d_shape": [1536],
    "k_shape": [12288],
    "d_file": "france_to_tokyo.d.bin",
    "k_file": "france_to_tokyo.k.bin",
    "dtype": "bfloat16",
    "provenance": {
      "source_prompt": "...",
      "old_token": " Paris",
      "new_token": " Tokyo",
      "scale": 2.25,
      "crown_delta": -14.19
    }
  }
  ```
- `france_to_tokyo.d.bin`: raw bytes, hidden_size × dtype
- `france_to_tokyo.k.bin`: raw bytes, intermediate_size × dtype (already normalized by k·k)

For rank>1 (MEMIT output), store as matrices with matching rank.

Total footprint: ~55KB for a Gemma 4 4B single edit in bf16.

### `larql apply-patch`

**Input:** model dir, patch file(s), optional prompt for immediate test.

**Behavior:** At load time, read patch files; modify `down_proj.weight` in memory only (no disk write). Optionally run inference on the given prompt.

**Reversibility:** Patches are additive. `apply-patch --reverse` undoes by subtracting.

## Out of scope for this RFC

- Attention head editing (would require extending the forward pass capture beyond MLP outputs)
- Edits targeting residual stream directly (vs MLP output)
- Cross-language edit consistency (future work: Chapter 20 open thread Phase 143)
- Full gradient-based ROME objective (we use the simpler activation-delta approximation)

## Product implications

This RFC turns LarQL into the **first mechanistic-interpretability-native fact-editing CLI**. Competing tools (EasyEdit, FastEdit, etc.) exist as Python libraries; LarQL's advantage is:

- **Static + dynamic triangulation**: LarQL's existing vindex analysis (Phase 117 / 120) can *predict* which facts will be editable cheaply
- **Rust speed + GGUF loading**: edits apply to ollama-compatible quantized models, not just HF safetensors
- **Self-calibrating pipeline**: `auto-scale` + per-edit crown discovery deliver reliable single-edit specificity without requiring ML expertise
- **Patch-based deployment**: patches are small (~55KB), reversible, stackable — fitting enterprise MLOps workflows

## Dependencies / prerequisites

- `larql-inference` already has the forward pass + capture hooks — verify it supports zeroing MLP output at a specific layer × position.
- `larql-models` supports GGUF loading with column-major fix (landed in 71897b7) — supports Gemma 4.
- Need to extend `larql-python` bindings so experiments can be scripted without writing Rust.
- Add a small `larql-edit/` crate for the rank-1 / MEMIT math (bf16 outer products, small-matrix inverses).

## Phased rollout

**Phase A (1-2 weeks):** `larql crown` — builds on existing inference capture. Smallest new code.
**Phase B (2-3 weeks):** `larql edit` single-fact — adds rank-1 math, patch file format, `apply-patch`.
**Phase C (3-4 weeks):** `larql memit` multi-fact — grouped MEMIT with validation.
**Phase D (2 weeks):** Python bindings for scripted experiments (`larql-python` crate extension).

## Open questions

1. **Patch file format**: should we use safetensors-compatible binary to leverage existing tooling?
2. **Crown scan range**: hardcode L18-L{N-2} or parameterize per-architecture?
3. **Attention-capture editing**: L29 attention was identified as a significant writer (Phase 125c). Should `edit`/`memit` support attention edits in v2?
4. **Vindex integration**: should `crown` prefer reading the vindex's `down_meta.bin` static features over running a live forward pass (faster but less accurate)?

## Appendix — Reproducibility

All findings referenced above are reproducible from:
- **Colab notebooks**: runtime artifacts `/content/phase125c_layer_module_ablation.json`, `/content/phase140c_rome_2x.json`, `/content/phase141c_memit.json`, `/content/phase143a_per_edit_crowns.json`
- **Chapters**: `notebooks/CHAPTER_15_GHOST_TRANSPLANT.md` through `CHAPTER_23_PER_EDIT_CROWN.md` in the Divinci-AI/server repo

Target model for initial implementation: `google/gemma-4-e2b-it` (4B dense, matches the published LarQL vindex at `Divinci-AI/gemma-4-4b-e2b-vindex`).
