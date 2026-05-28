# RFC-0002: Upstream Harvest — chrishayuk/larql → Divinci-AI/larql (2026-05-28)

**Status:** Proposed
**Author:** Divinci AI team
**Created:** 2026-05-28
**Related:** prior harvest commit `c037edf2` (2026-05-04, upstream `ae933753`)
**Upstream head at writing:** `b6d5e8d5` (Merge `#148` — BitNet 1.58)

## Summary

Bring 24 days of upstream activity on `chrishayuk/larql` (May 4 → May 28, 2026) into `Divinci-AI/larql` in four risk-graded waves. The harvest captures **111 net new upstream commits** beyond what our May 4 port already covered, spanning new quant decoders (Q3_K, Q5_K, BitNet 1.58), a substantial GGUF stack overhaul (modular split, multi-shard, streaming extract), MLA absorption for DeepSeek-V3 / Kimi-K2, multi-modal vision Phase 1+2, a KV/compute refactor, and a long tail of correctness fixes and cross-platform CI work. The fork retains its RFC-0001 fact-editing surface (`crown`/`edit`/`memit`/`apply-patch`), fp8-block-quant decode, and harvest checkpoint modules — none of those are touched by upstream.

## Motivation

1. **License risk** is acute: upstream pinned `evalexpr` to v11.3.1 (MIT) because v12+ became AGPL-3.0. We currently follow the unpinned crate and are at risk of pulling AGPL into our release artifacts. This is the single highest-priority item in the harvest.
2. **Correctness fixes** upstream target hot paths we use in production: Gemma-4 PLE sidecar validation, remote-FFN norms across every decode-loop branch, MoE empty-q4 dense-FFN guard, signed-probe thresholding, `lm_head` 32-bit vocab overflow, router `matmul_transb`. We are otherwise running on buggy code.
3. **Composes with our fork-only work**: Q3_K + Q5_K dequant complements our fp8-block-quant; MLA absorption + asymmetric GQA + DS-V3 metadata directly extend the model families our fp8 work was built for; GGUF streaming + multi-shard reader paves the way for the Kimi-K2 sized models our harvest checkpoint was designed to tolerate.
4. **Our recent PRs (#74–#81-equivalent) already landed upstream**, so most of our May 4 → May 28 fork commits will dedupe under a real merge rather than conflict. The window to harvest cleanly is now, before our and upstream's `extract/` paths diverge further.
5. **Strategic alignment**: the fork was built on the premise that we'd track upstream closely while layering RFC-0001 on top. Letting drift grow makes that contract harder to honor.

## Inventory of upstream changes since `ae933753`

### A. Model / quant support
- BitNet 1.58 ternary (`TQ1_0` / `TQ2_0` / `I2_S` decoders) — PR `#148` (just merged)
- Q3_K + Q5_K dequantization (ggml types 11 + 13) — PR `#103`
- MLA absorption — fuse DS-V3 low-rank Q/K/V into dense weight matrices (`2a1fc079`, `2d10daa8`)
- MLA metadata surfacing (DeepSeek-V2/V3, Kimi-K2) — `#67`
- Asymmetric GQA (`gqa_attention_asym`) for MLA-absorbed head dims (`d93797fe`)

### B. GGUF stack
- Multi-shard reader for `*-NNNNN-of-NNNNN.gguf` (`575963d5`)
- Modular `gguf/` directory split (consolidation PR `#145`, `c54875db`)
- `GgufTensorInfo` accessors exposed for streaming consumers (`80bd96b6`)
- Streaming GGUF in extract pipeline at browse-level (`fd6f0b43`)
- Per-layer `down_meta` incremental flush (`5294a3b2`)
- MoE-only `expert_feed_forward_length` fallback (`8c816dca`)
- Accept GGUF input (file or directory) in extract — PR `#133`

### C. Inference correctness fixes
- Gemma-4 PLE sidecars write + validate on `--quant none` — PR `#121`/`#125`
- Remote-FFN pre/post norms across **every** decode-loop dispatch branch — PR `#126`
- `--metal` flag wired into remote-FFN path + post-FFN norms — PR `#122`
- Signed probe: signed thresholding, full-depth scan, combined-index matching — PR `#150`
- MoE shards: empty q4 dense FFN guard + `--metal` wiring — PR `#152` (closes `#151`)
- `lm_head` vocab overflow on 32-bit hosts — PR `#100`
- Router uses `matmul_transb` for MoE expert scoring (`bcd63808`)

### D. Refactors
- KV engine retrieval trait split — PR `#142` (paves "mode 5")
- Compute refactor: metal decouple, modular KV engines, `larql-compute-metal` mega-split — PRs `#109`/`#120`
- KV-engine accuracy scoring (`ScoreOutcome` variants) — PRs `#140`/`#141`

### E. Multi-modal (vision)
- Phase 1: cross-architecture vision captioning — PR `#143`
- Phase 2: Granite Vision protocol, MLP connector, AnyRes tiler — PR `#144`
- `connectors/projector.rs` (renamed from prior location)

### F. Platform / build / infra
- **`evalexpr` pinned to v11.3.1** (MIT, not v12 AGPL-3.0) — PR `#92`
- HF cache scan recognises model-repo pulls — PR `#93`
- Windows: BLAS single-threaded, async attention K/V gate, six platform-specific test gates — PRs `#113`, `#87`, `#90`, `#124`, `#127`
- FreeBSD: OpenBLAS deps — PR `#58`
- Android aarch64 cross-compile — PR `#99`
- Nix flake — PR `#34`
- Swagger UI / OpenAPI at `/v1/openapi.json` — PR `#47`
- Dependabot config — PR `#101`
- Coverage backfill on hot streaming/HF/loading paths — PR `#84`

### G. Misc
- `shannon-layers` subcommand for per-layer bit measurement (`61c36629`)
- `extract/build.rs` restoration (`#46`)
- bench-regress CI routing to new crate homes (`#132`)

## Fork-only work that upstream does NOT have (do not lose during merge)

| Area | Fork commit | Notes |
|------|-------------|-------|
| fp8 block-quant decode (Kimi-K2 / DS-V3) | `dfd9fc9a` | Genuine fork-divergent; candidate for upstreaming |
| Harvest checkpoint + metadata + stage_labels | `c037edf2` | Verbatim port from upstream `ae933753`; candidate for upstreaming |
| RFC-0001 surface: `crown` / `edit` / `apply-patch` / `memit` | `2324af46`, `7c597f80`, `ed369cbf` | Core fork product |
| PyO3 bindings for the editing surface | `186019ca` | RFC-0001 Phase D |
| Per-layer `intermediate_size` for Gemma-4 double-wide MLP | `44d549bc` | PR `#10` |
| `LARQL_API_KEY` env-var for cloud-run secrets | `758a0523` | PR `#15` |
| Isolation-harness CI gates | `3266558f` | PR `#12` |

Naturally-deduping fork commits (already in upstream main as `#74–#81` etc): the eight MXFP4 / F8 dtype / DeepSeekV4Arch / metadata-resolve / HF model-repo / SVD-summary / down_meta-cap commits will collapse during merge. Expect zero conflicts on those.

## Design — Wave-by-wave plan

Branch convention: `feat/upstream-harvest-2026-05-28-wave{N}` off `main`, merged sequentially. Each wave produces a single squash-merge or merge-commit PR so reverts stay surgical.

### Wave 1 — License + correctness fixes (target: 1 evening)

**Branch:** `feat/upstream-harvest-2026-05-28-wave1`
**Style:** cherry-pick from upstream main. Each commit lands as a separate cherry-pick to preserve attribution.

Order (license first, then composable fixes, then additive features):

1. `fe25575c` — `fix(license): pin evalexpr to v11.3.1` ← do this first
2. `4e4f7b29` + `a4ea55f1` — HF cache scan recognises model repos (`#93`)
3. `716355fb` + `ae35058b` — Gemma-4 PLE sidecars validation (`#121`)
4. `83345ad5` — Apply PLE + layer_scalar in cached prefill/decode (`#125`)
5. `13b380a3` + `5ab2d078` — Remote-FFN norms + Metal flag (`#122`)
6. `4b8ac8e1` — Remote-FFN norms across every decode-loop branch (`#126`)
7. `834d0659` — `lm_head` 32-bit vocab overflow (`#100`)
8. `bcd63808` — Router `matmul_transb` for MoE expert scoring
9. `32f78fe2` — Signed-probe fix (`#150`)
10. `49403543` — MoE shards empty-q4 dense FFN guard + `--metal` (`#152`)
11. `f2a4c348` — Q3_K + Q5_K dequant (`#103`)
12. `58c849fa` — Accept GGUF input in extract (`#133`)

**Files touched (anticipated):** `Cargo.toml` / `Cargo.lock`, `crates/larql-cli/src/commands/`, `crates/larql-inference/src/`, `crates/larql-vindex/src/`, `crates/larql-models/src/quant/`. Avoids `crates/larql-vindex/src/extract/` heavy paths — those wait for Wave 2.

**Risk:** Low. None of these intersect RFC-0001 or fp8-block-quant. Expect cherry-picks to apply clean.

**Validation:** `make ci`. Spot-check: load a Gemma-4 vindex, run `larql infer` with `--metal`, confirm no panic; load a Q3_K GGUF, run `larql extract-index --browse`.

### Wave 2 — GGUF stack + MLA (target: 1–2 days)

**Branch:** `feat/upstream-harvest-2026-05-28-wave2`
**Style:** real merge of upstream's GGUF-related PRs as a coherent unit, then rebase fork-only `extract/` modules on top.

Order:

1. Merge upstream commits up through `c54875db` (modular `gguf/` directory consolidation PR `#145`).
2. Resolve the expected conflict: our `extract/checkpoint.rs`, `extract/metadata.rs`, `extract/stage_labels.rs`, the MXFP4 streaming gate-vectors path, and the fp8-block-quant decode path must be re-pointed at the new module layout.
3. Pull `9b56cd2e` + `2a1fc079` (MLA `qk_nope`/`rope`/`v_head_dim` + absorption) and `d93797fe` (asymmetric GQA) — these compose with our fp8-block-quant decode.
4. Pull `8f1c8f3f` + `45f473a4` (GGUF MLA metadata surfacing for DS-V2/V3/Kimi-K2).
5. Pull `d0b915b9` (map `deepseek_v4` / `deepseekv4` GGUF arch string to our `DeepSeekV4Arch`).

**Conflict zones (predicted):**
- `crates/larql-vindex/src/extract/streaming.rs` — our MoE SVD summary, down_meta cap, and stage_labels port all live here.
- `crates/larql-vindex/src/extract/checkpoint.rs` — fork-only, must survive untouched.
- `crates/larql-vindex/src/gguf*` — upstream split this file; our fp8-block-quant code reads from it.
- `crates/larql-models/src/arch/` — our `DeepSeekV4Arch` may collide with upstream's `deepseek_v4` arch string mapping.

**Risk:** Medium. Plan for ~half a day of conflict resolution. The fp8-block-quant path is the riskiest because it touches GGUF decoding directly.

**Validation:** `make ci` + load both a multi-shard GGUF (e.g. Kimi-K2) and a DS-V3 GGUF and confirm `extract-index --browse` resumes correctly from our harvest checkpoint.

### Wave 3 — KV / compute refactor (target: 2–3 days, gated)

**Branch:** `feat/upstream-harvest-2026-05-28-wave3`
**Trigger:** only execute if we have a concrete inference task scheduled in the next sprint. Otherwise defer.

Order:

1. Merge upstream up through `66c825a7` (KV engine reworked for mode 5).
2. Merge through `93c4ec58` (compute refactor consolidation, PR `#120`).
3. Pull KV-engine accuracy scoring (`52e4e0c3`, `3202067b`).

**Conflict zones:**
- `crates/larql-inference/src/` — broad surface change. Our RFC-0001 commands hook into inference forward-pass capture; expect to re-validate `crown` and `edit` end-to-end after this lands.
- `crates/larql-compute/`, `crates/larql-compute-metal/` (the latter may be new from the split).

**Risk:** High surface area, low logical risk. The refactor is a structure-preserving split; behavior should be unchanged, but our editing hooks may need to be re-pointed.

**Validation:** Run the RFC-0001 reproducibility suite (`crown` on Gemma-4 4B + `edit` for France→Tokyo + `apply-patch` test) before and after; results should be bit-identical.

### Wave 4 — Optional / opt-in (target: as time permits)

These are independent of one another; pick any subset:

- **Nix flake** (PR `#34`) — adds reproducible builds; useful for CI parity but doesn't block any current work.
- **Swagger UI / OpenAPI** for `larql-server` (PR `#47`) — only valuable if we're going to publish the server API externally.
- **Dependabot config** (PR `#101`) — accept the auto-PR load or skip.
- **Android aarch64 cross-compile** (PR `#99`) — only if mobile deployment is on the roadmap.
- **`shannon-layers` subcommand** (`61c36629`) — small, additive; pull if curious about per-layer bit measurements.
- **BitNet 1.58 ternary** (`#148`) — pull when we have a BitNet model to test against. Until then, the decoder is dead code.
- **Multi-modal Phase 1+2** (`#143`/`#144`) — significant new surface (`connectors/projector.rs`, Granite Vision protocol). Skip unless vision is on the product roadmap; RFC-0001 has no vision dependency.

**Risk:** Each item independently low. The risk is opportunity-cost — Wave 4 is where harvest fatigue causes us to merge things we don't actually need.

## Conflict mitigation strategy

For Wave 2 (the only wave with predicted conflicts):

1. Before merging upstream, snapshot our fork-only `extract/` modules to a scratch branch as a backstop.
2. Use `git merge upstream/main -X theirs` only on the GGUF subdirectory; resolve fork-specific paths manually.
3. After merge, re-run our existing extract tests against:
   - A Gemma-4 4B GGUF (validates our PLE + per-layer intermediate_size handling)
   - A DeepSeek-V3 fp8 safetensors (validates fp8-block-quant survived)
   - A Kimi-K2 multi-shard GGUF (validates the new multi-shard reader composes with our harvest checkpoint)

## Out of scope for this RFC

- **Upstreaming our fork-only work to chrishayuk/larql**. The fp8-block-quant decode and harvest checkpoint modules are good candidates, but submitting them is a separate workstream and shouldn't block this harvest.
- **CI parity**: upstream and our CI configurations have diverged independently. Reconciling them is its own RFC.
- **RFC-0001 surface expansion**: attention-head editing, residual-stream editing, gradient-based ROME (already deferred in RFC-0001).
- **Vendor lock-in audit**: the `evalexpr` pin is reactive; a proactive license audit of all transitive deps is out of scope here.

## Success criteria

- `make ci` passes on `main` after all merged waves.
- `evalexpr` v11.x is in `Cargo.lock` (no v12).
- RFC-0001 reproducibility (`crown` + `edit` + `apply-patch` on Gemma-4 4B France→Tokyo) produces bit-identical results before vs after harvest.
- A Kimi-K2 multi-shard GGUF resumes correctly from our harvest checkpoint after Wave 2.
- No regressions in the May 4 `extract/` ports (`checkpoint.rs`, `metadata.rs`, `stage_labels.rs`).

## Open questions

1. **When do we upstream fp8-block-quant?** It's been in our fork since `dfd9fc9a` and composes cleanly with upstream's MLA absorption. Sending a PR to chrishayuk now would reduce future divergence.
2. **Do we want Wave 3 at all?** The KV/compute refactor is structurally clean but expensive to absorb. If our roadmap is RFC-0001-heavy and inference-light for the next month, deferring Wave 3 saves a week.
3. **Multi-modal — yes or never?** Vision support is a real product question, not a harvest question. If the answer is "never," we can mark `#143`/`#144` as permanent skip and free ourselves from re-evaluating it each harvest.
4. **Squash vs merge commits per wave?** RFC-0001 used squash for the four phases. A merge commit per wave keeps upstream attribution intact and makes future harvests cheaper to plan; recommend merge commits here.

## Appendix — Useful commands

```bash
# What's still ahead of us after each wave
git log --oneline upstream/main ^main | wc -l

# What's fork-only (sanity check before merge)
git log --oneline main ^upstream/main

# Inspect a single upstream commit before cherry-pick
git show <sha> --stat

# Wave 1 cherry-pick template
git checkout -b feat/upstream-harvest-2026-05-28-wave1
for sha in fe25575c 4e4f7b29 a4ea55f1 716355fb ae35058b 83345ad5 \
           13b380a3 5ab2d078 4b8ac8e1 834d0659 bcd63808 32f78fe2 \
           49403543 f2a4c348 58c849fa; do
  git cherry-pick "$sha" || break
done
```

## Appendix — Prior-art reference

The May 4 harvest (`c037edf2`) used verbatim file ports rather than cherry-picks because the three modules (`checkpoint.rs`, `metadata.rs`, `stage_labels.rs`) were small, self-contained, and easy to attribute. **This harvest is large enough that cherry-pick (Wave 1) + real merge (Wave 2/3) is the right shape** — verbatim ports would lose upstream attribution and make the next harvest harder to plan.
