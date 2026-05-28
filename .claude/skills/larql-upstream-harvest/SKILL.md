---
name: larql-upstream-harvest
description: Bring `chrishayuk/larql` upstream changes into the `Divinci-AI/larql` fork. Use when the user says "harvest upstream", "sync with chrishayuk", "merge upstream", "catch up with upstream", "pull from chrishayuk", "RFC-000X harvest", or asks to compare fork-vs-upstream. Covers the survey-first triage, the cherry-pick-vs-reset-and-replay decision tree, our specific Commands/DevCommand conflict patterns, fp8-block-quant preservation, and the RFC-as-living-document convention. Captures lessons from the 2026-05-28 harvest (RFC-0002).
---

# LARQL Upstream Harvest

Bring `chrishayuk/larql` upstream changes into `Divinci-AI/larql`. This skill encodes the playbook from RFC-0002 (2026-05-28 harvest): survey before planning, default to reset-and-replay when structural drift is large, and keep fork divergence small.

## Step 0 — Survey before planning

**Always run these before proposing a wave plan.** They reveal whether the harvest is surgical or structural.

```bash
# Are we up to date with the upstream remote?
git fetch upstream --tags

# How much drift?
git log --oneline upstream/main ^main | wc -l   # commits behind
git log --oneline main ^upstream/main | wc -l   # commits ahead
git merge-base upstream/main main                # divergence point

# What's structurally changed? (The load-bearing signal — directory restructures
# break surgical cherry-picks. Compare directory layouts at the merge-base vs
# upstream HEAD.)
git diff --name-only $(git merge-base upstream/main main) upstream/main \
  | xargs -n1 dirname | sort -u | head -30

# What fork commits are genuinely fork-only (not dupes of upstream PRs)?
git log --oneline main ^upstream/main
```

**Red flags that mean reset-and-replay, not cherry-pick:**
- Upstream split a file into a directory (e.g. `gguf.rs` → `gguf/`, `detect.rs` → `detect/`, `extract/streaming.rs` → `extract/streaming/`).
- Upstream restructured the CLI surface (top-level `Commands` enum reshuffled).
- More than ~20 fork commits are "verbatim ports" of upstream PRs (these dedupe naturally under reset).
- Big-bang `git merge upstream/main` produces > 100 conflicting files.

## Strategy decision tree

```
Survey shows:                          → Approach:
─────────────────────────────────────────────────────────────────────────
< 30 upstream commits, no restructure  → Surgical cherry-picks per wave
< 30 commits, restructure              → Cherry-pick the restructure PR
                                         FIRST, then dependent fixes
> 100 commits, no restructure          → Real merge of upstream/main
                                         (rare; structural drift is the
                                         usual reason for big drift)
> 100 commits, restructure (typical)   → RESET-AND-REPLAY (default)
```

## Reset-and-replay procedure (the default)

This is the strategy that landed Wave 2 of the 2026-05-28 harvest. It trades
"preserve fork merge history" for "each conflict is bounded to one commit's
scope" — which is far more tractable.

```bash
# 1. Identify genuinely fork-only commits (skip PR dupes already in upstream).
#    Check each commit's content against upstream — verbatim ports and
#    independently-converged work should be skipped.
git log --oneline main ^upstream/main

# 2. Branch off main, reset to upstream HEAD.
git checkout -b feat/upstream-harvest-<date>
git reset --hard upstream/main

# 3. Replay fork-only commits in chronological order.
for sha in <fork-only-shas-oldest-first>; do
  git cherry-pick "$sha" || break    # resolve conflicts, then --continue
done

# 4. Build + test at each milestone, not just at the end.
cargo check --workspace
cargo test --package larql-models --lib
cargo test --package larql-vindex --lib
cargo test --package larql-inference --lib

# 5. Merge to main — destructive reset required because main diverged.
#    NEVER auto-push; flag this to the user.
git checkout main
git reset --hard feat/upstream-harvest-<date>
```

**`git reset --hard` on main is destructive and rewrites local history.** It requires
`--force-with-lease` to push. ALWAYS flag this to the user before doing the reset
on `main` itself; do the reset on a wave branch first and propose the main reset
explicitly.

## Conflict patterns specific to this fork

### Pattern 1 — `Commands` enum vs `DevCommand`

Upstream moved research/interpretability commands from top-level `Commands` into
`Dev(DevCommand)` subcommand group. Our fork's RFC-0001 commands (`crown`,
`edit`, `apply-patch`, `memit`) need to land in `DevCommand`, not `Commands`.

**Where to add:** `crates/larql-cli/src/main.rs`
1. Add the variant to `enum DevCommand { ... }` (the cherry-pick usually does this automatically).
2. Add the dispatch arm to `fn run_dev()` (often missed — the cherry-pick targets the old `Commands` match in `real_main`).
3. The `rewrite_legacy_argv` trampoline preserves the old `larql crown ...` invocation.

### Pattern 2 — `mod foo;` declarations conflict

When both sides add a module to the same `pub mod` list, the conflict is
mechanical. Keep BOTH sets of declarations, alphabetize. Common files:
- `crates/larql-cli/src/commands/extraction/mod.rs`
- `crates/larql-inference/src/ffn/mod.rs`
- `crates/larql-inference/src/lib.rs` (also has `pub use` re-exports)

### Pattern 3 — fork-only file modified, upstream-deleted

Upstream renames a file (e.g. `detect.rs` → `detect/mod.rs`) and the
cherry-pick wants to modify the old path. Resolution:
1. `git rm` the stale file.
2. Find the equivalent location in the new directory structure (`detect/parser.rs`, `detect/mod.rs`, etc.).
3. Manually port the fork change. **Verify the change isn't already in
   upstream** (independently-converged work — common for Gemma-4 fixes,
   `LARQL_API_KEY` env var, etc.).

### Pattern 4 — fp8-block-quant decode preservation

The fp8-block-quant decode (`dfd9fc9a`) is the highest-risk port. It lives in:
- `crates/larql-models/src/loading/safetensors.rs`: `dequantize_fp8_block_companions` + `decode_f8_e4m3` (must be `pub`).
- `crates/larql-vindex/src/extract/streaming/tensor_io.rs::get_tensor_f32`: parallel detection branch alongside the existing MXFP4 branch (`F8_E4M3` + `.weight_scale_inv` companion).
- `crates/larql-models/src/detect/mod.rs`: `"kimi_k2"` arm mapping to `DeepSeekArch`.

If upstream restructures these files again, port the fp8 logic by analogy with
the adjacent MXFP4 detection — same shape, different dtype companion.

**Validation: `cargo check` is necessary but NOT sufficient.** Test against a
real Kimi-K2 or DS-V3 fp8 vindex extract before declaring it done.

## What to SKIP during a harvest

Fork commits that are already in upstream (either as a PR we mirrored, or as
independently-converged work):
- `LARQL_API_KEY` env var (now in upstream's `larql-server/src/main.rs` with `#[arg(long, env = "LARQL_API_KEY")]`)
- Gemma-4 per-layer `intermediate_size` (upstream's `config.rs::intermediate_size_for_layer` + `use_double_wide_mlp`)
- `rebuild_overrides` regression test (upstream's `overlay_apply.rs`)
- May-4-style verbatim ports of upstream PRs `#74-#81` (MXFP4, F8 dtypes, DeepSeekV4Arch, metadata-only resolve, HF model-repo fallback, MoE SVD, down_meta cap)

When in doubt: `grep` upstream for the function name or symbol the fork
commit added. If it's already there, skip.

## Validation checklist (before merging to main)

1. `cargo check --workspace` clean (warnings OK, errors not).
2. `cargo test --package larql-models --lib` green.
3. `cargo test --package larql-vindex --lib` green.
4. `cargo test --package larql-inference --lib` green.
5. CLI smoke test: `larql --help`, `larql dev crown --help` (legacy alias still works).
6. PyO3: `cd crates/larql-python && uv run --no-sync maturin develop --release && uv run --no-sync pytest tests/`.
7. If fp8-block-quant touched: run a Kimi-K2 or DS-V3 fp8 extract.
8. Note any **force-push requirement** for `origin/main` — flag to user, don't auto-push.

## Post-harvest hygiene

After every harvest, the fork's job is to **stay small**:

1. **Upstream the fp8-block-quant work to `chrishayuk/larql`** if it isn't already a PR. Carrying it through another harvest is the same manual port work. Submit while the integration is fresh in memory.
2. **Retire verbatim ports.** If we ported an upstream PR ahead of merge (May 4 pattern), open a follow-up to delete our copy once the upstream PR lands and our merge-base advances past it. Verbatim ports create false divergence and inflate harvest cost.
3. **Update RFC inventory.** RFC-0001 is the fork's identity (crown/edit/memit/apply-patch). RFC-000N harvest RFCs document each catch-up. Add a new RFC for each substantive harvest with a "Wave execution log" section appended as work progresses.

## RFC-as-living-document convention

Each major harvest writes an RFC at `docs/rfcs/000N-upstream-harvest-<date>.md`.
The pattern (from RFC-0002):

1. **Summary + Motivation + Inventory** — written upfront after the survey.
2. **Wave-by-wave plan** — what cherry-picks or merges go in each batch.
3. **Conflict-mitigation strategy** — predicted conflict zones, resolution rules.
4. **Out of scope + Success criteria** — what's NOT in this harvest.
5. **Execution log (appended as work progresses)** — what actually landed,
   what was skipped and why, what manual ports were needed. This section is
   the highest-value retrospective surface; future harvests learn from it.

Pivots are normal. **Append the pivot reasoning to the execution log rather
than rewriting the plan** — the original plan + the pivot is more useful than
a clean retroactive plan.

## Carry-forward fork-only work (as of 2026-05-28)

These are the ~11 commits that define the fork on top of upstream:

| Area | Notes |
|------|-------|
| RFC-0001 surface | `crown` / `edit` / `apply-patch` / `memit` CLI in `DevCommand` group |
| PyO3 bindings | `crates/larql-python/src/edit_py.rs` + `lib.rs` registrations |
| fp8-block-quant decode | Kimi-K2 / DS-V3 fp8 — top candidate for upstreaming |
| README badges | Divinci-AI fork header |
| Isolation-harness CI | `.github/workflows/harness.yml` + `testdata/tiny-vindex/` |
| RFC-000N documents | `docs/rfcs/` harvest plans + execution logs |

If a future harvest touches any of these, this skill applies. If it doesn't,
this skill probably isn't needed.

## Meta — when this skill works and when it doesn't

**Works**: fork divergence is small (~10-30 product commits), upstream has done
structural refactors, most fork commits are dupes of upstream PRs. The
2026-05-28 harvest matched this profile exactly.

**Doesn't work**: fork has deeply forked from upstream (100+ divergent product
commits, fundamentally different architecture). At that point, the fork has
become a different project and harvest-style sync is the wrong frame — consider
a one-way "cherry from upstream when useful" model instead.
