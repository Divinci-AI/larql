# Republishing models — the 2026-08 recovery, and why it was manual

**Publishing belongs to the Vindex Factory.** `larql recipe build` already
runs PREFLIGHT → RELEASE: fetch a pinned revision, extract, slice each
declared output, verify checksums, publish **private**, then flip to public
only after verification passes. That is
[`docs/vindex-factory.md`](vindex-factory.md) §7–§8, and it is a stronger
pipeline than anything done by hand.

This document is not an alternative process. It records a republish that
**could not use the factory**, what that cost, and what needs to exist so the
next one does.

---

## 1. Why this republish was manual

PR #207/#212 fixed the ggml nibble layout, which invalidated every vindex
written before 2026-08-07 that stores Q4_0 or Q6_K. Five models needed
re-extracting and republishing.

**No recipe exists for any of them**, and `chuk-vindex-recipes` — the data
plane, per [`vindex-factory.md`](vindex-factory.md) §3 — is not checked out
on this machine. `larql recipe build` had nothing to run, so the work went
through `larql extract` + `larql publish` directly.

That is the gap. Everything below is a consequence of it.

---

## 2. What the factory would have prevented

Not hypothetical — these happened.

**A partial upload went public.** HuggingFace rate-limited a 1.65 GB upload
(`503 SlowDown`, part 35 of 109) and the publish aborted. The repo was left
holding the *old* weight file with no replacement — still serving, still
public, now inconsistent with its own manifests. The factory publishes
private and only flips at RELEASE after VERIFY-B (§8: "nothing goes public
unverified"), so this failure would have been contained to a private repo.

**Stale files accumulated.** `publish` only ever added files, so renaming a
weight file left both generations live and the loader chose between them by
name. 4.4 GB of superseded bytes across six repos, removed by hand. Fixed in
PR #215, but the factory's checksum verification compares published bytes
against the manifest, which is a second net under the same failure.

**Nothing pinned the source revision.** The re-extractions used whatever
snapshot happened to be in the local HF cache. A recipe pins the revision, so
a rebuild is reproducible; a manual run is only as reproducible as the cache.

---

## 3. The audit rule

Factory-independent and worth keeping: how to tell whether an existing vindex
is affected.

A vindex is affected **iff both** hold:

1. it stores `Q6_K` or `Q4_0` blocks, **and**
2. it was written before **2026-08-07**.

`Q4_K` was never affected. Two traps:

- **`lm_head_q4.bin` is Q4_K** despite the name (`quantize_q4_k` in
  `write_kquant/lm_head.rs`). A blind "delete the old-named files" cleanup
  would destroy a good 378 MB tensor. Guard every delete on its replacement
  actually being present.
- **Two filename generations exist** — current `*_kquant.bin` and legacy
  `*_q4k.bin`. An audit globbing only the first reported `qwen3-0.6b-q4k` as
  carrying no quantised weights; it carries `interleaved_q4k.bin` and was
  fully broken. Match both, or list the directory.

`interleaved_*.bin` written in `q4k` mode always carries Q6_K — the layout is
`[gate Q4_K | up Q4_K | down Q6_K]` (`write_kquant/ffn.rs`). `attn_weights_*`
carries Q6_K for the V projection.

Confirming is cheap and unambiguous — an affected vindex emits obvious
garbage:

```
$ larql run gemma3-4b-q4k-v2.vindex "The capital of France is"
 shaker peč mixtoவர கருதப்படுகிறதுstehungjö às części ladder Vase znač
```

---

## 4. The manual fallback

Use only when no recipe exists. Prefer `larql recipe build`.

```bash
# 1. Extract to a NEW path — never overwrite the only working copy.
larql extract <hf-snapshot> --output <name>-v2.vindex --quant q4k --down-top-k 10

# 2. Verify it generates. "Re-extracted" is not "re-extracted correctly".
larql run <name>-v2.vindex "The capital of France is" --max-tokens 10

# 3. Publish (prunes stale remote files since PR #215).
larql publish <name>-v2.vindex --repo chrishayuk/<repo> --slices none

# 4. Verify the published file list — current names present, legacy gone.

# 5. Archive. COPYFILE_DISABLE stops macOS writing AppleDouble sidecars.
COPYFILE_DISABLE=1 rsync -a SRC/ /Volumes/chrishayuk/vindexes/<org>/<name>.vindex/

# 6. Byte-compare before deleting local — size equality misses SMB corruption.
cmp SRC/interleaved_kquant.bin DST/interleaved_kquant.bin
```

Naming follows [`card::naming`](../crates/larql-factory/src/card/), which the
factory and its generated cards share: `<model>-q4k-vindex` quantised,
`<model>-vindex` unquantised, `<repo>-<preset>` for slices.

Two repos predate that convention and have no parent —
`gemma-4-26b-a4b-it-vindex-expert-server` and
`gemma-4-26b-a4b-client-vindex-client`. Reproduce with `--no-full` rather
than creating parents that never existed.

---

## 5. Operational traps

**Piping to `tail` masks the exit code.** `larql extract ... | tail -40`
reports `tail`'s status. One 26B extraction "succeeded" in two minutes having
done nothing — the binary was missing. Use `set -o pipefail`, or don't pipe.

**macOS writes AppleDouble sidecars to SMB.** A 15-file vindex arrives as 30,
each `._name` an xattr sidecar. Harmless for loading, but they would be
uploaded if the archive were ever published from.

**`--quant q4k` did not imply `--level all`** despite its help saying so, so
`index.json` recorded `inference` while the writer emitted an `all` vindex.
Fixed in PR #215.

---

## 6. Status — 2026-08-08

| model | affected | re-extracted | published | archived |
|---|---|---|---|---|
| gemma-3-4b-it (+5 slices) | yes | ✅ | ✅ pruned | kept local (testing) |
| gemma-4-26b-a4b | yes | ✅ | ⬜ | kept local (testing) |
| granite-4.1-3b | yes | ✅ | ✅ auto-pruned | ✅ local deleted |
| granite-4.1-8b | yes | ✅ | 🔄 | ⬜ |
| granite-4.1-30b | yes | ✅ | ⬜ | ⬜ |
| qwen3-0.6b-q4k | yes | ✅ | ⬜ | ⬜ |
| qwen3-0.6b (dense) | no | — | ⬜ | ⬜ |
| bitnet-b1.58-2b (ternary) | no | — | ⬜ | ⬜ |
| gemma3-4b-f16 | no | — | ⬜ | kept local |

---

## 7. What should exist

**Recipes for these nine models**, in `chuk-vindex-recipes`. That is the
deliverable this document is really arguing for: with them, the next
format-level fix is `larql recipe build` per model rather than a day of
`extract`/`publish`/`cmp`, and it gets revision pinning and the
private-until-verified gate for free.

**A nibble-layout version in the vindex format.** Nothing in a vindex records
which layout its bytes are in, which is the entire reason a fixed reader
turns old bytes into fluent garbage instead of erroring. A version field —
with the loader refusing, or converting, on mismatch — is the durable fix.
Everything in §3 is a workaround for its absence.

**An audit subcommand.** §3's rule is mechanical; `larql vindex doctor` could
answer it instead of a human globbing filenames and getting it wrong.

**Model-card notes on affected repos.** Anyone who pulled before 2026-08-07
holds broken bytes and has no way to know.
