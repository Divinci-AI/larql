# VINDEX3 registry/resolver — design notes

Status: **rung 1 implemented** (`crates/larql-vindex/src/registry/`).
§1–§7 below are the pre-implementation investigation this rung's
instruction required — *"Before implementation, report the existing
resolution seams and any places where the current code would force an
awkward compatibility compromise. Ground the design in the code rather
than adding a parallel resolver."* §8 records the three architecture
decisions that investigation surfaced, confirmed by the user on
2026-08-23. §9 records what was actually built against those decisions.

The scope is deliberately narrow (see §7): a versioned VINDEX3-only
registry manifest, a name/variant grammar, one shared resolver
abstraction, and a static local test registry. No website, no remote
registry service, no Mac app wiring, no runtime-lifecycle wiring.

## 1. What exists today — model-string resolution

**There are three independent "string → path" resolvers today, and
they already disagree with each other at the margins.** Any new
resolver must consolidate toward one of these becoming canonical, not
add a fourth:

1. **`crates/larql-cli/src/commands/primary/cache.rs::resolve_model`**
   — the "full" resolver: `hf://owner/name[@rev]` → download; an
   existing local directory → used as-is; a string containing `/` →
   checked against the merged cache, else prefixed `hf://` and
   downloaded; a bare shorthand name → `resolve_shorthand` requires a
   *unique* match across both caches. Used by `run`, `chat`, `show`,
   `slice`, and (pre-exec only) `serve`.
2. **`crates/larql-server/src/bootstrap/load.rs::load_artifact`** —
   inside the actual serving binary, much narrower: `is_hf_path`
   prefix check, else a literal filesystem path. **No knowledge of
   `~/.cache/larql/local/` shorthand names or bare `owner/name`
   HF-cache lookup at all.** It only "works" for those forms because
   `larql serve`'s CLI trampoline (`main.rs::run_serve`) pre-resolves
   the string before spawning the `larql-server` subprocess — and
   **silently falls back to the raw, unresolved string**
   (`.unwrap_or_else(|_| path.clone())`) if that pre-resolution
   errors, handing an unresolved/ambiguous string across the process
   boundary to the weaker resolver.
3. **`crates/larql-cli/src/commands/primary/pull_cmd.rs::looks_like_hf_repo`**
   — a *third*, independent `owner/name` heuristic ("exactly one `/`,
   neither side empty, no dot in the owner segment"), structurally
   similar to but not shared with either of the above. It rejects an
   owner containing `.`; `cache::resolve_model`'s slash-branch does
   not check that at all.

**`larql run`/`larql chat` have no VINDEX3 execution path at all** —
only `larql serve` does. `run_cmd::run` resolves via
`cache::resolve_model` and then unconditionally hands the result to
`load_vindex_config`/`walk_cmd`, both VINDEX2-only; a VINDEX3 result
fails several calls deep with a generic `VindexError::WrongContainerGeneration`,
not a purpose-built refusal at the top of `run` the way `slice`/`verify`
do it (`run_cmd.rs:996-998` says exactly this — "container completeness
is a separate rung"). **Consequence for this design**: a resolver that
returns a `ResolvedVindex3` pointing at a real VINDEX3 container is
correct and testable end-to-end against `serve`, but `larql run
qwen3.8` cannot actually execute yet regardless of what the resolver
returns — that gap is pre-existing and out of scope here (§7 says not
to wire runtime lifecycle in this rung).

**The local cache has no persisted manifest at all — the filesystem
*is* the registry.** `~/.cache/larql/local/` is a directory of
symlinks (`<name>.vindex` → target dir); `~/.cache/huggingface/hub/`
is the standard hf-hub layout. Both are scanned fresh on every command
(`scan_cached_vindexes`) into a `CachedVindex { repo, snapshot,
size_bytes, source: HuggingFace | Local }` — **which carries no
generation field**. `list`/`resolve_shorthand`/`resolve_cached`/`rm`
treat V2 and V3 entries identically; the only place generation is
ever discovered is a directory scan calling `detect_generation`
per-path, downstream of the cache layer entirely
(`larql-server`'s `--dir` discovery mixes V2 and V3 freely and sorts
them after the fact into two separate collections).

## 2. What exists today — VINDEX2/VINDEX3 generation detection

The sole discriminator, everywhere in the codebase, is a plain integer
field: `index.json`'s `version`. No marker file, no magic byte, no
directory-shape heuristic — deliberately (`format/generation.rs:1-33`
states this as policy).

```rust
// crates/larql-vindex/src/format/generation.rs
pub const V2_MIN_SCHEMA: u32 = 1;
pub const V2_CURRENT_SCHEMA: u32 = 2;
pub const V3_MIN_SCHEMA: u32 = 3;
pub const V3_CURRENT_SCHEMA: u32 = 4;

pub enum ContainerGeneration { V2, V3 }

pub fn detect_generation(dir: &Path) -> Result<ContainerGeneration, VindexError> {
    // reads only index.json's `version` field via a minimal probe struct —
    // deliberately does not fully deserialize, so a V3 index.json (whose
    // shape a V2 config struct can't model) reports "wrong generation"
    // instead of a confusing parse error.
}
```

This is a solid, well-tested pattern worth imitating for a new
ABI-version probe, not something to route around.

`crates/larql-server/src/bootstrap/load.rs::load_artifact` is already
the single choke point that decides V2 vs V3 for *serving* — detect
once, dispatch once, refuse-with-named-flags for any option a V3
binding can't honor (`unsupported_v3_options`). This is the right
shape to extend with a `Vindex3Abi` check inserted before
`load_v3_model` is called — not a resolver-side reimplementation.

## 3. The ABI gap — genuinely new, not a wrap-around

**No ABI/runtime-compatibility version concept exists for VINDEX3
today.** Grepped exhaustively: no `Abi` type, no `abi_version`, no
`min_larql_version`, no `runtime_version` field anywhere.
`format/vindex3/profile.rs:47`'s own comment: *"the VINDEX3 ABI is
explicitly not frozen yet."*

`Vindex3Index.version: u32` is a **schema** version (currently 4) —
"does this binary's `index.json` parser understand this shape", not
"was this container built for a runtime capability this binary lacks."
Two things are conflated by that one field today; a `Vindex3Abi` in
the registry's `ResolvedVindex3` is new, separate machinery.

The word "admissible" is used heavily in the codebase, but exclusively
for **pre-encode** semantic representability (`larql vindex3 plan`:
can this HF checkpoint become a valid VINDEX3 graph at all) and its
post-hoc drift re-check — never for "can this runtime load this
already-built container." `capability::scope`'s `DocumentCapabilities`/
`ProfileCapabilities` similarly answer "what can this resolved profile
serve given the bytes present", not ABI compatibility. **A load-time
admissibility/ABI gate for an already-built container would be new
machinery**, though the `plan::report::Finding`/"collect every
blocking fact, verdict = no blockers" pattern is a good template to
copy for it.

"CAP-0/CAP-1" (mentioned in prior project memory) does not exist
anywhere in the repo under that name.

## 4. The provenance gap — the reason `registry/*.json` cannot be started yet

This is the finding that matters most for sequencing, per the explicit
instruction not to start with JSON files until `publish` was inspected:

- `larql publish`/`larql slice` write **no manifest of their own** —
  they move/upload files and re-derive display titles from
  `index.json`. `PublishOptions` is upload plumbing only.
- **An authoritative, versioned provenance schema already exists**:
  `larql_vindex_spec::VindexManifest` / `Source`
  (`crates/larql-vindex-spec/src/lib.rs`) — HF repo, revision, base
  model SHA, per-shard checksums, extractor version/SHA, timestamp.
  JSON-Schema-mirrored; the crate's own README states "Rust types win"
  on conflict. **But it is wired only to VINDEX2's `index.json`** — its
  own module doc describes "a dense Gemma-shaped extraction."
- **`Vindex3Index`, VINDEX3's own root manifest, carries zero
  provenance fields** — no `source`, no `checksums`, no HF repo/revision,
  no extractor SHA, no timestamp. Only identity (`model`/`family`) plus
  structure (`segments`/`profiles`/`variants`).
- `larql-factory::Recipe` is the closest thing to a build-intent
  registry (pinned `hf_repo`+`revision`, extractor tool/version/level,
  output presets, publish target) — but it describes *what should be
  built*, not a queryable index of what already exists, and its own
  code comments call the build-driver side "aspirational."
- The one existing "official artifact" discovery mechanism —
  `library_name: larql` HF model-card frontmatter tag
  (`crates/larql-factory/src/card/frontmatter.rs`), intended to make
  vindexes filterable via `huggingface.co/models?library=larql` — is
  hard-gated to `vindex_spec_version: 1` (VINDEX2/v1-manifest only,
  the validator rejects any other value) and is itself flagged as an
  unresolved open question in `docs/vindex-factory.md` §13.1 (model
  repo vs. dataset repo tension).

**Verdict**: there is no existing VINDEX3-native provenance struct to
wrap. The registry design has an open choice — carry provenance
out-of-band in the registry entry (reusing `Source`'s *shape*, not its
VINDEX2 wiring), or extend `Vindex3Index` itself with a
`Source`-shaped field. The latter is a container-format change with a
much larger blast radius than this rung's stated scope. **§8 proposes
out-of-band as the default for this rung** and flags it as the one
decision most worth confirming before writing the schema.

## 5. HF pull/download — asymmetric with publish, VINDEX2-shaped

- `resolve_hf_vindex`/`resolve_hf_vindex_with_progress`/`download_hf_weights`
  (`crates/larql-vindex/src/format/huggingface/download/mod.rs`) fetch
  a **fixed, hardcoded VINDEX2 filename list**
  (`VINDEX_METADATA_FILES`/`VINDEX_BIN_FILES`/`VINDEX_WEIGHT_FILES`).
  A VINDEX3 repo's actual payload (`moe_manifest.json`,
  `routed/layer_NNN.lyrw`) is **not in that list** — pulling a VINDEX3
  repo via `larql pull` today would fetch `index.json` (schema 3/4)
  and silently miss the container. Confirmed wired exactly this way in
  `larql-lql`'s `USE "hf://..."` path: resolve, *then* detect
  generation locally — generation is discovered strictly after a
  possibly-incomplete download.
- `publish`'s upload side (`enumerate_publishable_files`) is already
  **generic** — it walks the source directory's actual shape (root
  files + one level of subdirectories), so it structurally handles a
  VINDEX3 directory fine. The asymmetry (generic upload, fixed-list
  download) is the sharpest HF-layer seam.
- **No org/namespace enforcement exists anywhere.** `is_hf_path` is a
  bare `"hf://"` prefix check. `chrishayuk/*-vindex` appears
  throughout docs/tests purely as the author's personal example, never
  checked or allow-listed in code. There is no `LARQL_HF_ORG`-style
  config. An official short-name → HF-repo mapping is greenfield.
- Sibling/preset conventions (`{repo}-{preset}`, `client`/`attn`/
  `embed`/`server`/`browse`) exist only because `larql slice` can
  carve V2 vindexes — `slice` explicitly refuses VINDEX3 containers.
  The naming *template* is reusable; the preset vocabulary is
  VINDEX2-file-layout-shaped and doesn't map onto VINDEX3 segments.

## 6. The existing analogue to a "variant" string

VINDEX3's own format already has almost exactly the vocabulary a
registry `variants` map needs, just not surfaced anywhere in the
CLI/cache layer yet:

```rust
// crates/larql-vindex/src/format/vindex3/variants.rs
pub struct StoredVariant { pub storage: String, pub fidelity: Fidelity }
pub struct RegionSetVariants { pub baseline: String, pub variants: BTreeMap<String, StoredVariant> }
pub struct VariantCatalogue { sets: BTreeMap<String, RegionSetVariants> } // #[serde(transparent)]

// crates/larql-vindex/src/format/vindex3/profile.rs
pub struct Profile { pub name: String, pub selects: BTreeMap<String, String> }
```

`Vindex3Index.select_profile(name)` is the resolution entry point;
`larql show <v3-dir>` is the *only* place in `larql-cli` that
currently exercises it, purely for display. **There is no CLI flag on
`run`/`chat`/`serve` to select a profile by name** — `load_v3_model`
always opens the container with no profile argument. A registry
`variant` string (e.g. `27b-nvfp4`) maps onto `Profile.name` almost
exactly; wiring a `--profile` flag through to `Vindex3Runtime::open`
is a natural, small follow-up but is not required for this rung
(the static test registry can name a profile in its manifest without
the CLI needing to pass it anywhere yet).

## 7. Scope for this rung (unchanged from the instruction, restated for reference)

1. Versioned registry manifest/schema — **VINDEX3-only**; `format:
   "vindex3"` structurally required, not a switchable field.
2. Model-name/variant grammar, defined and tested.
3. One shared resolver abstraction — `crate::ResolvedVindex3`, not a
   generic `ResolvedModel`.
4. A tiny static local test registry — no website, no remote registry
   service, no network API.
5. Deterministic default-variant selection; unknown-model/variant
   refusal; ABI/runtime compatibility refusal; explicit `hf://`/local
   resolution — all proven by tests.
6. No Mac app wiring, no runtime-lifecycle wiring.

**Explicitly not required by this rung** (raised in §1/§5, left as
follow-ups): consolidating the three existing resolvers into calling
the new one; wiring `--profile` through to `Vindex3Runtime::open`;
fixing `larql pull`'s VINDEX2-fixed-file-list download so a VINDEX3
repo actually round-trips; giving `larql run` a VINDEX3 execution path.

## 8. Confirmed architecture decisions (2026-08-23)

1. **Provenance placement (§4): out-of-band in the registry manifest.**
   `Vindex3Index` describes the container itself; the registry manifest
   answers a different question — "where did this published build come
   from" — and the two are deliberately not coupled. A registry variant
   carries `source: { repo, revision }`, a **V3-registry-native type**
   (not a reuse of `larql_vindex_spec::Source` — reusing a VINDEX2-wired
   type just because its fields happen to match would leak the exact
   coupling this decision removes), and `revision` must be an immutable
   pin, never `main`/`latest`/`HEAD`/unfrozen-branch for an official
   entry. If VINDEX3 itself ever needs embedded reproducibility
   provenance (e.g. detached/offline verification with no registry
   present), that is a deliberate, separate container-format decision —
   not a side effect of this rung.
2. **Consolidation scope: additive only, this rung.** The new resolver
   establishes the authoritative VINDEX3 reference semantics without
   becoming responsible for preserving every existing resolver's quirk;
   it reuses genuinely shared primitives (`detect_generation`,
   `is_hf_path`) but does not delegate its semantics back to
   `cache::resolve_model`, `load_artifact`, or
   `pull_cmd::looks_like_hf_repo` — otherwise the new abstraction would
   just inherit the inconsistencies §1 found. "Additive" must not mean
   "speculative dead code": this rung's tests pin the full contract
   (deterministic default variant, unknown model/variant, ABI refusal,
   explicit hf://\+local resolution, VINDEX2 refusal, malformed
   references) end-to-end. **Banked follow-up rule**: once this
   resolver's contract is proven, the three existing resolution paths
   are meant to converge onto it as a "resolver convergence" rung — they
   are not intended to remain permanently parallel.
3. **Module home: `larql-vindex`, as a dedicated `registry` module —
   not under `format::vindex3`.** `larql-vindex` already owns the
   adjacent concepts (generation detection, HF path handling) and is
   already a shared dependency of `larql-cli`/`larql-server`. The
   registry is not part of the on-disk VINDEX3 format — it's a
   distribution/identity layer *for* VINDEX3 — so it lives at
   `crates/larql-vindex/src/registry/`, a sibling of `format/`, not
   nested inside it. No new crate: per "don't extract crates
   speculatively", extraction becomes evidence-driven only if registry
   logic later grows substantial independent networking/caching/
   signing/publishing machinery.

## 9. Rung 1 — what was built

`crates/larql-vindex/src/registry/` (see its module doc for the
full picture): `reference.rs` (the four-form grammar — `ModelName`/
`VariantName` newtypes, `ModelReference`/`ExplicitReference`, disjoint
by construction because a `ModelName` structurally cannot contain `/`),
`manifest.rs` (`RegistryManifest`/`RegistryModel`/`RegistryVariant`/
`RegistryArtifactRef`/`Provenance`, schema-versioned,
`validate()`/`from_json()` reject a dangling default variant or an
unpinned revision before the manifest is usable), `abi.rs`
(`Vindex3Abi` — one supported value, exact match, deliberately no
compatibility range invented ahead of a second value existing),
`resolver.rs` (`resolve()` — the one entry point; `Vindex3Resolution::
{Registry(ResolvedVindex3), Explicit(ArtifactRef)}`, kept as two output
shapes rather than forcing name/variant/ABI/provenance placeholders
onto an explicit `hf://`/local reference that has no registry identity
to report), `error.rs` (`RegistryError`, wraps `VindexError` via
`#[from]` for the one place this resolver reuses a VINDEX3 primitive —
`detect_generation`, in the explicit-local-path arm, which refuses a
VINDEX2 directory even through the escape hatch), `fixtures.rs` (the
tiny static test registry — `qwen3.8` with two variants, public and
unconditional, following the `format::vindex3::fixtures` precedent so
`larql-cli`/`larql-server` tests can reuse it later without duplicating
data). Colocated `*_tests.rs` files per source file (the
`generation.rs`/`generation_tests.rs` precedent) plus an end-to-end
`crates/larql-vindex/tests/vindex3_registry.rs` against the public API.

Gates: 64 colocated unit tests + 7 integration tests, all green;
`cargo clippy --all-targets -- -D warnings` clean; `cargo fmt --check`
clean; 100% region/line/function coverage on all five new source files
(`abi.rs`, `fixtures.rs`, `manifest.rs`, `reference.rs`, `resolver.rs`)
via `cargo llvm-cov`, well above the 90% floor. Downstream crates
(`larql-cli`, `larql-server`, `larql-lql`) still `cargo check` clean
against the new crate-root re-exports. Not wired into `larql run`/
`serve`/`pull`/the three existing resolvers, and not wired into the Mac
app or runtime lifecycle — both explicitly out of scope for this rung
(§7).
