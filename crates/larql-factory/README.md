# larql-factory

The Vindex Factory driver: recipe schema, `build_id` canonicaliser,
structural validator, capability manifest, and Hub card generator for
[docs/vindex-factory.md](../../docs/vindex-factory.md).

This is the single implementation both a `chuk-vindex-recipes` GitHub
Action and a rig worker are meant to call (as `larql recipe`,
`larql capabilities`, `larql card render`) — see §3.1 of the spec for
why the driver lives here rather than in a separate repo.

```
crates/larql-factory/
├── src/
│   ├── recipe/          Recipe schema (§4) — one file per YAML section
│   ├── validate/         Structural validator (§6.1's non-network checks)
│   ├── build_id.rs        build_id canonicaliser (§5)
│   ├── capabilities/       Capability manifest (§15.2)
│   ├── card/                 Hub model-card generator (§9)
│   ├── estimate/               Size/cost estimate (§6.1 step 4) — the
│   │                           only module with network I/O
│   ├── constants.rs            Facts shared across modules
│   └── hex.rs                   Lowercase hex encoding
└── testdata/
    └── gemma-3-4b-it.yaml   Sample recipe used throughout the test suite
```

## What it does

- **Recipe schema** ([`recipe`]) — Rust types for the v0.1 YAML schema:
  `source`, `extractor`, `outputs`, `verify`, `publish`, `budget`.
  `Recipe::from_yaml` parses; `larql_vindex_spec::{ExtractLevel,
  StorageDtype, QuantFormat}` are reused directly rather than
  duplicated.
- **`build_id`** ([`build_id`]) — SHA-256 over exactly `apiVersion` +
  `source` + `extractor` + `outputs`. Changing `verify`/`publish`/
  `budget`/`metadata` doesn't change it — those don't change the
  produced bytes.
- **Structural validator** ([`validate`]) — everything §6.1's PR check
  needs that doesn't require network I/O: full-SHA revision, released-tag
  extractor version, known preset names, threshold ranges. Reports every
  problem in one pass, not just the first.
- **Capability manifest** ([`capabilities`]) — which architectures the
  running `larql` recognises and what each supports, built from
  `larql_models::detect::ARCHITECTURE_REGISTRY` (a real registry, not a
  hand-duplicated list — see that module's docs).
- **Card generator** ([`card`]) — a Hub model card: frontmatter, dims,
  slice table, `USE` snippet with a computed revision tag, verification
  summary, inlined recipe. `VerificationReport`/`SliceSummary` are a
  provisional shape — nothing produces them yet since the build driver
  (spec §7) doesn't exist.
- **Size/cost estimate** ([`estimate`]) — upstream download size, a
  coarse per-output byte estimate (dims via the same
  `larql_models::detect::detect_from_json` path the real extractor
  uses), an executor recommendation, and a cost band from
  `docs/dec-funnel-v0.2.md` §7's rate basis. Prices the recipe's own
  declared `budget.max_wall_minutes` rather than inventing a duration
  prediction — there's no real throughput data anywhere to ground one
  in. The only module here that touches the network.

## CLI usage

```bash
# Structural validation — prints every problem found, exits 1 if any
larql recipe validate my-recipe.yaml

# The content hash that determines whether a build is a no-op / verify-only / rebuild
larql recipe build-id my-recipe.yaml

# This release's capability manifest, as JSON
larql capabilities

# Render a Hub model card from a recipe + manifest + verification report
larql card render \
  --recipe my-recipe.yaml \
  --manifest index.json \
  --verification verification.json \
  --slices slices.json   # optional

# Upstream size, per-output size, executor recommendation, cost band —
# the only command here that hits the network (HF file listing + config.json)
larql recipe estimate my-recipe.yaml
```

## Not built here yet

The build-stage driver (§7: FETCH → EXTRACT → SLICE → MANIFEST → MIRROR
→ VERIFY-A → PUBLISH → VERIFY-B → RELEASE → REGISTER → TEARDOWN) and the
verify-from-hub harness are still open — see the spec's §14 build
inventory. Both need real credentials (HF write token, R2) and a real
`chuk-vindex-recipes` repo to be useful end-to-end, not just
unit-testable.

## Tests

```sh
cargo test -p larql-factory
```

Every source file is at or above the 90% floor (`coverage-policy.json`);
the large majority are at 100% — `estimate/mod.rs` (97.7%) and
`estimate/http.rs` (91.6%) are the two exceptions, both network-facing.

## CI

```sh
make larql-factory-ci
```

GitHub Actions: `.github/workflows/larql-factory.yml`
Platforms: **Linux · Windows · macOS** (all in CI)
