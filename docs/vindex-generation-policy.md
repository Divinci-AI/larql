# Container-generation policy: the VINDEX2 → VINDEX3 transition

Status: **contract adopted; default not yet flipped.**
The semantic catch-up phase closed 2026-08-22: VINDEX3 reached full LQL
parity with VINDEX2 (execution, inference, browse, mutation, patching,
compose + parity, COMPILE, logical DIFF, COMPACT), gated cross-platform,
with a real-model compose smoke green on granite-4.1-3b. VINDEX3 is the
**candidate primary generation**; VINDEX2 is the **compatibility
generation**.

## The migration contract

```text
New extraction        → VINDEX3 by default            (after the flip)
Existing VINDEX2      → continues to open and run
Generation detection  → bind once at load
Higher layers         → never care which generation was bound
New semantic features → V3 only, unless compatibility requires V2
V2                    → compatibility generation, no architectural expansion
Escape hatch          → explicit V2 extraction/binding during the transition
```

## The default-flip gate

The flip is a **named decision, made in exactly one place** — never a
side effect of a CLI default, a recipe template, or a surface-local
fallback:

- `crates/larql-vindex/src/format/generation.rs` —
  `DEFAULT_EXTRACTION_GENERATION` is the generation an extraction with no
  expressed preference writes. It is `V2` today.
- `crates/larql-vindex/src/format/generation_tests.rs` —
  `auto_extraction_resolves_to_v2_until_the_default_flip_is_decided` pins
  it. Flipping means changing the constant **and** this test in the same
  commit; that pair is the decision.

Every extraction surface resolves caller intent to a `GenerationRequest`
(`Auto` | `Explicit(generation)`) and passes it through
`admit_extraction_generation`. `Auto` gains its meaning only there.
"V3 was requested and refused" and "V3 was never requested" are distinct
by construction, and an explicit request is **never downgraded** — a
surface that cannot produce the requested generation refuses by name.

## Surfaces

| Surface | Request spelling | Today (`Auto` → V2) |
|---|---|---|
| LQL | `EXTRACT MODEL "m" INTO "o" [FORMAT VINDEX2\|VINDEX3]` | `FORMAT VINDEX3` refuses by name (producer not wired); `FORMAT VINDEX2` explicit; absent = policy |
| CLI | `larql extract --generation {v2\|v3}` | `--generation v3` refuses by name; omitted = policy |
| Factory | `extractor.options.generation: "v2"\|"v3"` → forwarded as `--generation` | absent = tool policy; a pin participates in `build_id` |
| V3 producer | `larql vindex3 encode <hf-artifacts>` | the only VINDEX3 producer today |

Binding needs no policy: `detect_generation` (`index.json` schema
version, the sole discriminator — no filename or shape sniffing) already
dispatches `USE`, the server bootstrap, and `DIFF` operands, and fails
closed on unknown schemas.

## Rungs to the flip

- **M1 (this document + the seam)** — policy type, pinned default,
  explicit request spellings on every surface, refusals instead of
  downgrades. No behaviour change.
- **M2 — V3 production reachable from the extraction surfaces.** Wire
  `FORMAT VINDEX3` / `--generation v3` to the V3 encoder for admissible
  sources; the V3 path must reach parity on the extraction side channels
  (tokenizer.json, HF metadata snapshot) so a default-produced container
  binds with full capability; post-extract auto-bind takes the V3 arm.
- **M3 — consumer readiness.** V2-only consumers either route by
  detection or refuse by name: `SHOW MODELS` must list V3 containers,
  `run`/`walk`/`describe`/`stats`, Vindexfile `FROM`, slice/publish/link,
  quant converters; `/v1/models` reports `generation` for both.
- **M4 — the flip.** `DEFAULT_EXTRACTION_GENERATION = V3` + the pinned
  test, in one commit. V2 becomes the explicitly-requested compatibility
  generation. Evidence rows required by then: the real-model compose
  smoke (done — granite-4.1-3b) and whatever Metal-path/scale rows the
  release checklist names.

After the flip, V2 receives no architectural expansion — compatibility
fixes only.
