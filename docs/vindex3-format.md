# VINDEX3 — Model-System Container Format

Status: **living spec**. §1–§5 describe what is implemented and gated as of
2026-08-11 (branch `feat/vindex3-glimmer-g0-g2`); §6–§8 pin the design for
the rungs currently being built (G3–G5). The experimental programme,
gates and run order live in `docs/vindex3-experiments.md`; the LYRW v2
routed-layer physical layout lives in `docs/lyrw-v2.md` and is incorporated
here by reference as one segment family.

> Older `spec §N` citations in `format/vindex3/{index,lyrw2,…}` code
> comments refer to the pre-G2 draft of this spec; where they conflict,
> this document is authoritative and those comments should migrate.

---

## 1. What VINDEX3 is

A modern model release is not a weights file. It is a **system**: a target
model, a perception tower physically embedded in the same checkpoint, a
speculative drafter shipped as a sibling artifact consuming the target's
hidden states through a declared tap interface, several quantization
profiles of the same logical operands. GGUF answers *"how do I store and
run these tensors?"* VINDEX3 answers:

> What are these model objects, what operations can consume them, which
> representations are equivalent, which parts should be resident, and what
> future computation will need them?

The pipeline is deliberately compiler-shaped:

```text
HF artifacts
    ↓
inventory            (G0 — source semantics, two authorities)
    ↓
representability     (G1/G2 — type checking; fail-closed)
    ↓
system graph         (G2 — semantic IR)
    ↓
physical encoding    (G3 — materialisation)
    ↓
verification         (G4 — Declared ≡ Resolved ≡ Graph ≡ Encoded)
    ↓
execution            (G5 — from the encoded description alone)
```

### 1.1 Principles

1. **Single semantic authority.** The system graph is built once
   (`graph::build_from_inventories`) and every downstream consumer — the
   planner, the encoder, eventually the executor — consumes *it*, never a
   private re-interpretation of the checkpoint. "Representable" has
   exactly one definition: *the graph builder placed it.* There is no
   capability registry to drift out of sync.
2. **Fail-closed.** A config key nobody has judged, a tensor group no
   placement rule owns, an interface whose producer cannot be resolved
   unambiguously — each blocks the plan. Unjudged is not admissible;
   ambiguity is refused, never guessed.
3. **No silent conversion.** Regions are placed as their producer wrote
   them; generation boundaries error precisely rather than convert; a
   defaulted value never impersonates a declared one.
4. **No model-family branches.** Roles, placements and policies derive
   from evidence (declared interfaces, shape arithmetic, component
   topologies), not from `model_type` matching. Glimmer went from 51
   blocking findings to zero without a single Muse-specific branch; that
   property is the point and every future architecture must preserve it.

### 1.2 Identity distinctions (load-bearing)

```text
artifact       ≠ component
tensor name    ≠ logical object
interface      ≠ implementing tensor
NoPE           ≠ rope(theta = 0)
logical object ≠ physical representation
representable  ≠ "parser consumed the key"
```

---

## 2. The G-ladder

| Rung | Question | Gate |
|---|---|---|
| G0 | What does the source declare? | `larql inspect-hf` emits the inventory |
| G1 | Can the schema describe it? | `larql vindex3 plan` — typed findings, non-zero exit on blockers |
| G2 | Generalise the schema until reality fits | plan over the artifact set: `blocking = 0, mismatched = 0, unknown = 0` |
| G3 | Materialise the graph | `encode` then `inspect` reconstructs the system **solely from the container** |
| G4 | Prove source ≡ encoded | four-authority comparison + payload-hash equality |
| G5 | Execute from the encoded description | forward pass with zero architecture branches |
| G6 | Drafter parity | speculative execution discovered from the `HiddenStateEdge` |
| G7 | Performance baseline | reference numbers on the target hardware class |
| G8 | Alternate physical/execution plans | LARQL-specific layouts/prediction over the same logical system |

G8 must not contaminate G0–G5: optimisation is an *alternate plan over the
same graph*, never a schema change.

---

## 3. G0 — the architecture inventory

`larql inspect-hf <dir>` → `ArchitectureInventory`
(`larql-models/src/inventory/`), schema version `INVENTORY_SCHEMA = 2`.

Two authorities, side by side; disagreement is the instrument:

- **`config_keys`** — every leaf of `config.json`, flattened to a dot
  path, classified `consumed` / `metadata` / `unconsumed`.
- **`resolved`** — what this build's detection would actually run: the
  full per-layer table (span, window, position policy, head geometry).

Classification honesty rules:

- Name-registry credit applies only inside containers the parser itself
  recurses into (`text_config`, `rope_scaling`, …). A leaf under any other
  container is never credited by name — `vision_config.hidden_size` shares
  a name with a consumed key but the text parser does not read it.
- Nested components (`vision_config`, any `*_config` sibling) are read by
  a generic component-topology reader whose **recorded reads are the
  consumed-credit**: a key is consumed iff a read actually stored it. The
  reader and the classification cannot diverge because there is only one
  artifact.
- A parser-sync test scans `detect/parser.rs` for key accesses and fails
  when the registry does not know one. The registry cannot rot silently.

The inventory also carries: identity, detection outcome (including the
generic-fallback flag — a model serving through the generic path with
unconsumed keys is the loudest red flag the report can raise), nested
component topologies, declared interfaces, and a tensor inventory read
from safetensors *headers only* (no payload I/O).

### 3.1 Position policy

`PositionPolicy = Rope { theta } | None`. Absence of positional rotation
is an intentional per-layer execution property, not a parameter value.
The HF `layer_rope_theta` spelling uses `0.0` as a NoPE sentinel; that
sentinel is interpreted at **exactly one boundary**
(`PositionPolicy::from_declared_theta`) and nowhere else. No zero theta
may circulate internally — `1/0^(i/d)` is degenerate, and a resolver
storing `0.0` where it means "none" has re-invented the magic value the
type exists to remove.

---

## 4. G1/G2 — the representability plan

`larql vindex3 plan <artifact>…` → `SystemPlan`
(`larql-vindex/src/format/vindex3/plan/`), schema `PLAN_SCHEMA = 2`.
Artifacts are checkpoint dirs and/or saved inventory JSONs, treated as one
model system. Exit is non-zero when the plan is inadmissible.

Findings are typed twice:

- **category** — `representable` / `mismatched` / `unrepresented` /
  `interface`;
- **semantic class** — `execution_semantic` / `tensor_semantic` /
  `interface_semantic` / `metadata_only` / `ignored_safe` / `unknown`.

Rules:

- `consumed` is never trusted. Comparators re-read declared values from
  the config facts and diff them against resolution (scalar topology,
  uniform rope θ in parser-precedence order, per-layer
  `layer_rope_theta` against per-layer policies, `layer_types`
  interleave). Equal → representable; different → `mismatched`, blocking.
- Unconsumed keys are graded by a semantic registry of *known HF field
  names*. A name the registry has never seen grades `unknown` — and
  blocks. Keep this painful; a "probably harmless" bucket is exactly how
  silent semantic loss returns.
- `ignored_safe` ships empty. Every future entry needs a per-entry
  justification comment.
- Tensor/topology/interface representability comes from the graph builder
  (§5): placed objects are representable with their graph ids as proof;
  `unplaced` groups and `unresolved_interfaces` are blocking findings.
- The plan embeds the built graph — the proof object G3 consumes.
- Verdict: `admissible ⇔ blocking == 0`.

Standing tripwire: a clean dense model (Llama-shaped fixture) plans
**admissible**. Its job is to stay that way; a regression that starts
blocking clean dense models fails there first.

---

## 5. G2 — the system graph

`larql-vindex/src/format/vindex3/graph/`, schema `GRAPH_SCHEMA = 1`.

```text
SystemGraph
├── components: [Component]        id, role, source_artifact,
│                                  num_layers, hidden_size,
│                                  attention: [AttentionLayerPolicy]?
├── objects:    [LogicalObject]    id, component, kind,
│                                  source_bindings: [SourceBinding],
│                                  representations: [Representation]
└── edges:      [HiddenStateEdge]  producer_component, producer_layers,
                                   consumer_component, consumer_object,
                                   block_size?
```

### 5.1 Components

Roles are **evidence-derived**: an artifact declaring `target_layer_ids`
is a `drafter`; a nested `*_config` component is `perception`; otherwise
`primary_text`. Ids are conceptual (`target`, `vision`, `draft`), with
numeric suffixes on collision — never directory names. The source
artifact is recorded for traceability, not identity.

`AttentionLayerPolicy { span: sliding|full, window, position }` per layer
is architectural liveness information: a KV planner reading it knows that
positions beyond `window` on a sliding layer are *architecturally* dead
before any semantic analysis runs. A component without per-layer
resolution (perception towers today) carries no table — absent is honest,
fabricated is not.

### 5.2 Logical objects

Kinds (architectural vocabulary, not familial): `embedding`,
`decoder_stack`, `final_norm`, `output_head`, `perception_tower`,
`perception_adapter`, `feature_projector`.

Identity is conceptual — `{component}.{kind}` — and **physical names may
bind objects but never define identity**. That is what later allows

```text
target.decoder_stack
    ├── canonical BF16
    ├── K-quant
    ├── Metal-native packed
    └── future derived representations
```

without unwinding a single id. `SourceBinding { artifact, tensor_prefix,
tensors, bytes }` carries the physical trace; `Representation
{ encoding, fidelity: canonical|approximate }` carries materialisations
(encodings are *observed* from shard headers, never invented).

### 5.3 Placement rules (the builder)

- Groups are classified by a name-fragment vocabulary (first match wins,
  specific before generic — a vision tower has `layers` segments too) and
  merged into `(component, kind)` objects with multi-binding.
- **The projector claim runs before name classification.** The consumer
  side of a declared interface is identified by shape evidence — a 2-D
  tensor of `len(taps)·hidden × hidden` (either orientation) — and claims
  every group sharing its first path segment by *structural adjacency*.
  Without this ordering, the projector's own norm
  (`encoder.output_norm_enc`) name-classifies into `final_norm`.
- The edge's producer must be **exactly one** other component deep enough
  to own every declared tap. Zero candidates or two candidates → the
  interface is unresolved, and blocks. Never guessed.
- Everything unplaced returns as data (`unplaced`,
  `unresolved_interfaces`); the planner converts it to blocking findings.

### 5.4 The edge is not the tensor

A `HiddenStateEdge` describes **logical flow** of residual states across a
component boundary. The fusion projector implementing its consumer side is
a tensor object referenced by id. Both are representable; they are
distinct facts and are never merged.

---

## 6. G3 — physical encoding (pinned design)

> Deliberately boring, deterministic, source-independent. No layout
> optimisation: one logical object → one canonical representation → one
> or more simple contiguous segments. Optimised layouts arrive at G8 as
> *additional* representations over the same logical objects.

```bash
larql vindex3 encode <artifact>… --output <container>/
```

The encoder consumes **the built graph** (via the plan pipeline), never a
private re-interpretation of the checkpoint, and refuses to encode an
inadmissible plan. Two semantic authorities immediately after eliminating
them would be a regression.

### 6.1 Container layout

```text
<container>/
├── index.json            sole root authority: format version,
│                         system_graph ref, object/representation
│                         directory (representation → segment files)
├── system_graph.json     the SystemGraph, verbatim
└── segments/
    ├── target.decoder_stack.bin
    ├── target.embedding.bin
    ├── target.output_head.bin
    ├── vision.perception_tower.bin
    ├── vision.perception_adapter.bin
    ├── draft.decoder_stack.bin
    ├── draft.feature_projector.bin
    └── …
```

- The graph references logical object ids and representation ids; the
  index's directory maps representation → segment bytes. Graph edges
  never reference safetensors names: **the HF checkpoint disappears as an
  authority once encoded.**
- Within a segment, tensors are addressed by a per-representation tensor
  table (name relative to the binding prefix, dtype, shape, offset,
  length), payloads concatenated in table order. Relative names are
  structural (`0.self_attn.q_proj.weight`), not artifact-global.
- Every canonical representation records a **source payload hash**
  (SHA-256, computed while copying) and the encoded segment records its
  own hash. These are G4's byte-equivalence inputs.
- `Vindex3Index.system_graph` is optional in the schema: absence means
  "no graph recorded", never "single-component assumed".

### 6.2 The G3 gate

After encoding, the validation path gets **no access to the source**:

```bash
larql vindex3 inspect <container>/
```

must reconstruct — solely from the container, with no transformers
config, no HF filenames, no safetensors headers, no architecture
registry —

```text
components: target (52 layers, 6656, 39 sliding / 13 full-NoPE)
            vision (50 layers, 1536)
            draft  (5 layers, taps [1,13,25,37,49], block 16)
edge:       target.hidden[1,13,25,37,49] → draft.feature_projector
objects:    …with sizes, encodings and hashes
```

---

## 7. G4 — verification (pinned design)

Four explicit authorities, compared structurally and semantically:

| Authority | Source |
|---|---|
| **Declared** | what HF said (`config.json`, shard headers) |
| **Resolved** | what LARQL interpreted (detection + policies) |
| **Graph** | the logical system constructed at G2 |
| **Encoded** | what the container actually contains |

Invariant: `Declared ≡ Resolved ≡ Graph ≡ Encoded` for everything
execution-semantic. Example instance:

```text
Declared  layer_rope_theta[3] = 0
Resolved  PositionPolicy::None
Graph     target.attention[3].position = none
Encoded   target.attention[3].position = none        PASS
```

Two equivalence claims are kept **separate**, because each can fail while
the other holds:

- **semantic equivalence** — the four-authority structural comparison;
- **byte payload equivalence** — source-binding payload hash ==
  encoded canonical segment hash, per representation.

"Tensor count before == after" is not verification and does not appear in
this format.

---

## 8. G5 — execution contract (pinned design)

Execute the system **using only semantic information present in the
container**. The executor owns kernels (attention, matvec, norm,
activation, drafter verification); it must not own architecture:

- no `if family == X` branches — span/window/position come from
  `AttentionLayerPolicy`, scaling scalars from the persisted config
  surface;
- no hardcoded tap constants — the drafter discovers
  `target.hidden[…] → draft.feature_projector` from the
  `HiddenStateEdge`, block size included.

When G5 holds, an architecture is an **instance** the container
describes, not an implementation the executor contains.

---

## 9. Relationship to existing formats

- **LYRW v2** (`docs/lyrw-v2.md`) remains the physical layout for routed
  MoE expert banks; under this spec it is one segment family a
  representation may use, selected by profiles/variants
  (`index.json` §profiles). Its region roles (`gate/up/down/bias/scales/
  latents`) are operand-level structure *inside* an FFN object, not a
  substitute for logical objects.
- **VINDEX2** containers are a different generation; loaders refuse
  cross-generation directories with a precise error naming both versions.
- The **MoE manifest** (`moe_manifest.json`) continues to describe routed
  programmes; the system graph does not replace it, it locates it.
