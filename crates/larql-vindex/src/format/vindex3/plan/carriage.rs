//! How far a declared fact travels — the VINDEX3-boundary authority gate.
//!
//! The inventory answers *did a parser read this key?*
//! ([`KeyStatus::Consumed`](larql_models::inventory::KeyStatus::Consumed)).
//! The plan used to treat that answer as *can VINDEX3 represent this
//! fact?* — a different question about a different object, and the gap
//! between them is silent by construction: a fact the parser reads into
//! `ModelConfig` and VINDEX3 then drops looks fully covered from the
//! plan's side.
//!
//! GPT-OSS is the witness. It declares `rope_scaling = {rope_type:
//! "yarn", factor: 32}` for a 131k context. Every one of those leaves
//! classifies `consumed` — the parser genuinely reads them. But
//! [`PositionPolicy`] expresses `Rope { theta } | None` and nothing
//! else, and no other field under `format/vindex3/` carries a scaling
//! block, so the model would plan, encode and execute as **plain rope at
//! θ=150000**, with the plan reporting no defect at all. (VINDEX1/2 do
//! carry it, as raw JSON — so this is a regression the older path does
//! not have.)
//!
//! ```text
//! config.json fact
//!    ↓  parsed        larql-models' parser stored it in ModelConfig
//!    ↓  represented   the VINDEX3 system graph persists it
//!    ↓  lowered       it reaches the generic op plan as an op parameter
//!    ↓  executed      an executor reads that op parameter
//! ```
//!
//! Each execution-semantic key needs a [`CarriageRule`] declaring which
//! of those stages it reaches. Rules claiming [`Carriage::Represented`]
//! or deeper carry a **probe** that reads the value back off the built
//! graph, so the claim is checked against the schema rather than
//! trusted; a probe that disagrees with the declaration blocks. Rules
//! that honestly stop at [`Carriage::Parsed`] must say why, and are
//! reported rather than hidden. A key with **no rule at all** blocks —
//! that is the state this module exists to abolish.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use larql_models::config::{score_scale_from_query_pre_attn_scalar, Activation};

use super::super::graph::Component;

/// How far a declared fact travels from `config.json` into execution.
///
/// Ordered: a deeper stage implies every shallower one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Carriage {
    /// A registered parser read the key into `ModelConfig`. This is what
    /// the inventory's `consumed` status means, and on its own it is not
    /// evidence of anything downstream.
    Parsed,
    /// The VINDEX3 system graph persists it: a container round-trips the
    /// fact, so encoding does not lose it.
    Represented,
    /// It reaches the generic op plan as an op parameter, so a backend
    /// receives it rather than re-deriving it.
    Lowered,
    /// An executor reads that op parameter on the path under test.
    Executed,
}

impl Carriage {
    /// The stage name as the report prints it.
    pub fn name(self) -> &'static str {
        match self {
            Self::Parsed => "parsed",
            Self::Represented => "represented",
            Self::Lowered => "lowered",
            Self::Executed => "executed",
        }
    }
}

/// What VINDEX3 claims about one execution-semantic config leaf, and the
/// means of checking the claim.
pub struct CarriageRule {
    /// Flattened config leaf name this rule governs (`rope_type`), matched
    /// after the container path — `text_config.rope_parameters.rope_type`
    /// and `rope_scaling.rope_type` share one rule, because they are the
    /// same fact under two spellings.
    pub leaf: &'static str,
    /// The deepest stage VINDEX3 carries this fact to.
    pub reaches: Carriage,
    /// Where in the schema it lands (or why it stops), printed in the
    /// finding so a reader never has to grep for the answer.
    pub site: &'static str,
    /// Reads the carried value back off the built component. `None` when
    /// the component cannot answer (no surface, no attention table); the
    /// gate then reports carriage without a value comparison rather than
    /// inventing a disagreement.
    ///
    /// Required for [`Carriage::Represented`] and deeper, and unused for
    /// [`Carriage::Parsed`] — a rule that stops at the parser has nothing
    /// to read back.
    pub probe: Option<fn(&Component) -> Option<Value>>,
}

/// The rules. Every leaf classified
/// [`ExecutionSemantic`](super::report::SemanticClass::ExecutionSemantic)
/// must appear here or block.
///
/// Adding a key here is a claim about the VINDEX3 schema, not about the
/// parser — which is the whole point of the module.
pub const CARRIAGE_RULES: &[CarriageRule] = &[
    // ── Position ────────────────────────────────────────────────────
    CarriageRule {
        leaf: "rope_theta",
        reaches: Carriage::Lowered,
        site: "Component.attention[].position (PositionPolicy::Rope) → AttentionOp.position",
        probe: Some(probe_rope_theta),
    },
    CarriageRule {
        leaf: "layer_rope_theta",
        reaches: Carriage::Lowered,
        site: "Component.attention[].position, per layer → AttentionOp.position",
        probe: Some(probe_layer_rope_theta),
    },
    CarriageRule {
        leaf: "rope_type",
        reaches: Carriage::Represented,
        // PositionPolicy is `Rope { theta } | None`. It can state that a
        // layer is rotary or has no position encoding, and nothing else —
        // so the only rope *class* it can represent is the unscaled one.
        site: "Component.attention[].position — PositionPolicy expresses unscaled rope only",
        probe: Some(probe_rope_type),
    },
    // ── Span policy ─────────────────────────────────────────────────
    CarriageRule {
        leaf: "layer_types",
        reaches: Carriage::Lowered,
        site: "Component.attention[].span → AttentionOp.span",
        probe: Some(probe_layer_types),
    },
    CarriageRule {
        leaf: "sliding_window",
        reaches: Carriage::Lowered,
        site: "Component.attention[].window → AttentionOp.window",
        probe: Some(probe_sliding_window),
    },
    // ── Norms ───────────────────────────────────────────────────────
    CarriageRule {
        leaf: "rms_norm_eps",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.norm.pre.eps → NormOp.eps",
        probe: Some(probe_pre_norm_eps),
    },
    CarriageRule {
        leaf: "layer_norm_eps",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.norm.pre.eps → NormOp.eps",
        probe: Some(probe_pre_norm_eps),
    },
    CarriageRule {
        leaf: "norm_epsilon",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.norm.pre.eps → NormOp.eps",
        probe: Some(probe_pre_norm_eps),
    },
    CarriageRule {
        leaf: "layer_norm_epsilon",
        reaches: Carriage::Lowered,
        // GPT-2's spelling; `detect/parser.rs:292` folds it into the same
        // `norm_eps` read as its three siblings above.
        site: "ExecutionSurface.norm.pre.eps → NormOp.eps",
        probe: Some(probe_pre_norm_eps),
    },
    CarriageRule {
        leaf: "post_norm_eps",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.norm.post.eps → NormOp.eps at the post sites",
        probe: Some(probe_post_norm_eps),
    },
    // ── FFN ─────────────────────────────────────────────────────────
    CarriageRule {
        leaf: "hidden_act",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.ffn.activation → FfnOp.activation",
        probe: Some(probe_activation),
    },
    CarriageRule {
        leaf: "hidden_activation",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.ffn.activation → FfnOp.activation",
        probe: Some(probe_activation),
    },
    // ── Attention/output scaling ────────────────────────────────────
    CarriageRule {
        leaf: "qk_scale_factor",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.attention.query_scale → AttentionOp.query_scale",
        probe: Some(probe_query_scale),
    },
    CarriageRule {
        leaf: "query_pre_attn_scalar",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.attention.score_scale → AttentionOp.score_scale",
        probe: Some(probe_score_scale),
    },
    CarriageRule {
        leaf: "attn_logit_softcapping",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.attention.logit_softcapping → AttentionOp.logit_softcapping",
        probe: Some(probe_attn_softcap),
    },
    CarriageRule {
        leaf: "final_logit_softcapping",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.head.final_logit_softcapping → OutputOp.softcapping",
        probe: Some(probe_final_softcap),
    },
    CarriageRule {
        leaf: "output_multiplier",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.head.output_multiplier → OutputOp.multiplier",
        probe: Some(probe_output_multiplier),
    },
    CarriageRule {
        leaf: "embedding_multiplier",
        reaches: Carriage::Lowered,
        // Granite's embedding-scale operation, wired through
        // `GraniteArch::embed_scale()` (`config/architecture.rs`) into
        // `HeadSurface.embed_scale` and on into `EmbeddingOp.scale`
        // (`opplan/build.rs`).
        site: "ExecutionSurface.head.embed_scale → EmbeddingOp.scale",
        probe: Some(probe_embed_scale),
    },
    CarriageRule {
        leaf: "attention_multiplier",
        reaches: Carriage::Lowered,
        // NOT `qk_scale_factor`/`query_scale` — Granite's attention_multiplier
        // *replaces* the standard 1/sqrt(head_dim) score scale rather than
        // multiplying on top of it (every legacy-path call site treats it
        // that way, and the declared value — 1/head_dim — confirms it
        // numerically). `ModelArchitecture::attention_scale`'s default
        // resolves it into `score_scale` accordingly.
        site: "ExecutionSurface.attention.score_scale → AttentionOp.score_scale",
        probe: Some(probe_score_scale),
    },
    CarriageRule {
        leaf: "logits_scaling",
        reaches: Carriage::Lowered,
        // Granite's spelling of `output_multiplier` — algebraically the
        // same operation (scaling commutes through the linear head, so
        // "before the vocab projection" and "on the logits" are the same
        // number), resolved by `ModelArchitecture::output_multiplier`'s
        // default the same way `attention_multiplier` resolves above.
        site: "ExecutionSurface.head.output_multiplier → OutputOp.multiplier",
        probe: Some(probe_output_multiplier),
    },
    CarriageRule {
        leaf: "residual_multiplier",
        reaches: Carriage::Lowered,
        // Granite's residual-stream scale: the sublayer's own output
        // (attention or FFN) is multiplied by this before its residual
        // add, at both sites — no other family in this registry scales
        // the residual stream, so this is new schema (A-11.3), not a
        // second spelling of an existing field.
        site: "ExecutionSurface.residual_scale → LayerPlan.residual_scale",
        probe: Some(probe_residual_scale),
    },
    // ── Facts that stop at the parser, reviewed ─────────────────────
    CarriageRule {
        leaf: "attention_bias",
        reaches: Carriage::Parsed,
        // VINDEX3 has no `attention_bias` field; what it has instead is
        // operand closure, which refuses any bias tensor it cannot
        // classify into a declared op. For a model that declares `false`
        // the two agree trivially. For one that declares `true` the bias
        // operands themselves block at G5b — a stronger check than a
        // boolean, and the reason this is judged rather than a hole.
        // MOE1 gives the projections explicit bias operands.
        site: "no schema field — carried instead as operand evidence, gated by G5b closure",
        probe: None,
    },
    CarriageRule {
        leaf: "mlp_bias",
        reaches: Carriage::Parsed,
        // Same argument as `attention_bias` immediately above: VINDEX3 has
        // no `mlp_bias` field, and operand closure over the FFN's actual
        // bias tensors (or their absence) is the real gate. Granite 4.1
        // declares `false` on 3B/8B/30B, which agrees trivially; a
        // checkpoint declaring `true` blocks at G5b if the projections
        // don't carry bias operands, not here.
        site: "no schema field — carried instead as operand evidence, gated by G5b closure",
        probe: None,
    },
    CarriageRule {
        leaf: "max_position_embeddings",
        reaches: Carriage::Parsed,
        // A serving/KV-allocation bound, not a forward-pass semantic: no
        // op reads it, and two checkpoints differing only here compute
        // identical logits for any prompt both can hold. Recorded so the
        // absence is a judgement on the report rather than a silence.
        site: "no schema field — a KV-allocation bound, read by no generic op",
        probe: None,
    },
];

/// The rule governing a config leaf, if any.
pub fn rule_for(leaf: &str) -> Option<&'static CarriageRule> {
    CARRIAGE_RULES.iter().find(|rule| rule.leaf == leaf)
}

/// Canonicalises a declared config value into the vocabulary a probe's
/// carried value uses, for leaves where VINDEX3 legitimately stores a
/// *renamed* or *derived* form of the same fact rather than the
/// checkpoint's own spelling.
///
/// This is not a tolerance knob: each arm reuses the one conversion the
/// parser (or the runtime) already applies, so agreement here means the
/// same fact was recognised twice by the same rule, not that comparison
/// was loosened. A leaf with no arm here falls through unchanged, so
/// [`super::values_agree`] still requires byte-for-byte (or f32-precision)
/// identity — this function only ever narrows a `mismatched` finding to
/// `representable`, never the reverse, and callers still show the raw
/// declared value in the finding regardless of what this returns.
pub fn canonical_declared(leaf: &str, declared: &Value) -> Value {
    match leaf {
        // HF spells the tanh-approximated GELU several ways
        // (`gelu_new`, `gelu_pytorch_tanh`); `Activation::from_hf_name` is
        // the one name↔variant table the parser itself reads, so a probe
        // reading back `Activation::GeluTanh` as `"gelu_tanh"` is the same
        // fact as a declared `"gelu_pytorch_tanh"`, not a dropped one.
        "hidden_act" | "hidden_activation" => declared
            .as_str()
            .and_then(Activation::from_hf_name)
            .and_then(|activation| serde_json::to_value(activation).ok())
            .unwrap_or_else(|| declared.clone()),
        // The checkpoint declares the raw scalar; VINDEX3's execution
        // surface stores the score scale execution actually reads —
        // `scalar.powf(-0.5)`, the identical formula
        // `ModelArchitecture::attention_scale` applies at runtime, called
        // through the one shared function rather than re-derived here.
        "query_pre_attn_scalar" => declared
            .as_f64()
            .map(|scalar| json!(score_scale_from_query_pre_attn_scalar(scalar)))
            .unwrap_or_else(|| declared.clone()),
        _ => declared.clone(),
    }
}

// ── Probes ──────────────────────────────────────────────────────────
//
// Each reads what the *built graph* holds, so a rule's claim is checked
// against the schema rather than believed. They return `None` when the
// component has no surface or table to answer from.

/// The uniform rope base across the attention table, when there is one.
/// A per-layer split (Muse-Glimmer's `layer_rope_theta`) answers `None`
/// here and is checked by [`probe_layer_rope_theta`] instead.
fn probe_rope_theta(component: &Component) -> Option<Value> {
    let table = component.attention.as_ref()?;
    let mut thetas = table.iter().filter_map(|l| l.position.rope_theta());
    let first = thetas.next()?;
    thetas.all(|t| t == first).then(|| json!(first))
}

/// Every layer's rope base in layer order, with NoPE layers as `0` —
/// the same sentinel spelling the checkpoints use.
fn probe_layer_rope_theta(component: &Component) -> Option<Value> {
    let table = component.attention.as_ref()?;
    Some(Value::Array(
        table
            .iter()
            .map(|l| json!(l.position.rope_theta().unwrap_or(0.0)))
            .collect(),
    ))
}

/// The rope *class* the schema can express. `PositionPolicy` has no
/// scaling variant, so an all-rotary (or NoPE) table can only mean
/// unscaled rope — which is exactly the claim to compare against a
/// declared `rope_type`.
fn probe_rope_type(component: &Component) -> Option<Value> {
    component.attention.as_ref()?;
    Some(json!("default"))
}

/// Per-layer span kinds in the checkpoint's own vocabulary, so the
/// comparison is against the declared spelling rather than a rendering
/// this probe invents.
fn probe_layer_types(component: &Component) -> Option<Value> {
    let table = component.attention.as_ref()?;
    Some(Value::Array(
        table
            .iter()
            .map(|l| json!(l.span.declared_name()))
            .collect(),
    ))
}

/// The uniform sliding window across sliding layers, when there is one.
fn probe_sliding_window(component: &Component) -> Option<Value> {
    let table = component.attention.as_ref()?;
    let mut windows = table.iter().filter_map(|l| l.window);
    let first = windows.next()?;
    windows.all(|w| w == first).then(|| json!(first))
}

fn probe_pre_norm_eps(component: &Component) -> Option<Value> {
    Some(json!(component.execution.as_ref()?.norm.pre.eps))
}

fn probe_post_norm_eps(component: &Component) -> Option<Value> {
    Some(json!(component.execution.as_ref()?.norm.post?.eps))
}

fn probe_activation(component: &Component) -> Option<Value> {
    let activation = component.execution.as_ref()?.ffn.activation;
    serde_json::to_value(activation).ok()
}

fn probe_query_scale(component: &Component) -> Option<Value> {
    Some(json!(component.execution.as_ref()?.attention.query_scale?))
}

fn probe_score_scale(component: &Component) -> Option<Value> {
    Some(json!(component.execution.as_ref()?.attention.score_scale))
}

fn probe_attn_softcap(component: &Component) -> Option<Value> {
    Some(json!(
        component.execution.as_ref()?.attention.logit_softcapping?
    ))
}

fn probe_final_softcap(component: &Component) -> Option<Value> {
    Some(json!(
        component
            .execution
            .as_ref()?
            .head
            .as_ref()?
            .final_logit_softcapping?
    ))
}

fn probe_output_multiplier(component: &Component) -> Option<Value> {
    Some(json!(
        component
            .execution
            .as_ref()?
            .head
            .as_ref()?
            .output_multiplier?
    ))
}

fn probe_embed_scale(component: &Component) -> Option<Value> {
    Some(json!(
        component.execution.as_ref()?.head.as_ref()?.embed_scale?
    ))
}

fn probe_residual_scale(component: &Component) -> Option<Value> {
    Some(json!(component.execution.as_ref()?.residual_scale?))
}
