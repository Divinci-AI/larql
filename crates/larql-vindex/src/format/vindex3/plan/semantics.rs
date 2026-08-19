//! Semantic classification of config keys the parser does not consume.
//!
//! The registry maps *known* HF config-field names to a semantic class.
//! It is a vocabulary of the HF config format, not of any model family —
//! `qk_scale_factor` means an attention-scale override whichever checkpoint
//! declares it. A name the registry has never seen classifies as
//! [`SemanticClass::Unknown`], which blocks the plan: an unjudged key must
//! not pass silently, because "unconsumed and unjudged" is exactly the
//! silent-default shape the whole instrument exists to catch.

use super::report::SemanticClass;

/// Keys that change what a forward pass computes: norms, activations,
/// position encoding, attention/output scaling, attention span policy.
pub const EXECUTION_SEMANTIC_KEYS: &[&str] = &[
    "layer_rope_theta",
    "qk_scale_factor",
    "output_multiplier",
    "post_norm_eps",
    "hidden_activation",
    "hidden_act",
    "attention_bias",
    "mlp_bias",
    "layer_norm_eps",
    "rms_norm_eps",
    "norm_epsilon",
    "rope_theta",
    "rope_type",
    "layer_types",
    "sliding_window",
    "max_position_embeddings",
    "num_kv_shared_layers",
    "query_pre_attn_scalar",
    "final_logit_softcapping",
    "attn_logit_softcapping",
    "partial_rotary_factor",
    // GPT-OSS clamps both halves of the fused gate/up projection at
    // ±this value before the GLU. It changes what the FFN computes, so
    // it is execution-semantic wherever it is declared.
    "swiglu_limit",
    // GPT-2's spelling of the norm epsilon `rms_norm_eps` etc. already
    // cover — same fact, fourth name; `parser.rs` folds all four into one
    // `norm_eps` read, so this shares `rms_norm_eps`'s carriage rule.
    "layer_norm_epsilon",
    // Per-layer attention geometry and behaviour (A-9/A-11 census,
    // 2026-08-18: these were `consumed` but absent from every registry
    // here, so they silently graded `representable` instead of blocking —
    // the exact "parsed but unjudged" shape this module exists to name).
    // Which layers are sliding vs full.
    "sliding_window_pattern",
    // A second rope base for local/sliding layers, alongside `rope_theta`.
    "rope_local_base_freq",
    // Whether K and V share storage — changes what attention reads.
    "attention_k_eq_v",
    // Whether the FFN routes through MoE at all.
    "enable_moe_block",
    // Whether router weights are renormalised after top-k selection.
    "norm_topk_prob",
    // Routing width: how many experts activate per token.
    "num_experts_per_tok",
    "num_experts_per_token",
    "top_k_experts",
    // The rope-scaling (YaRN / Llama-3-style) block's own leaves, besides
    // `rope_type` — every one of them is consumed and changes what rope
    // computes, and none has a schema field yet (the A-9.0 YaRN work).
    "type",
    "factor",
    "low_freq_factor",
    "high_freq_factor",
    "original_max_position_embeddings",
    "beta_fast",
    "beta_slow",
    "truncate",
    "mscale",
    "mscale_all_dim",
    // Granite-style scaling multipliers (A-11.1): consumed into
    // `ModelConfig` but not yet carried past it — `embedding_multiplier`
    // is the one exception, wired through `embed_scale()`. See A-11.2/.3
    // in ROADMAP.md for the schema work that gives the other three a
    // canonical home instead of borrowing `qk_scale_factor` /
    // `output_multiplier`'s names.
    "embedding_multiplier",
    "attention_multiplier",
    "residual_multiplier",
    "logits_scaling",
];

/// Keys that describe stored operands: widths, depths, head geometry,
/// patching — the shape of what a container would have to hold.
pub const TENSOR_SEMANTIC_KEYS: &[&str] = &[
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "vocab_size",
    "out_hidden_size",
    "projector_hidden_size",
    "projector_hidden_act",
    "merge_size",
    "patch_size",
    "patch_temporal",
    "pos_emb_height",
    "pos_emb_width",
    // GPT-2 aliases of shape fields above (`hidden_size`, `num_hidden_layers`,
    // `intermediate_size`, `num_attention_heads` respectively).
    "n_embd",
    "n_layer",
    "n_inner",
    "n_head",
    // Per-layer attention geometry (Gemma 4 style global/local split) —
    // widths, not behaviour.
    "global_head_dim",
    "num_global_key_value_heads",
    // Per-layer embedding width (PLE).
    "hidden_size_per_layer_input",
    // MoE operand counts: how many expert tensors exist, not how the
    // forward pass selects among them (that's `num_experts_per_tok` etc.,
    // in `EXECUTION_SEMANTIC_KEYS`).
    "n_routed_experts",
    "num_local_experts",
    "num_experts",
    "n_shared_experts",
    "moe_intermediate_size",
    // MLA (DeepSeek-style) head/rank geometry.
    "kv_lora_rank",
    "q_lora_rank",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "v_head_dim",
];

/// Keys that declare a cross-component contract: hidden-state taps, block
/// protocols, special-token roles in a multimodal or drafter interface.
pub const INTERFACE_SEMANTIC_KEYS: &[&str] = &[
    "target_layer_ids",
    "block_size",
    "mask_token_id",
    "image_token_id",
    "video_token_id",
];

/// Identity facts inert for a forward pass wherever they appear.
pub const METADATA_KEYS: &[&str] = &[
    "model_type",
    "tie_word_embeddings",
    // `rope_scaling` as a bare leaf (not recursed into) means its value is
    // not an object — in every checkpoint on hand, `null`. A non-null
    // object never reaches this leaf; it flattens into `rope_type`/
    // `factor`/etc. instead, covered above. So a bare `rope_scaling` fact
    // carries no scaling information to lose — the same claim
    // `max_position_embeddings` makes about itself, just true unconditionally
    // here rather than by schema absence.
    "rope_scaling",
    // HF's serving-time KV-cache implementation selector (`"hybrid"`,
    // `"static"`, …) — which cache *class* generation code should
    // instantiate to hold a mix of sliding/full attention layers
    // efficiently. It names a consequence of the per-layer attention
    // topology, not an independent forward-pass fact: the topology itself
    // is declared elsewhere (`sliding_window` + the architecture's layer
    // alternation, e.g. Gemma 2's fixed period-2 pattern) and VINDEX3
    // already carries *that*, per layer, in the attention table. Two
    // checkpoints differing only in `cache_implementation` compute
    // identical logits for any prompt both can hold.
    "cache_implementation",
];

/// Keys that parameterise *training* and are inert at inference. Each
/// entry must name the training-time path it belongs to, because "we
/// don't run that" is the entire justification for dropping it.
pub const TRAINING_ONLY_KEYS: &[&str] = &[
    // MoE load-balancing auxiliary loss: added to the training objective,
    // never read on a forward pass.
    "router_aux_loss_coef",
    // Whether the model *returns* router logits alongside hidden states —
    // a training/analysis output switch. It changes what is returned, not
    // what is computed, and generic execution returns logits only.
    "output_router_logits",
];

/// Redundant spellings: `alias → canonical`. An entry claims the same
/// fact is declared under `canonical` *in the same config* and read
/// there, which the gate verifies — so listing a key here cannot silence
/// it if the canonical spelling is missing or disagrees.
pub const ALIAS_KEYS: &[(&str, &str)] = &[
    // GPT-OSS declares both spellings, with the same value; the parser's
    // alias list reads `num_experts_per_tok`.
    ("experts_per_token", "num_experts_per_tok"),
    // The pre-scaling context length, also declared inside the rope
    // scaling block, which is where the parser reads it.
    (
        "initial_context_length",
        "rope_scaling.original_max_position_embeddings",
    ),
];

/// Reviewed-and-safe-to-drop keys. Empty by design until a key has actually
/// been reviewed; every future entry must carry a justification comment.
pub const IGNORED_SAFE_KEYS: &[&str] = &[];

/// The canonical spelling this leaf aliases, if it is a registered alias.
pub fn alias_canonical(leaf: &str) -> Option<&'static str> {
    ALIAS_KEYS
        .iter()
        .find(|(alias, _)| *alias == leaf)
        .map(|(_, canonical)| *canonical)
}

/// Classify an unconsumed config key by its leaf name.
pub fn classify_key(leaf: &str) -> SemanticClass {
    if EXECUTION_SEMANTIC_KEYS.contains(&leaf) {
        SemanticClass::ExecutionSemantic
    } else if TENSOR_SEMANTIC_KEYS.contains(&leaf) {
        SemanticClass::TensorSemantic
    } else if INTERFACE_SEMANTIC_KEYS.contains(&leaf) {
        SemanticClass::InterfaceSemantic
    } else if METADATA_KEYS.contains(&leaf) {
        SemanticClass::MetadataOnly
    } else if TRAINING_ONLY_KEYS.contains(&leaf) {
        SemanticClass::TrainingOnly
    } else if alias_canonical(leaf).is_some() {
        SemanticClass::Alias
    } else if IGNORED_SAFE_KEYS.contains(&leaf) {
        SemanticClass::IgnoredSafe
    } else {
        SemanticClass::Unknown
    }
}

/// Logical component a flattened config path belongs to.
///
/// `<name>_config.<rest>` attributes to `<name>` (`text_config.x` → `text`);
/// everything else is the artifact root.
pub fn component_of(path: &str) -> String {
    const CONFIG_SUFFIX: &str = "_config";
    const ROOT_COMPONENT: &str = "root";
    match path.split('.').next() {
        Some(first) if first.ends_with(CONFIG_SUFFIX) => {
            first[..first.len() - CONFIG_SUFFIX.len()].to_string()
        }
        _ => ROOT_COMPONENT.to_string(),
    }
}

/// Last dot-separated segment of a flattened path.
pub fn leaf_of(path: &str) -> &str {
    path.rsplit('.').next().unwrap_or(path)
}
