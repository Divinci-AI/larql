//! Output types for the architecture inventory — the JSON `larql inspect-hf`
//! emits.
//!
//! Two deliberately separate sections carry the same subject matter from two
//! authorities:
//!
//! - **`config_keys`** — what the checkpoint *declares*, verbatim, with a
//!   per-key statement of whether this build's config parser reads it.
//! - **`resolved`** — what this build's detection *does* with it: the
//!   per-layer policy table the serving path would actually run.
//!
//! Disagreement between the two sections is the instrument's whole point.
//! A config fact the parser never reads (`status: unconsumed`) is the
//! [§4.7.8] failure shape *before* it ships: a field with one behavioural
//! default silently answering for every family. Five distinct serving bugs
//! have had that shape; this report makes the sixth visible from
//! `config.json` alone, with no forward pass.
//!
//! [§4.7.8]: ../../../../docs/k3-funnel.md

use serde::{Deserialize, Serialize};

/// Current inventory schema. Bump on any breaking change to these types.
///
/// v2: [`LayerPolicy`] carries a [`PositionPolicy`](crate::config::PositionPolicy)
/// instead of a bare `rope_base` — NoPE layers are a policy variant, not a
/// numeric sentinel.
pub const INVENTORY_SCHEMA: u32 = 2;

/// Machine-readable architecture inventory of one HF checkpoint directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureInventory {
    /// Always [`INVENTORY_SCHEMA`].
    pub schema: u32,
    /// The inspected directory, as given by the caller.
    pub path: String,
    pub identity: Identity,
    pub detection: Detection,
    pub resolved: ResolvedTopology,
    /// Nested sibling components (`vision_config`, …), each read into a
    /// typed topology by the generic component reader. Empty for
    /// single-component checkpoints.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub nested_components: Vec<crate::inventory::components::ComponentTopology>,
    /// Every leaf in `config.json`, flattened to a dot path, classified.
    pub config_keys: Vec<ConfigKeyFact>,
    /// Declared cross-component interfaces (see
    /// [`super::config_keys::KNOWN_INTERFACE_KEYS`]).
    pub interfaces: Vec<InterfaceFact>,
    pub tensors: TensorInventory,
}

/// What the checkpoint says it is, before any interpretation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Identity {
    /// `model_type`, read from `text_config` first, then top level.
    pub model_type: String,
    /// HF `architectures` list, verbatim.
    pub architectures: Vec<String>,
    /// Checkpoint dtype (`dtype` or `torch_dtype`).
    pub dtype: Option<String>,
    pub transformers_version: Option<String>,
    /// Nested component configs present (`text_config`, `vision_config`, …).
    /// A flat config reports none.
    pub components: Vec<String>,
}

/// What this build's detection resolves the checkpoint to.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Detection {
    /// `ModelArchitecture::family()` of the detected implementation.
    pub family: String,
    /// True when `model_type` matched no registry entry and detection fell
    /// through to the generic architecture. A generic fallback on a model
    /// with unconsumed config keys is the loudest red flag this report can
    /// raise: the model will load and serve, and serve wrong.
    pub generic_fallback: bool,
    /// Attention kind from the registry entry, when one matched.
    pub attention_kind: Option<String>,
    /// `ModelArchitecture::validate()` findings, carried as data — an
    /// inventory of an unsupported model must describe it, not refuse it.
    pub validation_errors: Vec<String>,
}

/// The topology the serving path would run, including the per-layer
/// attention policy table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedTopology {
    pub num_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: Option<usize>,
    pub sliding_window: Option<usize>,
    pub attention: AttentionSummary,
    /// One entry per layer, in layer order.
    pub layers: Vec<LayerPolicy>,
}

/// Counts over [`ResolvedTopology::layers`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionSummary {
    pub sliding_layers: usize,
    pub full_layers: usize,
}

/// Per-layer attention policy as the architecture trait resolves it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerPolicy {
    pub layer: usize,
    /// `"sliding"` or `"full"`.
    pub attention: String,
    /// Window size when sliding, `None` on full-attention layers.
    pub window: Option<usize>,
    /// How this layer encodes position — rotary at a base, or not at all.
    pub position: crate::config::PositionPolicy,
    pub head_dim: usize,
    pub num_kv_heads: usize,
}

/// One flattened `config.json` leaf.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigKeyFact {
    /// Dot path from the config root, e.g. `text_config.qk_scale_factor`.
    pub path: String,
    /// The leaf value, verbatim. Arrays are leaves.
    pub value: serde_json::Value,
    pub status: KeyStatus,
}

/// Whether this build's config parser reads a key.
///
/// Scope: classification is against `larql-models`' `ModelConfig` parser
/// only. Keys other subsystems read (tokenizer ids, vision loaders) are not
/// credited here — `metadata` covers the identity/training facts known to be
/// inert for a text forward pass, and everything else unread is
/// `unconsumed`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum KeyStatus {
    /// The config parser reads this key (directly or via an alias list).
    Consumed,
    /// Identity or training-time fact with no bearing on a forward pass.
    Metadata,
    /// Declared by the checkpoint, read by nothing in the config parser.
    /// Every entry here is a potential silent-default serving bug.
    Unconsumed,
}

/// A config field that declares a cross-component interface — e.g. a
/// drafter's `target_layer_ids` naming the target hidden states it consumes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfaceFact {
    pub path: String,
    pub value: serde_json::Value,
}

/// Tensor-level inventory from safetensors headers. Shapes and dtypes come
/// from the shard headers alone; no tensor data is read.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorInventory {
    /// Shard filenames scanned, relative to the model dir, sorted.
    pub files: Vec<String>,
    pub total_tensors: usize,
    pub total_bytes: u64,
    /// Totals grouped by name prefix (path up to the first numeric
    /// segment), sorted by prefix.
    pub groups: Vec<TensorGroup>,
    /// Every tensor, sorted by name.
    pub tensors: Vec<TensorFact>,
}

/// Aggregate over one name prefix, e.g. `model.language_model.layers`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorGroup {
    pub prefix: String,
    pub tensors: usize,
    pub bytes: u64,
}

/// One stored tensor, as its shard header describes it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorFact {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub bytes: u64,
    /// Shard filename holding the bytes, relative to the model dir.
    pub file: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_status_serialises_lowercase() {
        assert_eq!(
            serde_json::to_string(&KeyStatus::Unconsumed).unwrap(),
            "\"unconsumed\""
        );
        assert_eq!(
            serde_json::to_string(&KeyStatus::Consumed).unwrap(),
            "\"consumed\""
        );
        assert_eq!(
            serde_json::to_string(&KeyStatus::Metadata).unwrap(),
            "\"metadata\""
        );
    }

    #[test]
    fn key_status_round_trips() {
        for status in [
            KeyStatus::Consumed,
            KeyStatus::Metadata,
            KeyStatus::Unconsumed,
        ] {
            let json = serde_json::to_string(&status).unwrap();
            let back: KeyStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(back, status);
        }
    }
}
