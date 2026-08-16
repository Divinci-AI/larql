//! Per-layer attention policy as the graph records it.

use larql_models::config::{
    PositionPolicy, LAYER_TYPE_FULL_ATTENTION, LAYER_TYPE_SLIDING_ATTENTION,
    LAYER_TYPE_WINDOW_ATTENTION,
};
use serde::{Deserialize, Serialize};

/// Attention span kind of one layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionSpan {
    /// Attends to the last `window` positions only.
    Sliding,
    /// Attends to the whole prefix.
    Full,
    /// Attends within a bounded region the component's own geometry
    /// defines — a perception tower's spatial window — rather than a
    /// trailing sequence window. No `window` count applies, because the
    /// extent is not a position count and the config does not declare
    /// one.
    ///
    /// Distinct from [`Self::Sliding`] on purpose. Aliasing the two would
    /// let a KV planner infer that positions beyond a window are dead,
    /// which is true of a sequence window and not of a spatial one;
    /// aliasing to [`Self::Full`] would erase the distinction the
    /// checkpoint actually declares (Muse-Glimmer's vision tower splits
    /// 37/13).
    Windowed,
}

impl AttentionSpan {
    /// The span a declared `layer_types` entry names, or `None` when the
    /// vocabulary does not contain it.
    ///
    /// Fail-closed by construction: an unrecognised spelling answers
    /// `None` so the caller refuses, rather than resolving to a
    /// behavioural default. That is the [§4.7.8] shape — `layer_types`
    /// was once parsed and validated but never consulted, so every model
    /// ran full attention on every layer — and the same shape one level
    /// up is what a "not sliding, therefore full" rule would reintroduce
    /// for any new spelling.
    ///
    /// [§4.7.8]: ../../../../../docs/k3-funnel.md
    pub fn from_declared(entry: &str) -> Option<Self> {
        if entry.eq_ignore_ascii_case(LAYER_TYPE_SLIDING_ATTENTION) {
            Some(Self::Sliding)
        } else if entry.eq_ignore_ascii_case(LAYER_TYPE_FULL_ATTENTION) {
            Some(Self::Full)
        } else if entry.eq_ignore_ascii_case(LAYER_TYPE_WINDOW_ATTENTION) {
            Some(Self::Windowed)
        } else {
            None
        }
    }

    /// The `layer_types` spelling this span corresponds to — the inverse
    /// of [`Self::from_declared`], used to compare what the graph carries
    /// against what the checkpoint declared.
    pub fn declared_name(self) -> &'static str {
        match self {
            Self::Sliding => LAYER_TYPE_SLIDING_ATTENTION,
            Self::Full => LAYER_TYPE_FULL_ATTENTION,
            Self::Windowed => LAYER_TYPE_WINDOW_ATTENTION,
        }
    }
}

/// One layer's attention policy: span, window, and positional encoding.
/// This is architectural liveness information — a KV planner reading it
/// knows that positions beyond `window` on a sliding layer are
/// *architecturally* dead, before any semantic analysis runs.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AttentionLayerPolicy {
    pub span: AttentionSpan,
    /// Window size when [`AttentionSpan::Sliding`]; `None` on full and
    /// windowed layers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub window: Option<usize>,
    /// How the layer encodes position — including intentional absence.
    pub position: PositionPolicy,
}
