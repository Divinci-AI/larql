//! Last-position-ablating FFN backend for crown-layer discovery.
//!
//! Wraps another `FfnBackend` and zeroes its output at the last-token row
//! for a single target layer. Used by `larql dev crown` to measure each
//! MLP's causal contribution to the final-token prediction — the layer
//! whose ablation maximally suppresses the expected token is the "crown"
//! writer.
//!
//! Implements the Phase 125c methodology from Divinci-AI's mechanistic
//! interpretability chapters (CHAPTER_17_CORONATION.md).

use std::error::Error;
use std::fmt;

use larql_compute::ffn::FfnActivations;
use larql_execution::{BoxRefusal, ExecutionRefusal, RefusalKind};
use ndarray::Array2;

use super::FfnBackend;

/// The MoE full-layer path returns `h_out` (post-residual), not the FFN
/// output these wrappers are defined over, so there is no row that
/// corresponds to "the FFN's contribution" to zero. Refusing is the honest
/// answer: silently returning the inner backend's unablated `h_out` would
/// report a zero causal effect for every MoE layer, which reads as a
/// measurement rather than a gap.
#[derive(Debug)]
pub(super) struct MoeFullLayerUnsupported {
    pub(super) wrapper: &'static str,
    pub(super) layer: usize,
}

impl fmt::Display for MoeFullLayerUnsupported {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} cannot intervene on the MoE full-layer path at layer {}: \
             that path returns the post-residual layer output, not the FFN \
             contribution the wrapper is defined over",
            self.wrapper, self.layer
        )
    }
}

impl Error for MoeFullLayerUnsupported {}

impl ExecutionRefusal for MoeFullLayerUnsupported {
    fn kind(&self) -> RefusalKind {
        RefusalKind::Unsupported
    }
}

/// FFN backend that ablates its inner backend's last-token output at a
/// specific target layer. All other layers pass through unchanged.
pub struct LastPositionAblatingFfn<'a> {
    inner: &'a dyn FfnBackend,
    target_layer: usize,
}

impl<'a> LastPositionAblatingFfn<'a> {
    /// Create a new ablating wrapper around an existing FFN backend.
    /// At `target_layer`, the last-position row of the FFN output is zeroed.
    pub fn new(inner: &'a dyn FfnBackend, target_layer: usize) -> Self {
        Self {
            inner,
            target_layer,
        }
    }

    fn maybe_ablate(&self, layer: usize, out: &mut Array2<f32>) {
        if layer == self.target_layer {
            let seq = out.shape()[0];
            if seq > 0 {
                let mut last_row = out.row_mut(seq - 1);
                last_row.fill(0.0);
            }
        }
    }
}

impl FfnBackend for LastPositionAblatingFfn<'_> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        let mut out = self.inner.forward(layer, x);
        self.maybe_ablate(layer, &mut out);
        out
    }

    fn forward_observed(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, FfnActivations) {
        // The activations are the inner backend's honest record of what it
        // computed; ablation edits the output, not the account of the path.
        let (mut out, act) = self.inner.forward_observed(layer, x);
        self.maybe_ablate(layer, &mut out);
        (out, act)
    }

    fn name(&self) -> &str {
        "last-pos-ablating"
    }

    fn forward_moe_full_layer(
        &self,
        layer: usize,
        h_post_attn: &Array2<f32>,
    ) -> Result<Option<Array2<f32>>, BoxRefusal> {
        let out = self.inner.forward_moe_full_layer(layer, h_post_attn)?;
        match (layer == self.target_layer, out) {
            // The inner backend does not serve this layer, so the caller
            // falls back to `forward` — which this wrapper does ablate.
            (_, None) => Ok(None),
            (false, Some(h_out)) => Ok(Some(h_out)),
            (true, Some(_)) => Err(Box::new(MoeFullLayerUnsupported {
                wrapper: "LastPositionAblatingFfn",
                layer,
            })),
        }
    }
}
