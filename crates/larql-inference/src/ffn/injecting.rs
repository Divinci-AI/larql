//! Last-position-injecting FFN backend for activation-steering search.
//!
//! Wraps another `FfnBackend` and ADDS a fixed delta vector to its output
//! at the last-token row for a single target layer. Symmetric to
//! [`LastPositionAblatingFfn`](super::LastPositionAblatingFfn) (ablating
//! zeroes; this adds).
//!
//! Used by `larql dev edit` to binary-search the minimum scale at which a
//! steering vector flips the prompt's top prediction — implements Phase 130
//! from CHAPTER_18_THE_EDIT.md in the Divinci-AI research series.

use larql_compute::ffn::FfnActivations;
use larql_execution::BoxRefusal;
use ndarray::Array2;

use super::ablating::MoeFullLayerUnsupported;
use super::FfnBackend;

/// FFN backend that adds a fixed `delta` vector to its inner backend's
/// output at the last-token row at a specific target layer. All other
/// layers (and other positions within the target layer) pass through.
pub struct LastPositionInjectingFfn<'a> {
    inner: &'a dyn FfnBackend,
    target_layer: usize,
    /// Vector of shape `[hidden_size]`, added to the last-position output.
    delta: Vec<f32>,
}

impl<'a> LastPositionInjectingFfn<'a> {
    /// Create a new injecting wrapper. `delta.len()` must equal the model's
    /// hidden size (verified at forward time against `out.shape()[1]`).
    pub fn new(inner: &'a dyn FfnBackend, target_layer: usize, delta: Vec<f32>) -> Self {
        Self {
            inner,
            target_layer,
            delta,
        }
    }

    fn maybe_inject(&self, layer: usize, out: &mut Array2<f32>) {
        if layer == self.target_layer {
            let seq = out.shape()[0];
            let hidden = out.shape()[1];
            if seq > 0 && hidden == self.delta.len() {
                let mut last_row = out.row_mut(seq - 1);
                for (i, val) in last_row.iter_mut().enumerate() {
                    *val += self.delta[i];
                }
            }
        }
    }
}

impl FfnBackend for LastPositionInjectingFfn<'_> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        let mut out = self.inner.forward(layer, x);
        self.maybe_inject(layer, &mut out);
        out
    }

    fn forward_observed(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, FfnActivations) {
        // Injection edits the output; the inner backend's activation record
        // still describes the path that actually ran.
        let (mut out, act) = self.inner.forward_observed(layer, x);
        self.maybe_inject(layer, &mut out);
        (out, act)
    }

    fn name(&self) -> &str {
        "last-pos-injecting"
    }

    fn forward_moe_full_layer(
        &self,
        layer: usize,
        h_post_attn: &Array2<f32>,
    ) -> Result<Option<Array2<f32>>, BoxRefusal> {
        let out = self.inner.forward_moe_full_layer(layer, h_post_attn)?;
        match (layer == self.target_layer, out) {
            (_, None) => Ok(None),
            (false, Some(h_out)) => Ok(Some(h_out)),
            (true, Some(_)) => Err(Box::new(MoeFullLayerUnsupported {
                wrapper: "LastPositionInjectingFfn",
                layer,
            })),
        }
    }
}
