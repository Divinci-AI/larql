//! Last-position-ablating FFN backend for crown-layer discovery.
//!
//! Wraps another `FfnBackend` and zeroes its output at the last-token row
//! for a single target layer. Used by `larql crown` to measure each MLP's
//! causal contribution to the final-token prediction — the layer whose
//! ablation maximally suppresses the expected token is the "crown" writer.
//!
//! Implements the Phase 125c methodology from Divinci-AI's mechanistic
//! interpretability chapters (CHAPTER_17_CORONATION.md).

use ndarray::Array2;

use super::FfnBackend;

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
        Self { inner, target_layer }
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

impl<'a> FfnBackend for LastPositionAblatingFfn<'a> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        let mut out = self.inner.forward(layer, x);
        self.maybe_ablate(layer, &mut out);
        out
    }

    fn forward_with_activation(
        &self,
        layer: usize,
        x: &Array2<f32>,
    ) -> (Array2<f32>, Array2<f32>) {
        let (mut out, act) = self.inner.forward_with_activation(layer, x);
        self.maybe_ablate(layer, &mut out);
        (out, act)
    }

    fn name(&self) -> &str {
        "last-pos-ablating"
    }
}
