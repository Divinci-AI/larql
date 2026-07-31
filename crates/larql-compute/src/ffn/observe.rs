//! Opt-in FFN activation observation — the `forward`/`forward_observed`
//! split (2026-07-30 vindex/WalkFFN review, item 15).
//!
//! Activations used to be a mandatory second return value on every
//! [`super::FfnBackend`] forward, which most paths didn't honestly
//! produce: sparse walk paths zero-filled a dense `seq_len ×
//! intermediate` buffer they mostly didn't write, cache hits fabricated
//! zeros, and ordinary generation paid the allocation just to discard
//! it. [`FfnActivations`] makes observation opt-in and honest:
//!
//! - [`FfnActivations::Dense`] — every feature was computed (dense
//!   paths, where the activation matrix is an intrinsic intermediate).
//! - [`FfnActivations::Sparse`] — only the listed features were
//!   computed; every unlisted feature contributed exactly zero to the
//!   executed output, so densifying is faithful to the execution.
//! - [`FfnActivations::Absent`] — the executed path observed nothing
//!   (cache hit, remote backend, MoE). Never zeros: consumers can tell
//!   "not observed" from "observed as zero".

use ndarray::Array2;

/// Reason for the trait-default [`super::FfnBackend::forward_observed`]:
/// the backend ran, but it does not implement activation observation.
pub const REASON_UNOBSERVED_BACKEND: &str =
    "backend does not observe activations (forward_observed not overridden)";

/// Honest record of the pre-down activations an executed FFN path
/// actually computed. See the module doc for variant semantics.
#[derive(Debug, Clone, PartialEq)]
pub enum FfnActivations {
    /// Every feature computed: `[seq_len, num_features]`.
    Dense(Array2<f32>),
    /// Only the recorded `(feature, activation)` pairs were computed.
    Sparse(SparseActivations),
    /// The executed path produced no observation. `reason` names the
    /// path that declined (cache hit, remote, unobserved backend).
    Absent { reason: &'static str },
}

impl FfnActivations {
    /// The trait-default observation for backends that don't observe.
    pub fn unobserved() -> Self {
        FfnActivations::Absent {
            reason: REASON_UNOBSERVED_BACKEND,
        }
    }

    /// True when the executed path produced no observation.
    pub fn is_absent(&self) -> bool {
        matches!(self, FfnActivations::Absent { .. })
    }

    /// The decline reason, when absent.
    pub fn absent_reason(&self) -> Option<&'static str> {
        match self {
            FfnActivations::Absent { reason } => Some(reason),
            _ => None,
        }
    }

    /// Materialise as a dense `[seq_len, num_features]` matrix.
    ///
    /// `Sparse` densification is faithful to the execution: features the
    /// path never selected contributed exactly zero to its output.
    /// `Absent` returns `None` — there is nothing honest to materialise.
    pub fn into_dense(self) -> Option<Array2<f32>> {
        match self {
            FfnActivations::Dense(a) => Some(a),
            FfnActivations::Sparse(s) => Some(s.to_dense()),
            FfnActivations::Absent { .. } => None,
        }
    }
}

/// Per-position `(feature, activation)` pairs from a sparse FFN path.
/// One entry per feature the path actually computed — no zero-fill.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseActivations {
    num_features: usize,
    /// `positions[s]` = pairs recorded for sequence position `s`.
    positions: Vec<Vec<(usize, f32)>>,
}

impl SparseActivations {
    /// Empty observation for `seq_len` positions over a layer with
    /// `num_features` features.
    pub fn new(seq_len: usize, num_features: usize) -> Self {
        Self {
            num_features,
            positions: vec![Vec::new(); seq_len],
        }
    }

    /// Record one computed activation. Panics on an out-of-range
    /// position or feature — a mis-indexed observation is a bug, not
    /// data.
    pub fn record(&mut self, position: usize, feature: usize, activation: f32) {
        assert!(
            feature < self.num_features,
            "SparseActivations::record: feature {feature} >= num_features {}",
            self.num_features
        );
        self.positions[position].push((feature, activation));
    }

    pub fn seq_len(&self) -> usize {
        self.positions.len()
    }

    pub fn num_features(&self) -> usize {
        self.num_features
    }

    /// Pairs recorded for one position.
    pub fn position(&self, position: usize) -> &[(usize, f32)] {
        &self.positions[position]
    }

    /// Densify to `[seq_len, num_features]`. Repeated records of the
    /// same feature accumulate — matching how repeated contributions
    /// would have summed into the executed output.
    pub fn to_dense(&self) -> Array2<f32> {
        let mut dense = Array2::<f32>::zeros((self.positions.len(), self.num_features));
        for (s, pairs) in self.positions.iter().enumerate() {
            for &(feature, activation) in pairs {
                dense[[s, feature]] += activation;
            }
        }
        dense
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unobserved_is_absent_with_the_named_reason() {
        let obs = FfnActivations::unobserved();
        assert!(obs.is_absent());
        assert_eq!(obs.absent_reason(), Some(REASON_UNOBSERVED_BACKEND));
        assert_eq!(obs.into_dense(), None);
    }

    #[test]
    fn dense_round_trips_through_into_dense() {
        let a = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let obs = FfnActivations::Dense(a.clone());
        assert!(!obs.is_absent());
        assert_eq!(obs.absent_reason(), None);
        assert_eq!(obs.into_dense(), Some(a));
    }

    #[test]
    fn sparse_records_and_densifies_only_what_was_computed() {
        let mut s = SparseActivations::new(2, 4);
        assert_eq!(s.seq_len(), 2);
        assert_eq!(s.num_features(), 4);
        s.record(0, 1, 0.5);
        s.record(1, 3, -2.0);
        assert_eq!(s.position(0), &[(1, 0.5)]);
        assert_eq!(s.position(1), &[(3, -2.0)]);
        let dense = FfnActivations::Sparse(s).into_dense().unwrap();
        assert_eq!(dense.shape(), &[2, 4]);
        assert_eq!(dense[[0, 1]], 0.5);
        assert_eq!(dense[[1, 3]], -2.0);
        // Every unrecorded slot is zero — the honest zero of "this
        // feature contributed nothing to the executed output".
        assert_eq!(dense.iter().filter(|v| **v != 0.0).count(), 2);
    }

    #[test]
    fn sparse_duplicate_records_accumulate() {
        let mut s = SparseActivations::new(1, 2);
        s.record(0, 0, 1.0);
        s.record(0, 0, 0.25);
        assert_eq!(s.to_dense()[[0, 0]], 1.25);
    }

    #[test]
    #[should_panic(expected = "feature 5 >= num_features 2")]
    fn sparse_record_panics_on_out_of_range_feature() {
        SparseActivations::new(1, 2).record(0, 5, 1.0);
    }

    #[test]
    fn sparse_equality_is_structural() {
        let mut a = SparseActivations::new(1, 3);
        a.record(0, 2, 1.0);
        let mut b = SparseActivations::new(1, 3);
        b.record(0, 2, 1.0);
        assert_eq!(FfnActivations::Sparse(a), FfnActivations::Sparse(b.clone()));
        b.record(0, 1, 4.0);
        assert!(FfnActivations::Sparse(b) != FfnActivations::unobserved());
    }
}
