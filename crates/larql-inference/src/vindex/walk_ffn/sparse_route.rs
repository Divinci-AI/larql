//! Per-position hit/route selection for the sparse walk.
//!
//! Decides WHICH features the walk touches at a position, in priority
//! order:
//!
//! 1. **Cell router** (task #22) — the residual picks its nearest cell,
//!    the cell's precomputed pool is the candidate set, scored via the
//!    O(|pool|) local gate; `rank_within_pool` optionally narrows to
//!    top-K.
//! 2. **Per-layer pool** — precomputed route (cheap O(|pool|) scoring
//!    when `precomputed_routing`) or the full-projection
//!    `pool_restricted_gate_knn` fallback.
//! 3. **Selector dispatch** — production `gate_walk` → `gate_knn_q4` →
//!    `gate_knn` chain for `GateOnly`, `joint_gate_knn` for the joint
//!    criteria.
//!
//! The scoring/selection primitives themselves live in `selector.rs`;
//! this file is only the decision tree the per-position loop calls.

use super::helpers::selection_weight_cmp_desc;
use super::WalkFfn;
use crate::vindex::walk_config::FeatureSelector;

impl<'a> WalkFfn<'a> {
    /// Select the features (and their gate scores) the sparse walk
    /// visits at one position. See the module doc for the priority
    /// order.
    pub(super) fn select_route_hits(
        &self,
        layer: usize,
        x_owned: &ndarray::Array1<f32>,
        x_slice: &[f32],
        top_k: usize,
    ) -> Vec<(usize, f32)> {
        if let Some(router) = self.config.cell_router.as_ref() {
            // Residual-cell content-addressed route (task #22): the
            // per-position residual picks its nearest cell, and that
            // cell's precomputed pool is the candidate set. Scored via
            // the O(|pool|) local gate (no full projection). Used
            // directly (the gate-KNN-union route) unless
            // `rank_within_pool` narrows it to top-K. Falls back to
            // gate-KNN when the layer has no cell pool.
            match router.pool_for(layer, x_slice) {
                Some(pool) => match self.local_pool_gate_knn(layer, x_slice, pool) {
                    Some(mut h) => {
                        if self.config.rank_within_pool && h.len() > top_k {
                            h.select_nth_unstable_by(top_k - 1, |a, b| {
                                selection_weight_cmp_desc(a.1.abs(), b.1.abs())
                            });
                            h.truncate(top_k);
                        }
                        h
                    }
                    None => self.pool_restricted_gate_knn(layer, x_owned, top_k, pool),
                },
                None => self
                    .index
                    .gate_walk(layer, x_owned, top_k)
                    .unwrap_or_else(|| self.index.gate_knn(layer, x_owned, top_k)),
            }
        } else if let Some(pool_per_layer) = self.config.pool_per_layer.as_ref() {
            let empty = Vec::new();
            let pool = pool_per_layer.get(layer).unwrap_or(&empty);
            if self.config.precomputed_routing {
                // Cheap routing: gate scored only for the pool features
                // (O(|pool|)), no full gate projection. Falls back to
                // the projection path if Q4K gate bytes are absent.
                match self.local_pool_gate_knn(layer, x_slice, pool) {
                    Some(mut h) => {
                        if self.config.rank_within_pool && h.len() > top_k {
                            // Two-stage: rank the candidate pool by
                            // |gate_score| and keep the real top-K
                            // (content-addressed within the pool).
                            h.select_nth_unstable_by(top_k - 1, |a, b| {
                                selection_weight_cmp_desc(a.1.abs(), b.1.abs())
                            });
                            h.truncate(top_k);
                        } else {
                            // Pure precomputed route — pool order.
                            h.truncate(top_k);
                        }
                        h
                    }
                    None => self.pool_restricted_gate_knn(layer, x_owned, top_k, pool),
                }
            } else {
                self.pool_restricted_gate_knn(layer, x_owned, top_k, pool)
            }
        } else {
            match self.config.selector {
                FeatureSelector::GateOnly => self
                    .index
                    .gate_walk(layer, x_owned, top_k)
                    .or_else(|| {
                        self.backend
                            .and_then(|be| self.index.gate_knn_q4(layer, x_owned, top_k, be))
                    })
                    .unwrap_or_else(|| self.index.gate_knn(layer, x_owned, top_k)),
                kind @ (FeatureSelector::GateXDownNorm
                | FeatureSelector::GateXUpDownNorm
                | FeatureSelector::GateXUpScore
                | FeatureSelector::ActXUpScoreXDownNorm
                | FeatureSelector::Random) => self.joint_gate_knn(layer, x_owned, top_k, kind),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::test_utils::{
        make_test_q4k_vindex, make_test_q4k_weights, make_test_vindex, make_test_weights,
    };
    use crate::vindex::{WalkFfn, WalkFfnConfig};
    use ndarray::Array2;

    fn x(seq: usize, hidden: usize) -> Array2<f32> {
        Array2::from_shape_vec(
            (seq, hidden),
            (0..seq * hidden).map(|i| (i as f32 + 1.0) * 0.02).collect(),
        )
        .unwrap()
    }

    /// Cheap routing (task #18): a precomputed per-layer pool with
    /// `precomputed_routing = true` drives the walk through
    /// `local_pool_gate_knn` (gate scored only for the route features),
    /// not `pool_restricted_gate_knn`. Output must be well-formed.
    #[test]
    fn walk_ffn_sparse_precomputed_routing_q4k_fixture() {
        use std::sync::Arc;
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let pool: Vec<usize> = vec![0, 1, 2];
        let cfg = WalkFfnConfig::sparse(weights.num_layers, 8)
            .with_pool_per_layer(Arc::new(vec![pool; weights.num_layers]))
            .with_precomputed_routing(true);
        let ffn = WalkFfn::from_config(&weights, &index, cfg);
        let (out, _act) = ffn
            .walk_ffn_sparse(0, &x(1, weights.hidden_size))
            .expect("precomputed-routing walk should produce output");
        assert_eq!(out.shape(), &[1, weights.hidden_size]);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    /// Residual-cell router (task #22): a per-position residual selects its
    /// nearest cell, whose pool becomes the route. With a single all-zero
    /// centroid every position lands in cell 0; output must be well-formed.
    #[test]
    fn walk_ffn_sparse_cell_router_q4k_fixture() {
        use crate::vindex::CellRouter;
        use std::sync::Arc;
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let hidden = weights.hidden_size;
        let nl = weights.num_layers;
        let router = CellRouter {
            centroids: vec![vec![0.0f32; hidden]; nl], // 1 cell/layer at origin
            n_cells: vec![1; nl],
            pools: vec![vec![vec![0usize, 1, 2]]; nl],
            hidden,
        };
        let cfg = WalkFfnConfig::sparse(nl, 8).with_cell_router(Arc::new(router));
        let ffn = WalkFfn::from_config(&weights, &index, cfg);
        let (out, _act) = ffn
            .walk_ffn_sparse(0, &x(1, hidden))
            .expect("cell-router walk should produce output");
        assert_eq!(out.shape(), &[1, hidden]);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    /// Cell router with an empty pool set for the layer falls back to the
    /// gate-KNN path (no cell pool available) — still produces output.
    #[test]
    fn walk_ffn_sparse_cell_router_empty_falls_back_to_gate_knn() {
        use crate::vindex::CellRouter;
        use std::sync::Arc;
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        // Router with no cells for any layer → pool_for returns None.
        let router = CellRouter {
            centroids: vec![Vec::new(); weights.num_layers],
            n_cells: vec![0; weights.num_layers],
            pools: vec![Vec::new(); weights.num_layers],
            hidden: weights.hidden_size,
        };
        let cfg = WalkFfnConfig::sparse(weights.num_layers, 8).with_cell_router(Arc::new(router));
        let ffn = WalkFfn::from_config(&weights, &index, cfg);
        let result = ffn.walk_ffn_sparse(0, &x(1, weights.hidden_size));
        assert!(result.is_some());
    }

    /// Two-stage routing: a candidate pool larger than K, ranked within by
    /// gate score (`rank_within_pool`), keeps the real top-K and still
    /// produces well-formed output.
    #[test]
    fn walk_ffn_sparse_rank_within_pool_q4k_fixture() {
        use std::sync::Arc;
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let feats = index.num_features(0);
        // Candidate pool = all features; K = 4 < pool, so ranking kicks in.
        let pool: Vec<usize> = (0..feats).collect();
        let cfg = WalkFfnConfig::sparse(weights.num_layers, 4)
            .with_pool_per_layer(Arc::new(vec![pool; weights.num_layers]))
            .with_precomputed_routing(true)
            .with_rank_within_pool(true);
        let ffn = WalkFfn::from_config(&weights, &index, cfg);
        let (out, _act) = ffn
            .walk_ffn_sparse(0, &x(1, weights.hidden_size))
            .expect("ranked two-stage walk should produce output");
        assert_eq!(out.shape(), &[1, weights.hidden_size]);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    /// `local_pool_gate_knn` scores only the pool features (O(K)), in
    /// pool order, and filters out-of-range indices. Empty pool → empty.
    #[test]
    fn local_pool_gate_knn_scores_only_pool_features() {
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let cfg = WalkFfnConfig::sparse(weights.num_layers, 8);
        let ffn = WalkFfn::from_config(&weights, &index, cfg);
        let x1 = x(1, weights.hidden_size);
        let x_slice = x1.row(0).to_vec();

        // Empty pool → empty hits.
        let empty = ffn.local_pool_gate_knn(0, &x_slice, &[]);
        assert_eq!(empty, Some(Vec::new()));

        // Valid pool + an out-of-range index that must be filtered.
        let huge = usize::MAX;
        let hits = ffn
            .local_pool_gate_knn(0, &x_slice, &[2, 0, 1, huge])
            .expect("q4k fixture exposes interleaved gate bytes");
        // huge filtered out; the three valid features kept in pool order.
        assert_eq!(
            hits.iter().map(|(f, _)| *f).collect::<Vec<_>>(),
            vec![2, 0, 1]
        );
        assert!(hits.iter().all(|(_, g)| g.is_finite()));
    }

    /// When the layer exposes no interleaved Q4K gate bytes (plain f32
    /// fixture), `local_pool_gate_knn` returns None so the caller routes
    /// to the `pool_restricted_gate_knn` projection fallback.
    #[test]
    fn local_pool_gate_knn_none_without_q4k_gate_bytes() {
        let weights = make_test_weights();
        let index = make_test_vindex(&weights);
        let cfg = WalkFfnConfig::sparse(weights.num_layers, 4);
        let ffn = WalkFfn::from_config(&weights, &index, cfg);
        let x1 = x(1, weights.hidden_size);
        let x_slice = x1.row(0).to_vec();
        assert!(ffn.local_pool_gate_knn(0, &x_slice, &[0, 1]).is_none());
    }
}
