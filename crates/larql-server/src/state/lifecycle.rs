//! Lifecycle-mutation invariant (§7 of `docs/runtime-lifecycle-design.md`):
//! dynamic model lifecycle is scoped to single-model topology only.
//!
//! `RouterTopology` freezes, at boot, which axum `Router` variant
//! `bootstrap::serve` actually built. That's a structural fact about
//! the running process — the route table it describes cannot change
//! shape without rebuilding the whole `Router`, which nothing in this
//! codebase does. `ModelSet`'s live count (via `AppState::is_multi_model`)
//! answers a different question — "how many models are bound right
//! now" — and once lifecycle mutation exists, that number can change
//! while `router_topology` never does. Conflating the two is exactly
//! the bug this module exists to make impossible: a multi-model boot
//! must refuse mutation outright, and a single-model boot must never
//! be allowed to grow past one binding, because axum was never given
//! a second route table to grow into.
//!
//! There is no mutation API yet (see the design doc's non-goals) —
//! this exists so a future load/unload endpoint has nowhere to go but
//! through [`AppState::validate_lifecycle_mutation`], rather than
//! reconstructing this invariant ad hoc once it's actually needed.

/// The router topology `bootstrap::serve` built, frozen at
/// `AppState` construction and never recomputed. See the module doc
/// for why this must not be derived from the live model count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouterTopology {
    /// `routes::single_model_router` — the route table this process
    /// was actually given. Dynamic lifecycle mutation is only ever
    /// permitted here, and only while the bound count stays 0 or 1.
    SingleModel,
    /// `routes::multi_model_router` — a route table sized for the
    /// boot-time model count. No lifecycle mutation is supported
    /// against it; growing or shrinking the bound set would require a
    /// router this process was never built with.
    MultiModel,
}

impl RouterTopology {
    /// The topology `bootstrap::serve` picks for `total_models` models
    /// at boot — the same threshold `AppState::is_multi_model` uses,
    /// applied once, before any mutation could exist. `bootstrap::serve`
    /// calls this directly so the router it builds and the topology
    /// `AppState` freezes can never disagree with each other.
    pub fn for_boot_count(total_models: usize) -> Self {
        if total_models > 1 {
            RouterTopology::MultiModel
        } else {
            RouterTopology::SingleModel
        }
    }
}

/// Why a proposed lifecycle mutation (load/unload/swap) was refused.
/// Both variants are permanent for the life of the process — neither
/// is a transient "try again" condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LifecycleError {
    /// This process booted multi-model — `bootstrap::serve` built
    /// `multi_model_router` for a fixed, boot-time set of
    /// `/v1/{id}/...` routes. No lifecycle mutation is attempted
    /// against it at all, regardless of what the mutation would do to
    /// the count.
    StaticMultiModelTopology,
    /// This process booted single-model, but the proposed mutation
    /// would leave more than one model bound — the
    /// `single_model_router` route table has nowhere for a second
    /// model to live.
    DynamicMultiModelUnsupported,
}

impl std::fmt::Display for LifecycleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LifecycleError::StaticMultiModelTopology => write!(
                f,
                "this server booted with a static multi-model router topology; \
                 dynamic model lifecycle mutation is not supported"
            ),
            LifecycleError::DynamicMultiModelUnsupported => write!(
                f,
                "this mutation would leave more than one model bound, which the \
                 single-model router topology this server booted with cannot serve"
            ),
        }
    }
}

impl std::error::Error for LifecycleError {}

/// The pure decision behind [`AppState::validate_lifecycle_mutation`],
/// split out so it's testable without constructing a real `AppState`.
fn check_topology_invariant(
    topology: RouterTopology,
    proposed_count: usize,
) -> Result<(), LifecycleError> {
    if topology == RouterTopology::MultiModel {
        return Err(LifecycleError::StaticMultiModelTopology);
    }
    if proposed_count > 1 {
        return Err(LifecycleError::DynamicMultiModelUnsupported);
    }
    Ok(())
}

impl crate::state::AppState {
    /// Would a lifecycle mutation that leaves `proposed_count` models
    /// bound be allowed? A `MultiModel` boot refuses every mutation
    /// outright; a `SingleModel` boot allows one so long as the bound
    /// count stays 0 or 1. Callers pass the count the mutation would
    /// produce (e.g. "load while idle" proposes 1; "unload the bound
    /// model" proposes 0) — this only ever judges the destination
    /// state, never the mechanics of getting there.
    pub fn validate_lifecycle_mutation(&self, proposed_count: usize) -> Result<(), LifecycleError> {
        check_topology_invariant(self.router_topology, proposed_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn for_boot_count_matches_is_multi_models_own_threshold() {
        assert_eq!(
            RouterTopology::for_boot_count(0),
            RouterTopology::SingleModel
        );
        assert_eq!(
            RouterTopology::for_boot_count(1),
            RouterTopology::SingleModel
        );
        assert_eq!(
            RouterTopology::for_boot_count(2),
            RouterTopology::MultiModel
        );
        assert_eq!(
            RouterTopology::for_boot_count(5),
            RouterTopology::MultiModel
        );
    }

    #[test]
    fn multi_model_topology_refuses_every_mutation() {
        // Even a mutation that would leave the count unchanged, or
        // drop it to 0 or 1, is refused — there's no router to serve
        // any post-mutation shape, so the proposed count never even
        // gets consulted.
        for proposed in [0, 1, 2, 3] {
            assert_eq!(
                check_topology_invariant(RouterTopology::MultiModel, proposed),
                Err(LifecycleError::StaticMultiModelTopology),
                "proposed count {proposed} must still be refused under MultiModel topology"
            );
        }
    }

    #[test]
    fn single_model_topology_allows_zero_or_one() {
        assert_eq!(
            check_topology_invariant(RouterTopology::SingleModel, 0),
            Ok(())
        );
        assert_eq!(
            check_topology_invariant(RouterTopology::SingleModel, 1),
            Ok(())
        );
    }

    #[test]
    fn single_model_topology_refuses_growing_past_one() {
        assert_eq!(
            check_topology_invariant(RouterTopology::SingleModel, 2),
            Err(LifecycleError::DynamicMultiModelUnsupported)
        );
        assert_eq!(
            check_topology_invariant(RouterTopology::SingleModel, 7),
            Err(LifecycleError::DynamicMultiModelUnsupported)
        );
    }

    #[test]
    fn lifecycle_error_display_names_the_reason() {
        assert!(format!("{}", LifecycleError::StaticMultiModelTopology).contains("static"));
        assert!(
            format!("{}", LifecycleError::DynamicMultiModelUnsupported).contains("single-model")
        );
    }

    #[test]
    fn lifecycle_error_is_a_std_error() {
        fn assert_error<E: std::error::Error>() {}
        assert_error::<LifecycleError>();
    }
}
