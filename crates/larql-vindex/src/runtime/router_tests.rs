//! Colocated tests for `router`.

use larql_compute::{MoeExpertScalePolicy, MoeTopKWeightPolicy};

use super::error::ExecutionError;
use super::router::{BoundRouter, SelectedExpert};
use super::test_support::{ascending, vector};

const POPULATION: u32 = 4;
const HIDDEN: u32 = 3;
const TOP_K: usize = 2;
const ROUTER: &str = "router";
const SCALE: &str = "router_scale";

fn router(per_expert_scale: Option<&[f32]>) -> BoundRouter<'static> {
    BoundRouter {
        weight: ascending(ROUTER, POPULATION, HIDDEN),
        top_k: TOP_K,
        selected_weight: MoeTopKWeightPolicy::RenormalizedSoftmax,
        expert_scale: if per_expert_scale.is_some() {
            MoeExpertScalePolicy::PerExpert
        } else {
            MoeExpertScalePolicy::None
        },
        per_expert_scale: per_expert_scale.map(|v| vector(SCALE, v)),
    }
}

#[test]
fn a_router_reports_the_population_and_width_it_scores() {
    let r = router(None);
    assert_eq!(r.population(), POPULATION as usize);
    assert_eq!(r.hidden_dim(), HIDDEN as usize);
}

#[test]
fn a_router_validates_against_the_bank_it_routes_into() {
    router(None)
        .validate(POPULATION as usize, HIDDEN as usize)
        .unwrap();
}

#[test]
fn a_router_addressing_a_different_population_is_refused() {
    let err = router(None)
        .validate(POPULATION as usize + 1, HIDDEN as usize)
        .unwrap_err();
    assert!(matches!(err, ExecutionError::DimensionMismatch { .. }));
}

#[test]
fn a_per_expert_scale_must_cover_the_whole_population() {
    // A short scale vector would silently leave the tail of the population
    // unscaled rather than fail.
    let short = router(Some(&[1.0, 1.0]));
    assert!(short
        .validate(POPULATION as usize, HIDDEN as usize)
        .is_err());

    let full = router(Some(&[1.0, 1.0, 1.0, 1.0]));
    full.validate(POPULATION as usize, HIDDEN as usize).unwrap();
}

#[test]
fn a_router_without_a_scale_validates_without_one() {
    assert!(router(None).per_expert_scale.is_none());
    router(None)
        .validate(POPULATION as usize, HIDDEN as usize)
        .unwrap();
}

#[test]
fn a_router_describes_its_depth_and_population() {
    let text = router(None).describe();
    assert!(text.contains(&format!("top-{TOP_K}")), "{text}");
    assert!(text.contains(&POPULATION.to_string()), "{text}");
}

#[test]
fn a_selected_expert_keeps_both_its_weights() {
    // Two selections can share a final weight after renormalisation while
    // having scored differently; the raw score is where that stays visible.
    let s = SelectedExpert {
        expert_id: 3,
        weight: 0.5,
        raw_score: 0.2,
    };
    assert_eq!(s.expert_id, 3);
    assert_ne!(s.weight, s.raw_score);
    assert_eq!(s, s);
}
