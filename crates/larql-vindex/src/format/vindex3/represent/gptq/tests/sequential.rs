//! Hand-derived numeric checks for [`EliminationPlan`], with `ridge =
//! 0.0` throughout so every intermediate is exact rather than perturbed
//! by damping — the damping-ridge behaviour itself is `SiteHessian`'s
//! concern and is tested in `tests/hessian.rs`.
//!
//! `H = [[2, 1], [1, 2]]` is the fixture for every propagation case
//! below because its Cholesky-of-inverse-Cholesky factor works out to a
//! clean closed form (derived once, here, not per test):
//!
//! ```text
//! L  = chol(H)      = [[√2, 0], [1/√2, √1.5]]
//! H⁻¹               = (1/3) [[2, -1], [-1, 2]]
//! L' = chol(H⁻¹)    = [[√(2/3), 0], [-√(1/6), √0.5]]
//! ```
//!
//! and the propagation factor `eliminate_row` applies from column 0 into
//! column 1, `(err[0] / L'[0,0]) * L'[1,0]`, collapses to exactly
//! `-0.5 * err[0]` because `L'[1,0] / L'[0,0] = -√(1/6) / √(2/3) =
//! -√(1/4) = -0.5`. So `wwork[1] += 0.5 * err[0]`, independent of
//! `err[0]`'s magnitude — every case below just picks a different
//! `err[0]` to land on a different point of the E2M1 grid.

use ndarray::array;

use super::*;

#[test]
fn zero_size_plan_does_no_work() {
    let h = ndarray::Array2::<f64>::zeros((0, 0));
    let plan = EliminationPlan::build(&h, 0.0).expect("0x0 Cholesky is trivially valid");
    assert_eq!(plan.n(), 0);
    let result = plan.eliminate_row(&[], &[]);
    assert!(result.codes.is_empty());
    assert!(result.saturated.is_empty());
}

#[test]
fn a_single_alive_column_receives_no_propagation_and_matches_plain_nearest() {
    let h = array![[3.0]];
    let plan = EliminationPlan::build(&h, 0.0).expect("1x1 SPD");
    let result = plan.eliminate_row(&[0.24], &[1.0]);
    // Nothing can propagate into or out of the only column there is, so
    // this must equal ordinary nearest rounding of the raw value.
    assert_eq!(
        result.codes,
        vec![larql_models::quant::fp4::f32_to_e2m1(0.24) & 0x0F]
    );
    assert_eq!(result.saturated, vec![false]);
}

#[test]
fn correlated_columns_propagate_and_can_flip_a_code() {
    let h = array![[2.0, 1.0], [1.0, 2.0]];
    let plan = EliminationPlan::build(&h, 0.0).expect("2x2 SPD");

    // Without compensation, 0.24 rounds to grid point 0.0 (code 0): it
    // is closer to 0.0 (distance 0.24) than to 0.5 (distance 0.26).
    assert_eq!(larql_models::quant::fp4::f32_to_e2m1(0.24) & 0x0F, 0);

    let result = plan.eliminate_row(&[0.24, 0.24], &[1.0, 1.0]);

    // Column 0 is eliminated first, so nothing has propagated into it
    // yet — it matches plain nearest rounding exactly, as it always
    // must for the first-eliminated column regardless of H.
    assert_eq!(result.codes[0], 0);
    // Column 1 receives err[0] = 0.24 - 0.0 = 0.24, so
    // wwork[1] = 0.24 + 0.5*0.24 = 0.36 — closer to grid point 0.5
    // (distance 0.14) than to 0.0 (distance 0.36), so it flips to
    // code 1, which independent nearest rounding of 0.24 alone never
    // would have chosen.
    assert_eq!(result.codes[1], 1);
    assert_eq!(result.saturated, vec![false, false]);
}

#[test]
fn propagated_compensation_can_saturate_a_column_that_alone_would_not() {
    let h = array![[2.0, 1.0], [1.0, 2.0]];
    let plan = EliminationPlan::build(&h, 0.0).expect("2x2 SPD");

    // 5.6 alone rounds to grid point 6.0 (code 7) without saturating:
    // its own magnitude never exceeds the grid top.
    let nearest_5_6 = larql_models::quant::fp4::f32_to_e2m1(5.6) & 0x0F;
    assert_eq!(nearest_5_6, 7);

    let result = plan.eliminate_row(&[5.0, 5.6], &[1.0, 1.0]);

    // Column 0: 5.0 is an exact tie between grid points 4.0 (index 6,
    // even) and 6.0 (index 7, odd); round-to-nearest-even picks 4.0.
    // Its own magnitude (5.0) never exceeds the grid top either.
    assert_eq!(result.codes[0], 6);
    assert!(!result.saturated[0]);

    // Column 1 receives err[0] = 5.0 - 4.0 = 1.0, so
    // wwork[1] = 5.6 + 0.5*1.0 = 6.1 — past the grid top (6.0) before
    // rounding clamps it to code 7. Nearest-v1 could never produce this
    // element's saturation, because its group scale is chosen so the
    // group's own amax lands exactly at the grid top; this is a real,
    // disclosed cost of freezing scales computed from W0 against values
    // GPTQ has since moved.
    assert_eq!(result.codes[1], 7);
    assert!(result.saturated[1]);
}

#[test]
fn an_exact_grid_value_has_zero_error_and_propagates_nothing() {
    // 0.5 is exactly on the E2M1 grid, so its quantisation error is
    // exactly zero — even with a strongly correlated H, nothing should
    // propagate into column 1, which must land on plain nearest
    // rounding of its own raw value.
    let h = array![[2.0, 1.0], [1.0, 2.0]];
    let plan = EliminationPlan::build(&h, 0.0).expect("2x2 SPD");
    let result = plan.eliminate_row(&[0.5, 0.24], &[1.0, 1.0]);
    assert_eq!(result.codes[0], 1); // 0.5 itself
    assert_eq!(
        result.codes[1],
        larql_models::quant::fp4::f32_to_e2m1(0.24) & 0x0F
    );
    assert_eq!(result.saturated, vec![false, false]);
}

#[test]
fn a_non_finite_or_non_positive_step_quantises_to_the_zero_code() {
    // Mirrors `quantize_row_into`'s own guard for a pathological
    // all-zero group: `inv = 0.0` rather than a division by zero.
    let h = array![[1.0]];
    let plan = EliminationPlan::build(&h, 0.0).expect("1x1 SPD");
    let result = plan.eliminate_row(&[3.7], &[0.0]);
    assert_eq!(result.codes, vec![0]);
    assert!(!result.saturated[0]);
}
