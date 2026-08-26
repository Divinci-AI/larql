//! `SiteHessian`'s dead/alive partition, damping ridge and reduced
//! sub-matrix — the exact dead-coordinate rule, tested on tiny synthetic
//! matrices with a manufactured dead column, per `ENCODER-R4.md`'s own
//! requirement ("tested on a tiny synthetic matrix with a manufactured
//! dead column before the encoder ships").

use ndarray::{array, Array2};

use super::*;

#[test]
fn dead_and_alive_partition_from_the_raw_diagonal() {
    let h = array![[1.0, 0.5, 0.0], [0.5, 2.0, 0.0], [0.0, 0.0, 0.0],];
    let site = SiteHessian::from_raw(h);
    assert_eq!(site.alive(), &[0, 1]);
    assert_eq!(site.dead(), &[2]);
}

#[test]
fn damping_ridge_is_one_percent_of_the_full_diagonal_mean() {
    let h = array![[2.0, 0.0], [0.0, 6.0]];
    let site = SiteHessian::from_raw(h);
    let ridge = site.damping_ridge();
    assert!((ridge - 0.04).abs() < 1e-12, "{ridge}");
}

#[test]
fn dead_columns_dilute_the_damping_mean_but_are_excluded_from_reduced() {
    let h = array![[4.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 4.0],];
    let site = SiteHessian::from_raw(h);
    // mean over all 3 diagonal entries, the dead one contributing an
    // exact zero -- the literal "0.01 x mean(diag(H))" reading, not
    // "mean over the alive submatrix".
    let expected = 0.01 * 8.0 / 3.0;
    assert!((site.damping_ridge() - expected).abs() < 1e-12);
    assert_eq!(site.alive(), &[0, 2]);
    assert_eq!(site.dead(), &[1]);
}

#[test]
fn reduced_gathers_alive_columns_in_original_k_order() {
    let h = array![[1.0, 0.0, 2.0], [0.0, 0.0, 0.0], [2.0, 0.0, 5.0],];
    let site = SiteHessian::from_raw(h);
    let r = site.reduced();
    assert_eq!(r.shape(), &[2, 2]);
    assert_eq!(r[[0, 0]], 1.0);
    assert_eq!(r[[0, 1]], 2.0);
    assert_eq!(r[[1, 0]], 2.0);
    assert_eq!(r[[1, 1]], 5.0);
}

#[test]
fn all_dead_reduces_to_an_empty_matrix_and_zero_ridge() {
    let h = Array2::<f64>::zeros((4, 4));
    let site = SiteHessian::from_raw(h);
    assert!(site.alive().is_empty());
    assert_eq!(site.dead().len(), 4);
    assert_eq!(site.reduced().shape(), &[0, 0]);
    assert_eq!(site.damping_ridge(), 0.0);
}

#[test]
fn a_zero_dimensional_hessian_has_zero_ridge_via_the_explicit_empty_case() {
    // Distinct from `all_dead_reduces_to_an_empty_matrix_and_zero_ridge`
    // above: that one is a 4x4 all-zero matrix (d=4, sum=0, so ridge is
    // 0.0 via the ordinary `0.01 * sum / d` arithmetic). This is a
    // genuinely 0-dimensional matrix (d=0), which takes the explicit
    // early-return guard against dividing by zero.
    let h = Array2::<f64>::zeros((0, 0));
    let site = SiteHessian::from_raw(h);
    assert_eq!(site.dim(), 0);
    assert_eq!(site.damping_ridge(), 0.0);
}

#[test]
fn no_dead_columns_leaves_reduced_equal_to_raw() {
    let h = array![[3.0, 1.0], [1.0, 4.0]];
    let site = SiteHessian::from_raw(h.clone());
    assert!(site.dead().is_empty());
    assert_eq!(site.reduced(), h);
}

#[test]
#[should_panic(expected = "must be square")]
fn from_raw_panics_on_a_non_square_matrix() {
    let h = Array2::<f64>::zeros((2, 3));
    let _ = SiteHessian::from_raw(h);
}
