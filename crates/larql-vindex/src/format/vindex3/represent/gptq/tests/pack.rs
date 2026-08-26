//! `quantize_nvfp4_gptq`'s implementation oracles, matched to the four
//! the ENCODER-R4 record asks for before any real calibration data
//! touches the encoder:
//!
//! 1. zero compensation: `nvfp4-gptq-v1` with an all-zero Hessian is
//!    byte-for-byte `nvfp4-nearest-v1`.
//! 2. normal GPTQ: scale bytes stay identical to nearest; only E2M1
//!    payload nibbles may differ.
//! 3. determinism: same weights + same Hessian → byte-identical pack.
//! 4. stored decode reproduces the encoder's own reconstructed values.
//!
//! plus the dead-coordinate rule exercised through the public API (the
//! narrower unit-level version lives in `tests/hessian.rs`).

use ndarray::Array2;

use larql_models::quant::fp4::{f32_to_e2m1, unpack_nibbles};
use larql_models::quant::fp8::e4m3_to_f32;
use larql_models::quant::nvfp4::{self, dequantize_into};

use super::*;

fn realistic_row_major(rows: usize, k: usize) -> Vec<f32> {
    (0..rows * k)
        .map(|i| ((i % 37) as f32 - 18.0) * 0.013)
        .collect()
}

/// Oracle 1 — zero compensation.
#[test]
fn zero_compensation_oracle_matches_nearest_v1_byte_for_byte() {
    let (rows, k) = (6, 64);
    let w0 = realistic_row_major(rows, k);
    let h_raw = Array2::<f64>::zeros((k, k));

    let nearest = nvfp4::quantize(&w0, rows, k).expect("nearest reference");
    let outcome =
        quantize_nvfp4_gptq(&w0, rows, k, &h_raw, "w").expect("gptq with no compensation");

    assert_eq!(
        outcome.dead_columns, k,
        "every column is dead — no calibration signal"
    );
    assert_eq!(outcome.alive_columns, 0);
    assert_eq!(outcome.saturated_elements, 0);
    assert_eq!(
        outcome.matrix.tensor_scale.to_bits(),
        nearest.tensor_scale.to_bits(),
        "tensor scale byte"
    );
    assert_eq!(
        outcome.matrix.scales, nearest.scales,
        "every E4M3 group scale byte"
    );
    assert_eq!(
        outcome.matrix.packed, nearest.packed,
        "the entire payload, including every E2M1 nibble"
    );
}

/// Oracles 2 and the dead-coordinate rule, together: a single 16-wide
/// group (one NVFP4 group, so there is exactly one frozen scale to
/// check) with columns 0/1 correlated and alive, columns 2..16 dead.
#[test]
fn correlated_alive_columns_may_diverge_while_scales_and_dead_columns_never_do() {
    let (rows, k) = (1usize, 16usize);
    #[rustfmt::skip]
    let w0: Vec<f32> = vec![
        0.24, 0.24, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 6.0,
    ];
    let mut h_raw = Array2::<f64>::zeros((k, k));
    h_raw[[0, 0]] = 2.0;
    h_raw[[1, 1]] = 2.0;
    h_raw[[0, 1]] = 1.0;
    h_raw[[1, 0]] = 1.0;
    // Columns 2..16 stay exactly dead (raw diagonal 0) — including
    // column 15, which defines the group's amax and therefore its
    // scale, deliberately left untouched by GPTQ.

    let nearest = nvfp4::quantize(&w0, rows, k).expect("nearest reference");
    let outcome = quantize_nvfp4_gptq(&w0, rows, k, &h_raw, "w").expect("gptq");

    assert_eq!(outcome.alive_columns, 2);
    assert_eq!(outcome.dead_columns, 14);

    // Frozen from W0 regardless of H: always byte-identical, whether or
    // not any column was alive.
    assert_eq!(
        outcome.matrix.tensor_scale.to_bits(),
        nearest.tensor_scale.to_bits()
    );
    assert_eq!(
        outcome.matrix.scales, nearest.scales,
        "the one frozen group scale byte"
    );

    let gptq_codes = unpack_nibbles(&outcome.matrix.packed);
    let nearest_codes = unpack_nibbles(&nearest.packed);

    // Every dead column keeps nearest's own code, exactly — the public
    // API's version of the dead-coordinate rule (the unit-level version
    // against `SiteHessian` directly lives in `tests/hessian.rs`).
    for j in 2..k {
        assert_eq!(
            gptq_codes[j], nearest_codes[j],
            "dead column {j} must match nearest"
        );
    }
    // At least one alive column's code differs: GPTQ's compensation
    // moves column 1 across the 0.0/0.5 grid boundary once column 0's
    // rounding error propagates into it. The exact numeric derivation
    // (ridge-free, so the arithmetic is exact rather than perturbed by
    // damping) is pinned in `gptq/tests/sequential.rs`; here the claim
    // is only that *something* changed relative to independent
    // per-element nearest rounding — i.e. only nibbles moved, per
    // oracle 2's "only E2M1 payload nibbles may differ".
    assert_ne!(
        (gptq_codes[0], gptq_codes[1]),
        (nearest_codes[0], nearest_codes[1]),
        "GPTQ must have exercised compensation on the alive pair"
    );
}

/// Oracle 3 — determinism.
#[test]
fn same_weights_and_hessian_produce_a_byte_identical_pack() {
    let (rows, k) = (4usize, 32usize);
    let w0 = realistic_row_major(rows, k);
    let mut h_raw = Array2::<f64>::zeros((k, k));
    for j in 0..k {
        h_raw[[j, j]] = 1.0 + (j % 5) as f64;
    }
    for j in 0..k - 1 {
        h_raw[[j, j + 1]] = 0.1;
        h_raw[[j + 1, j]] = 0.1;
    }

    let a = quantize_nvfp4_gptq(&w0, rows, k, &h_raw, "w").expect("first run");
    let b = quantize_nvfp4_gptq(&w0, rows, k, &h_raw, "w").expect("second run");

    assert_eq!(a.matrix.packed, b.matrix.packed);
    assert_eq!(a.matrix.scales, b.matrix.scales);
    assert_eq!(
        a.matrix.tensor_scale.to_bits(),
        b.matrix.tensor_scale.to_bits()
    );
    assert_eq!(a.alive_columns, b.alive_columns);
    assert_eq!(a.saturated_elements, b.saturated_elements);
}

/// Oracle 4 — stored decode reproduces the encoder's own reconstruction:
/// decoding the pack and re-quantising under the same frozen grid (no
/// scales recomputed) must land back on exactly the codes that were
/// stored. This exercises `pack.rs`'s own nibble-merge plumbing — the
/// per-value grid math it depends on is already covered where it lives,
/// in `larql-models`'s own NVFP4/FP4 round-trip tests.
#[test]
fn decoded_values_reproduce_the_stored_codes_under_the_same_frozen_grid() {
    let (rows, k) = (1usize, 16usize);
    #[rustfmt::skip]
    let w0: Vec<f32> = vec![
        0.24, 0.24, 0.03, -0.07, 0.0, 0.9, -1.4, 2.3,
        -3.1, 0.0,  0.0,  0.0,   0.0, 0.0, 0.0,  6.0,
    ];
    let mut h_raw = Array2::<f64>::zeros((k, k));
    h_raw[[0, 0]] = 2.0;
    h_raw[[1, 1]] = 2.0;
    h_raw[[0, 1]] = 1.0;
    h_raw[[1, 0]] = 1.0;
    h_raw[[5, 5]] = 3.0;
    h_raw[[6, 6]] = 3.0;
    h_raw[[5, 6]] = 1.5;
    h_raw[[6, 5]] = 1.5;

    let outcome = quantize_nvfp4_gptq(&w0, rows, k, &h_raw, "w").expect("gptq");
    let mut decoded = vec![0.0f32; rows * k];
    dequantize_into(&outcome.matrix, rows, k, &mut decoded).expect("decode");

    let stored_codes = unpack_nibbles(&outcome.matrix.packed);
    let groups = k / 16;
    for g in 0..groups {
        let step = outcome.matrix.tensor_scale * e4m3_to_f32(outcome.matrix.scales[g]);
        let inv = if step > 0.0 && step.is_finite() {
            1.0 / step
        } else {
            0.0
        };
        for i in 0..16 {
            let idx = g * 16 + i;
            let recoded = f32_to_e2m1(decoded[idx] * inv) & 0x0F;
            assert_eq!(
                recoded, stored_codes[idx],
                "index {idx}: decode-then-reencode under the same frozen step must \
                 reproduce the stored code"
            );
        }
    }
}

/// Refusals: shape mismatches are errors, not panics or silent padding
/// — the same contract `nvfp4-nearest-v1` and `PackLayout` already keep.
#[test]
fn a_value_count_that_does_not_fill_rows_by_k_is_refused() {
    let w0 = vec![0.0f32; 15]; // one short of 1x16
    let h = Array2::<f64>::zeros((16, 16));
    let err = quantize_nvfp4_gptq(&w0, 1, 16, &h, "w")
        .unwrap_err()
        .to_string();
    assert!(err.contains("do not fill"), "{err}");
}

#[test]
fn unaligned_k_is_refused() {
    let w0 = vec![0.0f32; 24];
    let h = Array2::<f64>::zeros((24, 24));
    let err = quantize_nvfp4_gptq(&w0, 1, 24, &h, "w")
        .unwrap_err()
        .to_string();
    assert!(err.contains("not a multiple"), "{err}");
}

#[test]
fn a_hessian_of_the_wrong_size_is_refused() {
    let w0 = vec![0.0f32; 16];
    let h = Array2::<f64>::zeros((8, 8));
    let err = quantize_nvfp4_gptq(&w0, 1, 16, &h, "w")
        .unwrap_err()
        .to_string();
    assert!(err.contains("16x16"), "{err}");
}
