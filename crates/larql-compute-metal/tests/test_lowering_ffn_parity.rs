//! G6b, first fragment: does the GPU-lowered gated FFN compute what the
//! interpreter's CPU-glue realisation computes?
//!
//! The lowered path moves the norm, the SiLU-GLU and the residual onto
//! the GPU, so unlike G6a's scheduling-only comparison the arithmetic
//! realisation genuinely changes and float reassociation is legitimate.
//! The bar is therefore a tolerance, judged in the same units the
//! production-parity work uses — max abs, relative RMS, cosine — not
//! bit equality.
//!
//! ## Controls
//!
//! Agreement alone would not show the lowering *read the plan*. A
//! lowering that ignored the norm epsilon, dropped the centred-norm
//! offset, or silently used the wrong activation would still produce
//! finite, plausible, nearly-correct numbers. So each judged fact gets a
//! negative arm that must break parity:
//!
//! - **norm weight offset** — Glimmer's centred convention (`1 + w`).
//!   Dropping it is the single likeliest silent lowering bug.
//! - **activation** — SiLU-GLU vs plain GLU.
//! - **residual** — present vs omitted.
//!
//! If a control does *not* break parity, the corresponding assertion in
//! the positive arm is vacuous and the test says so rather than passing.
//!
//! Control strength is judged **relative to the parity residual**, not
//! against an absolute constant. A fixed threshold is a guess about the
//! fixture: the residual control below moves rel_rms to 2.4e-3, which
//! looks small next to an arbitrary 1e-2 bar and is in fact 2500x the
//! 9.6e-7 the lowering itself achieves — overwhelmingly distinguishable.
//! What makes a control meaningful is that its effect dwarfs the noise
//! the positive arm tolerates, so that is what gets asserted.

#![cfg(target_os = "macos")]

use larql_compute_metal::lowering::ffn::{FfnScratch, FfnShape, FfnWeights};
use larql_models::quant::nvfp4;

const HIDDEN: usize = 512;
const INTER: usize = 1408;
const EPS: f32 = 1e-5;
/// Glimmer's centred-norm convention.
const NORM_OFFSET: f32 = 1.0;

fn deterministic(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(12345);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            ((s as f32 / u32::MAX as f32) - 0.5) * 0.6
        })
        .collect()
}

/// The reference: the same program on the CPU, in f32, written straight
/// from the plan's op order. Independent of the Metal code under test.
///
/// Takes every judged fact explicitly — including the two a control
/// perturbs — because a reference that hard-coded them could not model
/// the defects the controls exist to detect.
#[allow(clippy::too_many_arguments)]
fn cpu_reference(
    h: &[f32],
    norm_w: &[f32],
    gate: &[f32],
    up: &[f32],
    down: &[f32],
    offset: f32,
    silu: bool,
    residual: bool,
) -> Vec<f32> {
    let ms = h.iter().map(|v| v * v).sum::<f32>() / HIDDEN as f32;
    let inv = 1.0 / (ms + EPS).sqrt();
    let normed: Vec<f32> = h
        .iter()
        .zip(norm_w)
        .map(|(x, w)| x * inv * (offset + w))
        .collect();
    let mv = |m: &[f32], x: &[f32], n: usize, k: usize| -> Vec<f32> {
        (0..n)
            .map(|r| (0..k).map(|c| m[r * k + c] * x[c]).sum())
            .collect()
    };
    let g = mv(gate, &normed, INTER, HIDDEN);
    let u = mv(up, &normed, INTER, HIDDEN);
    let act: Vec<f32> = g
        .iter()
        .zip(&u)
        .map(|(gv, uv)| {
            if silu {
                (gv / (1.0 + (-gv).exp())) * uv
            } else {
                gv * uv
            }
        })
        .collect();
    let d = mv(down, &act, HIDDEN, INTER);
    if residual {
        h.iter().zip(&d).map(|(a, b)| a + b).collect()
    } else {
        d
    }
}

/// How far a control must exceed the parity residual to demonstrate that
/// the positive arm could have detected the corresponding defect.
const CONTROL_MARGIN: f64 = 100.0;

struct Metrics {
    max_abs: f32,
    rel_rms: f64,
    cosine: f64,
}

fn compare(reference: &[f32], got: &[f32]) -> Metrics {
    let max_abs = reference
        .iter()
        .zip(got)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let (mut num, mut den, mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (a, b) in reference.iter().zip(got) {
        let (a, b) = (*a as f64, *b as f64);
        num += (a - b) * (a - b);
        den += a * a;
        dot += a * b;
        na += a * a;
        nb += b * b;
    }
    Metrics {
        max_abs,
        rel_rms: (num / den).sqrt(),
        cosine: dot / (na.sqrt() * nb.sqrt()),
    }
}

/// A control must move the result far enough above the parity residual
/// that the positive arm would have caught the defect it models.
fn assert_control(what: &str, perturbed: &[f32], got: &[f32], parity_rel_rms: f64) {
    let c = compare(perturbed, got);
    let ratio = c.rel_rms / parity_rel_rms;
    eprintln!(
        "  control `{what}`: rel_rms {:.3e} = {ratio:.0}x the parity residual",
        c.rel_rms
    );
    assert!(
        ratio > CONTROL_MARGIN,
        "control `{what}` moves the result only {ratio:.1}x the parity residual          ({:.3e} vs {parity_rel_rms:.3e}) — the positive assertion cannot          distinguish this defect, so passing it proves nothing",
        c.rel_rms
    );
}

/// Run the lowered FFN once and return its output.
fn run_lowered(
    gpu: &larql_compute_metal::MetalBackend,
    h: &[f32],
    norm_w: &[f32],
    gate: &nvfp4::Nvfp4Matrix,
    up: &nvfp4::Nvfp4Matrix,
    down: &nvfp4::Nvfp4Matrix,
    offset: f32,
) -> Vec<f32> {
    let h_in = gpu.lowering_upload(h).expect("upload");
    let norm_buf = gpu.lowering_upload(norm_w).expect("upload");
    let h_out = gpu.lowering_scratch(HIDDEN);
    let (normed, g, u, a, d) = (
        gpu.lowering_scratch(HIDDEN),
        gpu.lowering_scratch(INTER),
        gpu.lowering_scratch(INTER),
        gpu.lowering_scratch(INTER),
        gpu.lowering_scratch(HIDDEN),
    );
    let w = FfnWeights {
        gate_packed: &gpu.lowering_weight(&gate.packed),
        gate_scales: &gpu.lowering_weight(&gate.scales),
        gate_tensor_scale: gate.tensor_scale,
        up_packed: &gpu.lowering_weight(&up.packed),
        up_scales: &gpu.lowering_weight(&up.scales),
        up_tensor_scale: up.tensor_scale,
        down_packed: &gpu.lowering_weight(&down.packed),
        down_scales: &gpu.lowering_weight(&down.scales),
        down_tensor_scale: down.tensor_scale,
        norm_weight: &norm_buf,
    };
    let s = FfnScratch {
        normed: &normed,
        gate: &g,
        up: &u,
        act: &a,
        down: &d,
    };
    let shape = FfnShape {
        hidden: HIDDEN,
        intermediate: INTER,
        norm_eps: EPS,
        norm_weight_offset: offset,
    };

    let cmd = gpu.new_lowering_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    gpu.encode_nvfp4_gated_ffn(enc, &h_in, &h_out, &w, &s, &shape);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let out = gpu.lowering_readback(&h_out, HIDDEN).expect("readback");
    for b in [h_in, norm_buf, h_out, normed, g, u, a, d] {
        gpu.recycle_lowering_scratch(b);
    }
    out
}

#[test]
fn lowered_ffn_matches_the_cpu_program_and_reads_its_judged_facts() {
    let Some(gpu) = larql_compute_metal::MetalBackend::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let h = deterministic(HIDDEN, 1);
    let norm_w = deterministic(HIDDEN, 2);
    let gate_f = deterministic(INTER * HIDDEN, 3);
    let up_f = deterministic(INTER * HIDDEN, 4);
    let down_f = deterministic(HIDDEN * INTER, 5);

    let gate = nvfp4::quantize(&gate_f, INTER, HIDDEN).unwrap();
    let up = nvfp4::quantize(&up_f, INTER, HIDDEN).unwrap();
    let down = nvfp4::quantize(&down_f, HIDDEN, INTER).unwrap();

    // The reference consumes the *quantised* weights, so the comparison
    // isolates the lowering from quantisation error — which Q2 already
    // measured separately and which would otherwise dominate here.
    let gate_q = nvfp4::round_trip(&gate_f, INTER, HIDDEN).unwrap();
    let up_q = nvfp4::round_trip(&up_f, INTER, HIDDEN).unwrap();
    let down_q = nvfp4::round_trip(&down_f, HIDDEN, INTER).unwrap();

    let reference = cpu_reference(
        &h,
        &norm_w,
        &gate_q,
        &up_q,
        &down_q,
        NORM_OFFSET,
        true,
        true,
    );
    let got = run_lowered(&gpu, &h, &norm_w, &gate, &up, &down, NORM_OFFSET);

    let m = compare(&reference, &got);
    eprintln!(
        "lowered FFN vs CPU program: max_abs {:.3e}  rel_rms {:.3e}  cosine {:.9}",
        m.max_abs, m.rel_rms, m.cosine
    );
    assert!(
        got.iter().all(|v| v.is_finite()),
        "lowered FFN produced non-finite output"
    );
    assert!(
        m.rel_rms < 1e-4 && m.cosine > 0.999_999,
        "lowered FFN disagrees with its own program: rel_rms {:.3e}, cosine {:.9}",
        m.rel_rms,
        m.cosine
    );

    // ── Control 1: the centred-norm offset is read ──────────────────
    let no_offset = cpu_reference(&h, &norm_w, &gate_q, &up_q, &down_q, 0.0, true, true);
    assert_control("centred-norm offset", &no_offset, &got, m.rel_rms);

    // ── Control 2: the activation is SiLU-GLU, not plain GLU ────────
    let plain_glu = cpu_reference(
        &h,
        &norm_w,
        &gate_q,
        &up_q,
        &down_q,
        NORM_OFFSET,
        false,
        true,
    );
    assert_control("SiLU-GLU activation", &plain_glu, &got, m.rel_rms);

    // ── Control 3: the residual is applied ──────────────────────────
    let no_residual = cpu_reference(
        &h,
        &norm_w,
        &gate_q,
        &up_q,
        &down_q,
        NORM_OFFSET,
        true,
        false,
    );
    assert_control("FFN residual", &no_residual, &got, m.rel_rms);
}
