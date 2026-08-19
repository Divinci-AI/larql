//! A-5b: the segmented NVFP4 GEMV (Q+K+V or gate+up in one dispatch) is
//! bit-identical to the per-matrix `x2` dispatches it replaces — rows
//! straddling a segment boundary, absent third segment, output offsets.

#![cfg(target_os = "macos")]

use larql_compute_metal::lowering::{MatvecOperands, Nvfp4Kernel, Nvfp4Segment};
use larql_compute_metal::MetalBackend;
use larql_models::quant::nvfp4;

struct Mat {
    m: nvfp4::Nvfp4Matrix,
    n: usize,
}

fn mat(n: usize, k: usize, seed: usize) -> Mat {
    let values: Vec<f32> = (0..n * k)
        .map(|i| (((i * 17 + seed) % 977) as f32 / 977.0) - 0.5)
        .collect();
    Mat {
        m: nvfp4::quantize(&values, n, k).expect("quantise"),
        n,
    }
}

/// Separate x2 dispatches vs one segmented dispatch, same x, same outs.
fn compare(gpu: &MetalBackend, k: usize, mats: &[Mat], offsets: &[u64]) {
    let x: Vec<f32> = (0..k).map(|i| (i % 11) as f32 * 0.02 - 0.1).collect();
    let xb = gpu.lowering_upload(&x).expect("x");
    let packed: Vec<_> = mats
        .iter()
        .map(|m| gpu.lowering_weight(&m.m.packed))
        .collect();
    let scales: Vec<_> = mats
        .iter()
        .map(|m| gpu.lowering_weight(&m.m.scales))
        .collect();
    let mut results: Vec<Vec<Vec<f32>>> = Vec::new();
    for fused in [false, true] {
        // Outputs sized for the offset plus the rows.
        let outs: Vec<_> = mats
            .iter()
            .zip(offsets)
            .map(|(m, off)| gpu.lowering_scratch(m.n + (*off as usize) / 4))
            .collect();
        let cmd = gpu.new_lowering_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        if fused {
            let segs: Vec<Nvfp4Segment<'_>> = (0..mats.len())
                .map(|i| Nvfp4Segment {
                    packed: &packed[i],
                    scales: &scales[i],
                    tensor_scale: mats[i].m.tensor_scale,
                    out: &outs[i],
                    out_offset: offsets[i],
                    n: mats[i].n,
                })
                .collect();
            gpu.encode_nvfp4_matvec_segments(enc, &xb, k, &segs);
        } else {
            for i in 0..mats.len() {
                gpu.encode_nvfp4_kernel(
                    Nvfp4Kernel::X2,
                    enc,
                    &MatvecOperands {
                        packed: &packed[i],
                        scales: &scales[i],
                        x: &xb,
                        out: &outs[i],
                        out_offset: offsets[i],
                        n: mats[i].n,
                        k,
                    },
                    mats[i].m.tensor_scale,
                );
            }
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let read: Vec<Vec<f32>> = outs
            .iter()
            .zip(mats)
            .zip(offsets)
            .map(|((o, m), off)| {
                let all = gpu
                    .lowering_readback(o, m.n + (*off as usize) / 4)
                    .expect("readback");
                all[(*off as usize) / 4..].to_vec()
            })
            .collect();
        for o in outs {
            gpu.recycle_lowering_scratch(o);
        }
        results.push(read);
    }
    gpu.recycle_lowering_scratch(xb);
    for (i, (a, b)) in results[0].iter().zip(&results[1]).enumerate() {
        assert_eq!(a, b, "segment {i} of {}: fused != separate", mats.len());
        assert!(a.iter().all(|v| v.is_finite()));
    }
}

#[test]
fn three_segments_match_separate_x2_dispatches_bit_for_bit() {
    let Some(gpu) = MetalBackend::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let k = 2816;
    // Odd first segment so a lane's row pair straddles the Q|K boundary;
    // K and V rows land at a cache-slot offset like the real lowering.
    let mats = [mat(4097, k, 1), mat(2048, k, 2), mat(2048, k, 3)];
    compare(&gpu, k, &mats, &[0, 4 * 2048 * 3, 4 * 2048 * 3]);
}

#[test]
fn two_segments_gate_up_match_and_third_is_absent() {
    let Some(gpu) = MetalBackend::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let k = 2560;
    let mats = [mat(8192, k, 4), mat(8192, k, 5)];
    compare(&gpu, k, &mats, &[0, 0]);
    // A single segment is just x2.
    let one = [mat(37, k, 6)];
    compare(&gpu, k, &one, &[0]);
}
