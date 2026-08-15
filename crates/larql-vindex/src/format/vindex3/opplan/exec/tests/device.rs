//! The device backend's seam, tested without any device.
//!
//! `DevicePlanBackend` is generic over `larql-compute`'s `MatMul`, so
//! its routing and fail-closed contract are testable with local trait
//! implementors: one whose gemv is a plain loop (parity against the
//! reference backend), one that widens f16 weights and loops (the f16
//! residency path's arithmetic), and one with no gemv kernel at all
//! (every matmul path must refuse, not fall back). Real-device parity
//! for `--backend metal` runs at the CLI layer, where the concrete
//! Metal backend is injected.

use larql_compute::backend::MatMul;
use larql_compute::cpu::ops::q4_common::f16_to_f32;
use ndarray::{Array2, ArrayView2};

use super::golden::{miniature_glimmer, G_TOKENS};
use crate::format::vindex3::encode::encode_system;
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::backend::WeightFormat;
use crate::format::vindex3::opplan::exec::device::DevicePlanBackend;
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::exec::{execute_plan, ExecutionTrace};
use crate::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan};

/// Above loop-vs-loop reassociation noise on the miniature's shapes,
/// far below any semantic effect.
const NOISE_CEILING: f32 = 1e-5;

/// The f16 realisation's tolerance. The miniature fixture stores f32
/// tensors, so its f16 load path rounds to nearest (2⁻¹¹ relative per
/// weight) and the error compounds through norms, softmax and two
/// layers to ~1% — unlike the real container's bf16 payload, which
/// converts exactly (pinned by the unit tests in `weights.rs`). This
/// test is a plumbing gate (layout, padding, dtype length — which fail
/// as garbage, orders of magnitude past this ceiling); the bit-exact
/// f16 decode-vs-batch parity test and the real-model table carry the
/// precision claims.
const F16_CEILING: f32 = 0.05;

/// A "device" whose gemv is a plain in-order loop. `pub(super)` so the
/// decode-parity tests can drive the same device through a session.
pub(super) struct LoopDevice;

impl MatMul for LoopDevice {
    fn matmul(&self, _a: ArrayView2<f32>, _b: ArrayView2<f32>) -> Array2<f32> {
        unimplemented!("the plan backend only dispatches gemv")
    }

    fn matmul_transb(&self, _a: ArrayView2<f32>, _b: ArrayView2<f32>) -> Array2<f32> {
        unimplemented!("the plan backend only dispatches gemv")
    }

    fn f32_gemv_force(&self, w: ArrayView2<f32>, x: &[f32]) -> Option<Vec<f32>> {
        let (n, k) = (w.shape()[0], w.shape()[1]);
        if x.len() != k {
            return None;
        }
        Some(
            (0..n)
                .map(|row| (0..k).map(|col| w[[row, col]] * x[col]).sum())
                .collect(),
        )
    }

    fn f16_gemv_force(&self, w_f16: &[u8], x: &[f32], n: usize, k: usize) -> Option<Vec<f32>> {
        if w_f16.len() < n * k * 2 || x.len() != k {
            return None;
        }
        Some(
            (0..n)
                .map(|row| {
                    (0..k)
                        .map(|col| {
                            let at = (row * k + col) * 2;
                            let bits = u16::from_le_bytes([w_f16[at], w_f16[at + 1]]);
                            f16_to_f32(bits) * x[col]
                        })
                        .sum()
                })
                .collect(),
        )
    }
}

/// A device with no gemv kernel — the trait default `None`.
struct KernellessDevice;

impl MatMul for KernellessDevice {
    fn matmul(&self, _a: ArrayView2<f32>, _b: ArrayView2<f32>) -> Array2<f32> {
        unimplemented!()
    }

    fn matmul_transb(&self, _a: ArrayView2<f32>, _b: ArrayView2<f32>) -> Array2<f32> {
        unimplemented!()
    }
}

fn fixture() -> (tempfile::TempDir, ComponentOpPlan, OperandStore) {
    let dir = tempfile::tempdir().unwrap();
    miniature_glimmer(dir.path());
    let inventory = larql_models::inventory::build_inventory(dir.path()).unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_system(&[("mini-glimmer".to_string(), inventory)], container.path()).unwrap();
    let inspection = inspect_container(container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, container.path(), "target").unwrap();
    assert!(outcome.closed(), "defects: {:?}", outcome.defects);
    let plan = outcome.plan.unwrap();
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    (container, plan, store)
}

fn max_abs(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    a.iter()
        .zip(b)
        .flat_map(|(ra, rb)| ra.iter().zip(rb).map(|(x, y)| (x - y).abs()))
        .fold(0.0, f32::max)
}

fn assert_traces_agree(a: &ExecutionTrace, b: &ExecutionTrace, ceiling: f32, label: &str) {
    for (index, (da, db)) in a.layers.iter().zip(&b.layers).enumerate() {
        let delta = max_abs(&da.post_layer, &db.post_layer);
        assert!(delta < ceiling, "{label}: layer {index} max_abs {delta}");
    }
    let logits_delta = max_abs(
        std::slice::from_ref(a.logits.as_ref().unwrap()),
        std::slice::from_ref(b.logits.as_ref().unwrap()),
    );
    assert!(
        logits_delta < ceiling,
        "{label}: logits max_abs {logits_delta}"
    );
}

#[test]
fn a_device_backend_matches_the_reference_layer_by_layer() {
    let (_c, plan, store) = fixture();
    let device = DevicePlanBackend::new(LoopDevice, "loop-device-test", WeightFormat::F32);
    let on_device = execute_plan(&plan, &store, &G_TOKENS, &device).unwrap();
    let on_reference = execute_plan(&plan, &store, &G_TOKENS, &ReferenceBackend::new()).unwrap();
    assert_traces_agree(
        &on_device,
        &on_reference,
        NOISE_CEILING,
        "f32 device vs reference",
    );
}

/// The f16 residency path: the interpreter loads f16 operands for a
/// backend that declares them, and the arithmetic stays within the
/// documented conversion tolerance of the f32 reference.
#[test]
fn an_f16_device_backend_matches_the_reference_within_the_conversion_floor() {
    let (_c, plan, store) = fixture();
    let device = DevicePlanBackend::new(LoopDevice, "loop-device-f16-test", WeightFormat::F16);
    let on_device = execute_plan(&plan, &store, &G_TOKENS, &device).unwrap();
    let on_reference = execute_plan(&plan, &store, &G_TOKENS, &ReferenceBackend::new()).unwrap();
    assert_traces_agree(
        &on_device,
        &on_reference,
        F16_CEILING,
        "f16 device vs reference",
    );
}

#[test]
fn a_kernelless_device_fails_closed_naming_the_shape() {
    let (_c, plan, store) = fixture();
    let device = DevicePlanBackend::new(KernellessDevice, "kernelless-test", WeightFormat::F32);
    let err = execute_plan(&plan, &store, &G_TOKENS, &device).unwrap_err();
    assert!(
        err.to_string().contains("f32_gemv") && err.to_string().contains("refused"),
        "{err}"
    );
}

/// An f16-declaring backend on a device with no f16 kernel must refuse,
/// never quietly widen and take the f32 path.
#[test]
fn a_kernelless_device_fails_closed_for_f16_too() {
    let (_c, plan, store) = fixture();
    let device = DevicePlanBackend::new(KernellessDevice, "kernelless-f16", WeightFormat::F16);
    let err = execute_plan(&plan, &store, &G_TOKENS, &device).unwrap_err();
    assert!(
        err.to_string().contains("f16_gemv") && err.to_string().contains("refused"),
        "{err}"
    );
}
