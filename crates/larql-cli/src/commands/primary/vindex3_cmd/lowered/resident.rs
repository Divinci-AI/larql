//! Resident device operands (matrices, norms, rope tables) and the
//! ablation flags the lowered session binds — loaded once, held for the
//! session's lifetime.

use larql_compute_metal::lowering::nvfp4_fusion_enabled;
use larql_compute_metal::lowering::DeviceBuffer;
use larql_compute_metal::MetalBackend;
use larql_models::config::{PositionPolicy, RotaryFrequencyBasis};
use larql_vindex::error::VindexError;
use larql_vindex::format::vindex3::opplan::exec::backend::WeightFormat;
use larql_vindex::format::vindex3::opplan::exec::operands::OperandStore;
use larql_vindex::format::vindex3::opplan::exec::weights::{
    load_weight, AlignedBytes, LoadedWeight,
};
use larql_vindex::format::vindex3::opplan::{NormOp, OperandRef};

use super::DeviceMatrix;

/// Stages this run omits, for marginal-cost profiling.
///
/// Every one of these is an operation the plan marks **optional**, so
/// omitting it exercises a path the lowering already supports rather
/// than a special diagnostic branch. The numbers are wrong by
/// construction; the *time difference* is the measurement.
///
/// Ablation is used because this hardware supports counter sampling only
/// at compute-pass boundaries (`AtDispatchBoundary` is false on M3), so
/// per-dispatch GPU timestamps are unavailable. Splitting stages into
/// separate encoders to get boundaries would change what can overlap;
/// ablation leaves the schedule of everything that remains intact.
#[derive(Clone, Copy, Default)]
pub(super) struct Ablation {
    pub no_query_scale: bool,
    pub no_rope: bool,
    pub no_qk_norm: bool,
    pub no_gate: bool,
    pub no_post_norms: bool,
}

impl Ablation {
    pub(super) fn from_env() -> Self {
        let on = |k: &str| std::env::var(k).is_ok();
        Self {
            no_query_scale: on("LARQL_ABLATE_QUERY_SCALE"),
            no_rope: on("LARQL_ABLATE_ROPE"),
            no_qk_norm: on("LARQL_ABLATE_QK_NORM"),
            no_gate: on("LARQL_ABLATE_GATE"),
            no_post_norms: on("LARQL_ABLATE_POST_NORMS"),
        }
    }

    pub(super) fn any(&self) -> bool {
        self.no_query_scale || self.no_rope || self.no_qk_norm || self.no_gate || self.no_post_norms
    }
}

/// Load one matrix operand as NVFP4 and hand it to the device.
///
/// The buffers are keyed on the `AlignedBytes` address, which lives for
/// the session, so `lowering_weight` caches them and the weight is
/// uploaded once rather than per position.
pub(super) fn resident_matrix(
    gpu: &MetalBackend,
    store: &OperandStore,
    operand: &OperandRef,
    format: WeightFormat,
    keep: &mut Vec<LoadedWeight>,
) -> Result<DeviceMatrix, VindexError> {
    let rows = operand.shape.first().copied().unwrap_or(0);
    let cols = operand.shape.get(1).copied().unwrap_or(0);
    let loaded = load_weight(store, operand, format)?;
    let m = match &loaded {
        LoadedWeight::Nvfp4 {
            packed,
            scales,
            tensor_scale,
        } => DeviceMatrix {
            packed: gpu.lowering_weight(packed.as_slice()),
            scales: gpu.lowering_weight(scales.as_slice()),
            packed_offset: 0,
            scales_offset: 0,
            read_bytes: packed.as_slice().len() + scales.as_slice().len(),
            tensor_scale: *tensor_scale,
            format: WeightFormat::Nvfp4,
            rows,
            cols,
        },
        LoadedWeight::Mxfp4 { packed, scales } => DeviceMatrix {
            packed: gpu.lowering_weight(packed.as_slice()),
            scales: gpu.lowering_weight(scales.as_slice()),
            packed_offset: 0,
            scales_offset: 0,
            read_bytes: packed.as_slice().len() + scales.as_slice().len(),
            tensor_scale: 1.0,
            format: WeightFormat::Mxfp4,
            rows,
            cols,
        },
        LoadedWeight::F16(bytes) => DeviceMatrix {
            packed: gpu.lowering_weight(bytes.as_slice()),
            scales: gpu.lowering_weight(&[]),
            packed_offset: 0,
            scales_offset: 0,
            read_bytes: bytes.as_slice().len(),
            tensor_scale: 1.0,
            format: WeightFormat::F16,
            rows,
            cols,
        },
        _ => {
            return Err(VindexError::Parse(format!(
                "operand `{}`: unsupported lowering format {format:?}",
                operand.tensor
            )))
        }
    };
    // The device buffers alias these allocations, so the session owns
    // them for its lifetime.
    keep.push(loaded);
    Ok(m)
}

/// Operator control for the QKV single-allocation rung: `LARQL_QKV_PACK=0`
/// keeps the three per-matrix allocations (the A/B control arm). Read
/// once.
pub(super) const QKV_PACK_ENV: &str = "LARQL_QKV_PACK";

fn qkv_pack_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var(QKV_PACK_ENV).as_deref() != Ok("0"))
}

/// Metal bind alignment the packed slices must land on: the x2 body
/// loads codes as `uint2`, so a segment's base (buffer offset included)
/// must be 16-byte aligned; scales are held to the same bound.
const PACK_OFFSET_ALIGN: usize = 16;

/// Load Q, K and V resident as slices of ONE allocation each for codes
/// and scales — the loader half of the seg3t rung: the fused projection
/// dispatch then streams one contiguous address range, exactly as the
/// flat single-matrix kernel does (`examples/qkv_seg3_probe.rs` arm D).
///
/// Packing applies only when the attention class is NVFP4, projection
/// fusion is on and every slice boundary meets the bind alignment; any
/// other case loads the three matrices separately, unchanged.
pub(super) fn resident_qkv(
    gpu: &MetalBackend,
    store: &OperandStore,
    q: &OperandRef,
    k: &OperandRef,
    v: &OperandRef,
    format: WeightFormat,
    keep: &mut Vec<LoadedWeight>,
) -> Result<(DeviceMatrix, DeviceMatrix, DeviceMatrix), VindexError> {
    let unpacked = |gpu, store, keep: &mut Vec<LoadedWeight>| {
        Ok((
            resident_matrix(gpu, store, q, format, keep)?,
            resident_matrix(gpu, store, k, format, keep)?,
            resident_matrix(gpu, store, v, format, keep)?,
        ))
    };
    if format != WeightFormat::Nvfp4 || !qkv_pack_enabled() || !nvfp4_fusion_enabled() {
        return unpacked(gpu, store, keep);
    }
    let ops = [q, k, v];
    let loaded = [
        load_weight(store, q, format)?,
        load_weight(store, k, format)?,
        load_weight(store, v, format)?,
    ];
    let parts: Vec<(&AlignedBytes, &AlignedBytes, f32)> = loaded
        .iter()
        .filter_map(|w| match w {
            LoadedWeight::Nvfp4 {
                packed,
                scales,
                tensor_scale,
            } => Some((packed, scales, *tensor_scale)),
            _ => None,
        })
        .collect();
    // NVFP4 was requested, so all three load as NVFP4; anything else is
    // a load_weight contract change — fall back rather than misbind.
    if parts.len() != 3 {
        return unpacked(gpu, store, keep);
    }
    // Slice boundaries are cumulative LOGICAL lengths; every boundary
    // must meet the bind alignment or the dispatch would misread rows.
    let aligned = parts.iter().all(|(p, s, _)| {
        p.logical_len() % PACK_OFFSET_ALIGN == 0 && s.logical_len() % PACK_OFFSET_ALIGN == 0
    });
    if !aligned {
        return unpacked(gpu, store, keep);
    }
    let mut packed_all = Vec::with_capacity(parts.iter().map(|(p, ..)| p.logical_len()).sum());
    let mut scales_all = Vec::with_capacity(parts.iter().map(|(_, s, _)| s.logical_len()).sum());
    let mut offsets = [(0u64, 0u64); 3];
    for (i, (p, s, _)) in parts.iter().enumerate() {
        offsets[i] = (packed_all.len() as u64, scales_all.len() as u64);
        packed_all.extend_from_slice(&p.as_slice()[..p.logical_len()]);
        scales_all.extend_from_slice(&s.as_slice()[..s.logical_len()]);
    }
    let packed_all = AlignedBytes::from_bytes(&packed_all);
    let scales_all = AlignedBytes::from_bytes(&scales_all);
    let packed_buf = gpu.lowering_weight(packed_all.as_slice());
    let scales_buf = gpu.lowering_weight(scales_all.as_slice());
    let mut out = Vec::with_capacity(3);
    for (i, (p, s, tensor_scale)) in parts.iter().enumerate() {
        out.push(DeviceMatrix {
            packed: packed_buf.clone(),
            scales: scales_buf.clone(),
            packed_offset: offsets[i].0,
            scales_offset: offsets[i].1,
            read_bytes: p.logical_len() + s.logical_len(),
            tensor_scale: *tensor_scale,
            format: WeightFormat::Nvfp4,
            rows: ops[i].shape.first().copied().unwrap_or(0),
            cols: ops[i].shape.get(1).copied().unwrap_or(0),
        });
    }
    // The device buffers alias the COMBINED allocations; the per-matrix
    // loads can drop, the pack owns the bytes for the session.
    keep.push(LoadedWeight::Nvfp4 {
        packed: packed_all,
        scales: scales_all,
        tensor_scale: 1.0,
    });
    let [q_m, k_m, v_m] = <[DeviceMatrix; 3]>::try_from(out)
        .map_err(|_| VindexError::Parse("qkv pack produced a wrong-arity matrix set".into()))?;
    Ok((q_m, k_m, v_m))
}

/// Upload an optional f32 vector operand (a bias or the sink logits) to
/// the device, or `None` when the plan carries none.
pub(super) fn resident_vector(
    gpu: &MetalBackend,
    store: &OperandStore,
    operand: Option<&OperandRef>,
) -> Result<Option<DeviceBuffer>, VindexError> {
    match operand {
        Some(op) => {
            let v = store.load(op)?;
            let buf = gpu
                .lowering_upload(&v)
                .ok_or_else(|| VindexError::Parse("vector operand upload failed".into()))?;
            Ok(Some(buf))
        }
        None => Ok(None),
    }
}

/// The `inv_freq` map key for a rotary policy — distinct per (theta,
/// scaled-or-plain) so YaRN and plain rope at the same base never share a
/// table; `None` for NoPE.
pub(super) fn rope_table_key(position: &PositionPolicy, head_dim: usize) -> Option<u64> {
    use std::hash::{Hash, Hasher};
    // The table is `head_dim/2` entries of `theta^(-2i/head_dim)`: two
    // layers at one theta but different head widths (Gemma 4's 256 vs
    // 512) need different tables, so the width is part of every key.
    let with_width = |discriminant: u64| {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        discriminant.hash(&mut h);
        head_dim.hash(&mut h);
        h.finish() | 1
    };
    match position {
        PositionPolicy::Rope { theta } => Some(with_width(theta.to_bits())),
        // The partial rotary's table is the full-head rotate-half table
        // with the top frequencies zero (head-width basis); fraction and
        // basis join the key.
        PositionPolicy::PartialRope {
            theta,
            rotary_fraction,
            basis,
        } => {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            theta.to_bits().hash(&mut h);
            rotary_fraction.to_bits().hash(&mut h);
            (*basis == RotaryFrequencyBasis::HeadWidth).hash(&mut h);
            head_dim.hash(&mut h);
            Some(h.finish() | 1)
        }
        // Fold the yarn block into the key so two different blocks (or a
        // block vs plain rope) at one theta get their own tables. The
        // block's f64 fields hash deterministically.
        PositionPolicy::Yarn { theta, scaling } => {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            theta.to_bits().hash(&mut h);
            head_dim.hash(&mut h);
            scaling.factor.to_bits().hash(&mut h);
            scaling.beta_fast.to_bits().hash(&mut h);
            scaling.beta_slow.to_bits().hash(&mut h);
            scaling
                .original_max_position_embeddings
                .to_bits()
                .hash(&mut h);
            scaling.truncate.hash(&mut h);
            Some(h.finish() | 1)
        }
        PositionPolicy::None => None,
    }
}

/// The inverse-frequency table for a rotary policy, matching the
/// interpreter kernel exactly: plain `theta^(-2i/d)` for rope, the YaRN
/// ramp for a scaled layer.
pub(super) fn rope_inv_freq_table(position: &PositionPolicy, head_dim: usize) -> Vec<f32> {
    match position {
        PositionPolicy::Rope { theta } => (0..head_dim / 2)
            .map(|i| theta.powf(-2.0 * i as f64 / head_dim as f64) as f32)
            .collect(),
        PositionPolicy::Yarn { theta, scaling } => {
            let (inv_freq, _amplitude) =
                larql_vindex::format::vindex3::opplan::exec::kernels::yarn_frequencies(
                    scaling, head_dim, *theta,
                );
            inv_freq.iter().map(|f| *f as f32).collect()
        }
        PositionPolicy::None => Vec::new(),
        // Head-width basis: the interpreter's own table (zeros above the
        // fraction → identity rotation on those pairs). The rotary-width
        // basis rotates a prefix as its own block, which the rope kernel
        // does not express — refused in `LoweredSession::new`.
        PositionPolicy::PartialRope {
            theta,
            rotary_fraction,
            basis: RotaryFrequencyBasis::HeadWidth,
        } => larql_vindex::format::vindex3::opplan::exec::kernels::partial_rotary_frequencies(
            head_dim,
            *rotary_fraction,
            *theta,
        )
        .iter()
        .map(|f| *f as f32)
        .collect(),
        PositionPolicy::PartialRope {
            basis: RotaryFrequencyBasis::RotaryWidth,
            ..
        } => unreachable!("RotaryWidth partial rotary is refused before the session is built"),
    }
}

pub(super) fn resident_norm(
    gpu: &MetalBackend,
    store: &OperandStore,
    op: &NormOp,
) -> Result<(DeviceBuffer, f32, f32), VindexError> {
    let w = store.load(&op.weight)?;
    let buf = gpu
        .lowering_upload(&w)
        .ok_or_else(|| VindexError::Parse("norm weight upload failed".into()))?;
    Ok((buf, op.eps as f32, op.weight_offset))
}
