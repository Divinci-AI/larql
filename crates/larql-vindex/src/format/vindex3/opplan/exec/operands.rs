//! Operand loading: an [`OperandRef`] to f32 values, from the container's
//! segments alone.
//!
//! Resolution is `object id → representation → segment → table entry →
//! payload bytes` — the same path closure verified, and no other. An
//! operand the store cannot resolve, or a dtype nobody has judged a
//! widening for, is an error naming the operand — never a zero-filled
//! buffer.

use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use super::super::super::encode::segment::{read_segment_header, SegmentTensor};
use super::super::super::encode::REPRESENTATION_ID_SEP;
use super::super::super::inspect::SystemInspection;
use super::super::OperandRef;
use crate::error::VindexError;

/// Safetensors dtype labels this reference executor can widen to f32.
const DTYPE_F32: &str = "F32";
const DTYPE_BF16: &str = "BF16";

/// One object's segment: file path, payload origin, and tensor table.
struct SegmentMap {
    path: PathBuf,
    payload_start: u64,
    tensors: BTreeMap<String, SegmentTensor>,
}

/// Operand store over one container.
pub struct OperandStore {
    segments: BTreeMap<String, SegmentMap>,
}

impl OperandStore {
    /// Open every canonical segment of every object in the inspection.
    pub fn open(root: &Path, inspection: &SystemInspection) -> Result<Self, VindexError> {
        let mut segments = BTreeMap::new();
        for object in &inspection.graph.objects {
            let Some(representation) = object.representations.first() else {
                continue;
            };
            let id = format!(
                "{}{REPRESENTATION_ID_SEP}{}",
                object.id, representation.encoding
            );
            let Some(entry) = inspection.index.representations.get(&id) else {
                continue;
            };
            let path = root.join(&entry.segment);
            let (header, payload_start) = read_segment_header(&path)?;
            segments.insert(
                object.id.clone(),
                SegmentMap {
                    path,
                    payload_start,
                    tensors: header
                        .tensors
                        .into_iter()
                        .map(|t| (t.name.clone(), t))
                        .collect(),
                },
            );
        }
        Ok(Self { segments })
    }

    /// Load one operand as f32 values.
    pub fn load(&self, operand: &OperandRef) -> Result<Vec<f32>, VindexError> {
        let segment = self.segments.get(&operand.object).ok_or_else(|| {
            VindexError::Parse(format!("no segment for object `{}`", operand.object))
        })?;
        let tensor = segment.tensors.get(&operand.tensor).ok_or_else(|| {
            VindexError::Parse(format!(
                "no tensor `{}` in `{}`'s segment",
                operand.tensor, operand.object
            ))
        })?;
        let mut file = std::fs::File::open(&segment.path)?;
        file.seek(SeekFrom::Start(segment.payload_start + tensor.offset))?;
        let mut bytes = vec![0u8; tensor.len as usize];
        file.read_exact(&mut bytes)?;
        widen(&tensor.dtype, &bytes, &operand.tensor)
    }
}

/// Widen stored bytes to f32 — judged dtypes only, fail-closed.
pub(super) fn widen(dtype: &str, bytes: &[u8], name: &str) -> Result<Vec<f32>, VindexError> {
    match dtype {
        DTYPE_F32 => Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        DTYPE_BF16 => Ok(bytes
            .chunks_exact(2)
            .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
            .collect()),
        other => Err(VindexError::Parse(format!(
            "tensor `{name}`: no judged f32 widening for dtype `{other}`"
        ))),
    }
}
