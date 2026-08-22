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
    /// How many operands have been read out of this store.
    ///
    /// Residency is an architectural claim ("a served model's operands
    /// are lowered once"), and a claim that can only be checked by
    /// stopwatch is a claim that regresses quietly. This counter lets a
    /// test assert the shape directly: prepare, then serve N requests,
    /// then assert the count did not move.
    loads: std::sync::atomic::AtomicU64,
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
        Ok(Self {
            segments,
            loads: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Load one operand as f32 values.
    pub fn load(&self, operand: &OperandRef) -> Result<Vec<f32>, VindexError> {
        let raw = self.load_raw(operand)?;
        widen(&raw.dtype, &raw.bytes, &operand.tensor)
    }

    /// How many operands have been read out of this store since it was
    /// opened. The residency gate reads this.
    pub fn load_count(&self) -> u64 {
        self.loads.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Load one operand's stored bytes and dtype, unwidened — for a
    /// caller that converts to a representation other than f32 (and for
    /// [`Self::load`] itself, so there is exactly one resolution path).
    pub fn load_raw(&self, operand: &OperandRef) -> Result<RawOperand, VindexError> {
        self.loads
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
        Ok(RawOperand {
            dtype: tensor.dtype.clone(),
            bytes,
        })
    }
}

/// One operand exactly as stored: payload bytes plus the dtype label
/// that says how to read them.
pub struct RawOperand {
    pub dtype: String,
    pub bytes: Vec<u8>,
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

/// One logical f32 edit to a stored operand (V3-LQL-3B compose): a row
/// or a column replaced by new values. Addressed semantically — the
/// operand's identity plus a slot index — never by byte offsets, so an
/// edit survives repacking or an alternative physical representation.
#[derive(Debug, Clone, PartialEq)]
pub enum OperandEdit {
    Row { index: usize, values: Vec<f32> },
    Column { index: usize, values: Vec<f32> },
}

/// Logical edits over stored operands, keyed by operand identity
/// (object + tensor). Applied inside [`OperandSource::load`] — after
/// widening to f32, before any backend requantization — so **every
/// weight format observes the same effective values** (`load_weight`
/// quantizes from the widened f32 buffer).
#[derive(Debug, Default, Clone)]
pub struct OperandOverrides {
    edits: BTreeMap<(String, String), Vec<OperandEdit>>,
}

impl OperandOverrides {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.edits.is_empty()
    }

    /// Record one edit for an operand; edits apply in insertion order.
    pub fn push(&mut self, operand: &OperandRef, edit: OperandEdit) {
        self.edits
            .entry((operand.object.clone(), operand.tensor.clone()))
            .or_default()
            .push(edit);
    }

    pub fn is_overridden(&self, operand: &OperandRef) -> bool {
        self.edits
            .contains_key(&(operand.object.clone(), operand.tensor.clone()))
    }

    /// Apply this operand's edits onto its widened f32 values.
    /// Row-major 2-D shape; an edit that does not fit the operand's
    /// declared shape is an error naming the operand — never a silent
    /// partial write.
    pub fn apply(&self, operand: &OperandRef, values: &mut [f32]) -> Result<(), VindexError> {
        let key = (operand.object.clone(), operand.tensor.clone());
        let Some(edits) = self.edits.get(&key) else {
            return Ok(());
        };
        let (rows, cols) = match operand.shape[..] {
            [rows, cols] => (rows, cols),
            _ => {
                return Err(VindexError::Parse(format!(
                    "operand `{}/{}` is not 2-D; overlay edits address rows/columns",
                    operand.object, operand.tensor
                )))
            }
        };
        for edit in edits {
            match edit {
                OperandEdit::Row { index, values: row } => {
                    if *index >= rows || row.len() != cols {
                        return Err(VindexError::Parse(format!(
                            "row edit {index} (len {}) does not fit `{}/{}` [{rows}, {cols}]",
                            row.len(),
                            operand.object,
                            operand.tensor
                        )));
                    }
                    values[index * cols..(index + 1) * cols].copy_from_slice(row);
                }
                OperandEdit::Column { index, values: col } => {
                    if *index >= cols || col.len() != rows {
                        return Err(VindexError::Parse(format!(
                            "column edit {index} (len {}) does not fit `{}/{}` [{rows}, {cols}]",
                            col.len(),
                            operand.object,
                            operand.tensor
                        )));
                    }
                    for (r, v) in col.iter().enumerate() {
                        values[r * cols + *index] = *v;
                    }
                }
            }
        }
        Ok(())
    }
}

/// The executor's operand resolver: base representation + overlay
/// override → effective operand. Execution asks this seam, never the
/// store directly, so a mutation can alter what execution computes
/// without touching the container's bytes — and a source with no
/// overrides resolves bit-identically to the bare store.
#[derive(Clone, Copy)]
pub struct OperandSource<'a> {
    base: &'a OperandStore,
    overrides: Option<&'a OperandOverrides>,
}

impl<'a> OperandSource<'a> {
    /// A source with overlay edits. An empty overrides value behaves
    /// exactly like the bare store.
    pub fn overlaid(base: &'a OperandStore, overrides: &'a OperandOverrides) -> Self {
        Self {
            base,
            overrides: (!overrides.is_empty()).then_some(overrides),
        }
    }

    /// Load one operand as f32, with any overlay edits applied.
    pub fn load(&self, operand: &OperandRef) -> Result<Vec<f32>, VindexError> {
        let mut values = self.base.load(operand)?;
        if let Some(overrides) = self.overrides {
            overrides.apply(operand, &mut values)?;
        }
        Ok(values)
    }

    /// Load one operand's stored bytes unwidened. Overlay edits are
    /// f32-space facts and cannot be represented in raw stored bytes,
    /// so an overridden operand refuses here rather than serving stale
    /// base bytes.
    pub fn load_raw(&self, operand: &OperandRef) -> Result<RawOperand, VindexError> {
        if let Some(overrides) = self.overrides {
            if overrides.is_overridden(operand) {
                return Err(VindexError::Parse(format!(
                    "operand `{}/{}` carries overlay edits — raw (unwidened) access would \
                     bypass them; load it widened instead",
                    operand.object, operand.tensor
                )));
            }
        }
        self.base.load_raw(operand)
    }
}

impl<'a> From<&'a OperandStore> for OperandSource<'a> {
    fn from(base: &'a OperandStore) -> Self {
        Self {
            base,
            overrides: None,
        }
    }
}
