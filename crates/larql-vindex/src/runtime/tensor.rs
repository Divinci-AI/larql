//! A bound tensor — resolved bytes plus everything needed to read them.
//!
//! This is the terminal product of binding. It carries no coordinate to look
//! up, no variant to choose and no candidate to prefer: the decisions are
//! already made, and what remains is arithmetic.
//!
//! # The view is part of the operand, not a caller's problem
//!
//! Storage shape and role shape differ whenever a view is in play — a tied
//! embedding serving an LM head is `[vocab, hidden]` on disk and `[hidden,
//! vocab]` to the operation that reads it. The view lives here so that every
//! consumer indexes in *role* coordinates and none of them has to know which
//! physical arrangement it got. That is the property that lets fused and
//! decomposed storage produce identical output through one executor.
//!
//! # Deliberately plain
//!
//! Decoding dispatches per element. That is the wrong shape for a hot path and
//! the right shape for a reference: it is obviously correct, it has no layout
//! special-cases to get wrong, and it is what a fast kernel gets checked
//! against. Production paths bind quantised regions to `larql-compute`
//! kernels; this decoder exists to say what the answer should have been.

use crate::format::capability::binding::{ComponentView, RepresentationIdentity};
use crate::format::capability::component::{ComponentContract, TensorKind};
use crate::format::lyrw2::region_format::RegionFormat;

use super::axis::Axis;
use super::consts::{BF16_BYTES, BF16_SHIFT, COL_DIM, F16_BYTES, F32_BYTES, MATRIX_RANK, ROW_DIM};
use super::error::{ExecutionError, OperandUnsuitability};

/// Resolved bytes, with the encoding and access pattern that read them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundTensor<'a> {
    /// Which catalogue declaration this came from. Diagnostics only — nothing
    /// in execution branches on it.
    representation: RepresentationIdentity,
    bytes: &'a [u8],
    format: RegionFormat,
    /// Shape as stored on disk.
    storage: ComponentContract,
    /// Shape the operation sees, after `view`.
    role: ComponentContract,
    view: ComponentView,
    /// Bytes per element under `format`. Resolved once, at bind time.
    stride: usize,
}

impl<'a> BoundTensor<'a> {
    /// Bind bytes to a role.
    ///
    /// Fails if the view cannot apply to the storage shape, or if the region
    /// is too short for the shape it claims — both binding faults, caught here
    /// rather than as an out-of-bounds read later.
    pub fn new(
        representation: RepresentationIdentity,
        bytes: &'a [u8],
        format: RegionFormat,
        storage: ComponentContract,
        view: ComponentView,
    ) -> Result<Self, ExecutionError> {
        let operand = representation.describe();
        let role = view
            .apply_to(&storage)
            .map_err(|_| ExecutionError::UnsupportedView {
                view: view.describe(),
                operand: operand.clone(),
            })?;
        let elements: usize = storage.shape.iter().map(|d| *d as usize).product();
        let stride = element_bytes(format, &operand)?;
        let needed = elements * stride;
        if bytes.len() < needed {
            return Err(ExecutionError::ShortRegion {
                operand,
                needed,
                found: bytes.len(),
            });
        }
        Ok(Self {
            representation,
            bytes,
            format,
            storage,
            role,
            view,
            stride,
        })
    }

    /// Convenience for the common case: bytes read exactly as stored.
    pub fn direct(
        representation: RepresentationIdentity,
        bytes: &'a [u8],
        format: RegionFormat,
        storage: ComponentContract,
    ) -> Result<Self, ExecutionError> {
        Self::new(
            representation,
            bytes,
            format,
            storage,
            ComponentView::Direct,
        )
    }

    /// The stored bytes, for tests that must confirm what is *behind* a view.
    #[cfg(test)]
    pub(crate) fn bytes_for_test(&self) -> &'a [u8] {
        self.bytes
    }

    /// This operand as a contiguous `f32` slice, if it genuinely is one.
    ///
    /// For binding an operand to a kernel that takes `&[f32]` in row-major
    /// order — the incumbent's BLAS scoring, for instance — **without
    /// reconstructing it**. A bridge that dequantised, repacked into an
    /// incumbent-shaped temporary and then called the kernel could reach
    /// numerical parity while proving nothing about the binding architecture.
    /// This hands over the stored bytes or refuses.
    ///
    /// The error distinguishes format, view, alignment and length, because
    /// each implies a different remedy: bind another variant, bind a
    /// view-aware kernel, take an aligned copy, or reject the index. Only the
    /// last is a defect.
    pub fn as_f32_slice(&self) -> Result<&'a [f32], OperandUnsuitability> {
        const WANTED: &str = "contiguous row-major f32";
        if self.format != RegionFormat::F32 {
            return Err(OperandUnsuitability::ElementFormat {
                found: self.format.name(),
                wanted: WANTED,
            });
        }
        if self.view != ComponentView::Direct {
            return Err(OperandUnsuitability::NonDirectView {
                view: self.view.describe(),
            });
        }
        // SAFETY: `align_to` is sound for any `T: Copy` with no invalid bit
        // patterns, which `f32` satisfies. The empty-prefix check is what
        // makes the result the *whole* slice rather than a shifted window.
        let (prefix, values, _) = unsafe { self.bytes.align_to::<f32>() };
        if !prefix.is_empty() {
            return Err(OperandUnsuitability::MisalignedBase { wanted: WANTED });
        }
        if values.len() < self.len() {
            return Err(OperandUnsuitability::Length {
                expected: self.len(),
                found: values.len(),
            });
        }
        Ok(&values[..self.len()])
    }

    pub fn representation(&self) -> &RepresentationIdentity {
        &self.representation
    }

    pub fn format(&self) -> RegionFormat {
        self.format
    }

    pub fn view(&self) -> &ComponentView {
        &self.view
    }

    /// The shape the operation sees.
    pub fn contract(&self) -> &ComponentContract {
        &self.role
    }

    pub fn describe(&self) -> String {
        self.representation.describe()
    }

    /// Rows in role coordinates.
    pub fn rows(&self) -> usize {
        self.role.shape.first().copied().unwrap_or(0) as usize
    }

    /// Columns in role coordinates. A vector has one column's worth per row.
    pub fn cols(&self) -> usize {
        match self.role.kind {
            TensorKind::Matrix => self.role.shape.get(COL_DIM).copied().unwrap_or(0) as usize,
            TensorKind::Vector => 1,
        }
    }

    pub fn len(&self) -> usize {
        self.role.shape.iter().map(|d| *d as usize).product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Assert this operand has the expected role shape.
    ///
    /// Binding checks contracts, so a failure here means binding accepted an
    /// operand it should have refused.
    pub fn require_matrix(&self, rows: usize, cols: usize) -> Result<(), ExecutionError> {
        if self.role.kind != TensorKind::Matrix || self.role.shape.len() != MATRIX_RANK {
            return Err(ExecutionError::NotAMatrix {
                operand: self.describe(),
                found: self.role.describe(),
            });
        }
        self.require_axis(Axis::Rows, rows, self.rows())?;
        self.require_axis(Axis::Columns, cols, self.cols())
    }

    /// Assert a vector operand's length.
    pub fn require_vector(&self, len: usize) -> Result<(), ExecutionError> {
        self.require_axis(Axis::Length, len, self.len())
    }

    fn require_axis(
        &self,
        axis: Axis,
        expected: usize,
        found: usize,
    ) -> Result<(), ExecutionError> {
        if expected == found {
            return Ok(());
        }
        Err(ExecutionError::DimensionMismatch {
            operand: self.describe(),
            axis,
            expected,
            found,
        })
    }

    /// Read one row, in role coordinates, into `out`.
    pub fn row_into(&self, row: usize, out: &mut [f32]) -> Result<(), ExecutionError> {
        if row >= self.rows() {
            return Err(ExecutionError::RowOutOfRange {
                operand: self.describe(),
                row,
                rows: self.rows(),
            });
        }
        self.require_axis(Axis::Columns, out.len(), self.cols())?;
        for (col, slot) in out.iter_mut().enumerate() {
            *slot = self.decode_at(self.storage_index(row, col)?)?;
        }
        Ok(())
    }

    /// Read one row, in role coordinates.
    pub fn row(&self, row: usize) -> Result<Vec<f32>, ExecutionError> {
        let mut out = vec![0.0f32; self.cols()];
        self.row_into(row, &mut out)?;
        Ok(out)
    }

    /// Read a vector operand whole.
    pub fn to_vec(&self) -> Result<Vec<f32>, ExecutionError> {
        (0..self.len()).map(|i| self.decode_at(i)).collect()
    }

    /// Map a role-coordinate cell to its index in storage.
    ///
    /// The whole point of the view: every caller indexes as the role, and the
    /// physical arrangement is resolved here once.
    fn storage_index(&self, row: usize, col: usize) -> Result<usize, ExecutionError> {
        let storage_cols = match self.storage.kind {
            TensorKind::Matrix => self.storage.shape.get(COL_DIM).copied().unwrap_or(0) as usize,
            TensorKind::Vector => 1,
        };
        Ok(match &self.view {
            ComponentView::Direct => row * storage_cols + col,
            ComponentView::Transpose => col * storage_cols + row,
            ComponentView::Slice { dim, start, .. } => {
                let start = *start as usize;
                match *dim {
                    ROW_DIM => (start + row) * storage_cols + col,
                    COL_DIM => row * storage_cols + start + col,
                    _ => {
                        return Err(ExecutionError::UnsupportedView {
                            view: self.view.describe(),
                            operand: self.describe(),
                        })
                    }
                }
            }
        })
    }

    /// Decode a single stored element.
    ///
    /// `stride` is resolved at bind time rather than here. Recomputing it per
    /// element also meant building the operand name per element — a `String`
    /// allocation on every scalar read, which dominated the reference decoder
    /// and had nothing to do with decoding.
    fn decode_at(&self, index: usize) -> Result<f32, ExecutionError> {
        let stride = self.stride;
        let at = index * stride;
        let raw = self
            .bytes
            .get(at..at + stride)
            .ok_or_else(|| ExecutionError::ShortRegion {
                operand: self.describe(),
                needed: at + stride,
                found: self.bytes.len(),
            })?;
        Ok(match self.format {
            RegionFormat::F32 => f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]),
            // Shared with the quantised kernels on purpose: its subnormal
            // branch fixes a 2× error a local reimplementation would repeat.
            RegionFormat::F16 => larql_compute::f16_to_f32(u16::from_le_bytes([raw[0], raw[1]])),
            RegionFormat::BF16 => {
                f32::from_bits(u32::from(u16::from_le_bytes([raw[0], raw[1]])) << BF16_SHIFT)
            }
            other => return Err(ExecutionError::unsupported_format(other, self.describe())),
        })
    }
}

/// Bytes per element, or a refusal naming the encoding.
fn element_bytes(format: RegionFormat, operand: &str) -> Result<usize, ExecutionError> {
    Ok(match format {
        RegionFormat::F32 => F32_BYTES,
        RegionFormat::F16 => F16_BYTES,
        RegionFormat::BF16 => BF16_BYTES,
        other => return Err(ExecutionError::unsupported_format(other, operand)),
    })
}
