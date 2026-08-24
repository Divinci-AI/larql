//! What a dense projection kernel must expose, and who threads it.

/// Who owns the machine while a primitive runs.
///
/// Declared by the kernel because only the kernel knows: a BLAS call has
/// already made its own arrangements, a hand-written SIMD loop has not,
/// and a literal reference transcription must stay single-threaded to
/// remain readable beside the source it transcribes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuParallelism {
    /// One call, one thread. The reference oracle.
    Serial,
    /// The implementation is already threaded internally — Accelerate's
    /// `sgemv`. The executor calls it ONCE over the whole output and does
    /// not partition: measured at 1.14x for the effort, and slower than
    /// serial on some shapes.
    LibraryOwned,
    /// The executor partitions output rows across its workers. Kernels
    /// that scale — the fused BF16 kernel went 34.3 -> 122.0 GB/s from
    /// one worker to twelve — and every future low-bit kernel.
    ExternalPool,
}

/// A dense projection `y = W x`, expressed as a row-range primitive.
///
/// Deliberately NOT `fn project(w, x) -> Vec<f32>`: that shape forces the
/// kernel to decide its own threading, which is the thing this design
/// forbids. `project_rows` computes a contiguous slice of the output from
/// exactly the weight rows that produce it, so the executor is free to
/// cut the work however measurement says it should.
pub trait DenseProjector: Sync {
    /// Who threads this kernel. See [`CpuParallelism`].
    fn parallelism(&self) -> CpuParallelism;

    /// Compute `out.len()` output rows.
    ///
    /// `weight_rows` is row-major and covers exactly `out.len()` rows of
    /// `x.len()` columns — the executor has already sliced it. The kernel
    /// must not look outside it, and must not spawn.
    fn project_rows(&self, weight_rows: WeightRows<'_>, x: &[f32], out: &mut [f32]);
}

/// A contiguous slab of weight rows, in whatever representation it is
/// resident as.
///
/// The representation stays COMPACT to this point. CPU-1B's decisive
/// finding was that widening to a scratch matrix first costs more than
/// the traffic it saves — 27.3 GB/s against 122.0 — so a compact format
/// must reach the kernel still compact and be decoded into registers.
#[derive(Clone, Copy)]
pub enum WeightRows<'a> {
    F32(&'a [f32]),
    /// Big-endian-agnostic bf16 code units: each is the top 16 bits of the
    /// f32 it denotes, so widening is `(bits as u32) << 16` — exact, no
    /// rounding, no table.
    Bf16(&'a [u16]),
}

impl WeightRows<'_> {
    /// Rows in this slab, given the column count.
    pub fn rows(&self, in_dim: usize) -> usize {
        match self {
            Self::F32(w) => w.len() / in_dim,
            Self::Bf16(w) => w.len() / in_dim,
        }
    }

    /// A sub-slab of `count` rows starting at `start`.
    pub fn slice_rows(&self, in_dim: usize, start: usize, count: usize) -> Self {
        let (a, b) = (start * in_dim, (start + count) * in_dim);
        match self {
            Self::F32(w) => Self::F32(&w[a..b]),
            Self::Bf16(w) => Self::Bf16(&w[a..b]),
        }
    }

    /// Bytes this slab actually occupies — what the roofline is measured
    /// against, and the whole point of keeping it compact.
    pub fn bytes(&self) -> usize {
        match self {
            Self::F32(w) => w.len() * 4,
            Self::Bf16(w) => w.len() * 2,
        }
    }
}
