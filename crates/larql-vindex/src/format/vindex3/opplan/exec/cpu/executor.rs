//! The persistent worker pool and the partitioning policy.

use super::projector::{CpuParallelism, DenseProjector, WeightRows};

/// LARQL's CPU worker pool.
///
/// Persistent on purpose: decode runs hundreds of projections per token
/// (64 layers x 3 FFN + 5 delta or 4 attention matrices), and rebuilding
/// a task graph for each would cost more than some of them.
pub struct CpuExecutor {
    pool: rayon::ThreadPool,
    workers: usize,
}

/// Below this many bytes a projection is not worth splitting — and,
/// separately, is not worth keeping compact either: a cache-resident
/// matrix has no RAM traffic to halve, and the measured `48 x 5120` case
/// runs 3.8x FASTER through BLAS f32 than through the fused bf16 kernel.
/// Worker count and format are both policy, and both belong here.
///
/// Below this many bytes a projection is not worth splitting.
///
/// Measured, not guessed: the `48 x 5120` delta projections are ~1 MB,
/// fit in cache, and ran at 262 GB/s as a single call — there is no
/// streaming to parallelise, only overhead to add.
const MIN_SPLIT_BYTES: usize = 4 * 1024 * 1024;

impl CpuExecutor {
    /// A pool sized to the machine's performance cores.
    ///
    /// Performance cores specifically: the fused BF16 kernel is a
    /// streaming load, and efficiency cores contribute little to memory
    /// throughput while still taking a share of the rows and finishing
    /// late. Falls back to the total core count where the split is not
    /// reported.
    pub fn new() -> Result<Self, String> {
        let workers = performance_cores();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .thread_name(|i| format!("larql-cpu-{i}"))
            .build()
            .map_err(|e| format!("could not build the CPU executor pool: {e}"))?;
        Ok(Self { pool, workers })
    }

    pub fn workers(&self) -> usize {
        self.workers
    }

    /// How many workers to cut this projection across.
    ///
    /// A policy, deliberately shaped by measurement rather than fixed at
    /// the core count: CPU-1B's large `10240 x 5120` kernel kept scaling
    /// to twelve, but small projections lose to the split, and a future
    /// Q4 kernel with more compute per byte will have its own curve. This
    /// is the one place that judgement lives.
    fn workers_for(&self, kind: CpuParallelism, bytes: usize) -> usize {
        match kind {
            // Already threaded — calling it once is the measured optimum.
            CpuParallelism::LibraryOwned | CpuParallelism::Serial => 1,
            CpuParallelism::ExternalPool => {
                if bytes < MIN_SPLIT_BYTES {
                    1
                } else {
                    self.workers
                }
            }
        }
    }

    /// Run `y = W x` under this executor's threading policy.
    pub fn project(
        &self,
        kernel: &dyn DenseProjector,
        weight: WeightRows<'_>,
        x: &[f32],
        out_dim: usize,
    ) -> Vec<f32> {
        let in_dim = x.len();
        let mut out = vec![0.0f32; out_dim];
        let workers = self.workers_for(kernel.parallelism(), weight.bytes());
        if workers <= 1 || out_dim < workers {
            kernel.project_rows(weight, x, &mut out);
            return out;
        }
        // Row-contiguous partitions: each worker streams one unbroken
        // slab of weight, which is what the memory system wants.
        let rows = out_dim.div_ceil(workers);
        self.pool.install(|| {
            use rayon::prelude::*;
            out.par_chunks_mut(rows).enumerate().for_each(|(i, slot)| {
                let slab = weight.slice_rows(in_dim, i * rows, slot.len());
                kernel.project_rows(slab, x, slot);
            });
        });
        out
    }
}

/// Performance-core count, or the total where the split is unknown.
fn performance_cores() -> usize {
    #[cfg(target_os = "macos")]
    {
        if let Some(n) = sysctl_usize("hw.perflevel0.logicalcpu") {
            return n.max(1);
        }
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

#[cfg(target_os = "macos")]
fn sysctl_usize(name: &str) -> Option<usize> {
    std::process::Command::new("sysctl")
        .args(["-n", name])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse().ok())
}
