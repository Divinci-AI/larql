//! Spin-barrier thread pool for the decode hot path.
//!
//! Rayon puts idle workers to sleep between parallel sections (the right call
//! for batch throughput, the wrong one for a tight decode loop). A 26B-A4B
//! token runs ~200 small fork-join sections — attention Q/K/V/O, dense
//! gate_up/down, the expert fold, lm_head, per layer — and a `/usr/bin/sample`
//! profile attributed ~30% of decode thread-time to the resulting churn:
//! workers asleep in `wait_until_cold`, the driver blocked in
//! `in_worker_cold -> LockLatch::wait_and_reset -> __psynch_cvwait`, plus the
//! condvar wake latency paid on *every* section.
//!
//! This pool keeps workers HOT. They spin on an epoch counter and only
//! [`park`](std::thread::park_timeout) after a long idle gap, so a
//! `for_each_chunk` dispatched microseconds after the previous one finds them
//! already spinning — ready in ~ns, no condvar round-trip. The dispatcher
//! participates as the n-th worker; chunks are owned by static contiguous-block
//! assignment (participant `p` runs one unbroken run of `num/n` chunks), which
//! keeps the `completed == num_chunks` barrier sound across back-to-back
//! dispatches *and* keeps each owner's reads sequential — see `run_chunks` for
//! why the block/stride choice is worth 1.87× at lm_head-class shapes.
//! When a worker has to wait it backs off spin → yield → park, so it stays
//! cooperative under contention. Modeled on llama.cpp's persistent thread
//! pool + `ggml_barrier`.
//!
//! [`enabled`] gates whether callers route through here or stay on rayon. It is
//! **on by default** (the yield backoff makes it safe on shared machines);
//! `LARQL_SPIN_POOL=0` forces the rayon path. Either way the arithmetic is
//! identical — only *which threads run which chunks* differs.

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::Duration;

thread_local! {
    /// True while this thread is executing a dispatched chunk body. Guards
    /// against reentrant `for_each_chunk` (a body that itself dispatches): the
    /// nested call runs serially inline rather than deadlocking on the pool.
    static IN_BODY: Cell<bool> = const { Cell::new(false) };
}

/// Adaptive-backoff thresholds (iterations of the wait loop) for a worker
/// waiting on the next dispatch. It escalates spin → yield → park:
///
/// - **spin** (`< SPIN_HOT`): `spin_loop()` for ~hundreds of µs. This is the
///   same pure-spin window that produced the measured decode win, so *active
///   decode behaviour is unchanged* — every inter-section gap within a token
///   stays in the spin phase, giving a ~ns wake.
/// - **yield** (`< YIELD_UNTIL`): `yield_now()` — cooperative bridge once a wait
///   outlives a whole token's worth of spinning (i.e. the loop went genuinely
///   idle, or another process is starving this core). Hands the core to other
///   runnable threads instead of burning it.
/// - **park** (otherwise): deep idle between requests / runs, ~0 CPU. The
///   dispatcher unparks all workers on every dispatch, so a parked worker wakes
///   immediately on the next section — parking is cheap to enter; the timeout
///   is only a shutdown-check backstop.
///
/// Net: spin = the win during active decode; yield+park = don't peg cores when
/// the decode loop is idle — which is what makes on-by-default safe.
const SPIN_HOT: u32 = 256_000;
const YIELD_UNTIL: u32 = 384_000;

/// Cross-thread dispatch state. Published to workers by the `epoch` release
/// store; workers read the task fields only after the matching acquire load,
/// so the plain `Relaxed` value stores are safe (epoch is the synchroniser).
struct Shared {
    /// Bumped once per `for_each_chunk`; workers wake when it changes.
    epoch: AtomicU64,
    /// Chunks finished this dispatch; the barrier waits for `== num_chunks`.
    /// With static block ownership each chunk is run exactly once, so this
    /// reaching `num_chunks` proves every trampoline call has returned — no
    /// worker can still touch the (about-to-drop) closure.
    completed: AtomicUsize,
    /// Chunk count for the current dispatch.
    num_chunks: AtomicUsize,
    /// Type-erased `&F` for the current dispatch (valid until the barrier).
    data: AtomicPtr<()>,
    /// `fn(*const (), usize)` trampoline that recovers `&F` and calls it.
    tramp: AtomicUsize,
    /// Set on drop; workers observe it and exit.
    shutdown: AtomicBool,
    /// Set when any chunk this dispatch panicked — a cheap flag the dispatcher
    /// checks after the barrier without locking on the happy path.
    panicked: AtomicBool,
    /// The first chunk panic's payload. A panicking body still increments
    /// `completed` (so the barrier finishes instead of hanging on a dead
    /// worker), and the dispatcher `resume_unwind`s this afterward — so a panic
    /// propagates to the caller exactly like rayon, rather than killing a
    /// worker thread and live-locking every future dispatch.
    panic_payload: Mutex<Option<Box<dyn std::any::Any + Send + 'static>>>,
}

/// A persistent spin-barrier pool. Owns `n-1` worker threads; the thread that
/// calls [`for_each_chunk`] is the n-th participant.
pub struct SpinPool {
    shared: Arc<Shared>,
    workers: Vec<thread::JoinHandle<()>>,
    n_threads: usize,
    /// Serializes dispatchers. Uncontended (≈one atomic CAS) for the normal
    /// single-driver decode loop; serializes the rare concurrent dispatch
    /// (`bench --concurrent N`, multi-threaded test harness) so the shared
    /// epoch/cursor state stays consistent.
    dispatch_lock: Mutex<()>,
}

/// Recover `&F` from the type-erased data pointer and invoke it for `chunk`.
///
/// # Safety
/// `data` must point to the live `F` published for the current epoch (the
/// dispatcher keeps it on its stack until the completion barrier passes), and
/// `F: Sync` (multiple threads call it concurrently).
fn trampoline<F: Fn(usize) + Sync>(data: *const (), chunk: usize) {
    // SAFETY: see fn docs — `data` is `&F` published under the epoch fence and
    // outlives every call within the dispatch.
    let f = unsafe { &*(data as *const F) };
    f(chunk);
}

fn worker_loop(shared: Arc<Shared>, worker_id: usize, n_participants: usize) {
    let mut seen_epoch = 0u64;
    loop {
        // Wait for a new dispatch (spin first, park if idle persists).
        let mut spins = 0u32;
        let epoch = loop {
            let e = shared.epoch.load(Ordering::Acquire);
            if e != seen_epoch {
                break e;
            }
            if shared.shutdown.load(Ordering::Relaxed) {
                return;
            }
            spins += 1;
            if spins < SPIN_HOT {
                std::hint::spin_loop();
            } else if spins < YIELD_UNTIL {
                std::thread::yield_now();
            } else {
                thread::park_timeout(Duration::from_micros(50));
            }
        };
        seen_epoch = epoch;
        if shared.shutdown.load(Ordering::Relaxed) {
            return;
        }
        run_chunks(&shared, worker_id, n_participants);
    }
}

/// Run this participant's statically-assigned chunks, as one **contiguous
/// block** of the chunk index space.
///
/// Static ownership — rather than a shared resettable cursor — is what makes
/// `completed == num_chunks` a sound barrier across back-to-back dispatches: no
/// participant can re-claim a chunk the next dispatch reset, so once the count
/// is reached every trampoline call has returned and the closure is safe to
/// drop. *Contiguous* blocks are as static as the round-robin stride this
/// previously used (`c += n_participants`) — each chunk still has exactly one
/// owner — so that argument is untouched by the change from stride to block.
///
/// **Why contiguous rather than strided.** Chunks map to contiguous byte ranges
/// of the weight slab, so a strided owner walks the *whole* slab at a stride of
/// `n_participants × chunk_bytes` and the hardware sequential prefetcher never
/// engages. Measured on `benches/q4k_q8k_matvec.rs::bench_mt_production`
/// (M3 Max, AC, 2026-07-28): at the lm_head-class shape 65536×2816 (415 MB,
/// 2048 chunks × 50 KB, 811 KB stride) the strided pool ran **58.5 GiB/s vs
/// rayon's 109.6** — a 1.87× loss on a default-on path — while every
/// per-layer shape *won* by 45–113%, because those slabs (3–33 MB) are
/// SLC-resident and striding costs nothing there. Blocking keeps the
/// small-shape win (dispatch overhead is what that one is about) and gives each
/// owner one sequential run at large sizes.
///
/// Chunk cost is uniform for every current caller (equal row counts per chunk),
/// so the load-balancing advantage strided ownership would have under uneven
/// chunk cost does not apply; llama.cpp's pool partitions rows the same way.
fn run_chunks(shared: &Shared, participant_id: usize, n_participants: usize) {
    let num = shared.num_chunks.load(Ordering::Relaxed);
    let tramp_addr = shared.tramp.load(Ordering::Relaxed);
    if tramp_addr == 0 || num == 0 || participant_id >= n_participants {
        return;
    }
    let data = shared.data.load(Ordering::Relaxed) as *const ();
    // SAFETY: `tramp_addr` is a `fn(*const (), usize)` stored by the dispatcher
    // before the epoch release; recovered here after the epoch acquire.
    let tramp: fn(*const (), usize) = unsafe { std::mem::transmute(tramp_addr) };
    // Contiguous block for this participant. The first `num % n` participants
    // take one extra chunk, so the assignment covers `0..num` exactly once with
    // a size spread of at most 1 — the property `completed == num_chunks`
    // relies on.
    let base = num / n_participants;
    let rem = num % n_participants;
    let start = participant_id * base + participant_id.min(rem);
    let end = start + base + usize::from(participant_id < rem);

    let mut c = start;
    while c < end {
        // `IN_BODY` makes a reentrant `for_each_chunk` (a body that dispatches)
        // fall back to serial instead of deadlocking. run_chunks is only
        // entered at top level, so the prior value is always false.
        IN_BODY.with(|b| b.set(true));
        // Catch a panicking body so we still `completed.fetch_add` below: a
        // worker that unwound out of the loop would never count its chunk and
        // the dispatcher would spin the barrier forever. The first payload is
        // kept and re-raised on the dispatcher.
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| tramp(data, c)));
        IN_BODY.with(|b| b.set(false));
        if let Err(payload) = r {
            if !shared.panicked.swap(true, Ordering::AcqRel) {
                *shared
                    .panic_payload
                    .lock()
                    .unwrap_or_else(|e| e.into_inner()) = Some(payload);
            }
        }
        shared.completed.fetch_add(1, Ordering::Release);
        c += 1;
    }
}

impl SpinPool {
    /// Build a pool with `n_threads` total participants (spawns `n_threads-1`
    /// persistent workers; the dispatcher is the n-th). `n_threads <= 1` makes
    /// [`for_each_chunk`] run inline with no workers.
    pub fn new(n_threads: usize) -> Self {
        let n_threads = n_threads.max(1);
        let shared = Arc::new(Shared {
            epoch: AtomicU64::new(0),
            completed: AtomicUsize::new(0),
            num_chunks: AtomicUsize::new(0),
            data: AtomicPtr::new(std::ptr::null_mut()),
            tramp: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
            panicked: AtomicBool::new(false),
            panic_payload: Mutex::new(None),
        });
        let workers = (1..n_threads)
            .map(|i| {
                let shared = Arc::clone(&shared);
                thread::Builder::new()
                    .name(format!("larql-spin-{i}"))
                    // Participant `i` of `n_threads`; the dispatcher is 0.
                    .spawn(move || worker_loop(shared, i, n_threads))
                    .expect("spawn spin-pool worker")
            })
            .collect();
        Self {
            shared,
            workers,
            n_threads,
            dispatch_lock: Mutex::new(()),
        }
    }

    /// Number of participating threads (workers + dispatcher).
    pub fn num_threads(&self) -> usize {
        self.n_threads
    }

    /// Run `body(chunk_idx)` for every `chunk_idx in 0..num_chunks`, across the
    /// pool, blocking until all chunks complete.
    ///
    /// `body` must only touch data disjoint per `chunk_idx` — exactly the
    /// contract of `slice::par_chunks_mut().enumerate().for_each()`, which this
    /// replaces. The calling thread participates, so this is *not* reentrant:
    /// `body` must not itself call `for_each_chunk` on the same pool.
    pub fn for_each_chunk<F: Fn(usize) + Sync>(&self, num_chunks: usize, body: F) {
        if num_chunks == 0 {
            return;
        }
        // No workers, or already inside a dispatched body (reentrant): run the
        // chunks serially on this thread. The reentrancy fallback also avoids
        // deadlocking against `dispatch_lock` if a body ever dispatches.
        if self.workers.is_empty() || IN_BODY.with(|b| b.get()) {
            for c in 0..num_chunks {
                body(c);
            }
            return;
        }
        // Serialize dispatchers so the shared epoch/cursor state is consistent;
        // uncontended in the single-driver decode loop.
        let _dispatch = self.dispatch_lock.lock().unwrap_or_else(|e| e.into_inner());
        let shared = &self.shared;
        // Publish the task, then release it to workers via the epoch bump.
        shared
            .data
            .store(&body as *const F as *mut (), Ordering::Relaxed);
        shared
            .tramp
            .store(trampoline::<F> as *const () as usize, Ordering::Relaxed);
        shared.num_chunks.store(num_chunks, Ordering::Relaxed);
        shared.completed.store(0, Ordering::Relaxed);
        shared.panicked.store(false, Ordering::Relaxed);
        shared.epoch.fetch_add(1, Ordering::Release);

        // Wake any worker that parked during an idle gap so the barrier never
        // stalls ~park_timeout waiting on its assigned block. Unparking a
        // still-spinning worker just sets its token (harmless). During tight
        // back-to-back decode dispatches workers stay spinning and this is a
        // no-op fast path.
        for w in &self.workers {
            w.thread().unpark();
        }

        // The dispatcher participates as participant 0.
        run_chunks(shared, 0, self.n_threads);

        // Completion barrier: spin until every chunk has finished. With static
        // block ownership, `completed == num_chunks` means every trampoline
        // call has returned (panics still count, see run_chunks), so it is safe
        // to let `body` drop as this returns.
        while shared.completed.load(Ordering::Acquire) < num_chunks {
            std::hint::spin_loop();
        }

        // Re-raise the first chunk panic on this (the dispatching) thread, so a
        // panicking body propagates to the caller like a serial loop or rayon —
        // instead of being swallowed on a worker. Drop the dispatch guard first
        // so the pool stays usable after the unwind.
        if shared.panicked.load(Ordering::Acquire) {
            let payload = shared
                .panic_payload
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .take();
            drop(_dispatch);
            if let Some(payload) = payload {
                std::panic::resume_unwind(payload);
            }
        }
    }
}

impl Drop for SpinPool {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::Relaxed);
        // Bump epoch so any spinning worker breaks out and re-checks shutdown.
        self.shared.epoch.fetch_add(1, Ordering::Release);
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

/// Number of performance ("P") cores, when the OS exposes a heterogeneous
/// topology. `None` on homogeneous machines and anywhere the query is
/// unavailable — callers then fall back to the plain thread count.
///
/// Apple silicon reports `hw.nperflevels > 1` with level 0 = performance and
/// level 1 = efficiency (M3 Max: 12 P + 4 E).
#[cfg(target_os = "macos")]
fn performance_cores() -> Option<usize> {
    fn sysctl_usize(name: &std::ffi::CStr) -> Option<usize> {
        let mut out: i32 = 0;
        let mut len = std::mem::size_of::<i32>();
        // SAFETY: `name` is a NUL-terminated C string; `out`/`len` are a
        // correctly sized i32 destination and its length, as sysctlbyname
        // requires. It writes at most `len` bytes and updates `len`.
        let rc = unsafe {
            libc::sysctlbyname(
                name.as_ptr(),
                (&mut out as *mut i32).cast(),
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        (rc == 0 && out > 0).then_some(out as usize)
    }

    // Only meaningful when there really are multiple performance levels.
    if sysctl_usize(c"hw.nperflevels")? < 2 {
        return None;
    }
    sysctl_usize(c"hw.perflevel0.logicalcpu")
}

#[cfg(not(target_os = "macos"))]
fn performance_cores() -> Option<usize> {
    None
}

/// Process-wide pool, built on first use.
///
/// Sized to the active rayon thread count, **capped at the performance-core
/// count on heterogeneous CPUs**.
///
/// The cap is load-bearing, not a tuning preference. This pool partitions
/// chunks *statically*, so an efficiency core is handed the same share as a
/// performance core and the completion barrier waits on the slowest
/// participant; rayon's work-stealing rebalances instead, which is why the
/// pathology is specific to this pool. Measured on M3 Max (12 P + 4 E), AC
/// power, `benches/q4k_q8k_matvec.rs::bench_mt_production` at 65536×2816 with
/// `--measurement-time 20`:
///
/// | participants | GiB/s |
/// |---|---|
/// | 16 (12 P + 4 E) | 33.0 |
/// | 12 | 124.0 |
/// | 8 | 126.7 |
///
/// End-to-end on `gemma4-26b-a4b-q4k` (`--cpu -n 50`): **10.6 tok/s at 16
/// participants vs 37.3 at 8** — a 3.5× collapse. Straggler cost scales with
/// the work per dispatch, so small per-layer matvecs still won (+45–113% vs
/// rayon) and only the lm_head-class shape fell over; that is why this hid.
///
/// It hid for a second reason worth recording: `configure_rayon_threads` is
/// called **only** from the CLI's bench path, where it picks 8 on Apple
/// silicon. `larql run` / `larql serve` configure nothing and inherit rayon's
/// default (`available_parallelism()` = 16). So the benchmark harness was
/// structurally unable to observe a bug that every non-bench path hit, and the
/// historical spin-pool numbers (+28%, ~35 tok/s) were all taken at 8.
///
/// Capping here rather than in the CLI fixes it for every consumer —
/// larql-server, embedders, tests — not just the one binary that happened to
/// set a thread count. Bandwidth-bound decode loses nothing: attainable read
/// bandwidth saturates at **two** threads (`examples/membw_probe.rs`), so the
/// E-cores were contributing ~nothing even before they became stragglers.
///
/// See `docs/diagnoses/memory-bandwidth-roofline.md`.
pub fn global() -> &'static SpinPool {
    static POOL: OnceLock<SpinPool> = OnceLock::new();
    POOL.get_or_init(|| {
        let rayon_threads = rayon::current_num_threads().max(1);
        // An explicit smaller thread count still wins — this only removes the
        // E-cores from an otherwise-unconstrained pool.
        let n = performance_cores().map_or(rayon_threads, |p| rayon_threads.min(p.max(1)));
        SpinPool::new(n)
    })
}

/// Whether the decode hot path routes parallel sections through the spin pool
/// instead of rayon. **On by default** — the spin-then-yield backoff makes it
/// safe on shared/contended machines — set `LARQL_SPIN_POOL=0` to force the
/// rayon path (e.g. for an A/B or a heavily oversubscribed host). Either path
/// is numerically identical; only *which threads run which chunks* differs.
pub fn enabled() -> bool {
    crate::options::spin_pool_enabled()
}

/// Drop-in for `out.par_chunks_mut(chunk).enumerate().for_each(|(ci, c)| body(ci, c))`
/// that routes through the spin pool when [`enabled`], else stays on rayon.
///
/// `body(chunk_idx, chunk)` receives each disjoint `chunk`-sized (last shorter)
/// slice of `out` and its index — identical semantics either way, so the
/// arithmetic is unchanged; only *which thread runs which chunk* differs.
pub fn par_chunks_mut<T, F>(out: &mut [T], chunk: usize, body: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync + Send,
{
    if chunk == 0 || out.is_empty() {
        return;
    }
    if enabled() {
        let total = out.len();
        let n = total.div_ceil(chunk);
        let base = out.as_mut_ptr() as usize;
        global().for_each_chunk(n, |ci| {
            let start = ci * chunk;
            // `start < total` holds by construction for `ci < n` - but this
            // feeds a raw `.add(start)` below with no bounds check of its
            // own, so `saturating_sub` (not `-`) turns any violation of that
            // invariant into an inert zero-length chunk instead of a wrapped
            // `usize` producing a wild slice length (release builds have no
            // overflow-checks).
            let len = chunk.min(total.saturating_sub(start));
            if len == 0 {
                return;
            }
            // SAFETY: chunk index `ci` owns the disjoint range
            // `[start, start+len)` of `out`; no two chunks overlap, and the
            // dispatch barrier keeps `out` borrowed for the whole call.
            let s = unsafe { std::slice::from_raw_parts_mut((base as *mut T).add(start), len) };
            body(ci, s);
        });
    } else {
        use rayon::prelude::*;
        out.par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(ci, c)| body(ci, c));
    }
}

/// Two-output sibling of [`par_chunks_mut`] for kernels that write `a` and `b`
/// at the same row index (e.g. the fused gate/up dual matvec). `a` and `b` must
/// have the same length; `body(chunk_idx, a_chunk, b_chunk)` gets the matching
/// disjoint slices.
pub fn par_chunks_mut2<T, F>(a: &mut [T], b: &mut [T], chunk: usize, body: F)
where
    T: Send,
    F: Fn(usize, &mut [T], &mut [T]) + Sync + Send,
{
    debug_assert_eq!(a.len(), b.len(), "par_chunks_mut2 needs equal-length a/b");
    if chunk == 0 || a.is_empty() {
        return;
    }
    if enabled() {
        let total = a.len();
        let n = total.div_ceil(chunk);
        let base_a = a.as_mut_ptr() as usize;
        let base_b = b.as_mut_ptr() as usize;
        global().for_each_chunk(n, |ci| {
            let start = ci * chunk;
            // See the matching comment in `par_chunks_mut` above.
            let len = chunk.min(total.saturating_sub(start));
            if len == 0 {
                return;
            }
            // SAFETY: disjoint per-chunk ranges of `a` and `b` (separate
            // buffers); barrier keeps both borrowed for the call.
            let sa = unsafe { std::slice::from_raw_parts_mut((base_a as *mut T).add(start), len) };
            let sb = unsafe { std::slice::from_raw_parts_mut((base_b as *mut T).add(start), len) };
            body(ci, sa, sb);
        });
    } else {
        use rayon::prelude::*;
        a.par_chunks_mut(chunk)
            .zip(b.par_chunks_mut(chunk))
            .enumerate()
            .for_each(|(ci, (ca, cb))| body(ci, ca, cb));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;

    #[test]
    fn runs_every_chunk_exactly_once() {
        let pool = SpinPool::new(4);
        let hits: Vec<AtomicU32> = (0..1000).map(|_| AtomicU32::new(0)).collect();
        pool.for_each_chunk(hits.len(), |c| {
            hits[c].fetch_add(1, Ordering::Relaxed);
        });
        for (i, h) in hits.iter().enumerate() {
            assert_eq!(h.load(Ordering::Relaxed), 1, "chunk {i} ran != once");
        }
    }

    /// Each participant must get ONE unbroken run of chunk indices.
    ///
    /// `runs_every_chunk_exactly_once` pins the correctness invariant the
    /// barrier needs, and it passes under round-robin striding too — so it
    /// cannot catch a revert to `c += n_participants`. Contiguity is the
    /// *performance* invariant: chunks map to contiguous byte ranges of the
    /// weight slab, and a strided owner defeats the sequential prefetcher
    /// (measured 1.87× slower at the 415 MB lm_head-class shape). This test is
    /// what makes that regression loud instead of silent.
    ///
    /// Uneven counts are included deliberately: the `rem` participants take one
    /// extra chunk, which is where an off-by-one would overlap or gap two
    /// participants' ranges.
    #[test]
    fn each_participant_runs_a_contiguous_block() {
        for (n_threads, num_chunks) in [(4usize, 1000usize), (4, 1001), (4, 3), (8, 8), (3, 100)] {
            let pool = SpinPool::new(n_threads);
            let seen: std::sync::Mutex<Vec<(std::thread::ThreadId, usize)>> =
                std::sync::Mutex::new(Vec::new());
            pool.for_each_chunk(num_chunks, |c| {
                seen.lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .push((std::thread::current().id(), c));
            });

            let mut by_thread: std::collections::HashMap<std::thread::ThreadId, Vec<usize>> =
                std::collections::HashMap::new();
            for (tid, c) in seen.into_inner().unwrap_or_else(|e| e.into_inner()) {
                by_thread.entry(tid).or_default().push(c);
            }

            let mut covered = 0usize;
            for (tid, mut chunks) in by_thread {
                chunks.sort_unstable();
                let (first, last) = (chunks[0], chunks[chunks.len() - 1]);
                assert_eq!(
                    last - first + 1,
                    chunks.len(),
                    "participant {tid:?} got a non-contiguous block {chunks:?} \
                     (n_threads={n_threads}, num_chunks={num_chunks}) — this is the \
                     strided-ownership regression; see run_chunks"
                );
                covered += chunks.len();
            }
            assert_eq!(
                covered, num_chunks,
                "blocks must cover 0..{num_chunks} exactly (n_threads={n_threads})"
            );
        }
    }

    /// The global pool must never include efficiency cores.
    ///
    /// Static partitioning makes the barrier wait on the slowest participant,
    /// so one E-core in the pool cost 3.5× end-to-end on the 26B. This asserts
    /// the cap rather than the mechanism, because the mechanism only shows up
    /// under a large-shape benchmark on a heterogeneous box — the exact
    /// combination that let the regression ship unnoticed.
    #[test]
    fn global_pool_excludes_efficiency_cores() {
        let n = global().num_threads();
        assert!(n >= 1, "pool must have at least one participant");
        if let Some(p) = performance_cores() {
            assert!(
                n <= p,
                "global pool has {n} participants but only {p} performance cores — \
                 an efficiency core in a statically-partitioned pool stalls the barrier \
                 (see global() docs)"
            );
        }
        assert!(
            n <= rayon::current_num_threads().max(1),
            "pool must never exceed an explicitly configured thread count"
        );
    }

    #[test]
    fn disjoint_mut_writes_match_serial() {
        // The production pattern: each chunk writes its disjoint row range of a
        // shared output buffer via a raw pointer (caller guarantees disjoint).
        let pool = SpinPool::new(4);
        let rows = 517usize;
        let chunk = 32usize;
        let n_chunks = rows.div_ceil(chunk);
        let mut out = vec![0u64; rows];
        let ptr = out.as_mut_ptr() as usize;
        pool.for_each_chunk(n_chunks, |ci| {
            let start = ci * chunk;
            let end = (start + chunk).min(rows);
            for r in start..end {
                // SAFETY: chunks are disjoint row ranges of `out`.
                unsafe { *(ptr as *mut u64).add(r) = (r as u64) * 3 + 1 };
            }
        });
        for (r, v) in out.iter().enumerate() {
            assert_eq!(*v, (r as u64) * 3 + 1);
        }
    }

    #[test]
    fn parallel_sum_matches_serial() {
        let pool = SpinPool::new(8);
        let n = 100_000usize;
        let partials: Vec<AtomicU64> = (0..64).map(|_| AtomicU64::new(0)).collect();
        let chunk = n.div_ceil(64);
        pool.for_each_chunk(64, |ci| {
            let start = ci * chunk;
            let end = (start + chunk).min(n);
            let s: u64 = (start as u64..end as u64).sum();
            partials[ci].store(s, Ordering::Relaxed);
        });
        let got: u64 = partials.iter().map(|a| a.load(Ordering::Relaxed)).sum();
        let want: u64 = (0..n as u64).sum();
        assert_eq!(got, want);
    }

    #[test]
    fn zero_chunks_is_noop() {
        let pool = SpinPool::new(4);
        pool.for_each_chunk(0, |_| panic!("must not run"));
    }

    #[test]
    fn single_thread_runs_inline() {
        let pool = SpinPool::new(1);
        let hits: Vec<AtomicU32> = (0..50).map(|_| AtomicU32::new(0)).collect();
        pool.for_each_chunk(hits.len(), |c| {
            hits[c].fetch_add(1, Ordering::Relaxed);
        });
        assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1));
    }

    #[test]
    fn chunk_panic_propagates_and_pool_stays_usable() {
        // A panicking body must (a) NOT hang the barrier (a dead worker would
        // never count its chunk → dispatcher spins forever) and (b) propagate
        // the panic to the dispatcher. Chunk 37 lands on a worker (37 % 4 != 0),
        // exercising the worker-side catch, not just the dispatcher's own.
        let pool = SpinPool::new(4);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool.for_each_chunk(50, |c| {
                if c == 37 {
                    panic!("boom at chunk {c}");
                }
            });
        }));
        assert!(
            result.is_err(),
            "a panicking chunk body must propagate to the dispatcher"
        );
        // The pool must still work after a panic (not poisoned / not hung).
        let hits: Vec<AtomicU32> = (0..20).map(|_| AtomicU32::new(0)).collect();
        pool.for_each_chunk(hits.len(), |c| {
            hits[c].fetch_add(1, Ordering::Relaxed);
        });
        assert!(
            hits.iter().all(|h| h.load(Ordering::Relaxed) == 1),
            "pool must stay usable after a chunk panic"
        );
    }

    #[test]
    fn concurrent_dispatchers_stay_consistent() {
        // Multiple driver threads dispatching on one shared pool (the
        // `--concurrent N` / multi-threaded-test shape). The dispatch lock
        // serializes them; each dispatch must still complete correctly.
        let pool = SpinPool::new(4);
        std::thread::scope(|s| {
            for _ in 0..3 {
                s.spawn(|| {
                    for round in 1..=50u64 {
                        let acc: Vec<AtomicU64> = (0..20).map(|_| AtomicU64::new(0)).collect();
                        pool.for_each_chunk(20, |c| {
                            acc[c].store(round * (c as u64 + 1), Ordering::Relaxed);
                        });
                        for (c, a) in acc.iter().enumerate() {
                            assert_eq!(a.load(Ordering::Relaxed), round * (c as u64 + 1));
                        }
                    }
                });
            }
        });
    }

    /// Cross-dispatch read-after-write — the real decode pipeline shape
    /// (dispatch A writes a buffer; the *next* dispatch B reads it and writes a
    /// derived buffer). Exercises the visibility the disjoint-write tests don't:
    /// workers running dispatch B must observe ALL of dispatch A's writes (the
    /// `barrier_A.Acquire → epoch_B.Release → worker_B.Acquire` chain). The pool
    /// is oversubscribed (more workers than cores) so the barrier routinely waits
    /// on a descheduled worker. Kept fast (a few hundred rounds) — under EXTREME
    /// oversubscription (2× burners, 4000 rounds) this and the disjoint-write
    /// path stayed correct, so this is a regression guard, not a repro.
    #[test]
    fn stress_cross_dispatch_read_after_write() {
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8);
        // Oversubscribe the pool itself (more workers than cores) so the barrier
        // routinely waits on a descheduled worker.
        let pool = SpinPool::new((cores + 2).max(4));
        let n = 61usize; // chunks; not a multiple of the thread count
        let mut a = vec![0u64; n];
        let mut b = vec![0u64; n];
        for round in 1..=400u64 {
            // Dispatch A: fill `a` with a round-derived pattern.
            let pa = a.as_mut_ptr() as usize;
            pool.for_each_chunk(n, |c| {
                // SAFETY: chunk c owns element c.
                unsafe { *(pa as *mut u64).add(c) = round.wrapping_mul(c as u64 + 1) | 1 };
            });
            // Dispatch B: read `a`, write `b = f(a)`. If B's workers don't see
            // all of A's writes, `b[c]` is wrong (or derived from a stale 0).
            let pa_r = a.as_ptr() as usize;
            let pb = b.as_mut_ptr() as usize;
            pool.for_each_chunk(n, |c| {
                // SAFETY: read element c (written by A's chunk c), write b[c].
                let av = unsafe { *(pa_r as *const u64).add(c) };
                unsafe { *(pb as *mut u64).add(c) = av.wrapping_mul(31).wrapping_add(7) };
            });
            for c in 0..n {
                let want_a = round.wrapping_mul(c as u64 + 1) | 1;
                assert_eq!(a[c], want_a, "round {round} chunk {c}: A wrong");
                assert_eq!(
                    b[c],
                    want_a.wrapping_mul(31).wrapping_add(7),
                    "round {round} chunk {c}: B read a stale/partial A"
                );
            }
        }
    }

    #[test]
    fn back_to_back_dispatches_reuse_workers() {
        // Exercises the epoch path: many tiny dispatches in a row (the decode
        // loop shape) must each complete fully.
        let pool = SpinPool::new(4);
        for round in 1..=200u64 {
            let acc: Vec<AtomicU64> = (0..16).map(|_| AtomicU64::new(0)).collect();
            pool.for_each_chunk(16, |c| {
                acc[c].store(round * (c as u64 + 1), Ordering::Relaxed);
            });
            for (c, a) in acc.iter().enumerate() {
                assert_eq!(a.load(Ordering::Relaxed), round * (c as u64 + 1));
            }
        }
    }

    // ── SIGSEGV reproduction attempt (three crash reports, all localizing
    // inside `par_chunks_mut`/`run_chunks`, fault addresses matching real
    // model dimensions or `0x1`) ────────────────────────────────────────────

    const RD_HIDDEN: usize = 2560;
    const RD_INTER: usize = 10240;
    const RD_Q_DIM: usize = 2048;
    const RD_KV_DIM: usize = 1024;
    const RD_ROWS_CHUNK: usize = 32;
    const RD_ELEM_CHUNK: usize = 256;

    fn rd_f32_encode(caller: u64, round: u64, tag: u64, idx: usize) -> f32 {
        // Cheap, collision-resistant-enough encoding of (caller, round, tag,
        // idx) into an f32 so a read of the wrong slot/epoch/caller's data is
        // caught as a mismatch rather than silently looking plausible.
        ((caller.wrapping_mul(998_244_353)
            ^ round.wrapping_mul(1_000_003)
            ^ tag.wrapping_mul(97)
            ^ idx as u64)
            % 1_000_000) as f32
    }
    fn rd_fill_chunk(chunk: &mut [f32], global_start: usize, caller: u64, round: u64, tag: u64) {
        for (local_i, v) in chunk.iter_mut().enumerate() {
            // Cheap but non-trivial busy-work per element (~tens of ns) so a
            // chunk's dispatched duration is closer to the real matvec
            // kernel's (many SDOTs per row) than a near-instant synthetic
            // write - in case the bug is timing/duration-dependent (spin ->
            // yield -> park transitions calibrated around real chunk cost).
            let mut acc = 0.0f32;
            for j in 0..64u32 {
                acc = acc * 1.0000001 + (j as f32).sin();
            }
            *v = rd_f32_encode(caller, round, tag, global_start + local_i) + acc * 0.0;
        }
    }
    fn rd_check(buf: &[f32], caller: u64, round: u64, tag: u64) {
        for (i, &v) in buf.iter().enumerate() {
            let want = rd_f32_encode(caller, round, tag, i);
            assert_eq!(
                v.to_bits(),
                want.to_bits(),
                "caller {caller} round {round} tag {tag} idx {i}: corrupted \
                 (got {v}, want {want}) - buffer len {}",
                buf.len()
            );
        }
    }

    /// One simulated decode step's worth of dispatches against `par_chunks_mut`
    /// - the actual public entry point production uses, with the REAL
    ///   gemma-3-4b-it row counts (q_dim=2048, kv_dim=1024, hidden=2560,
    ///   intermediate=10240 - all exact multiples of their chunk sizes, same as
    ///   production, so this can't exercise the underflow this file already
    ///   hardened against; it's targeting a different bug). `caller` tags every
    ///   value so concurrent callers (simulating overlapping requests) can tell
    ///   their own data apart from a caller they got mixed up with.
    fn rd_run(caller: u64, rounds: u64, layers: u64) {
        let mut q = vec![0.0f32; RD_Q_DIM];
        let mut k = vec![0.0f32; RD_KV_DIM];
        let mut v = vec![0.0f32; RD_KV_DIM];
        let mut o = vec![0.0f32; RD_HIDDEN];
        let mut gate = vec![0.0f32; RD_INTER];
        let mut up = vec![0.0f32; RD_INTER];
        let mut activated = vec![0.0f32; RD_INTER];
        let mut down = vec![0.0f32; RD_HIDDEN];

        for round in 0..rounds {
            for layer in 0..layers {
                let tag = round * layers + layer;
                for (buf, sub) in [
                    (&mut q, 0u64),
                    (&mut k, 1),
                    (&mut v, 2),
                    (&mut o, 3),
                    (&mut gate, 4),
                    (&mut up, 5),
                    (&mut down, 7),
                ] {
                    let t = tag.wrapping_mul(8) + sub;
                    par_chunks_mut(buf, RD_ROWS_CHUNK, |ci, chunk| {
                        rd_fill_chunk(chunk, ci * RD_ROWS_CHUNK, caller, round, t)
                    });
                    rd_check(buf, caller, round, t);
                }
                // Elementwise activation: 256-chunked, INTER-sized - the
                // production shape at kquant_forward/cached.rs:869.
                let t = tag.wrapping_mul(8) + 6;
                par_chunks_mut(&mut activated, RD_ELEM_CHUNK, |ci, chunk| {
                    rd_fill_chunk(chunk, ci * RD_ELEM_CHUNK, caller, round, t)
                });
                rd_check(&activated, caller, round, t);
            }
        }
    }

    /// Single sequential caller, at production scale (34 layers x 400
    /// simulated tokens). `--test-threads` contention from other tests
    /// sharing `global()` is part of the reproduction attempt - do not run
    /// this test in isolation only.
    #[test]
    fn stress_realistic_decode_shape_no_corruption() {
        rd_run(0, 40, 34);
    }

    /// Genuinely concurrent callers (unlike `concurrent_dispatchers_stay_
    /// consistent` above, which uses one FIXED dispatch size for every
    /// thread) - each thread runs its own independent `rd_run`-shaped decode
    /// loop against the SAME shared `global()` pool at the SAME time, so
    /// `dispatch_lock` has to actually serialize dispatchers with DIFFERENT
    /// `(total, chunk)` shapes mid-flight, not just identical ones.
    #[test]
    fn stress_concurrent_realistic_decode_shape_no_corruption() {
        std::thread::scope(|s| {
            for caller in 0..6u64 {
                s.spawn(move || rd_run(caller, 8, 34));
            }
        });
    }
}
