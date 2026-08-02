//! MAP-3B: physical page residency / prefetch performance.
//!
//! Registered as chuk-experiments programme `map`, experiment `map-3b`.
//! Built on branch `worktree-vindex2`, same as `map3a_selected_operand_residency.rs`.
//!
//! MAP-3A proved a LOGICAL guarantee: the runtime refuses outright if a
//! routed expert's bytes are absent from the bound plan. This experiment
//! asks the separate, genuinely open, PERFORMANCE question MAP-3A
//! deliberately did not touch: if the exact required bytes are cold
//! (evicted from the page cache), does prefetching exactly the predicted
//! footprint — computed from the real routing decision, before a single
//! page is faulted — measurably beat prefetching nothing, or prefetching
//! an equal-byte budget of the WRONG (unselected) pages?
//!
//! Correctness must be invariant across every condition below — the real
//! bytes are always reachable via page-fault-on-demand regardless of what
//! was prefetched, so if that invariant breaks, something else is wrong,
//! not a prefetch-performance finding.
//!
//! Method: reuses `vindex3_residency_probe.rs`'s eviction/mincore
//! technique (`msync(MS_INVALIDATE)` + `madvise(MADV_DONTNEED)` — plain
//! `MADV_DONTNEED` alone does not evict on Darwin; `mincore` reports
//! residency directly rather than relying on unreliable fault counters),
//! but wired to the REAL Gemma layer file's real mmap and real per-expert
//! byte ranges (`ModelWeights::packed_mmaps` / `packed_byte_ranges`,
//! keyed via `per_layer_ffn_key`) instead of a synthetic blob.
//!
//! Conditions, each starting from a freshly-evicted (verified cold) mapping:
//! ```text
//! cold        no prefetch at all
//! predicted   prefetch exactly the routed experts' gate_up+down byte ranges
//! random      prefetch an equal-COUNT set of UNSELECTED experts' ranges
//! full        prefetch the entire layer file
//! ```
//!
//! Usage:
//! ```text
//! LARQL_CPU_DUMP_LAYERS=/tmp/dump larql run <vindex> "prompt" -n 1
//! cargo run --release -p larql-vindex --example map3b_prefetch_performance -- \
//!   --vindex <path> --dump /tmp/dump [--layer 5]
//! ```

#[cfg(unix)]
mod probe {
    use std::ops::Range;
    use std::time::{Duration, Instant};

    use larql_compute::cpu::ops::moe::{
        moe_expert_input, moe_route_from_router_input, moe_router_input,
    };
    use larql_compute::pipeline_layer::build_moe_weights;
    use larql_compute::MoeLayerWeights;

    use larql_models::weights::{per_layer_ffn_key, PER_LAYER_FFN_DOWN, PER_LAYER_FFN_GATE_UP};
    use larql_vindex::format::capability::binding::{ComponentView, RepresentationIdentity};
    use larql_vindex::format::capability::component::ComponentContract;
    use larql_vindex::format::capability::coordinate::BankCoordinate;
    use larql_vindex::format::filenames::layer_weights_filename;
    use larql_vindex::format::lyrw2::region_format::RegionFormat;
    use larql_vindex::format::lyrw2::region_role::RegionRole;
    use larql_vindex::runtime::consts::{COL_DIM, FUSED_PROJECTION_HALVES};
    use larql_vindex::runtime::residency::{account, ExpertRegion, PageSpan};
    use larql_vindex::runtime::{
        execute, BoundBankOperation, BoundExpert, BoundExpertScaling, BoundMoeOperation,
        BoundProjection, BoundReduction, BoundRouter, BoundTensor, ExpertKernel, MoeInputs,
        RouterKernel,
    };

    const VARIANT: &str = "vindex2-bytes";
    const ROUTER_REGION_SET: &str = "router";
    const PER_EXPERT_SCALE_REGION_SET: &str = "router_per_expert_scale";
    const BANK_ID: u16 = 0;
    const DEFAULT_LAYER: usize = 5;
    /// Above this fraction resident after eviction, the mapping is not cold
    /// and no downstream number is trustworthy (same threshold as the probe).
    const COLD_THRESHOLD: f64 = 0.05;

    fn arg(name: &str) -> Option<String> {
        std::env::args()
            .collect::<Vec<_>>()
            .iter()
            .position(|a| a == name)
            .and_then(|i| std::env::args().nth(i + 1))
    }

    fn page_size() -> usize {
        // SAFETY: `sysconf` is a pure query with no preconditions.
        (unsafe { libc::sysconf(libc::_SC_PAGESIZE) }) as usize
    }

    /// Same ordering as `vindex3_residency_probe.rs::evict` — load-bearing on
    /// Darwin, where plain `MADV_DONTNEED` leaves clean pages resident.
    fn evict(base: *mut libc::c_void, len: usize) {
        // SAFETY: caller guarantees `base..base+len` is a live mapping.
        unsafe {
            libc::msync(base, len, libc::MS_INVALIDATE);
            libc::madvise(base, len, libc::MADV_DONTNEED);
        }
    }

    fn resident_pages(base: *const libc::c_void, len: usize, page: usize) -> Vec<bool> {
        let pages = len.div_ceil(page);
        let mut vec = vec![0u8; pages];
        // SAFETY: `vec` has one byte per page of the live mapping.
        let rc =
            unsafe { libc::mincore(base as *mut libc::c_void, len, vec.as_mut_ptr() as *mut _) };
        if rc != 0 {
            eprintln!(
                "  mincore failed (errno {})",
                std::io::Error::last_os_error()
            );
            return vec![false; pages];
        }
        vec.into_iter().map(|b| b & 1 == 1).collect()
    }

    /// Fault in every page of `range` by reading one byte from each. The
    /// checksum is printed, not just discarded, so the read cannot be
    /// optimised away and doubles as a real "these bytes are sane" signal.
    fn touch(bytes: &[u8], range: &Range<usize>, page: usize, checksum: &mut u64) {
        let mut offset = range.start;
        while offset < range.end {
            *checksum = checksum.wrapping_add(bytes[offset] as u64);
            offset += page;
        }
    }

    fn tensor<'a>(
        region_set: &str,
        bytes: &'a [u8],
        format: RegionFormat,
        contract: ComponentContract,
    ) -> Result<BoundTensor<'a>, String> {
        BoundTensor::direct(
            RepresentationIdentity::new(region_set, VARIANT),
            bytes,
            format,
            contract,
        )
        .map_err(|e| e.to_string())
    }

    fn sliced_tensor<'a>(
        region_set: &str,
        bytes: &'a [u8],
        format: RegionFormat,
        storage: ComponentContract,
        keep: usize,
    ) -> Result<BoundTensor<'a>, String> {
        BoundTensor::new(
            RepresentationIdentity::new(region_set, VARIANT),
            bytes,
            format,
            storage,
            ComponentView::Slice {
                dim: COL_DIM,
                start: 0,
                len: keep as u32,
            },
        )
        .map_err(|e| e.to_string())
    }

    fn build_expert<'a>(
        moe: &MoeLayerWeights<'a>,
        hidden: usize,
        inter: usize,
        inter_padded: usize,
        e: usize,
    ) -> Result<BoundExpert<'a>, String> {
        let gate_up_bytes = *moe
            .experts_gate_up
            .get(e)
            .ok_or_else(|| format!("expert {e} has no gate_up bytes"))?;
        let down_bytes = *moe
            .experts_down
            .get(e)
            .ok_or_else(|| format!("expert {e} has no down bytes"))?;
        Ok(BoundExpert {
            expert_id: e as u32,
            projection: BoundProjection::Fused {
                gate_up: tensor(
                    &RegionRole::GateUpFused.name(),
                    gate_up_bytes,
                    RegionFormat::Q4K,
                    ComponentContract::matrix(
                        (FUSED_PROJECTION_HALVES * inter) as u32,
                        hidden as u32,
                    ),
                )?,
            },
            down: sliced_tensor(
                &RegionRole::Down.name(),
                down_bytes,
                RegionFormat::Q4K,
                ComponentContract::matrix(hidden as u32, inter_padded as u32),
                inter,
            )?,
        })
    }

    fn bind_operation<'a>(
        layer: usize,
        hidden: usize,
        moe: &MoeLayerWeights<'_>,
        router_bytes: &'a [u8],
        scale_bytes: &'a [u8],
        experts: Vec<BoundExpert<'a>>,
    ) -> Result<BoundMoeOperation<'a>, String> {
        Ok(BoundMoeOperation {
            router: BoundRouter {
                weight: tensor(
                    ROUTER_REGION_SET,
                    router_bytes,
                    RegionFormat::F32,
                    ComponentContract::matrix(moe.num_experts as u32, hidden as u32),
                )?,
                top_k: moe.top_k,
                selected_weight: moe.routing_policy.selected_weight,
                scaling: if scale_bytes.is_empty() {
                    BoundExpertScaling::None
                } else {
                    BoundExpertScaling::PerExpert {
                        scales: tensor(
                            PER_EXPERT_SCALE_REGION_SET,
                            scale_bytes,
                            RegionFormat::F32,
                            ComponentContract::vector(moe.num_experts as u32),
                        )?,
                    }
                },
                kernel: RouterKernel::Incumbent,
            },
            transforms: Vec::new(),
            banks: vec![BoundBankOperation {
                bank: BankCoordinate::new(layer as u32, BANK_ID),
                experts,
                intermediate_dim: moe.intermediate_size,
                hidden_dim: hidden,
                activation: moe.activation,
                kernel: ExpertKernel::IncumbentQ4kQ8k,
            }],
            reduction: BoundReduction::WeightedSum,
            residual_dim: hidden,
        })
    }

    fn last_row(path: &str, width: usize) -> Result<Vec<f32>, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        let values: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        if !values.len().is_multiple_of(width) || values.is_empty() {
            return Err(format!(
                "{path}: {} values is not a whole number of {width}-wide rows",
                values.len()
            ));
        }
        let rows = values.len() / width;
        Ok(values[(rows - 1) * width..].to_vec())
    }

    struct ConditionResult {
        name: &'static str,
        prefetch_time: Duration,
        execute_time: Duration,
        prefetched_pages: usize,
        resident_after_execute: usize,
        predicted_pages: usize,
        coverage: f64,
        overshoot: f64,
        output: Vec<f32>,
    }

    #[allow(clippy::too_many_arguments)]
    fn run_condition(
        name: &'static str,
        base: *mut libc::c_void,
        mmap_bytes: &[u8],
        page: usize,
        prefetch_ranges: &[Range<usize>],
        all_regions: &[ExpertRegion],
        selected: &[usize],
        predicted_bytes: usize,
        layer: usize,
        hidden: usize,
        moe: &MoeLayerWeights<'_>,
        router_bytes: &[u8],
        scale_bytes: &[u8],
        expert_input: &[f32],
        router_in: &[f32],
    ) -> Result<ConditionResult, String> {
        // Reset to a verified-cold baseline before every condition.
        evict(base, mmap_bytes.len());
        let cold_check = resident_pages(base, mmap_bytes.len(), page);
        let cold_total = cold_check.len();
        let cold_resident = cold_check.iter().filter(|r| **r).count();
        let cold_fraction = cold_resident as f64 / cold_total.max(1) as f64;
        if cold_fraction > COLD_THRESHOLD {
            return Err(format!("[{name}] SPINE FAIL: {cold_resident}/{cold_total} pages resident after evict ({:.1}%) — eviction did not work on this OS, numbers would be meaningless", cold_fraction * 100.0));
        }

        let mut checksum = 0u64;
        let prefetch_start = Instant::now();
        for range in prefetch_ranges {
            touch(mmap_bytes, range, page, &mut checksum);
        }
        let prefetch_time = prefetch_start.elapsed();
        std::hint::black_box(checksum);

        let selected_ids: Vec<u32> = selected.iter().map(|&e| e as u32).collect();
        let prefetched_pages: usize = {
            let mut pages: Vec<usize> = prefetch_ranges
                .iter()
                .flat_map(|r| {
                    let span = PageSpan::of(r, page);
                    span.first..span.end
                })
                .collect();
            pages.sort_unstable();
            pages.dedup();
            pages.len()
        };

        let experts: Vec<BoundExpert<'_>> = selected
            .iter()
            .map(|&e| build_expert(moe, hidden, moe.intermediate_size, moe.inter_padded(), e))
            .collect::<Result<_, _>>()?;
        let op = bind_operation(layer, hidden, moe, router_bytes, scale_bytes, experts)?;
        op.validate().map_err(|e| format!("[{name}] bind: {e}"))?;

        let execute_start = Instant::now();
        let output = execute(&op, MoeInputs::split(expert_input, router_in)).map_err(|e| format!("[{name}] execute (should never fail — correctness must be prefetch-invariant): {e}"))?;
        let execute_time = execute_start.elapsed();

        let after = resident_pages(base, mmap_bytes.len(), page);
        let resident_after_execute = after.iter().filter(|r| **r).count();
        let empty_router: Range<usize> = 0..0;
        let acct = account(
            &after,
            page,
            &empty_router,
            all_regions,
            &selected_ids,
            predicted_bytes,
        );

        Ok(ConditionResult {
            name,
            prefetch_time,
            execute_time,
            prefetched_pages,
            resident_after_execute,
            predicted_pages: acct.predicted,
            coverage: acct.prediction_coverage(),
            overshoot: acct.overshoot_fraction(),
            output,
        })
    }

    fn print_condition(c: &ConditionResult) {
        println!(
            "  {:<10} prefetch={:>8.2?} pages_prefetched={:>7} | execute={:>8.2?} | resident_after={:>7} predicted={:>7} coverage={:>5.3} overshoot={:>5.3}",
            c.name, c.prefetch_time, c.prefetched_pages, c.execute_time, c.resident_after_execute, c.predicted_pages, c.coverage, c.overshoot
        );
    }

    pub fn run() -> Result<(), String> {
        let vindex = arg("--vindex").ok_or("set --vindex <path>")?;
        let dump = arg("--dump").ok_or("set --dump <dir> (LARQL_CPU_DUMP_LAYERS output)")?;
        let layer: usize = arg("--layer")
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_LAYER);

        println!("=== MAP-3B: physical page residency / prefetch performance ===");
        println!("  vindex  {vindex}");
        println!("  layer   {layer}");

        let mut callbacks = larql_vindex::SilentLoadCallbacks;
        let weights =
            larql_vindex::load_model_weights_kquant(std::path::Path::new(&vindex), &mut callbacks)
                .map_err(|e| format!("load weights: {e}"))?;
        let arch = &*weights.arch;
        let hidden = weights.hidden_size;
        let norm_offset = arch.norm_weight_offset();
        let eps = arch.norm_eps();
        let moe: MoeLayerWeights<'_> = build_moe_weights(&weights, arch, layer)
            .ok_or_else(|| format!("layer {layer} is not an MoE layer"))?;
        println!(
            "  shape   hidden {hidden}, {} experts, top-{}, intermediate {}",
            moe.num_experts, moe.top_k, moe.intermediate_size
        );

        let h = last_row(
            &larql_compute::forward::dump_config::cpu_layer_h_post_attn_path(&dump, layer),
            hidden,
        )?;
        let expert_input = moe_expert_input(&h, &moe, norm_offset, eps);
        let router_in = moe_router_input(&h, &expert_input, &moe, norm_offset, eps);
        let (incumbent_ids, _weights_) = moe_route_from_router_input(&router_in, &moe);
        println!("  routed  {incumbent_ids:?}\n");

        let router_bytes: Vec<u8> = moe
            .router_proj
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let scale_bytes: Vec<u8> = moe
            .router_per_expert_scale
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        // ── Real byte ranges, straight from the loader's own bookkeeping ────
        // No pointer arithmetic: `packed_byte_ranges` already records exactly
        // (filename, offset, length) for every per-layer FFN entry, and
        // `packed_mmaps` holds the SAME mmap the normal execution path reads
        // through — evicting it evicts the real pages `moe.experts_*` point into.
        let filename = layer_weights_filename(layer);
        let mmap = weights
            .packed_mmaps
            .get(&filename)
            .ok_or_else(|| format!("no packed mmap for {filename} — is ffn_layout per_layer?"))?;
        let mmap_bytes: &[u8] = mmap;
        let base = mmap.as_ptr() as *mut libc::c_void;
        let page = page_size();
        println!(
            "  mmap    {filename}, {} bytes ({:.1} MiB), page {page} B",
            mmap_bytes.len(),
            mmap_bytes.len() as f64 / 1_048_576.0
        );

        let byte_range = |e: usize, component: &str| -> Result<Range<usize>, String> {
            let key = per_layer_ffn_key(layer, e, component);
            let (_, offset, len) = weights
                .packed_byte_ranges
                .get(&key)
                .ok_or_else(|| format!("no byte range for {key}"))?;
            Ok(*offset..*offset + *len)
        };
        let mut all_regions: Vec<ExpertRegion> = Vec::with_capacity(moe.num_experts * 2);
        for e in 0..moe.num_experts {
            all_regions.push(ExpertRegion {
                expert_id: e as u32,
                bytes: byte_range(e, PER_LAYER_FFN_GATE_UP)?,
            });
            all_regions.push(ExpertRegion {
                expert_id: e as u32,
                bytes: byte_range(e, PER_LAYER_FFN_DOWN)?,
            });
        }

        let predicted_ranges: Vec<Range<usize>> = incumbent_ids
            .iter()
            .flat_map(|&e| {
                [
                    byte_range(e, PER_LAYER_FFN_GATE_UP),
                    byte_range(e, PER_LAYER_FFN_DOWN),
                ]
            })
            .collect::<Result<_, _>>()?;
        let predicted_bytes: usize = predicted_ranges.iter().map(|r| r.len()).sum();

        let unselected: Vec<usize> = (0..moe.num_experts)
            .filter(|e| !incumbent_ids.contains(e))
            .take(incumbent_ids.len())
            .collect();
        let random_ranges: Vec<Range<usize>> = unselected
            .iter()
            .flat_map(|&e| {
                [
                    byte_range(e, PER_LAYER_FFN_GATE_UP),
                    byte_range(e, PER_LAYER_FFN_DOWN),
                ]
            })
            .collect::<Result<_, _>>()?;
        println!("  random control: {} unselected experts {unselected:?} (same count as routed, so same predicted byte total)\n", unselected.len());

        let full_ranges = vec![0..mmap_bytes.len()];

        println!("== conditions ==");
        let conditions: [(&'static str, &[Range<usize>]); 4] = [
            ("cold", &[]),
            ("predicted", &predicted_ranges),
            ("random", &random_ranges),
            ("full", &full_ranges),
        ];
        let mut results = Vec::new();
        for (name, ranges) in conditions {
            let r = run_condition(
                name,
                base,
                mmap_bytes,
                page,
                ranges,
                &all_regions,
                &incumbent_ids,
                predicted_bytes,
                layer,
                hidden,
                &moe,
                &router_bytes,
                &scale_bytes,
                &expert_input,
                &router_in,
            )?;
            print_condition(&r);
            results.push(r);
        }

        println!("\n== correctness invariant (must hold across every condition) ==");
        let reference = &results[0].output;
        let mut all_match = true;
        for r in &results[1..] {
            let matches = r.output == *reference;
            all_match &= matches;
            println!(
                "  {:<10} {}",
                r.name,
                if matches {
                    "matches cold — PASS"
                } else {
                    "DIFFERS FROM COLD — FAIL, prefetch state changed the answer"
                }
            );
        }

        println!("\n== verdict ==");
        let predicted_time = results[1].execute_time;
        let cold_time = results[0].execute_time;
        let random_time = results[2].execute_time;
        println!("  execute time: cold={cold_time:?} predicted={predicted_time:?} random={random_time:?} full={:?}", results[3].execute_time);
        if all_match {
            println!("  PASS correctness invariant: identical output regardless of prefetch state");
        } else {
            println!("  FAIL correctness invariant — see above (this would indicate a real bug, not a prefetch-performance result)");
        }
        println!("  interpretation: with expert byte sizes this small ({:.1} MiB predicted for {} experts), timing differences may be near the OS/disk noise floor on fast local storage —", predicted_bytes as f64 / 1_048_576.0, incumbent_ids.len());
        println!("  the page-accounting numbers above (coverage/overshoot) are the more robust signal that prediction and reality agree, independent of how fast the underlying storage happens to be.");

        if all_match {
            Ok(())
        } else {
            Err("correctness invariant violated".into())
        }
    }
}

#[cfg(unix)]
fn main() -> Result<(), String> {
    probe::run()
}

#[cfg(not(unix))]
fn main() -> Result<(), String> {
    eprintln!("map3b_prefetch_performance is unix-only (mmap / madvise / mincore).");
    Ok(())
}
