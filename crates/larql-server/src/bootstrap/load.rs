//! Vindex/model loading — the V2/V3 artifact fork, the single-vindex
//! loader, ownership manifests, and `--dir` discovery.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use larql_vindex::format::filenames::*;
use larql_vindex::{
    load_vindex_config, load_vindex_embeddings, load_vindex_tokenizer, PatchedVindex,
    SilentLoadCallbacks, VectorIndex,
};
use tokio::sync::RwLock;
use tracing::{info, warn};

use crate::state::{load_probe_labels, model_id_from_name, LoadedModel};

use super::BoxError;

#[derive(Clone, Default)]
pub struct LoadVindexOptions {
    pub no_infer: bool,
    pub ffn_only: bool,
    pub embed_only: bool,
    pub layer_range: Option<(usize, usize)>,
    pub max_gate_cache_layers: usize,
    pub max_q4k_cache_layers: usize,
    pub hnsw: Option<usize>,
    pub warmup_hnsw: bool,
    pub release_mmap_after_request: bool,
    pub expert_filter: Option<(usize, usize)>,
    /// Fine-grained per-(layer, expert) ownership.  When `Some`, takes
    /// precedence over `expert_filter` for `run_expert`'s ownership check
    /// and for the HNSW / Metal warmup loops.  Loaded from `--units` JSON.
    pub unit_filter: Option<Arc<std::collections::HashSet<(usize, usize)>>>,
    /// Server-side remote MoE backend. When `Some`, the walk-ffn handler
    /// delegates MoE expert dispatch to remote shard servers.
    pub moe_remote: Option<Arc<larql_inference::ffn::RemoteMoeBackend>>,
}

/// JSON layout for the `--units` manifest.  Each value is a list of inclusive
/// `[start, end]` expert-id ranges, keyed by layer index (as a string for
/// JSON-object compatibility).
#[derive(serde::Deserialize)]
pub struct UnitManifest {
    pub layer_experts: std::collections::BTreeMap<String, Vec<[usize; 2]>>,
}

impl UnitManifest {
    /// Expand the per-layer range list into the flat `(layer, expert_id)`
    /// set used by ownership checks.  Reports the first malformed entry in
    /// the error path so the operator can fix it without grepping.
    pub fn into_unit_set(self) -> Result<std::collections::HashSet<(usize, usize)>, BoxError> {
        let mut units = std::collections::HashSet::new();
        for (layer_str, ranges) in self.layer_experts {
            let layer: usize = layer_str.parse().map_err(|_| -> BoxError {
                format!("--units: layer key '{layer_str}' is not a valid usize").into()
            })?;
            for [start, end] in ranges {
                if end < start {
                    return Err(format!(
                        "--units: layer {layer}: end ({end}) must be >= start ({start})"
                    )
                    .into());
                }
                for eid in start..=end {
                    units.insert((layer, eid));
                }
            }
        }
        Ok(units)
    }
}

/// Parse `--units PATH` into the canonical `(layer, expert_id)` ownership set.
pub fn parse_unit_manifest(
    path: &Path,
) -> Result<std::collections::HashSet<(usize, usize)>, BoxError> {
    let bytes = std::fs::read(path)
        .map_err(|e| -> BoxError { format!("--units: read {}: {e}", path.display()).into() })?;
    let manifest: UnitManifest = serde_json::from_slice(&bytes)
        .map_err(|e| -> BoxError { format!("--units: parse {}: {e}", path.display()).into() })?;
    manifest.into_unit_set()
}

/// One bound model artifact — which runtime family serves it. The
/// V2/V3 decision is made HERE, at binding time, from the container's
/// own generation marker; nothing downstream re-detects it.
pub enum LoadedArtifact {
    V2(Box<LoadedModel>),
    V3(Box<crate::vindex3::V3Model>),
}

/// Detect the artifact's container generation and bind it with the
/// matching loader. A VINDEX3 container binds as an executable
/// program ([`crate::vindex3::load_v3_model`]) — it structurally
/// cannot take the V2 path, whose `load_vindex_config` refuses
/// non-V2 generations.
pub fn load_artifact(path_str: &str, opts: LoadVindexOptions) -> Result<LoadedArtifact, BoxError> {
    let path = if larql_vindex::is_hf_path(path_str) {
        larql_vindex::resolve_hf_vindex(path_str)?
    } else {
        PathBuf::from(path_str)
    };
    match larql_vindex::format::generation::detect_generation(&path)? {
        larql_vindex::format::generation::ContainerGeneration::V3 => {
            info!("Loading VINDEX3 container: {}", path.display());
            Ok(LoadedArtifact::V3(Box::new(crate::vindex3::load_v3_model(
                &path,
            )?)))
        }
        larql_vindex::format::generation::ContainerGeneration::V2 => Ok(LoadedArtifact::V2(
            Box::new(load_single_vindex(path_str, opts)?),
        )),
    }
}

pub fn load_single_vindex(
    path_str: &str,
    opts: LoadVindexOptions,
) -> Result<LoadedModel, BoxError> {
    let path = if larql_vindex::is_hf_path(path_str) {
        info!("Resolving HuggingFace path: {}", path_str);
        larql_vindex::resolve_hf_vindex(path_str)?
    } else {
        PathBuf::from(path_str)
    };

    info!("Loading: {}", path.display());

    let config = load_vindex_config(&path)?;
    let model_name = config.model.clone();
    let id = model_id_from_name(&model_name);

    let mut cb = SilentLoadCallbacks;
    let mut index = VectorIndex::load_vindex_with_range(&path, &mut cb, opts.layer_range)?;
    if opts.max_gate_cache_layers > 0 {
        index.set_gate_cache_max_layers(opts.max_gate_cache_layers);
        info!(
            "  Gate cache: LRU, max {} layers",
            opts.max_gate_cache_layers
        );
    }
    if opts.max_q4k_cache_layers > 0 {
        index.set_kquant_ffn_cache_max_layers(opts.max_q4k_cache_layers);
        info!(
            "  Q4K FFN cache: LRU, max {} layers",
            opts.max_q4k_cache_layers
        );
    }
    if let Some(ef) = opts.hnsw {
        index.enable_hnsw(ef);
        info!("  HNSW gate KNN: enabled (ef_search={ef})");
        if opts.warmup_hnsw {
            let t0 = std::time::Instant::now();
            index.warmup_hnsw_all_layers();
            let owned = match opts.layer_range {
                Some((s, e)) => e - s,
                None => config.num_layers,
            };
            info!(
                "  HNSW warmup: built {} owned layer(s) in {:.2?}",
                owned,
                t0.elapsed()
            );
        }
    }
    let total_features: usize = config.layers.iter().map(|l| l.num_features).sum();

    let has_weights = config.has_model_weights
        || config.extract_level == larql_vindex::ExtractLevel::Inference
        || config.extract_level == larql_vindex::ExtractLevel::All;

    if let Some((start, end)) = opts.layer_range {
        info!("  Layers: {start}–{} (of {})", end - 1, config.num_layers);
    }
    info!(
        "  Model: {} ({} layers, {} features)",
        model_name, config.num_layers, total_features
    );

    if !opts.embed_only {
        match index.load_down_features(&path) {
            Ok(()) => info!("  Down features: loaded (mmap walk enabled)"),
            Err(_) => info!("  Down features: not available"),
        }
        if let Ok(()) = index.load_up_features(&path) {
            info!("  Up features: loaded (full mmap FFN)")
        }
        if index.has_down_features_kquant() {
            info!(
                "  Down features Q4K: loaded (W2 — per-feature decode skips kquant_ffn_layer cache)"
            );
        }

        // For inference-capable vindexes (`/v1/completions`,
        // `/v1/chat/completions`, `/v1/infer mode=walk`), load the
        // attention + interleaved-FFN slices the inference path needs.
        // Mirrors `larql_inference::open_inference_vindex` — without
        // these the Q4K decode panics with "attn Q4K slices missing".
        //
        // `--ffn-only` skips attention weights (no infer path) but MUST
        // still mmap interleaved_kquant so per-layer walk-ffn requests can
        // call `kquant_ffn_forward_layer`.
        let need_ffn_mmap = opts.ffn_only || (!opts.no_infer && has_weights);
        if !opts.no_infer && !opts.ffn_only && has_weights {
            if path.join(LM_HEAD_BIN).is_file() {
                let _ = index.load_lm_head(&path);
            }
            if has_kquant_lm_head(&path) {
                let _ = index.load_lm_head_kquant(&path);
            }
            if has_kquant_attn_weights(&path) {
                if let Err(e) = index.load_attn_kquant(&path) {
                    warn!("  Attn k-quant: failed to load ({e}) — generation may not work");
                } else {
                    info!("  Attn k-quant: loaded (inference path enabled)");
                }
            } else if path.join(ATTN_WEIGHTS_Q8_BIN).is_file() {
                if let Err(e) = index.load_attn_q8(&path) {
                    warn!("  Attn Q8: failed to load ({e}) — generation may not work");
                }
            }
        }
        if need_ffn_mmap {
            if has_kquant_interleaved(&path) {
                if let Err(e) = index.load_interleaved_kquant(&path) {
                    warn!("  Interleaved k-quant: failed to load ({e})");
                } else if opts.ffn_only {
                    info!("  Interleaved k-quant: loaded (ffn-service)");
                }
            } else if path.join(INTERLEAVED_Q4_BIN).is_file() {
                if let Err(e) = index.load_interleaved_q4(&path) {
                    warn!("  Interleaved Q4: failed to load ({e})");
                }
            }
        }
    }

    if opts.ffn_only || opts.embed_only {
        let reason = if opts.embed_only {
            "--embed-only"
        } else {
            "--ffn-only"
        };
        info!("  Warmup: skipped ({reason})");
    } else {
        index.warmup();
        info!("  Warmup: done");
    }

    let (embeddings, embed_scale) = load_vindex_embeddings(&path)?;
    info!(
        "  Embeddings: {}x{}",
        embeddings.shape()[0],
        embeddings.shape()[1]
    );

    let embed_store = if opts.embed_only {
        match crate::embed_store::EmbedStoreF16::open(
            &path,
            embed_scale,
            config.vocab_size,
            config.hidden_size,
            5_000,
        ) {
            Ok(store) => {
                let f16_bytes = config.vocab_size * config.hidden_size * 2;
                info!(
                    "  Embed store: f16 mmap ({:.1} GB, L1 cap 5000 tokens)",
                    f16_bytes as f64 / 1e9
                );
                Some(Arc::new(store))
            }
            Err(e) => {
                info!("  Embed store: f16 mmap unavailable ({e}), using f32 heap");
                None
            }
        }
    } else {
        None
    };

    let tokenizer = load_vindex_tokenizer(&path)?;
    let patched = PatchedVindex::new(index);

    let probe_labels = load_probe_labels(&path);
    if !probe_labels.is_empty() {
        info!("  Labels: {} probe-confirmed", probe_labels.len());
    }

    let infer_disabled = opts.no_infer || opts.ffn_only || opts.embed_only;
    if opts.embed_only {
        info!("  Mode: embed-service (--embed-only)");
        info!("  Infer: disabled (embed-service mode)");
    } else if opts.ffn_only {
        info!("  Mode: ffn-service (--ffn-only)");
        info!("  Infer: disabled (FFN-service mode)");
    } else if opts.no_infer {
        info!("  Infer: disabled (--no-infer)");
    } else if has_weights {
        info!("  Infer: available (weights detected, will lazy-load on first request)");
    } else {
        info!("  Infer: not available (no model weights in vindex)");
    }

    if opts.release_mmap_after_request {
        info!("  Mmap release: enabled (MADV_DONTNEED after each walk-ffn request)");
    }

    if let Some((start, end)) = opts.expert_filter {
        info!("  Experts: {start}–{end} (shard filter)");
        info!("  Endpoints: POST /v1/expert/batch, /v1/experts/layer-batch, GET /v1/stats");
    }

    let num_layers = config.num_layers;
    Ok(LoadedModel {
        id,
        path,
        config,
        patched: Arc::new(RwLock::new(patched)),
        embeddings,
        embed_scale,
        tokenizer,
        infer_disabled,
        ffn_only: opts.ffn_only,
        embed_only: opts.embed_only,
        embed_store,
        release_mmap_after_request: opts.release_mmap_after_request,
        weights: std::sync::OnceLock::new(),
        weights_init: std::sync::Mutex::new(()),
        probe_labels,
        ffn_l2_cache: crate::ffn_l2_cache::FfnL2Cache::new(num_layers),
        layer_latency_tracker: std::sync::Arc::new(crate::metrics::LayerLatencyTracker::new()),
        requests_in_flight: std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0)),
        requests_total: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
        expert_filter: opts.expert_filter,
        unit_filter: opts.unit_filter.clone(),
        moe_remote: opts.moe_remote.clone(),
        #[cfg(all(feature = "metal-experts", target_os = "macos"))]
        metal_backend: std::sync::OnceLock::new(),
        #[cfg(all(feature = "metal-experts", target_os = "macos"))]
        moe_scratches: std::sync::Mutex::new(std::collections::HashMap::new()),
        #[cfg(all(feature = "metal-experts", target_os = "macos"))]
        metal_ffn_layer_bufs: std::sync::OnceLock::new(),
    })
}

pub fn discover_vindexes(dir: &Path) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() && p.join(INDEX_JSON).exists() {
                paths.push(p);
            }
        }
    }
    paths.sort();
    paths
}
