//! I/O-bound replay driver: sweeps batch × wire × dispatch against a live
//! expert server using frames built from a capture pool. Coverage-excluded
//! (needs a running server); every pure decision lives in `replay.rs`.
//!
//! Dispatch parity: `streaming` = sequential per-layer POSTs (step time =
//! Σ layer RTTs, mirroring `generate_with_remote_ffn`'s per-layer round
//! trips); `batch` = all layers fired in parallel via `std::thread::scope`
//! (mirroring `LayerShardedBackend::forward_predispatch_all`). The
//! multi-layer wire frame is deliberately NOT used — the production client
//! never sends per-layer-distinct residuals through it.

use super::args::ReplayArgs;
use super::capture_format::CapturePool;
use super::output::{CaptureSummary, DecBenchJsonResult};
use super::pulse;
use super::replay::{
    build_q8k_frame, build_walk_ffn_frame, expand_sweep, movement_ratio, parse_batch_list,
    parse_layer_range, summarize, DispatchMode, RequestSample, SweepPoint, SweepPointStats,
    WireArm, weight_bytes_per_token,
};

use larql_inference::ffn::remote::{STATS_PATH, WALK_FFN_PATH, WALK_FFN_Q8K_PATH};

pub(super) fn run_replay(args: &ReplayArgs) -> Result<(), Box<dyn std::error::Error>> {
    let pool = CapturePool::open(&args.capture)?;

    let batches = parse_batch_list(&args.batch)?;
    if let Some(&max_b) = batches.iter().max() {
        if max_b > pool.num_prompts() {
            return Err(format!(
                "max batch {max_b} exceeds pool prompt count {} — capture more prompts",
                pool.num_prompts()
            )
            .into());
        }
    }
    let wires = WireArm::parse_list(&args.wire)?;
    let dispatches = DispatchMode::parse_list(&args.dispatch)?;
    let steps = match args.steps {
        Some(s) => {
            if s == 0 || s > pool.manifest.steps {
                return Err(format!(
                    "--steps {s} out of range (pool has {} steps)",
                    pool.manifest.steps
                )
                .into());
            }
            s
        }
        None => pool.manifest.steps,
    };

    let base_url = args.ffn.trim_end_matches('/').to_string();
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(args.timeout_secs))
        .build()?;

    let stats_before = fetch_stats(&client, &base_url)?;
    let server_layers = stats_before
        .get("layers")
        .and_then(|v| v.as_u64())
        .unwrap_or(pool.manifest.num_layers as u64) as usize;
    if server_layers != pool.manifest.num_layers {
        eprintln!(
            "[dec-bench] WARNING: server reports {server_layers} layers, pool captured {} — \
             replaying the pool's layer space",
            pool.manifest.num_layers
        );
    }
    let layers = parse_layer_range(args.layers.as_deref(), pool.manifest.num_layers)?;

    let (weight_bytes_tok, weight_missing) =
        match weight_bytes_per_token(&stats_before, &layers) {
            Some((bytes, missing)) => (Some(bytes), missing),
            None => {
                eprintln!(
                    "[dec-bench] server /v1/stats has no ffn_weights block — \
                     dec/movement_ratio will be omitted"
                );
                (None, 0)
            }
        };

    let sweep = expand_sweep(&batches, &wires, &dispatches);
    eprintln!(
        "[dec-bench] replaying {} points: batch {:?} × wire {:?} × dispatch {:?}, \
         {} layers × {} steps × {} repeats",
        sweep.len(),
        batches,
        wires.iter().map(|w| w.label()).collect::<Vec<_>>(),
        dispatches.iter().map(|d| d.label()).collect::<Vec<_>>(),
        layers.len(),
        steps,
        args.repeats,
    );

    let mut points = Vec::with_capacity(sweep.len());
    let mut pulse_lines = Vec::with_capacity(sweep.len());
    for (idx, point) in sweep.iter().enumerate() {
        let stats = run_point(&client, &base_url, &pool, point, &layers, steps, args)?;
        let summary = summarize(point, &stats);
        if summary.served_wire != vec![summary.wire_format.clone()] {
            eprintln!(
                "[dec-bench] NOTE: {} arm was served as {:?} (Accept fallback) — \
                 bandwidth numbers belong to the served format",
                summary.wire_format, summary.served_wire
            );
        }
        let ratio = weight_bytes_tok
            .and_then(|w| movement_ratio(summary.payload_bytes_tok, w));
        eprintln!(
            "[dec-bench] point {}/{}: batch={} wire={} dispatch={} → \
             step p50 {:.2} ms, p99 {:.2} ms, {:.1} tok/s, {:.0} B/tok{}",
            idx + 1,
            sweep.len(),
            summary.batch,
            summary.wire_format,
            summary.dispatch_mode,
            summary.step_ms_p50,
            summary.step_ms_p99,
            summary.tok_s,
            summary.payload_bytes_tok,
            ratio
                .map(|r| format!(", movement {r:.2e}"))
                .unwrap_or_default(),
        );
        pulse_lines.push(pulse::pulse_line(
            idx,
            &summary,
            weight_bytes_tok,
            ratio,
            args.net_rtt_ms,
            args.net_gbps,
            args.pulse_per_layer,
        ));
        points.push(summary);
    }

    let stats_after = fetch_stats(&client, &base_url)?;

    if let Some(path) = &args.pulse_file {
        std::fs::write(path, pulse::to_jsonl(&pulse_lines))
            .map_err(|e| format!("write pulse file {}: {e}", path.display()))?;
        eprintln!("[dec-bench] pulse → {}", path.display());
    }

    let record = DecBenchJsonResult {
        timestamp: {
            let secs = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            format!("{secs}")
        },
        endpoint: "walk-ffn".into(),
        ffn_url: base_url.clone(),
        capture: CaptureSummary::from(&pool.manifest),
        stats_before,
        stats_after,
        layers,
        steps,
        repeats: args.repeats,
        warmup_passes: args.warmup_passes,
        weight_bytes_tok,
        weight_bytes_missing_layers: weight_missing,
        net_rtt_ms: args.net_rtt_ms,
        net_gbps: args.net_gbps,
        points,
    };
    let json = serde_json::to_string_pretty(&record)?;
    match &args.output_file {
        Some(path) => {
            std::fs::write(path, &json)
                .map_err(|e| format!("write output file {}: {e}", path.display()))?;
            eprintln!("[dec-bench] run record → {}", path.display());
        }
        None => println!("{json}"),
    }
    Ok(())
}

fn fetch_stats(
    client: &reqwest::blocking::Client,
    base_url: &str,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let resp = client.get(format!("{base_url}{STATS_PATH}")).send()?;
    if !resp.status().is_success() {
        return Err(format!("/v1/stats returned {}", resp.status()).into());
    }
    Ok(resp.json()?)
}

/// Run one sweep point: warmup, then `repeats × steps` full passes over
/// `layers`.
fn run_point(
    client: &reqwest::blocking::Client,
    base_url: &str,
    pool: &CapturePool,
    point: &SweepPoint,
    layers: &[usize],
    steps: usize,
    args: &ReplayArgs,
) -> Result<SweepPointStats, Box<dyn std::error::Error>> {
    // Warmup: full passes over ALL replayed layers, discarded. Per-layer
    // warming matters — the server's FFN weights are mmap-backed, and a
    // cold layer's first touch pays page-fault cost that would otherwise
    // land in the first measured step's p99 (observed: 4s p99 on an
    // otherwise ~120ms point).
    for _ in 0..args.warmup_passes {
        for &layer in layers {
            let rows = pool.rows(point.batch, 0, layer)?;
            let _ = send_one(client, base_url, point, layer, &rows, pool)?;
        }
    }

    let mut stats = SweepPointStats::default();
    for _repeat in 0..args.repeats.max(1) {
        for step in 0..steps {
            match point.dispatch {
                DispatchMode::Streaming => {
                    let mut step_ms = 0.0f64;
                    for &layer in layers {
                        let rows = pool.rows(point.batch, step, layer)?;
                        let s = send_one(client, base_url, point, layer, &rows, pool)?;
                        step_ms += s.client_ms;
                        stats.samples.push(s);
                    }
                    stats.step_ms.push(step_ms);
                }
                DispatchMode::Batch => {
                    // Pre-build all frames outside the timed window so the
                    // fan-out wall time measures transport + server, not
                    // frame encoding (mirrors predispatch, which reuses
                    // already-materialised residuals).
                    let mut frames = Vec::with_capacity(layers.len());
                    for &layer in layers {
                        let rows = pool.rows(point.batch, step, layer)?;
                        frames.push((layer, rows));
                    }
                    let t0 = std::time::Instant::now();
                    let results: Vec<Result<RequestSample, String>> =
                        std::thread::scope(|scope| {
                            let handles: Vec<_> = frames
                                .iter()
                                .map(|(layer, rows)| {
                                    scope.spawn(move || {
                                        send_one(client, base_url, point, *layer, rows, pool)
                                            .map_err(|e| e.to_string())
                                    })
                                })
                                .collect();
                            handles
                                .into_iter()
                                .map(|h| {
                                    h.join().unwrap_or_else(|_| {
                                        Err("replay worker panicked".into())
                                    })
                                })
                                .collect()
                        });
                    let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
                    for r in results {
                        stats.samples.push(r.map_err(|e| -> Box<dyn std::error::Error> {
                            e.into()
                        })?);
                    }
                    stats.step_ms.push(wall_ms);
                }
            }
        }
    }
    Ok(stats)
}

/// Send one B-row request for one layer; validate the decoded shape through
/// the same codecs the production client uses.
fn send_one(
    client: &reqwest::blocking::Client,
    base_url: &str,
    point: &SweepPoint,
    layer: usize,
    rows: &[f32],
    pool: &CapturePool,
) -> Result<RequestSample, Box<dyn std::error::Error>> {
    let hidden = pool.manifest.hidden_size;
    let batch = point.batch;

    match point.wire {
        WireArm::Q8k => {
            let body = build_q8k_frame(layer, rows, batch, hidden);
            let bytes_sent = body.len() as u64;
            let t0 = std::time::Instant::now();
            let resp = client
                .post(format!("{base_url}{WALK_FFN_Q8K_PATH}"))
                .header(reqwest::header::CONTENT_TYPE, larql_inference::Q8K_BATCH_CT)
                .body(body)
                .send()?;
            let status = resp.status();
            let resp_body = resp.bytes()?;
            let client_ms = t0.elapsed().as_secs_f64() * 1000.0;
            if !status.is_success() {
                return Err(format!(
                    "q8k replay layer {layer} batch {batch}: server returned {status}"
                )
                .into());
            }
            let entries =
                larql_inference::decode_q8k_batch_response_entries(&resp_body)?;
            if entries.len() != batch
                || entries.iter().any(|(l, v)| *l != layer || v.len() != hidden)
            {
                return Err(format!(
                    "q8k replay layer {layer}: expected {batch} entries × hidden {hidden}, \
                     got {} entries",
                    entries.len()
                )
                .into());
            }
            Ok(RequestSample {
                layer,
                client_ms,
                server_ms: None,
                bytes_sent,
                bytes_recv: resp_body.len() as u64,
                served_wire: "q8k".into(),
            })
        }
        arm => {
            let body = build_walk_ffn_frame(layer, rows, batch);
            let bytes_sent = body.len() as u64;
            let accept = arm.accept().expect("non-q8k arm has an accept CT");
            let t0 = std::time::Instant::now();
            let resp = client
                .post(format!("{base_url}{WALK_FFN_PATH}"))
                .header(reqwest::header::CONTENT_TYPE, larql_inference::BINARY_CT)
                .header(reqwest::header::ACCEPT, accept)
                .body(body)
                .send()?;
            let status = resp.status();
            let resp_ct = resp
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .unwrap_or(larql_inference::BINARY_CT)
                .to_string();
            let resp_body = resp.bytes()?;
            let client_ms = t0.elapsed().as_secs_f64() * 1000.0;
            if !status.is_success() {
                return Err(format!(
                    "walk-ffn replay layer {layer} batch {batch}: server returned {status}"
                )
                .into());
            }
            let (resp_layer, server_ms, floats) =
                larql_inference::decode_single_response(&resp_ct, &resp_body, hidden)?;
            if resp_layer != layer || floats.len() != batch * hidden {
                return Err(format!(
                    "walk-ffn replay layer {layer}: expected {batch}×{hidden} floats for \
                     layer {layer}, got {} floats for layer {resp_layer}",
                    floats.len()
                )
                .into());
            }
            Ok(RequestSample {
                layer,
                client_ms,
                server_ms: Some(server_ms),
                bytes_sent,
                bytes_recv: resp_body.len() as u64,
                served_wire: super::replay::wire_label_for_content_type(&resp_ct),
            })
        }
    }
}
