//! WS /v1/stream — WebSocket streaming for layer-by-layer DESCRIBE.
//!
//! Client sends JSON messages, server streams results back layer by layer.
//!
//! Protocol:
//!   → {"type": "describe", "entity": "France"}
//!   ← {"type": "layer", "layer": 14, "edges": [...]}
//!   ← {"type": "layer", "layer": 15, "edges": [...]}
//!   ← {"type": "done", "total_edges": 6, "latency_ms": 12.3}

use std::sync::Arc;

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::Response;

use crate::state::AppState;

pub async fn handle_stream(
    State(state): State<Arc<AppState>>,
    ws: WebSocketUpgrade,
) -> Response {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(mut socket: WebSocket, state: Arc<AppState>) {
    while let Some(Ok(msg)) = socket.recv().await {
        let text = match msg {
            Message::Text(t) => t,
            Message::Close(_) => break,
            _ => continue,
        };

        let request: serde_json::Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(e) => {
                let _ = socket
                    .send(Message::Text(
                        serde_json::json!({"type": "error", "message": e.to_string()}).to_string().into(),
                    ))
                    .await;
                continue;
            }
        };

        let msg_type = request["type"].as_str().unwrap_or("");
        match msg_type {
            "describe" => {
                handle_stream_describe(&mut socket, &state, &request).await;
            }
            _ => {
                let _ = socket
                    .send(Message::Text(
                        serde_json::json!({
                            "type": "error",
                            "message": format!("unknown message type: {msg_type}. Supported: describe")
                        })
                        .to_string().into(),
                    ))
                    .await;
            }
        }
    }
}

async fn handle_stream_describe(
    socket: &mut WebSocket,
    state: &Arc<AppState>,
    request: &serde_json::Value,
) {
    let entity = match request["entity"].as_str() {
        Some(e) => e.to_string(),
        None => {
            let _ = socket
                .send(Message::Text(
                    serde_json::json!({"type": "error", "message": "missing entity"}).to_string().into(),
                ))
                .await;
            return;
        }
    };

    let model = match state.model(None) {
        Some(m) => Arc::clone(m),
        None => {
            let _ = socket
                .send(Message::Text(
                    serde_json::json!({"type": "error", "message": "no model loaded"}).to_string().into(),
                ))
                .await;
            return;
        }
    };

    let band = request["band"].as_str().unwrap_or("all");

    // Run the describe in a blocking task and stream results layer by layer.
    let start = std::time::Instant::now();

    let encoding = match model.tokenizer.encode(entity.as_str(), false) {
        Ok(e) => e,
        Err(e) => {
            let _ = socket
                .send(Message::Text(
                    serde_json::json!({"type": "error", "message": e.to_string()}).to_string().into(),
                ))
                .await;
            return;
        }
    };
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    if token_ids.is_empty() {
        let _ = socket
            .send(Message::Text(
                serde_json::json!({"type": "done", "total_edges": 0, "latency_ms": 0}).to_string().into(),
            ))
            .await;
        return;
    }

    let hidden = model.embeddings.shape()[1];
    let query = if token_ids.len() == 1 {
        model.embeddings.row(token_ids[0] as usize).mapv(|v| v * model.embed_scale)
    } else {
        let mut avg = larql_vindex::ndarray::Array1::<f32>::zeros(hidden);
        for &tok in &token_ids {
            avg += &model.embeddings.row(tok as usize).mapv(|v| v * model.embed_scale);
        }
        avg /= token_ids.len() as f32;
        avg
    };

    let config = &model.config;
    let last = config.num_layers.saturating_sub(1);
    let bands = config
        .layer_bands
        .clone()
        .or_else(|| larql_vindex::LayerBands::for_family(&config.family, config.num_layers))
        .unwrap_or(larql_vindex::LayerBands {
            syntax: (0, last),
            knowledge: (0, last),
            output: (0, last),
        });

    let patched = model.patched.read().await;
    let all_layers = patched.loaded_layers();

    let scan_layers: Vec<usize> = match band {
        "syntax" => all_layers.iter().copied()
            .filter(|l| *l >= bands.syntax.0 && *l <= bands.syntax.1)
            .collect(),
        "knowledge" => all_layers.iter().copied()
            .filter(|l| *l >= bands.knowledge.0 && *l <= bands.knowledge.1)
            .collect(),
        "output" => all_layers.iter().copied()
            .filter(|l| *l >= bands.output.0 && *l <= bands.output.1)
            .collect(),
        _ => all_layers,
    };

    let entity_lower = entity.to_lowercase();
    let mut total_edges = 0;

    // Stream layer by layer.
    for &layer in &scan_layers {
        let hits = patched.gate_knn(layer, &query, 20);
        let mut edges = Vec::new();

        for (feature, gate_score) in &hits {
            if *gate_score < 5.0 {
                continue;
            }
            if let Some(meta) = patched.feature_meta(layer, *feature) {
                let tok = meta.top_token.trim();
                if tok.is_empty() || tok.len() < 2 || tok.to_lowercase() == entity_lower {
                    continue;
                }
                let mut edge = serde_json::json!({
                    "target": tok,
                    "gate_score": (*gate_score * 10.0).round() / 10.0,
                    "feature": feature,
                });
                if let Some(label) = model.probe_labels.get(&(layer, *feature)) {
                    edge["relation"] = serde_json::json!(label);
                    edge["source"] = serde_json::json!("probe");
                }
                edges.push(edge);
            }
        }

        total_edges += edges.len();

        let msg = serde_json::json!({
            "type": "layer",
            "layer": layer,
            "edges": edges,
        });

        if socket.send(Message::Text(msg.to_string().into())).await.is_err() {
            return; // Client disconnected.
        }
    }

    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
    let done_msg = serde_json::json!({
        "type": "done",
        "entity": entity,
        "total_edges": total_edges,
        "latency_ms": (latency_ms * 10.0).round() / 10.0,
    });
    let _ = socket.send(Message::Text(done_msg.to_string().into())).await;
}
