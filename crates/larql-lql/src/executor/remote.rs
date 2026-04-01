/// Remote executor — forwards LQL queries to a larql-server via HTTP.

use crate::ast::*;
use crate::error::LqlError;
use super::Session;
use super::Backend;

impl Session {
    /// Connect to a remote larql-server.
    pub(crate) fn exec_use_remote(&mut self, url: &str) -> Result<Vec<String>, LqlError> {
        let url = url.trim_end_matches('/').to_string();

        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|e| LqlError::Execution(format!("failed to create HTTP client: {e}")))?;

        // Verify the server is reachable by hitting /v1/stats.
        let stats_url = format!("{url}/v1/stats");
        let resp = client
            .get(&stats_url)
            .send()
            .map_err(|e| LqlError::Execution(format!("failed to connect to {url}: {e}")))?;

        if !resp.status().is_success() {
            return Err(LqlError::Execution(format!(
                "server returned {}: {}",
                resp.status(),
                resp.text().unwrap_or_default()
            )));
        }

        let stats: serde_json::Value = resp
            .json()
            .map_err(|e| LqlError::Execution(format!("invalid response from server: {e}")))?;

        let model = stats["model"].as_str().unwrap_or("unknown");
        let layers = stats["layers"].as_u64().unwrap_or(0);
        let features = stats["features"].as_u64().unwrap_or(0);

        self.backend = Backend::Remote {
            url: url.clone(),
            client,
            local_patches: Vec::new(),
        };
        self.patch_recording = None;
        self.auto_patch = false;

        Ok(vec![format!(
            "Connected: {} ({} layers, {} features)\n  Remote: {}",
            model, layers, features, url,
        )])
    }

    /// Check if the backend is remote.
    pub(crate) fn is_remote(&self) -> bool {
        matches!(&self.backend, Backend::Remote { .. })
    }

    /// Get the remote URL and client, or error.
    fn require_remote(&self) -> Result<(&str, &reqwest::blocking::Client), LqlError> {
        match &self.backend {
            Backend::Remote { url, client, .. } => Ok((url, client)),
            _ => Err(LqlError::Execution("not connected to a remote server".into())),
        }
    }

    /// Number of local patches applied to the remote session.
    #[allow(dead_code)]
    fn local_patch_count(&self) -> usize {
        match &self.backend {
            Backend::Remote { local_patches, .. } => local_patches.len(),
            _ => 0,
        }
    }

    // ── Remote query forwarding ──

    pub(crate) fn remote_describe(
        &self,
        entity: &str,
        band: Option<LayerBand>,
        verbose: bool,
    ) -> Result<Vec<String>, LqlError> {
        let (url, client) = self.require_remote()?;

        let band_str = match band {
            Some(LayerBand::Syntax) => "syntax",
            Some(LayerBand::Knowledge) => "knowledge",
            Some(LayerBand::Output) => "output",
            Some(LayerBand::All) => "all",
            None => "knowledge",
        };

        let resp = client
            .get(format!("{url}/v1/describe"))
            .query(&[("entity", entity), ("band", band_str), ("verbose", if verbose { "true" } else { "false" })])
            .send()
            .map_err(|e| LqlError::Execution(format!("request failed: {e}")))?;

        let body: serde_json::Value = resp
            .json()
            .map_err(|e| LqlError::Execution(format!("invalid response: {e}")))?;

        let mut out = vec![entity.to_string()];

        if let Some(edges) = body["edges"].as_array() {
            if edges.is_empty() {
                out.push("  (no edges found)".into());
            } else {
                for edge in edges {
                    let target = edge["target"].as_str().unwrap_or("?");
                    let gate = edge["gate_score"].as_f64().unwrap_or(0.0);
                    let layer = edge["layer"].as_u64().unwrap_or(0);
                    let relation = edge["relation"].as_str().unwrap_or("");
                    let source = edge["source"].as_str().unwrap_or("");

                    let label = if !relation.is_empty() {
                        format!("{:<12}", relation)
                    } else {
                        format!("{:<12}", "")
                    };

                    let tag = if source == "probe" { "  (probe)" } else { "" };

                    out.push(format!(
                        "    {} → {:20} {:>7.1}  L{:<3}{}",
                        label, target, gate, layer, tag,
                    ));
                }
            }
        }

        if let Some(ms) = body["latency_ms"].as_f64() {
            out.push(format!("\n{:.1}ms (remote)", ms));
        }

        // Overlay local patch edges.
        if let Backend::Remote { local_patches, .. } = &self.backend {
            if !local_patches.is_empty() {
                let entity_lower = entity.to_lowercase();
                let mut local_edges = Vec::new();
                for patch in local_patches {
                    for op in &patch.operations {
                        if let larql_vindex::PatchOp::Insert {
                            entity: ent,
                            target,
                            relation,
                            layer,
                            confidence,
                            ..
                        } = op
                        {
                            if ent.to_lowercase() == entity_lower {
                                local_edges.push((
                                    relation.as_deref().unwrap_or(""),
                                    target.as_str(),
                                    *layer,
                                    confidence.unwrap_or(0.9),
                                ));
                            }
                        }
                    }
                }
                if !local_edges.is_empty() {
                    out.push("  Local patch edges:".into());
                    for (relation, target, layer, conf) in &local_edges {
                        let label = if relation.is_empty() {
                            format!("{:<12}", "")
                        } else {
                            format!("{:<12}", relation)
                        };
                        out.push(format!(
                            "    {} → {:20} {:>7.2}  L{:<3}  (local)",
                            label, target, conf, layer,
                        ));
                    }
                }
            }
        }

        Ok(out)
    }

    pub(crate) fn remote_walk(
        &self,
        prompt: &str,
        top: Option<u32>,
        layers: Option<&Range>,
    ) -> Result<Vec<String>, LqlError> {
        let (url, client) = self.require_remote()?;

        let top_k = top.unwrap_or(10);
        let mut params = vec![
            ("prompt".to_string(), prompt.to_string()),
            ("top".to_string(), top_k.to_string()),
        ];
        if let Some(r) = layers {
            params.push(("layers".to_string(), format!("{}-{}", r.start, r.end)));
        }

        let resp = client
            .get(format!("{url}/v1/walk"))
            .query(&params)
            .send()
            .map_err(|e| LqlError::Execution(format!("request failed: {e}")))?;

        let body: serde_json::Value = resp
            .json()
            .map_err(|e| LqlError::Execution(format!("invalid response: {e}")))?;

        let mut out = Vec::new();
        out.push(format!("Feature scan for {:?}", prompt));
        out.push(String::new());

        if let Some(hits) = body["hits"].as_array() {
            for hit in hits {
                let layer = hit["layer"].as_u64().unwrap_or(0);
                let feature = hit["feature"].as_u64().unwrap_or(0);
                let gate = hit["gate_score"].as_f64().unwrap_or(0.0);
                let target = hit["target"].as_str().unwrap_or("?");

                out.push(format!(
                    "  L{:2}: F{:<5} gate={:+.1}  top={:?}",
                    layer, feature, gate, target,
                ));
            }
        }

        if let Some(ms) = body["latency_ms"].as_f64() {
            out.push(format!("\n{:.1}ms (remote)", ms));
        }

        Ok(out)
    }

    pub(crate) fn remote_infer(
        &self,
        prompt: &str,
        top: Option<u32>,
        compare: bool,
    ) -> Result<Vec<String>, LqlError> {
        let (url, client) = self.require_remote()?;

        let mode = if compare { "compare" } else { "walk" };
        let body = serde_json::json!({
            "prompt": prompt,
            "top": top.unwrap_or(5),
            "mode": mode,
        });

        let resp = client
            .post(format!("{url}/v1/infer"))
            .json(&body)
            .send()
            .map_err(|e| LqlError::Execution(format!("request failed: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().unwrap_or_default();
            return Err(LqlError::Execution(format!("infer failed ({}): {}", status, text)));
        }

        let result: serde_json::Value = resp
            .json()
            .map_err(|e| LqlError::Execution(format!("invalid response: {e}")))?;

        let mut out = Vec::new();

        if compare {
            for mode in &["walk", "dense"] {
                if let Some(preds) = result[mode].as_array() {
                    out.push(format!("Predictions ({mode}):"));
                    for (i, p) in preds.iter().enumerate() {
                        let tok = p["token"].as_str().unwrap_or("?");
                        let prob = p["probability"].as_f64().unwrap_or(0.0);
                        out.push(format!("  {:2}. {:20} ({:.2}%)", i + 1, tok, prob * 100.0));
                    }
                    if let Some(ms) = result[format!("{mode}_ms")].as_f64() {
                        out.push(format!("  {:.0}ms", ms));
                    }
                    out.push(String::new());
                }
            }
        } else if let Some(preds) = result["predictions"].as_array() {
            out.push("Predictions (walk FFN):".into());
            for (i, p) in preds.iter().enumerate() {
                let tok = p["token"].as_str().unwrap_or("?");
                let prob = p["probability"].as_f64().unwrap_or(0.0);
                out.push(format!("  {:2}. {:20} ({:.2}%)", i + 1, tok, prob * 100.0));
            }
        }

        if let Some(ms) = result["latency_ms"].as_f64() {
            out.push(format!("{:.0}ms (remote)", ms));
        }

        Ok(out)
    }

    pub(crate) fn remote_stats(&self) -> Result<Vec<String>, LqlError> {
        let (url, client) = self.require_remote()?;

        let resp = client
            .get(format!("{url}/v1/stats"))
            .send()
            .map_err(|e| LqlError::Execution(format!("request failed: {e}")))?;

        let body: serde_json::Value = resp
            .json()
            .map_err(|e| LqlError::Execution(format!("invalid response: {e}")))?;

        let mut out = Vec::new();
        out.push(format!("Model: {}", body["model"].as_str().unwrap_or("?")));
        out.push(format!("Family: {}", body["family"].as_str().unwrap_or("?")));
        out.push(format!("Layers: {}", body["layers"].as_u64().unwrap_or(0)));
        out.push(format!("Features: {}", body["features"].as_u64().unwrap_or(0)));
        out.push(format!("Hidden: {}", body["hidden_size"].as_u64().unwrap_or(0)));
        out.push(format!("Dtype: {}", body["dtype"].as_str().unwrap_or("?")));
        out.push(format!("Extract level: {}", body["extract_level"].as_str().unwrap_or("?")));

        if let Some(bands) = body.get("layer_bands") {
            if let (Some(s), Some(k), Some(o)) = (
                bands.get("syntax"),
                bands.get("knowledge"),
                bands.get("output"),
            ) {
                out.push(format!(
                    "Bands: syntax {}-{}, knowledge {}-{}, output {}-{}",
                    s[0], s[1], k[0], k[1], o[0], o[1]
                ));
            }
        }

        if let Some(loaded) = body.get("loaded") {
            out.push(format!(
                "Loaded: browse={}, inference={}",
                loaded["browse"].as_bool().unwrap_or(false),
                loaded["inference"].as_bool().unwrap_or(false),
            ));
        }

        out.push(format!("Remote: {url}"));

        Ok(out)
    }

    pub(crate) fn remote_show_relations(&self) -> Result<Vec<String>, LqlError> {
        let (url, client) = self.require_remote()?;

        let resp = client
            .get(format!("{url}/v1/relations"))
            .send()
            .map_err(|e| LqlError::Execution(format!("request failed: {e}")))?;

        let body: serde_json::Value = resp
            .json()
            .map_err(|e| LqlError::Execution(format!("invalid response: {e}")))?;

        let mut out = Vec::new();
        out.push(format!(
            "{:<25} {:>8}",
            "Token", "Count"
        ));
        out.push("-".repeat(35));

        if let Some(rels) = body["relations"].as_array() {
            for rel in rels {
                let name = rel["name"].as_str().unwrap_or("?");
                let count = rel["count"].as_u64().unwrap_or(0);
                out.push(format!("{:<25} {:>8}", name, count));
            }
        }

        Ok(out)
    }

    // ── Remote mutations (forwarded to server as patches) ──

    pub(crate) fn remote_insert(
        &self,
        entity: &str,
        relation: &str,
        target: &str,
        layer: Option<u32>,
        confidence: Option<f32>,
    ) -> Result<Vec<String>, LqlError> {
        let (url, client) = self.require_remote()?;

        let op = larql_vindex::PatchOp::Insert {
            layer: layer.unwrap_or(26) as usize,
            feature: 0, // server assigns
            relation: Some(relation.to_string()),
            entity: entity.to_string(),
            target: target.to_string(),
            confidence,
            gate_vector_b64: None,
            down_meta: Some(larql_vindex::patch::core::PatchDownMeta {
                top_token: target.to_string(),
                top_token_id: 0,
                c_score: confidence.unwrap_or(0.9),
            }),
        };

        let patch = larql_vindex::VindexPatch {
            version: 1,
            base_model: String::new(),
            base_checksum: None,
            created_at: String::new(),
            description: Some(format!("INSERT ({entity}, {relation}, {target})")),
            author: None,
            tags: vec![],
            operations: vec![op],
        };

        let resp = client
            .post(format!("{url}/v1/patches/apply"))
            .json(&serde_json::json!({"patch": patch}))
            .send()
            .map_err(|e| LqlError::Execution(format!("request failed: {e}")))?;

        if !resp.status().is_success() {
            let text = resp.text().unwrap_or_default();
            return Err(LqlError::Execution(format!("INSERT failed: {text}")));
        }

        Ok(vec![format!(
            "Inserted: ({entity}, {relation}, {target}) → remote server"
        )])
    }

    pub(crate) fn remote_delete(
        &self,
        conditions: &[crate::ast::Condition],
    ) -> Result<Vec<String>, LqlError> {
        let (url, client) = self.require_remote()?;

        // Build delete operations from conditions.
        let mut ops = Vec::new();
        let layer = conditions
            .iter()
            .find(|c| c.field == "layer")
            .and_then(|c| match &c.value {
                crate::ast::Value::Integer(n) => Some(*n as usize),
                _ => None,
            })
            .unwrap_or(0);
        let feature = conditions
            .iter()
            .find(|c| c.field == "feature")
            .and_then(|c| match &c.value {
                crate::ast::Value::Integer(n) => Some(*n as usize),
                _ => None,
            })
            .unwrap_or(0);

        ops.push(larql_vindex::PatchOp::Delete {
            layer,
            feature,
            reason: Some("remote DELETE".into()),
        });

        let patch = larql_vindex::VindexPatch {
            version: 1,
            base_model: String::new(),
            base_checksum: None,
            created_at: String::new(),
            description: Some(format!("DELETE L{layer} F{feature}")),
            author: None,
            tags: vec![],
            operations: ops,
        };

        let resp = client
            .post(format!("{url}/v1/patches/apply"))
            .json(&serde_json::json!({"patch": patch}))
            .send()
            .map_err(|e| LqlError::Execution(format!("request failed: {e}")))?;

        if !resp.status().is_success() {
            let text = resp.text().unwrap_or_default();
            return Err(LqlError::Execution(format!("DELETE failed: {text}")));
        }

        Ok(vec![format!("Deleted: L{layer} F{feature} → remote server")])
    }

    // ── Local patch management (client-side overlay) ──

    pub(crate) fn remote_apply_local_patch(&mut self, path: &str) -> Result<Vec<String>, LqlError> {
        let patch_path = std::path::PathBuf::from(path);
        if !patch_path.exists() {
            return Err(LqlError::Execution(format!("patch not found: {path}")));
        }

        let patch = larql_vindex::VindexPatch::load(&patch_path)
            .map_err(|e| LqlError::Execution(format!("failed to load patch: {e}")))?;

        let (ins, upd, del) = patch.counts();
        let total = patch.len();

        match &mut self.backend {
            Backend::Remote { local_patches, .. } => {
                local_patches.push(patch);
                Ok(vec![format!(
                    "Applied locally: {path} ({total} ops: {ins} ins, {upd} upd, {del} del)\n\
                     Patch stays client-side — server never sees it."
                )])
            }
            _ => Err(LqlError::Execution("not connected to a remote server".into())),
        }
    }

    pub(crate) fn remote_show_patches(&self) -> Result<Vec<String>, LqlError> {
        let local_patches = match &self.backend {
            Backend::Remote { local_patches, .. } => local_patches,
            _ => return Err(LqlError::Execution("not connected to a remote server".into())),
        };

        let mut out = Vec::new();
        if local_patches.is_empty() {
            out.push("  (no local patches)".into());
        } else {
            out.push("Local patches (client-side only):".into());
            for (i, patch) in local_patches.iter().enumerate() {
                let (ins, upd, del) = patch.counts();
                let name = patch.description.as_deref().unwrap_or("(unnamed)");
                out.push(format!(
                    "  {}. {:<40} {} ops ({} ins, {} upd, {} del)",
                    i + 1, name, patch.len(), ins, upd, del,
                ));
            }
        }
        Ok(out)
    }

    pub(crate) fn remote_remove_local_patch(&mut self, name: &str) -> Result<Vec<String>, LqlError> {
        let local_patches = match &mut self.backend {
            Backend::Remote { local_patches, .. } => local_patches,
            _ => return Err(LqlError::Execution("not connected to a remote server".into())),
        };

        let pos = local_patches
            .iter()
            .position(|p| p.description.as_deref().unwrap_or("unnamed") == name);

        match pos {
            Some(i) => {
                local_patches.remove(i);
                Ok(vec![format!("Removed local patch: {name}")])
            }
            None => Err(LqlError::Execution(format!("local patch not found: {name}"))),
        }
    }
}
