//! Diagnostic, not a test: prints what `query=residual` sees on the fixture.
//! `cargo test -p larql-server --test residual_probe -- --ignored --nocapture`
#[test]
#[ignore]
fn probe() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-vindex");
    let m = larql_server::bootstrap::load::load_single_vindex(
        root.to_str().unwrap(),
        larql_server::bootstrap::load::LoadVindexOptions {
            no_infer: false,
            ..Default::default()
        },
    )
    .expect("load");
    for entity in ["[5]", "[124]", "[5] [124] [69]"] {
        for q in ["embedding", "residual"] {
            let params = larql_server::routes::describe::DescribeParams {
                entity: entity.into(),
                band: "all".into(),
                verbose: true,
                limit: 20,
                window: 20,
                min_score: 0.0,
                coherence: false,
                min_coherence: 0.0,
                relabel: false,
                relevance: true,
                background: Some("vocabulary".into()),
                query: q.into(),
                baseline: None,
            };
            let r = larql_server::routes::describe::describe_entity_with(
                &m,
                &m.patched.blocking_read(),
                &params,
            );
            match r {
                Ok(v) => {
                    let edges = v["edges"].as_array().unwrap();
                    let mut scores: Vec<f64> = edges
                        .iter()
                        .filter_map(|e| e["gate_score"].as_f64())
                        .collect();
                    scores.sort_by(|a, b| b.partial_cmp(a).unwrap());
                    let layers: std::collections::BTreeSet<u64> =
                        edges.iter().filter_map(|e| e["layer"].as_u64()).collect();
                    let top: Vec<String> = edges
                        .iter()
                        .take(6)
                        .map(|e| format!("L{}:{}", e["layer"], e["target"].as_str().unwrap_or("?")))
                        .collect();
                    eprintln!("{q:<9} {entity:<16} n={:<3} score max={:?} min={:?} layers={:?} residual_layers={} scanned={} top={:?}",
                        edges.len(), scores.first(), scores.last(), layers, v["residual_layers"], v["scanned_layers"], top);
                }
                Err(e) => eprintln!("{q:<9} {entity:<16} ERR {e:?}"),
            }
        }
    }
}
