//! LQL over a VINDEX3 container — the control plane end to end.
//!
//! Encodes the miniature judged-semantics checkpoint into a real
//! VINDEX3 container (with a tokenizer), then drives an actual LQL
//! session through every statement the V3 binding serves — plus the
//! refusals, because on a V3 binding a refusal is a capability
//! statement, not a failure:
//!
//! ```text
//! USE          bind once, on the container's own generation marker
//! STATS        the container's own authority
//! SHOW LAYERS  per-layer facts off the executable plan
//! INFER        top-k from batch-prefill logits
//! INFER … GENERATE   greedy continuation through the runtime seam
//! EXPLAIN INFER      the executable plan, statically
//! TRACE        observe the canonical executor while it runs
//! WALK / SELECT / DESCRIBE   browse via semantic roles (V3-LQL-3A)
//! ```
//!
//! Run: cargo run -p larql-demos --example vindex3_demo

use larql_inference::test_utils::synthetic_tokenizer_json;
use larql_lql::{parse, Session};
use larql_vindex::format::vindex3::fixtures::{
    encode_fixture_container, miniature_glimmer, G_VOCAB,
};

fn run(session: &mut Session, stmt: &str) {
    println!("larql> {stmt}");
    match parse(stmt) {
        Ok(parsed) => match session.execute(&parsed) {
            Ok(lines) => {
                for line in lines {
                    println!("{line}");
                }
            }
            Err(e) => println!("Error: {e}"),
        },
        Err(e) => println!("Parse error: {e}"),
    }
    println!();
}

fn main() {
    println!("=== LQL x VINDEX3 Demo ===\n");

    // A real container: the miniature Glimmer anatomy (sliding+full
    // attention split, four-norm placement) encoded into VINDEX3, plus
    // a tokenizer so the text statements work. This is the same
    // fixture the executor's parity gates certify.
    let checkpoint = tempfile::tempdir().expect("tempdir");
    let container = tempfile::tempdir().expect("tempdir");
    encode_fixture_container(
        miniature_glimmer,
        checkpoint.path(),
        container.path(),
        "demo-glimmer",
    );
    std::fs::write(
        container.path().join("tokenizer.json"),
        synthetic_tokenizer_json(G_VOCAB),
    )
    .expect("write tokenizer");

    let mut session = Session::new();

    // ── Bind once; everything after consumes declared facts ──
    run(
        &mut session,
        &format!("USE \"{}\";", container.path().display()),
    );

    // ── The container's own authority ──
    run(&mut session, "STATS;");
    run(&mut session, "SHOW LAYERS;");

    // ── Inference through the proven runtime seam ──
    run(&mut session, r#"INFER "[3]" TOP 5;"#);
    run(&mut session, r#"INFER "[3]" GENERATE 16;"#);

    // ── Explain the program that will run; observe it running ──
    run(&mut session, r#"EXPLAIN INFER "[3]";"#);
    run(&mut session, r#"TRACE "[3]";"#);

    // ── Browse: the model as a database, via semantic roles ──
    run(&mut session, r#"WALK "[3]" TOP 3;"#);
    run(&mut session, r#"SELECT * FROM FEATURES WHERE layer = 0 LIMIT 5;"#);
    run(&mut session, r#"DESCRIBE "[3]";"#);

    // ── Refusals are capability statements (mutation is V3-LQL-3B) ──
    run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a", "b", "c");"#,
    );

    println!("=== Done ===");
}
