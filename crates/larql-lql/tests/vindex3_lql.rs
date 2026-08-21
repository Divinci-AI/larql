//! LQL-1 gates: USE / SHOW / INFER against a VINDEX3 container.
//!
//! The invariant under test: **LQL binds once, then operates on the
//! runtime's declared facts and capabilities** — no statement executor
//! reconstructs architecture from weights or family metadata.
//!
//! The headline gate completes the equality chain the earlier rungs
//! built. Three arms run here — the direct runtime stack, the
//! larql-inference driver, and LQL `INFER … GENERATE` — and must agree
//! id-for-id on 16 greedy tokens. The fourth entry point,
//! larql-server, is already pinned token-for-token against the same
//! direct arm by the SERVE-1 gates, so the chain composes:
//!
//! ```text
//! direct V3 runtime == larql-inference == larql-server == LQL INFER
//! ```

use std::path::Path;

use larql_inference::test_utils::synthetic_tokenizer_json;
use larql_inference::vindex3::{continue_session, generate_session, Vindex3Runtime};
use larql_inference::{EosConfig, SamplingConfig};
use larql_kv::CanonicalKvState;
use larql_lql::{parse, Session};
use larql_vindex::format::load::load_vindex_config;
use larql_vindex::format::vindex3::fixtures::{
    encode_fixture_container, miniature_glimmer, G_VOCAB,
};
use larql_vindex::format::vindex3::opplan::exec::production::ProductionBackend;

const NEW_TOKENS: usize = 16;
const PROMPT: &str = "[3]";
const COMPONENT: &str = "target";

/// Encode the miniature container under its own name, with a servable
/// tokenizer.
fn v3_container() -> tempfile::TempDir {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(
        miniature_glimmer,
        checkpoint.path(),
        container.path(),
        "lql-fixture",
    );
    std::fs::write(
        container.path().join("tokenizer.json"),
        synthetic_tokenizer_json(G_VOCAB),
    )
    .unwrap();
    container
}

fn run(session: &mut Session, stmt: &str) -> Vec<String> {
    let parsed = parse(stmt).unwrap_or_else(|e| panic!("parse {stmt}: {e}"));
    session
        .execute(&parsed)
        .unwrap_or_else(|e| panic!("execute {stmt}: {e}"))
}

fn bound_session(container: &Path) -> Session {
    let mut session = Session::new();
    let use_stmt = format!("USE \"{}\";", container.display());
    run(&mut session, &use_stmt);
    session
}

fn prompt_ids(container: &Path) -> Vec<u32> {
    let tokenizer = larql_vindex::load_vindex_tokenizer(container).unwrap();
    tokenizer.encode(PROMPT, true).unwrap().get_ids().to_vec()
}

/// Arm A: the direct runtime stack, by hand.
fn direct_arm(container: &Path) -> Vec<u32> {
    let runtime = Vindex3Runtime::open(container, COMPONENT, ProductionBackend::new()).unwrap();
    let ids = prompt_ids(container);
    let mut kv = CanonicalKvState::new();
    let prefill = runtime.prefill_into(&ids, &mut kv).unwrap();
    let mut session = runtime.session_with_kv(&mut kv).unwrap();
    continue_session(
        &mut session,
        prefill,
        NEW_TOKENS,
        SamplingConfig::greedy(),
        &EosConfig::builtin(),
        |_| {},
    )
    .unwrap()
    .tokens
}

/// Arm B: the larql-inference generation driver over a fresh session.
fn inference_arm(container: &Path) -> Vec<u32> {
    let runtime = Vindex3Runtime::open(container, COMPONENT, ProductionBackend::new()).unwrap();
    let ids = prompt_ids(container);
    let mut session = runtime.session().unwrap();
    generate_session(
        &mut session,
        &ids,
        NEW_TOKENS,
        SamplingConfig::greedy(),
        &EosConfig::builtin(),
        |_| {},
    )
    .unwrap()
    .tokens
}

/// Arm C: LQL, over the statement surface. Ids come back on the
/// `  ids:  a,b,c` line of `INFER … GENERATE`.
fn lql_arm(container: &Path) -> Vec<u32> {
    let mut session = bound_session(container);
    let out = run(
        &mut session,
        &format!("INFER \"{PROMPT}\" GENERATE {NEW_TOKENS};"),
    );
    let ids_line = out
        .iter()
        .find_map(|line| line.trim_start().strip_prefix("ids:"))
        .unwrap_or_else(|| panic!("no ids line in {out:?}"));
    ids_line
        .trim()
        .split(',')
        .map(|id| id.parse().unwrap())
        .collect()
}

#[test]
fn the_equality_chain_holds_through_lql_infer_generate() {
    let container = v3_container();
    let direct = direct_arm(container.path());
    assert_eq!(direct.len(), NEW_TOKENS, "fixture must fill the budget");
    let inference = inference_arm(container.path());
    let lql = lql_arm(container.path());
    assert_eq!(direct, inference, "larql-inference arm diverges");
    assert_eq!(direct, lql, "LQL arm diverges");
}

#[test]
fn use_binds_a_v3_container_under_its_own_name() {
    let container = v3_container();
    let mut session = Session::new();
    let out = run(
        &mut session,
        &format!("USE \"{}\";", container.path().display()),
    );
    let banner = out.join("\n");
    assert!(banner.contains("VINDEX3"), "{banner}");
    // The container names itself — never the temp directory's name.
    assert!(banner.contains("lql-fixture"), "{banner}");
    assert!(banner.contains("execution closed"), "{banner}");
}

#[test]
fn infer_without_generate_prices_the_next_token() {
    let container = v3_container();
    let first_greedy = direct_arm(container.path())[0];
    let mut session = bound_session(container.path());
    let out = run(&mut session, &format!("INFER \"{PROMPT}\" TOP 3;"));
    assert!(out[0].contains("VINDEX3"), "{out:?}");
    // Top-1 must be the greedy continuation's first id.
    assert!(
        out[1].contains(&format!("[id {first_greedy}]")),
        "top-1 must match the greedy id: {out:?}"
    );
    // TOP 3 → three prediction rows plus header and timing.
    assert_eq!(out.len(), 5, "{out:?}");
}

#[test]
fn stats_reports_the_containers_own_authority() {
    let container = v3_container();
    let mut session = bound_session(container.path());
    let out = run(&mut session, "STATS;").join("\n");
    assert!(out.contains("lql-fixture (VINDEX3)"), "{out}");
    assert!(out.contains("Generation:      3"), "{out}");
    assert!(out.contains("1 sliding / 1 full"), "{out}");
    assert!(out.contains("kv_dim 4"), "{out}");
    assert!(out.contains("closed"), "{out}");
    assert!(out.contains("Output head:     present"), "{out}");
    assert!(out.contains("Tokenizer:       present"), "{out}");
}

#[test]
fn show_layers_reads_the_plan_not_a_family_registry() {
    let container = v3_container();
    let mut session = bound_session(container.path());
    let out = run(&mut session, "SHOW LAYERS;");
    // Layer 0: sliding window 3; layer 1: full — the miniature's mixed
    // anatomy, straight off the executable plan.
    let layer0 = out.iter().find(|l| l.starts_with("0")).unwrap();
    assert!(layer0.contains("sliding"), "{layer0}");
    assert!(layer0.contains('3'), "{layer0}");
    let layer1 = out.iter().find(|l| l.starts_with("1")).unwrap();
    assert!(layer1.contains("full"), "{layer1}");
}

/// The negative control: the container LQL is happily executing cannot
/// be opened by the V2 config loader at all — binding resolved V3 from
/// the container's own marker, not by falling back from a failed V2
/// load into some reconstruction.
#[test]
fn the_v2_path_refuses_the_container_lql_serves() {
    let container = v3_container();
    assert!(load_vindex_config(container.path()).is_err());
    // …and the same path binds and executes through LQL.
    let mut session = bound_session(container.path());
    let out = run(&mut session, &format!("INFER \"{PROMPT}\" GENERATE 2;"));
    assert!(out[0].contains("Generated (2 tokens"), "{out:?}");
}

#[test]
fn browse_and_mutation_statements_refuse_with_capabilities() {
    let container = v3_container();
    let mut session = bound_session(container.path());
    for stmt in ["SHOW RELATIONS;", "WALK \"[3]\";"] {
        let parsed = parse(stmt).unwrap();
        let err = session.execute(&parsed).unwrap_err().to_string();
        assert!(
            err.contains("not supported on a VINDEX3 container"),
            "{stmt}: {err}"
        );
        assert!(
            err.contains("INFER"),
            "refusal must name capabilities: {err}"
        );
    }
}

#[test]
fn generate_requires_a_v3_binding() {
    let mut session = Session::new();
    let parsed = parse(&format!("INFER \"{PROMPT}\" GENERATE 4;")).unwrap();
    let err = session.execute(&parsed).unwrap_err().to_string();
    assert!(err.contains("VINDEX3"), "{err}");
}

/// A container without tokenizer.json still binds and answers STATS —
/// but INFER refuses with a capability error naming the missing fact.
/// Capabilities gate statements; the binding itself never requires
/// more than the executable program.
#[test]
fn a_tokenizerless_container_binds_but_refuses_text_infer() {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(
        miniature_glimmer,
        checkpoint.path(),
        container.path(),
        "ids-only-fixture",
    );

    let mut session = Session::new();
    let out = run(
        &mut session,
        &format!("USE \"{}\";", container.path().display()),
    );
    assert!(
        out.join("\n").contains("token-id capability only"),
        "{out:?}"
    );

    let stats = run(&mut session, "STATS;").join("\n");
    assert!(
        stats.contains("absent (token-id capability only)"),
        "{stats}"
    );

    let parsed = parse(&format!("INFER \"{PROMPT}\" TOP 3;")).unwrap();
    let err = session.execute(&parsed).unwrap_err().to_string();
    assert!(err.contains("no tokenizer.json"), "{err}");
}

#[test]
fn an_empty_prompt_is_refused_before_the_runtime() {
    let container = v3_container();
    let mut session = bound_session(container.path());
    let parsed = parse("INFER \"\" GENERATE 4;").unwrap();
    let err = session.execute(&parsed).unwrap_err().to_string();
    assert!(err.contains("tokenises to empty"), "{err}");
}

// ── LQL-2: EXPLAIN INFER / TRACE ──

/// EXPLAIN INFER renders the executable authority — statically,
/// deterministically, and with the miniature's real anatomy: the
/// sliding(3)/full split, the four-norm op order, operand bindings,
/// and plan-derived continuation geometry. Repeated runs render
/// identically (the structured value is `PartialEq`-gated below the
/// renderer too).
#[test]
fn explain_infer_renders_the_executable_plan() {
    let container = v3_container();
    let mut session = bound_session(container.path());
    let first = run(&mut session, &format!("EXPLAIN INFER \"{PROMPT}\";"));
    let second = run(&mut session, &format!("EXPLAIN INFER \"{PROMPT}\";"));
    assert_eq!(first, second, "explain must be deterministic");

    let text = first.join("\n");
    assert!(text.contains("MODEL"), "{text}");
    assert!(text.contains("name: lql-fixture"), "{text}");
    assert!(text.contains("generation: 3"), "{text}");
    assert!(text.contains("execution: closed"), "{text}");
    assert!(text.contains("mode sliding window 3"), "{text}");
    assert!(text.contains("mode full window -"), "{text}");
    // Four-norm placement is explicit ops, not implied structure.
    assert!(text.contains("post_attention_norm"), "{text}");
    assert!(text.contains("post_ffn_norm"), "{text}");
    // Operand provenance: object::tensor @dtype bindings.
    assert!(text.contains("self_attn.q_proj.weight @F32"), "{text}");
    assert!(text.contains("mlp.up_proj.weight @F32"), "{text}");
    // Continuation geometry from the plan.
    assert!(text.contains("layer 0: kv_dim 4 window 3"), "{text}");
    assert!(text.contains("layer 1: kv_dim 4 window -"), "{text}");
    assert!(text.contains("output_head: present"), "{text}");
    // No family/architecture reconstruction anywhere on this path —
    // the V2 loader cannot even open the container.
    assert!(load_vindex_config(container.path()).is_err());
}

/// TRACE observes the canonical executor: every explained layer
/// appears in the observed stream, and enabling observation changes
/// nothing — the traced next token equals INFER's.
#[test]
fn trace_is_observational_and_agrees_with_explain() {
    let container = v3_container();
    let mut session = bound_session(container.path());

    let trace = run(&mut session, &format!("TRACE \"{PROMPT}\";"));
    let text = trace.join("\n");
    // One prompt position, both layers observed, in order.
    assert!(text.contains("position 0"), "{text}");
    assert!(text.contains("layer 0: attention"), "{text}");
    assert!(text.contains("layer 0: ffn"), "{text}");
    assert!(text.contains("layer 1: attention"), "{text}");
    assert!(text.contains("layer 1: ffn"), "{text}");
    assert!(text.contains("output_head (vocab 29)"), "{text}");

    // Explain/execution agreement: the explained layer set is exactly
    // the observed layer set.
    let explain = run(&mut session, &format!("EXPLAIN INFER \"{PROMPT}\";"));
    let explained_layers = explain
        .iter()
        .filter(|l| l.trim_start().starts_with("layer ") && l.contains("kv_dim"))
        .count();
    let observed_layers = trace.iter().filter(|l| l.contains(": ffn")).count();
    assert_eq!(explained_layers, observed_layers, "explain/trace disagree");

    // Observation is observational: the traced greedy token is INFER's.
    let first_greedy = direct_arm(container.path())[0];
    assert!(
        text.contains(&format!("next token {first_greedy} ")),
        "traced token diverges from the untraced run: {text}"
    );
}

#[test]
fn trace_with_options_refuses_on_v3() {
    let container = v3_container();
    let mut session = bound_session(container.path());
    let parsed = parse(&format!("TRACE \"{PROMPT}\" DECOMPOSE;")).unwrap();
    let err = session.execute(&parsed).unwrap_err().to_string();
    assert!(
        err.contains("not supported on a VINDEX3 container"),
        "{err}"
    );
}

/// The whole-language sweep: every LQL statement, executed against a
/// V3 binding, must do something sensible — execute, or refuse with a
/// message that names what the binding supports. Never a panic, and
/// never the misleading "No backend loaded" (a backend IS loaded).
///
/// This is the gate that keeps the capability model honest as the
/// language grows: a new statement that reaches a V3 session without
/// a deliberate decision fails here.
#[test]
fn every_statement_is_sensible_on_a_v3_binding() {
    let container = v3_container();
    let mut session = bound_session(container.path());

    // (statement, expectation): Ok = must succeed, Refuse = must error
    // mentioning the VINDEX3 capability set, Err = any helpful error.
    enum Expect {
        Ok,
        Refuse,
        Err,
    }
    use Expect::*;
    let cases: Vec<(&str, Expect)> = vec![
        // ── Serves on V3 ──
        (r#"STATS;"#, Ok),
        (r#"SHOW LAYERS;"#, Ok),
        (r#"INFER "[3]" TOP 3;"#, Ok),
        (r#"INFER "[3]" GENERATE 2;"#, Ok),
        (r#"EXPLAIN INFER "[3]";"#, Ok),
        (r#"TRACE "[3]";"#, Ok),
        (r#"SHOW MODELS;"#, Ok),             // registry listing, backend-free
        (r#"SHOW COMPACT STATUS;"#, Refuse), // LSM state is a vindex concept
        (r#"SHOW PATCHES;"#, Refuse),
        // ── Browse: refuses with capabilities ──
        (r#"WALK "[3]";"#, Refuse),
        (r#"DESCRIBE "x";"#, Refuse),
        (r#"SELECT * FROM EDGES LIMIT 5;"#, Refuse),
        (r#"SELECT * FROM FEATURES WHERE layer = 0;"#, Refuse),
        (r#"SELECT * FROM ENTITIES;"#, Refuse),
        (r#"EXPLAIN WALK "[3]";"#, Refuse),
        (r#"SHOW RELATIONS;"#, Refuse),
        (r#"SHOW FEATURES 0;"#, Refuse),
        (r#"SHOW ENTITIES;"#, Refuse),
        // ── Mutation: refuses with capabilities ──
        (
            r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a", "b", "c");"#,
            Refuse,
        ),
        (r#"DELETE FROM EDGES WHERE layer = 0;"#, Refuse),
        (
            r#"UPDATE EDGES SET confidence = 0.5 WHERE layer = 0;"#,
            Refuse,
        ),
        (r#"MERGE "other.vindex";"#, Err), // refuses at source validation
        // Vacuously true on V3: INSERT refuses, so there is never
        // anything to rebalance and the no-op report is honest.
        (r#"REBALANCE;"#, Ok),
        (r#"COMPACT MINOR;"#, Refuse),
        (r#"COMPACT MAJOR;"#, Refuse),
        // ── Patch lifecycle: refuses (a V3 binding has no overlay) ──
        (r#"BEGIN PATCH "p.vlp";"#, Refuse),
        (r#"SAVE PATCH;"#, Refuse),
        (r#"APPLY PATCH "p.vlp";"#, Err), // refuses at file validation
        (r#"REMOVE PATCH "p.vlp";"#, Refuse),
        // ── Lifecycle that targets other artifacts: any helpful error ──
        (r#"COMPILE "x.vindex" INTO MODEL "out";"#, Err),
        (r#"DIFF "a.vindex" "b.vindex";"#, Err),
        (r#"TRACE "[3]" DECOMPOSE;"#, Refuse),
    ];

    for (stmt, expect) in cases {
        let parsed = match parse(stmt) {
            std::result::Result::Ok(p) => p,
            std::result::Result::Err(e) => panic!("sweep statement fails to parse: {stmt}: {e}"),
        };
        let outcome = session.execute(&parsed);
        match (expect, outcome) {
            (Ok, std::result::Result::Ok(_)) => {}
            (Ok, std::result::Result::Err(e)) => panic!("{stmt}: expected success, got: {e}"),
            (Refuse, std::result::Result::Err(e)) => {
                let msg = e.to_string();
                assert!(
                    msg.contains("not supported on a VINDEX3 container"),
                    "{stmt}: refusal must name the capability set, got: {msg}"
                );
                assert!(
                    msg.contains("INFER"),
                    "{stmt}: refusal must list what IS supported: {msg}"
                );
            }
            (Refuse, std::result::Result::Ok(out)) => {
                panic!("{stmt}: must refuse on a V3 binding, got: {out:?}")
            }
            (Err, std::result::Result::Err(e)) => {
                let msg = e.to_string();
                assert!(
                    !msg.contains("No backend loaded"),
                    "{stmt}: a backend IS loaded; misleading error: {msg}"
                );
            }
            (Err, std::result::Result::Ok(out)) => {
                panic!("{stmt}: expected an error, got: {out:?}")
            }
        }
    }
}
