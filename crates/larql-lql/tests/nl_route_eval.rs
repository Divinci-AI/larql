//! Live evaluation of English → LQL routing against TypeSafe's Jev.
//!
//! `#[ignore]`d: it sends every case to api.typesafe.ai and needs
//! TYPESAFE_API_KEY. Run it explicitly:
//!
//!   TYPESAFE_API_KEY=… cargo test -p larql-lql --test nl_route_eval -- --ignored --nocapture
//!
//! Two held-out sets, reported separately and never pooled:
//!   A. doc_pairs.json — English written by larql's authors above an example
//!      statement in the docs. Scores the statement KIND only.
//!   B. authored.json  — one author, full arguments, compared as parsed ASTs,
//!      including requests no statement serves (must decline).

use std::collections::HashMap;
use std::io::Write;

use larql_lql::nl::catalog::{kind_of, Access};
use larql_lql::nl::{candidates, jev, route, HttpTransport, Outcome, RouterContext, Transport};
use serde_json::{json, Value};

fn fixture(name: &str) -> Value {
    let p = format!("{}/tests/fixtures/nl/{name}", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{p}: {e}")))
        .unwrap_or_else(|e| panic!("{p}: {e}"))
}

fn outcome_kind(o: &Outcome) -> String {
    match o {
        Outcome::Proposed(p) => p.kind.key().to_string(),
        Outcome::Incomplete { kind, .. } | Outcome::Unparseable { kind, .. } => {
            kind.key().to_string()
        }
        Outcome::NoMatch { .. } => "none".to_string(),
    }
}

/// Every English request in both fixtures, in a fixed order.
fn all_requests() -> Vec<String> {
    let a = fixture("doc_pairs.json");
    let b = fixture("authored.json");
    a["pairs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p["english"].as_str().unwrap().to_string())
        .chain(
            b["cases"]
                .as_array()
                .unwrap()
                .iter()
                .map(|c| c["english"].as_str().unwrap().to_string()),
        )
        .collect()
}

/// Score routing through any transport. The live run, and a replay of recorded
/// responses from another endpoint, go through this one function, so the two
/// can only differ in the answers — never in how they are scored.
fn run_eval(t: &dyn Transport) {
    let ctx = RouterContext::default();

    // ── A: doc pairs, statement kind ──
    let a = fixture("doc_pairs.json");
    let pairs = a["pairs"].as_array().unwrap();
    let (mut a_ok, mut a_none) = (0, 0);
    println!(
        "\n== A. doc pairs (larql authors' English, kind only) n={} ==",
        pairs.len()
    );
    for p in pairs {
        let english = p["english"].as_str().unwrap();
        let want = p["expected_kind"].as_str().unwrap();
        let got = route(english, &ctx, t)
            .map(|o| outcome_kind(&o))
            .unwrap_or_else(|e| format!("ERR {e}"));
        if got == want {
            a_ok += 1;
        } else {
            if got == "none" {
                a_none += 1;
            }
            println!("  MISS want={want:<18} got={got:<18} {english:.70}");
        }
    }
    println!(
        "  kind accuracy {a_ok}/{} = {:.3}   (declined {a_none})",
        pairs.len(),
        a_ok as f64 / pairs.len() as f64
    );

    // ── B: authored, full arguments + declines ──
    let b = fixture("authored.json");
    let cases = b["cases"].as_array().unwrap();
    let (mut exact, mut kind_ok, mut incomplete, mut in_scope) = (0, 0, 0, 0);
    let (mut decl_ok, mut decl_n, mut false_decline, mut writes_proposed_for_declines) =
        (0, 0, 0, 0);
    println!("\n== B. authored (single author) n={} ==", cases.len());
    for c in cases {
        let english = c["english"].as_str().unwrap();
        let out = match route(english, &ctx, t) {
            Ok(o) => o,
            Err(e) => {
                println!("  ERR  {english:.60}  {e}");
                continue;
            }
        };
        match c["expected_lql"].as_str() {
            None => {
                decl_n += 1;
                match &out {
                    Outcome::NoMatch { .. } => decl_ok += 1,
                    other => {
                        if let Outcome::Proposed(p) = other {
                            if p.access == Access::Write {
                                writes_proposed_for_declines += 1;
                            }
                        }
                        println!(
                            "  SHOULD DECLINE  got={:<14} {english:.60}",
                            outcome_kind(other)
                        );
                    }
                }
            }
            Some(want_lql) => {
                in_scope += 1;
                let want = larql_lql::parse(want_lql)
                    .unwrap_or_else(|e| panic!("fixture `{want_lql}`: {e}"));
                let want_kind = kind_of(&want).unwrap().key().to_string();
                if outcome_kind(&out) == want_kind {
                    kind_ok += 1;
                }
                match &out {
                    Outcome::Proposed(p) if format!("{:?}", p.statement) == format!("{want:?}") => {
                        exact += 1
                    }
                    Outcome::Proposed(p) => {
                        println!("  ARGS  want `{want_lql}`\n        got  `{}`", p.lql)
                    }
                    Outcome::Incomplete {
                        template, missing, ..
                    } => {
                        incomplete += 1;
                        println!("  INCOMPLETE {missing:?}  `{template}`  ← {english:.50}");
                    }
                    Outcome::NoMatch { .. } => {
                        false_decline += 1;
                        println!("  FALSE DECLINE  want `{want_lql}`");
                    }
                    Outcome::Unparseable { lql, error, .. } => {
                        println!("  UNPARSEABLE `{lql}`: {error}")
                    }
                }
            }
        }
    }
    println!("  in scope {in_scope}: kind {kind_ok}/{in_scope} = {:.3}   exact (kind + every argument) {exact}/{in_scope} = {:.3}",
        kind_ok as f64 / in_scope as f64, exact as f64 / in_scope as f64);
    println!("            incomplete (asked user to fill a value) {incomplete}   false declines {false_decline}");
    println!("  out of scope {decl_n}: declined {decl_ok}/{decl_n} = {:.3}   writes proposed for them {writes_proposed_for_declines}",
        decl_ok as f64 / decl_n as f64);
}

#[test]
#[ignore = "live: sends requests to api.typesafe.ai"]
fn route_eval() {
    let t = HttpTransport::from_env().unwrap_or_else(|e| panic!("{e}"));
    run_eval(&t);
}

/// Write the exact request body for every case, WITHOUT sending anything, so the
/// same bodies can be replayed against another Jev-compatible endpoint.
///
///   NL_DUMP=bodies.jsonl cargo test -p larql-lql --test nl_route_eval dump_bodies -- --ignored
#[test]
#[ignore = "writes a file named by NL_DUMP"]
fn dump_bodies() {
    let path = std::env::var("NL_DUMP").expect("set NL_DUMP to an output path");
    let mut f = std::fs::File::create(&path).unwrap();
    let mut seen = std::collections::HashSet::new();
    for english in all_requests() {
        // Identical English appears in both sets at most rarely; one body serves both.
        if !seen.insert(english.clone()) {
            continue;
        }
        let (body, _) = jev::build_request(&english, &candidates::extract(&english), &[]);
        let line =
            json!({ "id": format!("larql:{}", seen.len() - 1), "suite": "larql", "body": body });
        writeln!(f, "{line}").unwrap();
    }
    eprintln!("wrote {} bodies to {path}", seen.len());
}

/// Answers each request with a recorded response, matched on the request text.
struct Recorded(HashMap<String, Value>);

impl Transport for Recorded {
    fn post(&self, body: &Value) -> Result<Value, String> {
        let request = body["state"]["request"].as_str().unwrap_or_default();
        self.0
            .get(request)
            .cloned()
            .ok_or_else(|| format!("no recorded response for {request:?}"))
    }
}

/// Score recorded responses from another endpoint (e.g. a self-hosted djev):
///
///   NL_BODIES=bodies.jsonl NL_REPLAY=responses.jsonl \
///     cargo test -p larql-lql --test nl_route_eval route_replay -- --ignored --nocapture
#[test]
#[ignore = "reads recorded responses named by NL_REPLAY"]
fn route_replay() {
    let bodies = std::env::var("NL_BODIES").expect("set NL_BODIES");
    let replay = std::env::var("NL_REPLAY").expect("set NL_REPLAY");
    let mut request_of: HashMap<String, String> = HashMap::new();
    for line in std::fs::read_to_string(&bodies).unwrap().lines() {
        let v: Value = serde_json::from_str(line).unwrap();
        // The bodies file may carry other suites (flagger, collateral) whose
        // state has no `request`; only the larql suite is scored here.
        if v["suite"] != "larql" {
            continue;
        }
        request_of.insert(
            v["id"].as_str().unwrap().to_string(),
            v["body"]["state"]["request"].as_str().unwrap().to_string(),
        );
    }
    let mut map = HashMap::new();
    let mut failed = 0;
    for line in std::fs::read_to_string(&replay).unwrap().lines() {
        let v: Value = serde_json::from_str(line).unwrap();
        let id = v["id"].as_str().unwrap();
        let Some(request) = request_of.get(id) else {
            continue;
        };
        if v["response"].is_null() {
            failed += 1;
            continue;
        }
        map.insert(request.clone(), v["response"].clone());
    }
    println!(
        "replaying {} recorded responses from {replay} ({failed} requests had no response)",
        map.len()
    );
    run_eval(&Recorded(map));
}
