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

use larql_lql::nl::catalog::{kind_of, Access};
use larql_lql::nl::{route, HttpTransport, Outcome, RouterContext};
use serde_json::Value;

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

#[test]
#[ignore = "live: sends requests to api.typesafe.ai"]
fn route_eval() {
    let t = match HttpTransport::from_env() {
        Ok(t) => t,
        Err(e) => panic!("{e}"),
    };
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
        let got = route(english, &ctx, &t)
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
        let out = match route(english, &ctx, &t) {
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
