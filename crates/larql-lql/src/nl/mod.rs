//! English → LQL.
//!
//! `route` asks TypeSafe's Jev to pick the statement (or decline) and to
//! select each argument from spans code found in the request, renders LQL,
//! and hands it to the real parser. The parser is the last word: nothing the
//! router returns reaches the executor except as a parsed `Statement`.
//!
//! Every request is SENT TO api.typesafe.ai. That is a deliberate product
//! decision (2026-09-22) and the only network path in this module.
//!
//! Writes are proposed, never run: [`execute`] refuses a statement whose
//! parsed form is a write unless the caller passes `confirmed = true`.

pub mod candidates;
pub mod catalog;
pub mod jev;
pub mod render;

use crate::ast::Statement;
use crate::executor::Session;
use catalog::{Access, Kind, NONE_KEY};
use render::{Rendered, Slots};

pub use jev::{HttpTransport, Transport};

#[derive(Debug, thiserror::Error)]
pub enum NlError {
    #[error("router request failed: {0}")]
    Transport(String),
    #[error("router response was malformed: {0}")]
    Malformed(String),
    #[error("this is a write ({lql}) — re-run with confirmation to execute it")]
    NeedsConfirmation { lql: String },
    #[error(transparent)]
    Lql(#[from] crate::LqlError),
}

/// Context the router may use beyond the request text.
#[derive(Debug, Clone, Default)]
pub struct RouterContext {
    /// Relation labels the loaded vindex actually has. Offered first, so a
    /// request's wording can resolve to a stored relation name.
    pub relations: Vec<String>,
}

/// A statement the router proposes. Holds the PARSED statement, and its access
/// class is computed from that — not from the router's own choice.
#[derive(Debug, Clone)]
pub struct Proposal {
    pub lql: String,
    pub statement: Statement,
    pub kind: Kind,
    pub access: Access,
    /// The router's confidence in the statement choice.
    pub confidence: f64,
}

#[derive(Debug, Clone)]
pub enum Outcome {
    /// Parsed, ready to run (reads) or to confirm (writes).
    Proposed(Proposal),
    /// A statement was chosen, but a required value is not in the request.
    Incomplete {
        kind: Kind,
        template: String,
        missing: Vec<&'static str>,
        confidence: f64,
    },
    /// The router judged that no LQL statement serves the request.
    NoMatch { confidence: f64 },
    /// Rendered text the parser rejected. A renderer defect, reported as one
    /// rather than papered over.
    Unparseable {
        kind: Kind,
        lql: String,
        error: String,
    },
}

fn pick(offered: &jev::Offered, answers: &serde_json::Value, slot: &str) -> Option<String> {
    let a = jev::read_choice(answers, slot)?;
    if a.choice == NONE_KEY {
        return None;
    }
    // Only a value that was actually offered is accepted. The router cannot
    // introduce text of its own through a slot.
    offered
        .options(slot)?
        .iter()
        .find(|o| **o == a.choice)
        .cloned()
}

fn pick_num(offered: &jev::Offered, answers: &serde_json::Value, slot: &str) -> Option<u32> {
    pick(offered, answers, slot)?.parse().ok()
}

/// Route one English request to a proposed LQL statement, or decline.
pub fn route(
    english: &str,
    ctx: &RouterContext,
    transport: &dyn Transport,
) -> Result<Outcome, NlError> {
    let cands = candidates::extract(english);
    let (body, offered) = jev::build_request(english, &cands, &ctx.relations);
    let resp = transport.post(&body).map_err(NlError::Transport)?;
    let answers = resp
        .get("answers")
        .ok_or_else(|| NlError::Malformed("no `answers` field".into()))?;

    let st = jev::read_choice(answers, "statement")
        .ok_or_else(|| NlError::Malformed("no `statement` choice".into()))?;
    if st.choice == NONE_KEY {
        return Ok(Outcome::NoMatch {
            confidence: st.confidence,
        });
    }
    let kind = Kind::from_key(&st.choice).ok_or_else(|| {
        NlError::Malformed(format!("statement `{}` was never offered", st.choice))
    })?;

    let count = pick_num(&offered, answers, "count");
    let mut slots = Slots {
        prompt: pick(&offered, answers, "prompt"),
        entity: pick(&offered, answers, "entity"),
        relation: pick(&offered, answers, "relation"),
        target: pick(&offered, answers, "target"),
        source: pick(&offered, answers, "source"),
        dest: pick(&offered, answers, "dest"),
        other: pick(&offered, answers, "other"),
        layer: pick_num(&offered, answers, "layer"),
        top: count,
        limit: count,
        compile_into_model: jev::read_noul(answers, "into_model").is_some_and(|p| p >= 0.5),
        explain_walk: jev::read_noul(answers, "explain_walk").is_some_and(|p| p >= 0.5),
    };

    // One span cannot fill two roles that must differ. The router answers
    // each slot independently and can pick the same span twice; where the
    // roles are distinct, a duplicate is dropped so the statement comes back
    // Incomplete (the user is asked) rather than wrong. For EXTRACT that is
    // the difference between asking and overwriting the source with itself.
    //
    // Scoped to the statements that USE both roles. Every slot is answered on
    // every request, so for `USE a.vindex` source and dest both pick the only
    // path there is — and clearing them there would break a correct route.
    if kind == Kind::Extract && slots.source.is_some() && slots.source == slots.dest {
        slots.source = None;
        slots.dest = None;
    }
    if kind == Kind::Trace && slots.target.is_some() && slots.target == slots.prompt {
        slots.target = None;
    }

    match render::render(kind, &slots) {
        Rendered::Incomplete { template, missing } => Ok(Outcome::Incomplete {
            kind,
            template,
            missing,
            confidence: st.confidence,
        }),
        Rendered::Complete(lql) => match crate::parse(&lql) {
            Ok(statement) => Ok(Outcome::Proposed(Proposal {
                access: catalog::access(&statement),
                lql,
                statement,
                kind,
                confidence: st.confidence,
            })),
            Err(e) => Ok(Outcome::Unparseable {
                kind,
                lql,
                error: e.to_string(),
            }),
        },
    }
}

/// Run a proposal. A write runs only with `confirmed = true`; the check is on
/// the parsed statement's access class and happens before the session is
/// touched.
pub fn execute(
    p: &Proposal,
    session: &mut Session,
    confirmed: bool,
) -> Result<Vec<String>, NlError> {
    if catalog::access(&p.statement) == Access::Write && !confirmed {
        return Err(NlError::NeedsConfirmation { lql: p.lql.clone() });
    }
    Ok(session.execute(&p.statement)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    /// Answers every question with a fixed choice, as the router would.
    struct Scripted(Value);
    impl Transport for Scripted {
        fn post(&self, _body: &Value) -> Result<Value, String> {
            Ok(json!({ "answers": self.0.clone() }))
        }
    }

    fn c(choice: &str) -> Value {
        json!({ "type": "choice", "choice": choice, "confidence": 0.9 })
    }

    #[test]
    fn a_decline_is_a_no_match_not_a_guess() {
        let t = Scripted(json!({ "statement": c("none") }));
        let out = route("tell me a joke", &RouterContext::default(), &t).unwrap();
        assert!(matches!(out, Outcome::NoMatch { .. }));
    }

    #[test]
    fn a_describe_request_routes_to_parsed_lql() {
        let t = Scripted(json!({ "statement": c("describe"), "entity": c("France") }));
        let Outcome::Proposed(p) = route("describe France", &RouterContext::default(), &t).unwrap()
        else {
            panic!()
        };
        assert_eq!(p.lql, r#"DESCRIBE "France";"#);
        assert_eq!(p.access, Access::Read);
    }

    /// The router may only select values that code offered. Answering a slot
    /// with text that was never a candidate yields no value, not that text.
    #[test]
    fn a_slot_answer_that_was_never_offered_is_ignored() {
        let t = Scripted(json!({ "statement": c("describe"), "entity": c("Germany") }));
        let out = route("describe France", &RouterContext::default(), &t).unwrap();
        assert!(matches!(out, Outcome::Incomplete { .. }), "{out:?}");
    }

    #[test]
    fn a_statement_that_was_never_offered_is_malformed() {
        let t = Scripted(json!({ "statement": c("drop_table") }));
        assert!(matches!(
            route("drop everything", &RouterContext::default(), &t),
            Err(NlError::Malformed(_))
        ));
    }

    /// EXTRACT with the same span as model and output would overwrite the
    /// source with itself. It must come back asking, not proposing.
    #[test]
    fn the_same_span_in_two_distinct_roles_is_not_accepted() {
        let t = Scripted(json!({
            "statement": c("extract"), "source": c("a.vindex"), "dest": c("a.vindex")
        }));
        let out = route("extract a.vindex", &RouterContext::default(), &t).unwrap();
        assert!(
            matches!(
                out,
                Outcome::Incomplete {
                    kind: Kind::Extract,
                    ..
                }
            ),
            "{out:?}"
        );
    }

    /// The duplicate-role rule must not fire where only one role is used:
    /// a lone path answering both slots is the normal case for USE.
    #[test]
    fn a_single_path_still_routes_use() {
        let t = Scripted(json!({
            "statement": c("use"), "source": c("a.vindex"), "dest": c("a.vindex")
        }));
        let Outcome::Proposed(p) = route("open a.vindex", &RouterContext::default(), &t).unwrap()
        else {
            panic!("USE with one path was not proposed")
        };
        assert_eq!(p.lql, r#"USE "a.vindex";"#);
    }

    #[test]
    fn a_delete_without_its_subject_is_incomplete() {
        let t = Scripted(json!({ "statement": c("delete") }));
        let out = route("forget that fact", &RouterContext::default(), &t).unwrap();
        assert!(matches!(
            out,
            Outcome::Incomplete {
                kind: Kind::Delete,
                ..
            }
        ));
    }

    /// The gate that stands between a misroute and a weight edit. An
    /// unconfirmed write is refused BEFORE the session is touched: an empty
    /// session would otherwise fail with an executor error, and it must not
    /// get the chance.
    #[test]
    fn an_unconfirmed_write_is_refused_before_execution() {
        let t = Scripted(json!({ "statement": c("delete"), "entity": c("John Coyle") }));
        let Outcome::Proposed(p) =
            route("erase John Coyle", &RouterContext::default(), &t).unwrap()
        else {
            panic!()
        };
        assert_eq!(p.access, Access::Write);
        let mut session = Session::new();
        let err = execute(&p, &mut session, false).unwrap_err();
        assert!(matches!(err, NlError::NeedsConfirmation { .. }), "{err}");
    }

    /// Access is read from the PARSED statement. A proposal whose recorded
    /// kind and access both claim "read" but whose statement is a DELETE is
    /// still refused.
    #[test]
    fn the_gate_reads_the_statement_not_the_label() {
        let statement = crate::parse(r#"DELETE FROM EDGES WHERE entity = "John Coyle";"#).unwrap();
        let forged = Proposal {
            lql: "DESCRIBE \"John Coyle\";".into(),
            statement,
            kind: Kind::Describe,
            access: Access::Read,
            confidence: 1.0,
        };
        let err = execute(&forged, &mut Session::new(), false).unwrap_err();
        assert!(matches!(err, NlError::NeedsConfirmation { .. }));
    }

    #[test]
    fn a_read_is_not_held_for_confirmation() {
        let p = Proposal {
            lql: "STATS;".into(),
            statement: crate::parse("STATS;").unwrap(),
            kind: Kind::Stats,
            access: Access::Read,
            confidence: 1.0,
        };
        // No vindex is loaded, so the executor itself errors — but it is an
        // executor error, which proves the gate let a read through.
        assert!(
            !matches!(
                execute(&p, &mut Session::new(), false),
                Err(NlError::NeedsConfirmation { .. })
            ),
            "a read was held for confirmation"
        );
    }

    #[test]
    fn transport_failure_is_reported_as_itself() {
        struct Down;
        impl Transport for Down {
            fn post(&self, _: &Value) -> Result<Value, String> {
                Err("connection refused".into())
            }
        }
        let err = route("stats", &RouterContext::default(), &Down).unwrap_err();
        assert!(matches!(err, NlError::Transport(ref m) if m.contains("refused")));
    }
}
