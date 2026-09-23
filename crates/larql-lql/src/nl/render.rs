//! Selected argument slots → LQL text.
//!
//! Rendering is deliberately dumb string assembly. Whether the result is
//! valid LQL is decided afterwards by the real parser, never here, so this
//! module cannot quietly widen what the language accepts.

use super::catalog::Kind;

/// Argument values chosen by the router, all copied from the request.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Slots {
    pub prompt: Option<String>,
    pub entity: Option<String>,
    pub relation: Option<String>,
    pub target: Option<String>,
    /// What the statement reads from or acts on: a vindex, model id, or patch.
    pub source: Option<String>,
    /// What the statement writes or creates.
    pub dest: Option<String>,
    /// DIFF only: the second vindex to compare against.
    pub other: Option<String>,
    pub layer: Option<u32>,
    pub top: Option<u32>,
    pub limit: Option<u32>,
    /// COMPILE only: `INTO MODEL` rather than `INTO VINDEX`.
    pub compile_into_model: bool,
    /// EXPLAIN only: explain a WALK rather than an INFER.
    pub explain_walk: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Rendered {
    Complete(String),
    /// A statement was chosen but a value it requires was not found in the
    /// request. The template shows the user what to fill in; it never parses,
    /// so it can never execute.
    Incomplete {
        template: String,
        missing: Vec<&'static str>,
    },
}

/// Quote a value as an LQL string, or refuse. A value carrying a quote or a
/// line break is refused rather than escaped: it came from free text, and an
/// escaping rule this module invents is one the parser may not share.
fn q(v: &Option<String>) -> Option<String> {
    let v = v.as_ref()?;
    if v.is_empty() || v.contains('"') || v.contains('\n') {
        return None;
    }
    Some(format!("\"{v}\""))
}

struct Builder {
    parts: Vec<String>,
    missing: Vec<&'static str>,
}

impl Builder {
    fn new(head: &str) -> Self {
        Builder {
            parts: vec![head.to_string()],
            missing: Vec::new(),
        }
    }
    fn lit(mut self, s: &str) -> Self {
        self.parts.push(s.to_string());
        self
    }
    fn req_str(mut self, name: &'static str, v: &Option<String>) -> Self {
        match q(v) {
            Some(s) => self.parts.push(s),
            None => {
                self.parts.push(format!("<{name}>"));
                self.missing.push(name);
            }
        }
        self
    }
    fn req_num(mut self, name: &'static str, v: Option<u32>) -> Self {
        match v {
            Some(n) => self.parts.push(n.to_string()),
            None => {
                self.parts.push(format!("<{name}>"));
                self.missing.push(name);
            }
        }
        self
    }
    fn opt_num(mut self, kw: &str, v: Option<u32>) -> Self {
        if let Some(n) = v {
            self.parts.push(format!("{kw} {n}"));
        }
        self
    }
    fn finish(self) -> Rendered {
        let text = format!("{};", self.parts.join(" "));
        if self.missing.is_empty() {
            Rendered::Complete(text)
        } else {
            Rendered::Incomplete {
                template: text,
                missing: self.missing,
            }
        }
    }
}

/// WHERE clause over entity and (optionally) relation. Entity is required:
/// a DELETE or UPDATE with no subject would match everything.
fn where_entity(b: Builder, s: &Slots) -> Builder {
    let mut b = b.lit("WHERE entity =").req_str("entity", &s.entity);
    if let Some(r) = q(&s.relation) {
        b = b.lit("AND relation =").lit(&r);
    }
    b
}

pub fn render(kind: Kind, s: &Slots) -> Rendered {
    match kind {
        Kind::Extract => Builder::new("EXTRACT MODEL")
            .req_str("model", &s.source)
            .lit("INTO")
            .req_str("output", &s.dest)
            .finish(),
        Kind::Compile => Builder::new("COMPILE CURRENT INTO")
            .lit(if s.compile_into_model {
                "MODEL"
            } else {
                "VINDEX"
            })
            .req_str("output", &s.dest)
            .finish(),
        Kind::Diff => {
            let b = Builder::new("DIFF").req_str("vindex", &s.source);
            match q(&s.other) {
                Some(other) => b.lit(&other).finish(),
                None => b.lit("CURRENT").finish(),
            }
        }
        Kind::Use => {
            let is_model_id = s
                .source
                .as_deref()
                .is_some_and(|p| p.contains('/') && !p.ends_with(".vindex") && !p.ends_with('/'));
            Builder::new(if is_model_id { "USE MODEL" } else { "USE" })
                .req_str("vindex", &s.source)
                .finish()
        }
        Kind::Walk => Builder::new("WALK")
            .req_str("prompt", &s.prompt)
            .opt_num("TOP", s.top)
            .finish(),
        Kind::Infer => Builder::new("INFER")
            .req_str("prompt", &s.prompt)
            .opt_num("TOP", s.top)
            .finish(),
        Kind::Select => {
            let mut b = Builder::new("SELECT * FROM EDGES");
            let e = q(&s.entity);
            let r = q(&s.relation);
            match (e, r) {
                (Some(e), Some(r)) => b = b.lit(&format!("WHERE entity = {e} AND relation = {r}")),
                (Some(e), None) => b = b.lit(&format!("WHERE entity = {e}")),
                (None, Some(r)) => b = b.lit(&format!("WHERE relation = {r}")),
                (None, None) => {}
            }
            b.opt_num("LIMIT", s.limit).finish()
        }
        Kind::Describe => Builder::new("DESCRIBE")
            .req_str("entity", &s.entity)
            .opt_num("AT LAYER", s.layer)
            .finish(),
        Kind::Explain => Builder::new(if s.explain_walk {
            "EXPLAIN WALK"
        } else {
            "EXPLAIN INFER"
        })
        .req_str("prompt", &s.prompt)
        .opt_num("TOP", s.top)
        .finish(),
        Kind::Insert => {
            let b = Builder::new("INSERT INTO EDGES (entity, relation, target) VALUES (")
                .req_str("entity", &s.entity)
                .lit(",")
                .req_str("relation", &s.relation)
                .lit(",")
                .req_str("target", &s.target)
                .lit(")");
            b.opt_num("AT LAYER", s.layer).finish()
        }
        Kind::Delete => where_entity(Builder::new("DELETE FROM EDGES"), s).finish(),
        Kind::Update => {
            let b = Builder::new("UPDATE EDGES SET target =").req_str("target", &s.target);
            where_entity(b, s).finish()
        }
        Kind::Merge => Builder::new("MERGE").req_str("vindex", &s.source).finish(),
        Kind::Rebalance => Builder::new("REBALANCE").finish(),
        Kind::ShowRelations => Builder::new("SHOW RELATIONS")
            .opt_num("AT LAYER", s.layer)
            .finish(),
        Kind::ShowLayers => Builder::new("SHOW LAYERS").finish(),
        Kind::ShowFeatures => Builder::new("SHOW FEATURES")
            .req_num("layer", s.layer)
            .opt_num("LIMIT", s.limit)
            .finish(),
        Kind::ShowEntities => Builder::new("SHOW ENTITIES")
            .opt_num("LIMIT", s.limit)
            .finish(),
        Kind::ShowModels => Builder::new("SHOW MODELS").finish(),
        Kind::Stats => Builder::new("STATS").finish(),
        Kind::ShowCompactStatus => Builder::new("SHOW COMPACT STATUS").finish(),
        Kind::CompactInto => Builder::new("COMPACT INTO VINDEX")
            .req_str("output", &s.dest)
            .finish(),
        Kind::CompactMinor => Builder::new("COMPACT MINOR").finish(),
        Kind::CompactMajor => Builder::new("COMPACT MAJOR").finish(),
        Kind::BeginPatch => Builder::new("BEGIN PATCH")
            .req_str("patch", &s.dest.clone().or_else(|| s.source.clone()))
            .finish(),
        Kind::SavePatch => Builder::new("SAVE PATCH").finish(),
        Kind::ApplyPatch => Builder::new("APPLY PATCH")
            .req_str("patch", &s.source)
            .finish(),
        Kind::ShowPatches => Builder::new("SHOW PATCHES").finish(),
        Kind::RemovePatch => Builder::new("REMOVE PATCH")
            .req_str("patch", &s.source)
            .finish(),
        Kind::Trace => {
            let b = Builder::new("TRACE").req_str("prompt", &s.prompt);
            match q(&s.target) {
                Some(t) => b.lit("FOR").lit(&t).finish(),
                None => b.finish(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nl::catalog::{access, kind_of, Access, ALL};

    fn full() -> Slots {
        Slots {
            prompt: Some("The capital of France is".into()),
            entity: Some("France".into()),
            relation: Some("capital".into()),
            target: Some("Paris".into()),
            source: Some("a.vindex".into()),
            dest: Some("out.vindex".into()),
            other: Some("b.vindex".into()),
            layer: Some(26),
            top: Some(5),
            limit: Some(10),
            compile_into_model: false,
            explain_walk: false,
        }
    }

    /// Every routable kind, fully slotted, renders LQL the REAL parser accepts
    /// and that parses back to the same kind. A template that drifts from the
    /// grammar fails here, offline, rather than as a routing miss in the eval.
    #[test]
    fn every_kind_renders_lql_the_parser_accepts() {
        for kind in ALL {
            let Rendered::Complete(lql) = render(kind, &full()) else {
                panic!("{kind:?} was incomplete with every slot filled");
            };
            let stmt = crate::parse(&lql)
                .unwrap_or_else(|e| panic!("{kind:?}: `{lql}` did not parse: {e}"));
            assert_eq!(
                kind_of(&stmt),
                Some(kind),
                "`{lql}` parsed as a different statement"
            );
        }
    }

    #[test]
    fn the_alternate_forms_parse_too() {
        let mut s = full();
        s.compile_into_model = true;
        s.explain_walk = true;
        s.source = Some("google/gemma-3-4b-it".into());
        for kind in [Kind::Compile, Kind::Explain, Kind::Use] {
            let Rendered::Complete(lql) = render(kind, &s) else {
                panic!()
            };
            crate::parse(&lql).unwrap_or_else(|e| panic!("`{lql}`: {e}"));
        }
    }

    #[test]
    fn a_missing_required_value_never_renders_runnable_lql() {
        for kind in ALL {
            if let Rendered::Incomplete { template, missing } = render(kind, &Slots::default()) {
                assert!(!missing.is_empty());
                assert!(
                    crate::parse(&template).is_err(),
                    "{kind:?} template `{template}` parsed"
                );
            }
        }
    }

    /// DELETE and UPDATE with no entity would match everything. They must come
    /// back incomplete, not as `DELETE FROM EDGES;`.
    #[test]
    fn delete_and_update_without_a_subject_are_incomplete() {
        for kind in [Kind::Delete, Kind::Update] {
            let mut s = full();
            s.entity = None;
            assert!(
                matches!(render(kind, &s), Rendered::Incomplete { .. }),
                "{kind:?}"
            );
        }
    }

    #[test]
    fn a_value_containing_a_quote_is_refused_not_escaped() {
        let mut s = full();
        s.entity = Some("Fr\"ance".into());
        assert!(matches!(
            render(Kind::Describe, &s),
            Rendered::Incomplete { .. }
        ));
    }

    #[test]
    fn rendered_writes_classify_as_writes() {
        for kind in [
            Kind::Insert,
            Kind::Delete,
            Kind::Update,
            Kind::Compile,
            Kind::Merge,
        ] {
            let Rendered::Complete(lql) = render(kind, &full()) else {
                panic!()
            };
            assert_eq!(access(&crate::parse(&lql).unwrap()), Access::Write, "{lql}");
        }
    }
}
