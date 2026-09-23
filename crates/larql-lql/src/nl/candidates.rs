//! Candidate argument values found in an English request by code.
//!
//! The router cannot write text; it can only CHOOSE among options it is
//! offered. So every string or number an LQL statement needs must first be
//! proposed here, as a span that actually occurs in the request, and the
//! router selects one (or "none"). A value that is not proposed cannot be
//! chosen — which is the coverage limit of this design, and why these
//! extractors favour recall over precision.

/// Upper bound per slot. The router accepts 255 options; this leaves room for
/// "none" and keeps each question small.
pub const MAX_PER_SLOT: usize = 60;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Candidates {
    /// Text inside matching quotes, in order of appearance.
    pub quoted: Vec<String>,
    /// Runs of capitalised words ("France", "New York", "Atlantis").
    pub proper: Vec<String>,
    /// Path- or model-id-shaped tokens (`x.vindex`, `a/b`, `out/`).
    pub paths: Vec<String>,
    /// Lower-case content words and hyphenated tokens (relation names).
    pub words: Vec<String>,
    /// The remainder of the request after a lead-in ("after", "for", ":").
    pub tails: Vec<String>,
    /// Integers that appear as standalone tokens.
    pub numbers: Vec<u32>,
}

const QUOTE_PAIRS: [(char, char); 3] = [('"', '"'), ('\'', '\''), ('\u{201C}', '\u{201D}')];

const STOPWORDS: [&str; 40] = [
    "the", "and", "for", "with", "that", "this", "what", "which", "does", "from", "into", "about",
    "show", "list", "give", "tell", "please", "model", "vindex", "can", "you", "all", "any", "are",
    "how", "its", "their", "there", "have", "has", "was", "were", "will", "would", "should",
    "could", "then", "than", "when", "where",
];

const TAIL_LEADS: [&str; 7] = [
    "after", "for", "on", "prompt", "text", "predict", "continue",
];

const PATH_SUFFIXES: [&str; 5] = [".vindex", ".vlp", ".gguf", ".safetensors", ".json"];

fn push_unique<T: PartialEq>(v: &mut Vec<T>, x: T) {
    if v.len() < MAX_PER_SLOT && !v.contains(&x) {
        v.push(x);
    }
}

fn trim_punct(tok: &str) -> &str {
    tok.trim_matches(|c: char| matches!(c, ',' | ';' | ':' | '?' | '!' | '(' | ')' | '"' | '\''))
        .trim_end_matches('.')
}

fn quoted_spans(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for (open, close) in QUOTE_PAIRS {
        let mut rest = text;
        while let Some(start) = rest.find(open) {
            let after = &rest[start + open.len_utf8()..];
            // An apostrophe inside a word ("model's") is not a quote. Skip
            // past it BEFORE looking for a closing mark — pairing it first
            // swallows the opening quote of the next real quoted span.
            if open == '\'' && rest[..start].ends_with(char::is_alphanumeric) {
                rest = after;
                continue;
            }
            let Some(end) = after.find(close) else { break };
            let inner = after[..end].trim();
            if !inner.is_empty() {
                push_unique(&mut out, inner.to_string());
            }
            rest = &after[end + close.len_utf8()..];
        }
    }
    out
}

fn is_path_like(tok: &str) -> bool {
    let lower = tok.to_ascii_lowercase();
    if PATH_SUFFIXES.iter().any(|s| lower.ends_with(s)) {
        return true;
    }
    if tok.ends_with('/') && tok.len() > 1 {
        return true;
    }
    // model id: "google/gemma-3-4b-it" — one or more slashes, no spaces,
    // not a URL scheme fragment.
    tok.contains('/') && !tok.starts_with('/') && !tok.contains("://") && tok.len() > 2
}

/// "France's" names France. Without this the possessive becomes part of the
/// entity, and `WHERE entity = "France's"` matches nothing.
fn strip_possessive(tok: &str) -> &str {
    tok.strip_suffix("'s")
        .or_else(|| tok.strip_suffix("\u{2019}s"))
        .unwrap_or(tok)
}

fn starts_upper(tok: &str) -> bool {
    tok.chars().next().is_some_and(char::is_uppercase)
}

/// Propose every candidate value the request could be supplying.
pub fn extract(request: &str) -> Candidates {
    let mut c = Candidates {
        quoted: quoted_spans(request),
        ..Candidates::default()
    };

    let tokens: Vec<&str> = request.split_whitespace().collect();
    let clean: Vec<&str> = tokens
        .iter()
        .map(|t| strip_possessive(trim_punct(t)))
        .collect();

    for t in &clean {
        if t.is_empty() {
            continue;
        }
        if is_path_like(t) {
            push_unique(&mut c.paths, (*t).to_string());
        }
        if let Ok(n) = t.parse::<u32>() {
            push_unique(&mut c.numbers, n);
        }
        let lower = t.to_ascii_lowercase();
        let wordish = t
            .chars()
            .all(|ch| ch.is_alphanumeric() || ch == '-' || ch == '_');
        if wordish
            && !starts_upper(t)
            && t.len() >= 3
            && t.parse::<u32>().is_err()
            && !STOPWORDS.contains(&lower.as_str())
        {
            push_unique(&mut c.words, lower);
        }
    }

    // Capitalised runs. A run that begins the request also yields itself
    // without its first word, since sentence-initial capitals are not names.
    let mut i = 0;
    while i < clean.len() {
        if starts_upper(clean[i]) && !is_path_like(clean[i]) {
            let start = i;
            while i < clean.len() && starts_upper(clean[i]) && !is_path_like(clean[i]) {
                i += 1;
            }
            let run = clean[start..i].join(" ");
            push_unique(&mut c.proper, run);
            if start == 0 && i - start > 1 {
                push_unique(&mut c.proper, clean[start + 1..i].join(" "));
            }
        } else {
            i += 1;
        }
    }

    // Tails: the rest of the request after a lead-in word or a colon. These
    // are how an unquoted prompt ("predict the next word after The capital of
    // France is") becomes a selectable span.
    for (idx, raw) in tokens.iter().enumerate() {
        let lead = trim_punct(raw).to_ascii_lowercase();
        if (TAIL_LEADS.contains(&lead.as_str()) || raw.ends_with(':')) && idx + 1 < tokens.len() {
            let tail = tokens[idx + 1..].join(" ");
            let tail = tail.trim().trim_end_matches(['?', '.', '!']).trim();
            let tail = tail.trim_matches(|ch| matches!(ch, '"' | '\'' | '\u{201C}' | '\u{201D}'));
            if !tail.is_empty() {
                push_unique(&mut c.tails, tail.to_string());
            }
        }
    }

    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quoted_text_is_proposed_verbatim() {
        let c = extract(r#"run "The capital of France is" and show the top 5"#);
        assert_eq!(c.quoted, vec!["The capital of France is"]);
        assert!(c.numbers.contains(&5));
    }

    #[test]
    fn an_apostrophe_is_not_a_quote() {
        let c = extract("what is the model's view of 'Atlantis'");
        assert_eq!(c.quoted, vec!["Atlantis"]);
    }

    #[test]
    fn paths_and_model_ids_are_proposed() {
        let c = extract("extract google/gemma-3-4b-it into gemma3-4b.vindex please");
        assert!(c.paths.contains(&"google/gemma-3-4b-it".to_string()));
        assert!(c.paths.contains(&"gemma3-4b.vindex".to_string()));
    }

    #[test]
    fn sentence_initial_capital_does_not_swallow_the_name() {
        let c = extract("Describe France");
        assert!(c.proper.contains(&"France".to_string()), "{:?}", c.proper);
    }

    #[test]
    fn multi_word_names_stay_together() {
        let c = extract("what does it know about New York City");
        assert!(
            c.proper.contains(&"New York City".to_string()),
            "{:?}",
            c.proper
        );
    }

    #[test]
    fn an_unquoted_prompt_is_reachable_as_a_tail() {
        let c = extract("predict what comes after The capital of France is");
        assert!(
            c.tails.contains(&"The capital of France is".to_string()),
            "{:?}",
            c.tails
        );
    }

    #[test]
    fn a_possessive_names_the_entity() {
        let c = extract("change it so France's capital is Lyon");
        assert!(c.proper.contains(&"France".to_string()), "{:?}", c.proper);
        assert!(!c.proper.iter().any(|p| p.contains("'s")), "{:?}", c.proper);
    }

    #[test]
    fn relation_words_exclude_stopwords() {
        let c = extract("delete the lives-in relation for John");
        assert!(c.words.contains(&"lives-in".to_string()));
        assert!(!c.words.contains(&"the".to_string()));
    }

    #[test]
    fn every_slot_is_capped() {
        let many: String = (0..500).map(|i| format!("w{i}x ")).collect();
        let c = extract(&many);
        assert!(c.words.len() <= MAX_PER_SLOT);
    }
}
