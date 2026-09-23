//! One request to TypeSafe's System One (`jev-latest`): a `choice` over every
//! routable statement plus "none", and a `choice` per argument slot over the
//! candidates code found in the request. Questions run in parallel and cannot
//! see each other, so every slot is asked up front and only the chosen
//! statement's slots are read back.

use serde_json::{json, Map, Value};

use super::candidates::Candidates;
use super::catalog::{ALL, NONE_KEY};

pub const DEFAULT_URL: &str = "https://api.typesafe.ai/v1/systemone";
pub const MODEL: &str = "jev-latest";

/// Sends a request body and returns the parsed response body. A trait so the
/// router can be tested without the network.
pub trait Transport {
    fn post(&self, body: &Value) -> Result<Value, String>;
}

/// The real transport. The key is read from the environment and never logged.
pub struct HttpTransport {
    url: String,
    key: String,
    client: reqwest::blocking::Client,
}

impl HttpTransport {
    pub fn from_env() -> Result<Self, String> {
        let key = std::env::var("TYPESAFE_API_KEY").map_err(|_| {
            "TYPESAFE_API_KEY is not set — `larql ask` sends the request to api.typesafe.ai"
                .to_string()
        })?;
        if key.trim().is_empty() {
            return Err("TYPESAFE_API_KEY is empty".to_string());
        }
        let url = std::env::var("TYPESAFE_URL").unwrap_or_else(|_| DEFAULT_URL.to_string());
        let url = validate_endpoint(&url)?;
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            // The key rides every request. A redirect could move it — reqwest
            // strips Authorization across hosts, but not on a same-host
            // https -> http downgrade — and the API never needs to redirect.
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|e| format!("building HTTP client: {e}"))?;
        Ok(HttpTransport { url, key, client })
    }
}

/// Where the bearer key may be sent. TYPESAFE_URL is an override for pointing
/// at a self-hosted Jev-compatible server or a local mock, and every request
/// carries TYPESAFE_API_KEY — so a mistyped or hostile value would hand the
/// key to whoever answers. HTTPS always; plain HTTP only to loopback, which is
/// the one legitimate unencrypted case (a mock or a server on this machine).
pub fn validate_endpoint(raw: &str) -> Result<String, String> {
    let url =
        reqwest::Url::parse(raw.trim()).map_err(|e| format!("TYPESAFE_URL is not a URL ({e})"))?;
    if !url.username().is_empty() || url.password().is_some() {
        return Err("TYPESAFE_URL must not carry credentials in the URL".to_string());
    }
    let host = url.host_str().unwrap_or_default();
    match url.scheme() {
        "https" if !host.is_empty() => Ok(url.to_string()),
        "http" if matches!(host, "localhost" | "127.0.0.1" | "[::1]" | "::1") => Ok(url.to_string()),
        "http" => Err(format!(
            "refusing to send the TypeSafe key over plain http to {host}; use https (http is allowed only to localhost)"
        )),
        other => Err(format!("TYPESAFE_URL scheme {other:?} is not allowed; use https")),
    }
}

impl Transport for HttpTransport {
    fn post(&self, body: &Value) -> Result<Value, String> {
        let resp = self
            .client
            .post(&self.url)
            .bearer_auth(&self.key)
            .json(body)
            .send()
            .map_err(|e| format!("request to {} failed: {e}", self.url))?;
        let status = resp.status();
        // Read as text first: an error page or an empty body must surface as
        // itself, not as an opaque JSON parse error.
        let text = resp.text().map_err(|e| format!("reading response: {e}"))?;
        if !status.is_success() {
            let head: String = text.chars().take(300).collect();
            return Err(format!("{} returned {status}: {head}", self.url));
        }
        serde_json::from_str(&text).map_err(|e| {
            let head: String = text.chars().take(200).collect();
            format!("response was not JSON ({e}): {head}")
        })
    }
}

/// Which candidates were offered for each slot, so an answer key can be
/// mapped back to a value.
#[derive(Debug, Clone, Default)]
pub struct Offered {
    pub slots: Vec<(&'static str, Vec<String>)>,
}

impl Offered {
    pub fn options(&self, slot: &str) -> Option<&[String]> {
        self.slots
            .iter()
            .find(|(s, _)| *s == slot)
            .map(|(_, v)| v.as_slice())
    }
}

fn merged(sources: &[&[String]]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for src in sources {
        for v in src.iter() {
            // "none" is the decline key; a candidate spelled the same would be
            // indistinguishable from declining.
            if v != NONE_KEY && !out.contains(v) && out.len() < 250 {
                out.push(v.clone());
            }
        }
    }
    out
}

fn choice(instructions: &str, options: &[String], none_means: &str) -> Value {
    let mut criteria = Map::new();
    for o in options {
        criteria.insert(o.clone(), Value::Null);
    }
    criteria.insert(NONE_KEY.to_string(), Value::String(none_means.to_string()));
    json!({ "type": "choice", "instructions": instructions, "criteria": criteria })
}

/// Build the single request, and record what each slot was offered.
pub fn build_request(
    request: &str,
    c: &Candidates,
    relation_roster: &[String],
) -> (Value, Offered) {
    let mut questions = Map::new();

    let mut statements = Map::new();
    for k in ALL {
        statements.insert(
            k.key().to_string(),
            Value::String(k.description().to_string()),
        );
    }
    statements.insert(
        NONE_KEY.to_string(),
        Value::String(
            "No LQL statement serves this request: it is not about inspecting, querying or editing \
             a model's knowledge or its vindex files, or it asks for something LQL cannot do."
                .to_string(),
        ),
    );
    questions.insert(
        "statement".to_string(),
        json!({
            "type": "choice",
            "instructions": "Which LQL statement does `request` ask for? LQL inspects, queries and edits \
                             what a transformer model knows, stored as a vindex.",
            "criteria": statements,
        }),
    );

    let numbers: Vec<String> = c.numbers.iter().map(u32::to_string).collect();
    let slot_specs: [(&'static str, Vec<String>, &str); 9] = [
        (
            "prompt",
            merged(&[&c.quoted, &c.tails]),
            "Which span is the exact text the user wants run through the model — the prompt to walk, \
             infer, explain or trace?",
        ),
        (
            "entity",
            merged(&[&c.quoted, &c.proper]),
            "Which span names the entity (the subject) the request is about — the thing to describe, \
             or whose fact is inserted, changed or deleted?",
        ),
        (
            "relation",
            merged(&[relation_roster, &c.quoted, &c.words]),
            "Which span is a relation the request explicitly NAMES as the link between an entity and \
             its target, e.g. capital-of or lives-in? A verb describing what the user wants done \
             (knows, erase, change) is not a relation. If no relation is named, answer none.",
        ),
        (
            "target",
            merged(&[&c.quoted, &c.proper]),
            "Which span is the target value of the fact — what the entity's relation should point to, \
             or the token to track in a trace?",
        ),
        (
            "source",
            merged(&[&c.paths, &c.quoted]),
            "Which span is the vindex, model id or patch file the request reads from or acts on?",
        ),
        (
            "dest",
            merged(&[&c.paths, &c.quoted]),
            "Which span is the file or directory the request wants written or created?",
        ),
        (
            "other",
            merged(&[&c.paths, &c.quoted]),
            "Which span is a SECOND vindex the request wants to compare against?",
        ),
        (
            "layer",
            numbers.clone(),
            "Which number is the layer the request refers to?",
        ),
        (
            "count",
            numbers,
            "Which number is how many results the request wants (top N, or a limit)?",
        ),
    ];

    let mut offered = Offered::default();
    for (slot, options, instructions) in slot_specs {
        if !options.is_empty() {
            questions.insert(
                slot.to_string(),
                choice(instructions, &options, "The request does not state this."),
            );
        }
        offered.slots.push((slot, options));
    }

    questions.insert(
        "into_model".to_string(),
        json!({
            "type": "noul",
            "instructions": "Does the request want a model checkpoint file (safetensors or GGUF) rather than a vindex?",
            "criteria": { "true": "It asks for a model file or checkpoint.", "false": "It does not." },
        }),
    );
    questions.insert(
        "explain_walk".to_string(),
        json!({
            "type": "noul",
            "instructions": "Does the request ask about a feature walk (which features fire) rather than about the model's prediction?",
            "criteria": { "true": "It is about which features fire.", "false": "It is about the prediction." },
        }),
    );

    let body = json!({
        "model": MODEL,
        "state": { "request": request },
        "questions": questions,
    });
    (body, offered)
}

/// One `choice` answer.
#[derive(Debug, Clone, PartialEq)]
pub struct ChoiceAnswer {
    pub choice: String,
    pub confidence: f64,
}

pub fn read_choice(answers: &Value, q: &str) -> Option<ChoiceAnswer> {
    let a = answers.get(q)?;
    Some(ChoiceAnswer {
        choice: a.get("choice")?.as_str()?.to_string(),
        confidence: a.get("confidence").and_then(Value::as_f64).unwrap_or(0.0),
    })
}

pub fn read_noul(answers: &Value, q: &str) -> Option<f64> {
    answers.get(q)?.get("noul")?.as_f64()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nl::candidates::extract;

    #[test]
    fn the_key_is_only_sent_over_https_or_to_loopback() {
        assert!(validate_endpoint(DEFAULT_URL).is_ok());
        assert!(validate_endpoint("https://jev.internal.example/v1/systemone").is_ok());
        assert!(validate_endpoint("http://127.0.0.1:8011/v1/systemone").is_ok());
        assert!(validate_endpoint("http://localhost:8011/v1/systemone").is_ok());
        assert!(validate_endpoint("http://[::1]:8011/v1/systemone").is_ok());
        // Plain http anywhere else would put the bearer key on the wire.
        assert!(validate_endpoint("http://api.typesafe.ai/v1/systemone").is_err());
        assert!(validate_endpoint("http://10.0.0.5/v1/systemone").is_err());
        // A lookalike that merely STARTS with a loopback name is not loopback.
        assert!(validate_endpoint("http://localhost.evil.example/v1/systemone").is_err());
        assert!(validate_endpoint("http://127.0.0.1.evil.example/v1/systemone").is_err());
        assert!(validate_endpoint("ftp://api.typesafe.ai/").is_err());
        assert!(validate_endpoint("not a url").is_err());
        assert!(validate_endpoint("https://user:pw@api.typesafe.ai/v1/systemone").is_err());
    }

    #[test]
    fn the_request_offers_every_statement_and_a_decline() {
        let (body, _) = build_request("describe France", &extract("describe France"), &[]);
        let crit = &body["questions"]["statement"]["criteria"];
        for k in ALL {
            assert!(crit.get(k.key()).is_some(), "{} missing", k.key());
        }
        assert!(
            crit.get(NONE_KEY).is_some(),
            "the router must be able to decline"
        );
    }

    #[test]
    fn a_slot_with_no_candidates_is_not_asked() {
        let (body, offered) = build_request("show the layers", &extract("show the layers"), &[]);
        assert!(body["questions"].get("layer").is_none());
        assert!(offered.options("layer").is_some_and(<[String]>::is_empty));
    }

    #[test]
    fn every_offered_slot_can_decline() {
        let r = r#"insert "Atlantis" capital-of "Poseidon" at layer 24"#;
        let (body, _) = build_request(r, &extract(r), &[]);
        for (name, q) in body["questions"].as_object().unwrap() {
            if q["type"] == "choice" {
                assert!(
                    q["criteria"].get(NONE_KEY).is_some(),
                    "{name} cannot decline"
                );
            }
        }
    }

    #[test]
    fn a_candidate_spelled_none_is_not_confused_with_declining() {
        let r = r#"describe "none""#;
        let (_, offered) = build_request(r, &extract(r), &[]);
        assert!(!offered
            .options("entity")
            .unwrap()
            .iter()
            .any(|o| o == NONE_KEY));
    }

    #[test]
    fn the_relation_roster_is_offered_before_free_words() {
        let r = "which entities have the seat relation";
        let roster = vec!["capital".to_string()];
        let (_, offered) = build_request(r, &extract(r), &roster);
        assert_eq!(offered.options("relation").unwrap()[0], "capital");
    }
}
