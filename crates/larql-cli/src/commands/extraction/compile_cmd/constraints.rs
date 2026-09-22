//! Refuse to write a checkpoint that reached its target by breaking the model.
//!
//! `install_edge` writes an edge at whatever magnitude `--alpha` asks for, and until now nothing
//! downstream asked what that cost. Measured on 2026-09-20 against `gemma-4-E2B-it`: at alpha 3
//! the compiled checkpoint answers the installed fact AND has forgotten the capital of Italy, and
//! `larql compile` reports success. The operating window found by the M17 sweep is narrow
//! (1.5 <= alpha <= 2.0) and the tool's own default of 0.240 sits four orders of magnitude below
//! it, so "the caller passed an alpha" is not evidence that the alpha was survivable.
//!
//! `scripts/experiments/surgical-insert.py` has refused in this situation since it was written:
//! it accepts an alpha only when every installed prompt reaches its probability AND every control
//! keeps its top-1 AND the neutral paragraph's perplexity ratio stays under a bound, and it exits
//! non-zero having written nothing when no alpha clears all three. This brings the first two of
//! those to the shipping compiler.
//!
//! ## Why top-1 and not a probability threshold
//!
//! `larql_inference::forward::predict` is systematically "peakier" than HF transformers' forward
//! pass on the same weights — that is why `--max-iters` defaults to 0 and the balancer is opt-in.
//! A control gate written as `p(expected) >= 0.5` would inherit exactly that calibration gap and
//! reject or accept on a number that does not describe deployed inference. Argmax is far more
//! robust to peakiness than probability mass is: a distribution can sharpen a long way without
//! moving which token is on top. So the gate asks "does this prompt still answer Rome?", which is
//! the same primitive `surgical-insert.py` uses (`control_top1_intact`).
//!
//! ## Why the judgment is a pure function
//!
//! Measuring a control needs a model and a forward pass. Deciding what the measurements MEAN does
//! not, and keeping the decision separate is what makes it testable without a 10 GB checkpoint.
//! `judge()` takes measurements and returns a verdict; `measure_controls()` produces the
//! measurements. Everything that can be got wrong about the RULE is in the half with no I/O.

use std::fmt;

/// A control prompt and the answer it produced BEFORE the edge was installed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ControlSpec {
    pub prompt: String,
    pub expected: String,
}

/// What a control prompt answered AFTER the edge was installed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ControlResult {
    pub prompt: String,
    pub expected: String,
    pub got: String,
    pub intact: bool,
}

/// What the balancer did, as distinct from what it was asked to do.
///
/// `Disabled` and `NotConverged` are NOT the same state and must never collapse into one.
/// `--max-iters 0` is the documented default and makes no claim about the install; running the
/// balancer and failing to land in `[floor, ceiling]` is a failed claim, and before this module
/// existed both of them fell out of the loop into the same unconditional write.
#[derive(Debug, Clone, PartialEq)]
pub enum BalancerOutcome {
    Disabled,
    Converged { prob: f64 },
    NotConverged { iters: u32, last_prob: f64, floor: f64, ceiling: f64 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    Pass,
    Refuse(String),
}

impl fmt::Display for Verdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Verdict::Pass => write!(f, "pass"),
            Verdict::Refuse(why) => write!(f, "refuse: {why}"),
        }
    }
}

/// Parse a `--control "prompt=>expected"` argument.
///
/// `=>` rather than `=` because a control prompt is a sentence and sentences contain `=` far
/// less often than they contain nothing at all; splitting on the FIRST `=>` keeps an expected
/// answer that itself contains one from truncating the prompt.
pub fn parse_control(raw: &str) -> Result<ControlSpec, String> {
    let (prompt, expected) = raw
        .split_once("=>")
        .ok_or_else(|| format!("control {raw:?} has no `=>`; expected \"prompt=>expected answer\""))?;
    let prompt = prompt.trim();
    let expected = expected.trim();
    if prompt.is_empty() {
        return Err(format!("control {raw:?} has an empty prompt"));
    }
    if expected.is_empty() {
        return Err(format!("control {raw:?} has an empty expected answer"));
    }
    Ok(ControlSpec { prompt: prompt.to_string(), expected: expected.to_string() })
}

/// Decide whether this install may be written.
///
/// `requested` is how many controls the CALLER asked for, and is checked against how many were
/// actually measured. That comparison is the point of the parameter: "no control reported a
/// break" is also what you get when no control ran, and an analysis that could not tell those
/// apart reported PASS on an empty comparison earlier in this programme. A gate whose failure
/// mode is indistinguishable from its success mode is not a gate.
pub fn judge(requested: usize, results: &[ControlResult], balancer: &BalancerOutcome) -> Verdict {
    if let BalancerOutcome::NotConverged { iters, last_prob, floor, ceiling } = balancer {
        return Verdict::Refuse(format!(
            "the balancer ran {iters} iteration(s) and never landed in [{floor:.2}, {ceiling:.2}] \
             (last p = {last_prob:.3}). The install did not reach the probability it was asked \
             for, so the checkpoint would misrepresent what was compiled; nothing written"
        ));
    }

    if results.len() != requested {
        return Verdict::Refuse(format!(
            "{requested} control(s) requested but {} measured — refusing rather than reading \
             silence as success; nothing written",
            results.len()
        ));
    }

    let broken: Vec<&ControlResult> = results.iter().filter(|r| !r.intact).collect();
    if !broken.is_empty() {
        let detail = broken
            .iter()
            .map(|r| format!("  {:?} -> expected {:?}, got {:?}", r.prompt, r.expected, r.got))
            .collect::<Vec<_>>()
            .join("\n");
        return Verdict::Refuse(format!(
            "{} of {} control prompt(s) changed their answer, so this alpha reached its target by \
             damaging the model:\n{detail}\nnothing written",
            broken.len(),
            results.len()
        ));
    }

    Verdict::Pass
}

/// What a control prompt answered BEFORE the edge was installed.
///
/// A control only carries information if the base model already gives the expected answer. One
/// that does not is not a strict control — it is a control that cannot fire, and counting it
/// would inflate the apparent coverage of the gate while detecting nothing. The battery in
/// `notebooks/2026-09-16-erasure-readiness-matrix.md` reports attacks that cannot separate a
/// knowing model from a never-knew model as UNINFORMATIVE rather than as passes; this is the
/// same rule applied to controls.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ControlBaseline {
    pub prompt: String,
    pub expected: String,
    pub base_got: String,
}

impl ControlBaseline {
    pub fn usable(&self) -> bool {
        self.base_got == self.expected
    }
}

/// Check the controls describe the base model, BEFORE anything is installed.
///
/// Runs before the edge so a mis-specified control costs the caller an early refusal rather than
/// a compile that fails at the end for a reason that looks like damage.
pub fn judge_baselines(requested: usize, baselines: &[ControlBaseline]) -> Verdict {
    if baselines.len() != requested {
        return Verdict::Refuse(format!(
            "{requested} control(s) requested but {} measured against the base; nothing written",
            baselines.len()
        ));
    }
    let unusable: Vec<&ControlBaseline> = baselines.iter().filter(|b| !b.usable()).collect();
    if !unusable.is_empty() {
        let detail = unusable
            .iter()
            .map(|b| format!("  {:?} -> base already answers {:?}, not {:?}", b.prompt, b.base_got, b.expected))
            .collect::<Vec<_>>()
            .join("\n");
        return Verdict::Refuse(format!(
            "{} control(s) do not describe the BASE model, so they cannot detect damage done to \
             it and would provide coverage that is not there:\n{detail}\n\
             Fix the expected answers (or drop these controls) and re-run; nothing written",
            unusable.len()
        ));
    }
    Verdict::Pass
}

/// The provenance line describing what collateral checking actually happened.
///
/// An install with no controls is a legitimate thing to ask for and an illegitimate thing to
/// leave IMPLICIT: the absence of a failure message reads as a clean bill of health. This states
/// the absence out loud so a receipt cannot be mistaken for one that was checked.
pub fn collateral_provenance(requested: usize, balancer: &BalancerOutcome) -> String {
    let controls = match requested {
        0 => "collateral check: NOT PERFORMED (no --control given; this checkpoint is unverified \
              against any prompt but its own)"
            .to_string(),
        n => format!("collateral check: {n} control prompt(s) kept their top-1 answer"),
    };
    let bal = match balancer {
        BalancerOutcome::Disabled => {
            "balancer: disabled (--max-iters 0); installed at the caller's alpha".to_string()
        }
        BalancerOutcome::Converged { prob } => format!("balancer: converged at p = {prob:.3}"),
        BalancerOutcome::NotConverged { .. } => "balancer: DID NOT CONVERGE".to_string(),
    };
    format!("{controls}\n  {bal}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn intact(p: &str, a: &str) -> ControlResult {
        ControlResult { prompt: p.into(), expected: a.into(), got: a.into(), intact: true }
    }
    fn broken(p: &str, want: &str, got: &str) -> ControlResult {
        ControlResult {
            prompt: p.into(),
            expected: want.into(),
            got: got.into(),
            intact: false,
        }
    }

    fn base(p: &str, want: &str, got: &str) -> ControlBaseline {
        ControlBaseline { prompt: p.into(), expected: want.into(), base_got: got.into() }
    }

    #[test]
    fn a_control_the_base_model_already_fails_is_refused_up_front() {
        // The base says "Roma"; the caller wrote "Rome". This control can never fire, so counting
        // it would report coverage the gate does not have.
        let b = [base("capital of Italy?", "Rome", "Roma")];
        match judge_baselines(1, &b) {
            Verdict::Refuse(why) => {
                assert!(why.contains("do not describe the BASE model"), "{why}");
                assert!(why.contains("Roma"), "refusal must show what the base actually says: {why}");
            }
            v => panic!("an unusable control must refuse before install, got {v}"),
        }
    }

    #[test]
    fn controls_that_describe_the_base_pass_the_precheck() {
        let b = [base("capital of Italy?", "Rome", "Rome"), base("capital of Japan?", "Tokyo", "Tokyo")];
        assert_eq!(judge_baselines(2, &b), Verdict::Pass);
    }

    #[test]
    fn the_precheck_also_refuses_a_count_mismatch() {
        assert!(matches!(judge_baselines(2, &[base("q", "a", "a")]), Verdict::Refuse(_)));
    }

    #[test]
    fn all_controls_intact_and_balancer_disabled_passes() {
        let r = [intact("capital of Italy?", "Rome"), intact("capital of Japan?", "Tokyo")];
        assert_eq!(judge(2, &r, &BalancerOutcome::Disabled), Verdict::Pass);
    }

    #[test]
    fn a_broken_control_refuses_and_names_it() {
        let r = [intact("capital of Japan?", "Tokyo"), broken("capital of Italy?", "Rome", "Ormond")];
        match judge(2, &r, &BalancerOutcome::Disabled) {
            Verdict::Refuse(why) => {
                assert!(why.contains("Italy"), "refusal must name the broken control: {why}");
                assert!(why.contains("Ormond"), "refusal must say what it answered instead: {why}");
                assert!(!why.contains("Japan"), "refusal must not blame the intact control: {why}");
            }
            v => panic!("expected a refusal, got {v}"),
        }
    }

    /// The defect this whole module exists to prevent, in its purest form: an empty measurement
    /// set must not read as a clean one.
    #[test]
    fn zero_measured_against_a_nonzero_request_refuses() {
        match judge(3, &[], &BalancerOutcome::Disabled) {
            Verdict::Refuse(why) => assert!(
                why.contains("3 control(s) requested but 0 measured"),
                "refusal must say the count did not match: {why}"
            ),
            v => panic!("silence must not read as success, got {v}"),
        }
    }

    #[test]
    fn asking_for_no_controls_is_allowed_but_says_so() {
        assert_eq!(judge(0, &[], &BalancerOutcome::Disabled), Verdict::Pass);
        let line = collateral_provenance(0, &BalancerOutcome::Disabled);
        assert!(line.contains("NOT PERFORMED"), "absence must be explicit: {line}");
    }

    /// `Disabled` and `NotConverged` both fall out of the balancer loop. Only one of them is a
    /// failure, and before this function existed both wrote the checkpoint.
    #[test]
    fn a_disabled_balancer_is_not_a_failed_one() {
        assert_eq!(judge(0, &[], &BalancerOutcome::Disabled), Verdict::Pass);
        let nc = BalancerOutcome::NotConverged {
            iters: 12,
            last_prob: 0.07,
            floor: 0.40,
            ceiling: 0.85,
        };
        match judge(0, &[], &nc) {
            Verdict::Refuse(why) => {
                assert!(why.contains("12 iteration"), "refusal must say how hard it tried: {why}");
                assert!(why.contains("0.070"), "refusal must report where it got to: {why}");
            }
            v => panic!("a balancer that never converged must refuse, got {v}"),
        }
    }

    /// Non-convergence outranks intact controls: the install failed on its own terms, and a model
    /// that kept every control while never learning the fact is not a success.
    #[test]
    fn non_convergence_refuses_even_when_every_control_is_intact() {
        let r = [intact("capital of Italy?", "Rome")];
        let nc = BalancerOutcome::NotConverged {
            iters: 5,
            last_prob: 0.01,
            floor: 0.40,
            ceiling: 0.85,
        };
        assert!(matches!(judge(1, &r, &nc), Verdict::Refuse(_)));
    }

    #[test]
    fn parses_a_control_and_trims_it() {
        let c = parse_control("  What is the capital of Italy?  =>  Rome ").unwrap();
        assert_eq!(c.prompt, "What is the capital of Italy?");
        assert_eq!(c.expected, "Rome");
    }

    #[test]
    fn splits_on_the_first_arrow_so_the_answer_may_contain_one() {
        let c = parse_control("2+2=>4=>four").unwrap();
        assert_eq!(c.prompt, "2+2");
        assert_eq!(c.expected, "4=>four");
    }

    #[test]
    fn rejects_a_control_with_no_arrow_or_an_empty_half() {
        assert!(parse_control("no arrow here").is_err());
        assert!(parse_control("=>Rome").is_err());
        assert!(parse_control("capital of Italy?=>").is_err());
    }

    #[test]
    fn the_provenance_line_distinguishes_checked_from_unchecked() {
        let checked = collateral_provenance(5, &BalancerOutcome::Converged { prob: 0.62 });
        assert!(checked.contains("5 control prompt(s) kept"));
        assert!(checked.contains("converged"));
        let unchecked = collateral_provenance(0, &BalancerOutcome::Disabled);
        assert!(unchecked.contains("NOT PERFORMED"));
        assert!(unchecked.contains("disabled"));
    }
}
