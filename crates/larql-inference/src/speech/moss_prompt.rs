//! MOSS-TTS-Realtime prompt construction — the `[T, 1+rvq]` matrix the
//! generation loop prefls.
//!
//! Mirrors the reference processor exactly (`MossTTSRealtimeProcessor` in
//! the OpenMOSS repo): system prompt, optional voice-clone context whose
//! `<|audio_pad|>` run is overwritten channel-wise with reference codec
//! tokens, assistant header, then the text lead ([`DELAY_TOKENS_LEN`]
//! ids) with audio BOS on the last lead row. Byte-exactness of the
//! protocol strings is pinned by an ignored parity test against the
//! step-0 dump's prompt matrix — do not "tidy" the whitespace.
//!
//! Temporal policy stops here: this module builds state, the loop
//! consumes it, and nothing below the runtime knows about clocks.

use ndarray::Array2;

use larql_models::detect::ModelError;
use larql_models::speech::moss_tts_realtime::MossTtsRealtimeConfig;

/// How many text ids the prefill consumes before the first frame — the
/// text channel's lead over the audio channels. A protocol constant of
/// the checkpoint family (`delay_tokens_len` in the reference processor);
/// not present in `config.json`, so it is spelled here with that
/// provenance.
pub const DELAY_TOKENS_LEN: usize = 12;

/// The reference's system prompt, byte-exact (trailing space on the
/// second line included).
const SYSTEM_PROMPT: &str = "<|im_start|>system\nYou are a highly expressive text-to-speech (TTS) engine developed by Mosi Intelligence. \nYou possess natural language understanding, emotional modeling, and multi-style speech generation capabilities, allowing you to generate the corresponding speech based on the text given in the assistant.<|im_end|>\n";

const CONTEXT_PREFIX: &str =
    "<|im_start|>context\nThe assistant section should be synthesized using the following voice timbre:";
const CONTEXT_SUFFIX: &str = "<|im_end|>\n";
const ASSISTANT_HEADER: &str = "<|im_start|>assistant\n";
const AUDIO_PAD_TEXT: &str = "<|audio_pad|>";
const TEXT_PAD_TEXT: &str = "<|text_pad|>";

/// A built prompt: everything the generation loop needs.
#[derive(Debug)]
pub struct MossPrompt {
    /// `[T, 1+rvq]` — the prefill input, text lead and audio BOS placed.
    pub prefill_matrix: Array2<u32>,
    /// Text ids not consumed by the lead; fed one per frame.
    pub text_queue: Vec<u32>,
    /// Fed once `text_queue` is exhausted.
    pub text_pad_id: u32,
}

/// Build a voice-clone prompt. `reference_codes` is `[T, rvq]` (codec
/// tokens of the reference audio); `None` builds the no-clone prompt
/// (the model's unconditioned voice).
pub fn build_prompt(
    tokenizer: &tokenizers::Tokenizer,
    config: &MossTtsRealtimeConfig,
    reference_codes: Option<&Array2<u32>>,
    text: &str,
) -> Result<MossPrompt, ModelError> {
    let rvq = config.rvq;
    let pad = config.audio_pad_token as u32;
    let audio_pad_id = single_token_id(tokenizer, AUDIO_PAD_TEXT)?;
    let text_pad_id = single_token_id(tokenizer, TEXT_PAD_TEXT)?;

    // ── System (+ voice-clone context), tokenized as ONE string, as the
    // reference does — encoding across the concatenation boundary must
    // match its BPE exactly. ──
    let system_text = match reference_codes {
        Some(codes) => format!(
            "{SYSTEM_PROMPT}{CONTEXT_PREFIX}{}{CONTEXT_SUFFIX}",
            AUDIO_PAD_TEXT.repeat(codes.nrows())
        ),
        None => SYSTEM_PROMPT.to_string(),
    };
    let system_ids = encode(tokenizer, &system_text)?;
    let assistant_ids = encode(tokenizer, ASSISTANT_HEADER)?;
    let text_ids = encode(tokenizer, text)?;
    if text_ids.is_empty() {
        return Err(ModelError::Parse("empty text".into()));
    }

    let lead = text_ids.len().min(DELAY_TOKENS_LEN);
    let rows = system_ids.len() + assistant_ids.len() + lead;
    let mut matrix = Array2::<u32>::from_elem((rows, 1 + rvq), pad);

    for (row, &id) in system_ids.iter().chain(assistant_ids.iter()).enumerate() {
        matrix[[row, 0]] = id;
    }

    // ── Voice-clone splice: overwrite channels 1..=rvq of exactly the
    // contiguous <|audio_pad|> rows with the reference codes. ──
    if let Some(codes) = reference_codes {
        if codes.ncols() != rvq {
            return Err(ModelError::Parse(format!(
                "reference codes have {} channels, model expects {rvq}",
                codes.ncols()
            )));
        }
        let pad_rows: Vec<usize> = system_ids
            .iter()
            .enumerate()
            .filter(|(_, &id)| id == audio_pad_id)
            .map(|(row, _)| row)
            .collect();
        if pad_rows.len() != codes.nrows() {
            return Err(ModelError::Parse(format!(
                "tokenized prompt carries {} audio-pad rows for {} reference frames — \
                 the pad text did not tokenize one-to-one",
                pad_rows.len(),
                codes.nrows()
            )));
        }
        for (frame, &row) in pad_rows.iter().enumerate() {
            for channel in 0..rvq {
                matrix[[row, 1 + channel]] = codes[[frame, channel]];
            }
        }
    }

    // ── Text lead, audio BOS on its last row. ──
    let lead_start = system_ids.len() + assistant_ids.len();
    for (offset, &id) in text_ids[..lead].iter().enumerate() {
        matrix[[lead_start + offset, 0]] = id;
    }
    matrix[[rows - 1, 1]] = config.audio_bos_token() as u32;

    Ok(MossPrompt {
        prefill_matrix: matrix,
        text_queue: text_ids[lead..].to_vec(),
        text_pad_id,
    })
}

fn encode(tokenizer: &tokenizers::Tokenizer, text: &str) -> Result<Vec<u32>, ModelError> {
    Ok(tokenizer
        .encode(text, false)
        .map_err(|e| ModelError::Parse(format!("tokenize failed: {e}")))?
        .get_ids()
        .to_vec())
}

fn single_token_id(tokenizer: &tokenizers::Tokenizer, text: &str) -> Result<u32, ModelError> {
    let ids = encode(tokenizer, text)?;
    match ids.as_slice() {
        [id] => Ok(*id),
        other => Err(ModelError::Parse(format!(
            "{text:?} tokenizes to {} ids, expected a single special token",
            other.len()
        ))),
    }
}
