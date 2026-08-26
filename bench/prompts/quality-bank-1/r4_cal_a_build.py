#!/usr/bin/env python3
"""R4.0-CAL-A — freeze the calibration and validation pools for
ENCODER-R4 / nvfp4-gptq-v1, before any calibration-aware encoder exists.
One shot; no revision afterwards (see ENCODER-R4.md).

    python3 r4_cal_a_build.py build   <granite-4.1-3b checkpoint dir>
    python3 r4_cal_a_build.py verify  <granite-4.1-3b checkpoint dir>

`build` downloads (or reuses, checking sha256) four public-domain
Project Gutenberg texts, extracts and cleans their prose, tokenises
with the container's own tokenizer, verifies disjointness from Q-BANK-1
and between the two pools, and writes
`r4-cal-a-calibration-pool.json` / `r4-cal-a-validation-pool.json`.
`verify` regenerates from the same pinned sources and **refuses** if
anything moved — the digests already embedded in those two files are
what everything downstream (R4.0-CAL-B's sufficiency ladder) trusts.

Why two pools, and why calibration extends the EXISTING 458-position
bank rather than starting fresh
--------------------------------------------------------------------
`N0 = 458` is not a target this script hits — it IS
`calibration-disjoint.json`'s existing 12 prompts, unchanged, already
frozen by `freeze_calibration.py`. The nested ladder (`N0` through
`N4 = 65,536`) is built by appending MORE prompts, in a fixed
deterministic order, after that unchanged prefix — so every smaller
rung's prompt list is a strict prefix of every larger rung's, and N0
itself never has to be reproduced or re-verified here.

Calibration and validation are sourced from ENTIRELY DIFFERENT books
(Walden + On the Origin of Species vs. Pride and Prejudice + Twenty
Thousand Leagues Under the Sea) specifically so cross-pool content
overlap is impossible by construction, not merely checked for after
the fact — though it is also checked, exactly, not trusted to that
design choice alone.

Why public-domain text at all, rather than more hand-authored prompts
--------------------------------------------------------------------
The existing Q-BANK-1 / SENSITIVITY-1B' banks are hand-authored by
design, at a scale (69 / 12 prompts) that stays practical to write by
hand. R4's calibration ladder needs up to 65,536 real positions —
roughly 40x that scale — and genuine prose gives more realistic
activation statistics at that volume than the same amount of synthetic
writing would. Decided explicitly, not defaulted into (see
ENCODER-R4.md's R4.0-CAL-A section).
"""
import hashlib
import json
import os
import re
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
QBANK1_PATH = os.path.join(HERE, "prompts.json")
SENS_BANK_PATH = os.path.join(HERE, "calibration-disjoint.json")
CAL_OUT = os.path.join(HERE, "r4-cal-a-calibration-pool.json")
VAL_OUT = os.path.join(HERE, "r4-cal-a-validation-pool.json")
CACHE_DIR = os.path.join(HERE, ".r4-cal-a-source-cache")

# (key, Gutenberg URL, expected sha256, title, author, gutenberg_id) —
# the sha256 pin is what makes `verify` meaningful: if Gutenberg's
# hosted text ever changes, this refuses rather than silently rebuilding
# a different pool under the same file name.
SOURCES = {
    "walden": {
        "url": "https://www.gutenberg.org/cache/epub/205/pg205.txt",
        "sha256": "2d9a76a2e3e8195c69430516ebd33c4d0757a53ad432ff6186b7b794e6fe99f9",
        "title": "Walden, and On The Duty Of Civil Disobedience",
        "author": "Henry David Thoreau",
        "gutenberg_id": 205,
    },
    "origin_of_species": {
        "url": "https://www.gutenberg.org/cache/epub/1228/pg1228.txt",
        "sha256": "ededa9c0bf8761efed092c303b46c1c92de956838cba6249a33bedfd6d7363b4",
        "title": "On the Origin of Species By Means of Natural Selection",
        "author": "Charles Darwin",
        "gutenberg_id": 1228,
    },
    "pride_and_prejudice": {
        "url": "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
        "sha256": "74f2665d6e6925fc2c17dec644bec9e87df478a0f1836822125e8acbb3777806",
        "title": "Pride and Prejudice",
        "author": "Jane Austen",
        "gutenberg_id": 1342,
    },
    "twenty_thousand_leagues": {
        "url": "https://www.gutenberg.org/cache/epub/164/pg164.txt",
        "sha256": "3562e063ad9027725d1feb2f945e86a0a90804a8ab19f8cb272884c436ebc34f",
        "title": "Twenty Thousand Leagues under the Sea",
        "author": "Jules Verne",
        "gutenberg_id": 164,
    },
}
LICENSE = "public domain (US) — Project Gutenberg"
LADDER_TARGETS = [458, 2048, 8192, 32768, 65536]


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def fetch_source(key):
    """Download (once, cached) and verify against the pinned sha256."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, key + ".txt")
    meta = SOURCES[key]
    if not os.path.exists(path):
        with urllib.request.urlopen(meta["url"]) as resp:
            data = resp.read()
        open(path, "wb").write(data)
    else:
        data = open(path, "rb").read()
    digest = sha256_bytes(data)
    if digest != meta["sha256"]:
        raise SystemExit(
            f"REFUSED: {key} does not match its pinned sha256.\n"
            f"  pinned      {meta['sha256']}\n"
            f"  downloaded  {digest}\n"
            f"Gutenberg's hosted text for this work has changed since this pool was frozen."
        )
    return data.decode("utf-8")


def load_body(text):
    start_m = re.search(r"\*\*\* START OF.*?\*\*\*", text)
    end_m = re.search(r"\*\*\* END OF.*?\*\*\*", text)
    return text[start_m.end():end_m.start()]


def clean_paragraphs(body):
    """Split into paragraphs, drop headings/illustration captions/
    chapter-argument lists, keep only genuine prose."""
    raw_paras = re.split(r"\n\s*\n", body)
    out = []
    for p in raw_paras:
        p = " ".join(p.split())
        if len(p) < 200:
            continue
        if p.startswith("[Illustration") or p.startswith("CHAPTER") or p.startswith("PART "):
            continue
        if re.match(r"^[IVXLCDM]+\.?$", p):
            continue
        # Table-of-contents blocks (Walden: one long paragraph of Title
        # Case chapter names, no blank lines between them) carry almost
        # no sentence-ending punctuation relative to length.
        if p.count(".") < len(p) / 200:
            continue
        # Chapter-opening "argument" lists (On the Origin of Species:
        # dense runs of short topic-sentence fragments, each ending in
        # a period, so the check above alone doesn't catch them) — real
        # prose has both a healthy average AND median sentence length;
        # a block diluted by a few long sentences among many short
        # fragments fails the median even when the average looks fine.
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", p) if s.strip()]
        if sentences:
            avg_words = sum(len(s.split()) for s in sentences) / len(sentences)
            median_words = sorted(len(s.split()) for s in sentences)[len(sentences) // 2]
            if avg_words < 12 or median_words < 10:
                continue
        out.append(p)
    return out


def chunk_to_prompts(paragraphs, source_key, id_prefix, target_chars):
    prompts = []
    buf, buf_len, idx = [], 0, 0
    for p in paragraphs:
        buf.append(p)
        buf_len += len(p)
        if buf_len >= target_chars:
            prompts.append({"id": f"{id_prefix}-{idx:04d}", "category": "longform", "source": source_key, "text": " ".join(buf)})
            idx += 1
            buf, buf_len = [], 0
    if buf and buf_len >= 400:
        prompts.append({"id": f"{id_prefix}-{idx:04d}", "category": "longform", "source": source_key, "text": " ".join(buf)})
    return prompts


def tokenize_all(prompts, tokenizer):
    for p in prompts:
        ids = tokenizer.encode(p["text"], add_special_tokens=True).ids
        p["ids"] = ids
        p["n_positions"] = len(ids)
    return prompts


def check_disjoint(pool_a, pool_b, label_a, label_b):
    texts_a = {p["text"] for p in pool_a}
    texts_b = {p["text"] for p in pool_b}
    exact_overlap = texts_a & texts_b
    substring_hits = [
        (pa["id"], pb["id"])
        for pa in pool_a
        for pb in pool_b
        if pa["text"] != pb["text"] and (pa["text"] in pb["text"] or pb["text"] in pa["text"])
    ]
    print(f"  disjoint check {label_a} vs {label_b}: exact_overlap={len(exact_overlap)} substring_hits={len(substring_hits)}")
    if exact_overlap or substring_hits:
        raise SystemExit(f"REFUSED: {label_a} vs {label_b} are not disjoint.")


def build(container):
    from tokenizers import Tokenizer

    tokenizer_path = os.path.join(container, "tokenizer.json")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer_sha256 = sha256_bytes(open(tokenizer_path, "rb").read())

    sens_bank = json.load(open(SENS_BANK_PATH))
    n0_prompts = []
    for p in sens_bank["prompts"]:
        ids = tokenizer.encode(p["text"], add_special_tokens=True).ids
        n0_prompts.append({"id": p["id"], "category": p["category"], "source": "sensitivity-bank-458 (existing, frozen)", "text": p["text"], "ids": ids, "n_positions": len(ids)})
    n0_positions = sum(p["n_positions"] for p in n0_prompts)
    print(f"N0 (existing bank): {len(n0_prompts)} prompts, {n0_positions} positions")

    per_source = {}
    for key in ["walden", "origin_of_species"]:
        text = fetch_source(key)
        per_source[key] = chunk_to_prompts(clean_paragraphs(load_body(text)), key, f"cal-{key}", target_chars=1800)
    cal_extra = []
    iters = [iter(per_source["walden"]), iter(per_source["origin_of_species"])]
    exhausted = [False, False]
    while not all(exhausted):
        for i, it in enumerate(iters):
            if exhausted[i]:
                continue
            try:
                cal_extra.append(next(it))
            except StopIteration:
                exhausted[i] = True
    tokenize_all(cal_extra, tokenizer)
    print(f"calibration extension: {len(cal_extra)} prompts, {sum(p['n_positions'] for p in cal_extra)} positions available (interleaved)")

    val_prompts = []
    for key in ["pride_and_prejudice", "twenty_thousand_leagues"]:
        text = fetch_source(key)
        val_prompts.extend(chunk_to_prompts(clean_paragraphs(load_body(text)), key, f"val-{key}", target_chars=700))
    tokenize_all(val_prompts, tokenizer)
    print(f"validation pool: {len(val_prompts)} prompts, {sum(p['n_positions'] for p in val_prompts)} positions")

    qbank1 = json.load(open(QBANK1_PATH))["prompts"]
    all_cal = n0_prompts + cal_extra
    check_disjoint(all_cal, qbank1, "calibration", "Q-BANK-1")
    check_disjoint(val_prompts, qbank1, "validation", "Q-BANK-1")
    check_disjoint(all_cal, val_prompts, "calibration", "validation")

    nested = {}
    running, running_total = list(n0_prompts), n0_positions
    extra_iter = iter(cal_extra)
    for target in LADDER_TARGETS:
        while running_total < target:
            try:
                nxt = next(extra_iter)
            except StopIteration:
                break
            running.append(nxt)
            running_total += nxt["n_positions"]
        nested[str(target)] = {"n_prompts": len(running), "achieved_positions": running_total, "prompt_ids": [p["id"] for p in running]}
        print(f"N={target}: achieved {running_total} positions from {len(running)} prompts")
    if running_total < LADDER_TARGETS[-1]:
        raise SystemExit(f"REFUSED: only reached {running_total} positions, short of {LADDER_TARGETS[-1]}")

    full_cal_pool = list(running)
    return {
        "n0_prompts": n0_prompts,
        "cal_extra": cal_extra,
        "full_cal_pool": full_cal_pool,
        "val_prompts": val_prompts,
        "nested": nested,
        "tokenizer_sha256": tokenizer_sha256,
    }


def cmd_build(container):
    r = build(container)
    cal_text_digest = sha256_text(canonical_json([{"id": p["id"], "text": p["text"]} for p in r["full_cal_pool"]]))
    cal_token_digest = sha256_text(canonical_json([{"id": p["id"], "ids": p["ids"]} for p in r["full_cal_pool"]]))
    val_text_digest = sha256_text(canonical_json([{"id": p["id"], "text": p["text"]} for p in r["val_prompts"]]))
    val_token_digest = sha256_text(canonical_json([{"id": p["id"], "ids": p["ids"]} for p in r["val_prompts"]]))

    calibration_out = {
        "note": "R4.0-CAL-A frozen calibration pool for ENCODER-R4 / nvfp4-gptq-v1. Written before any calibration-aware encoder exists. One shot; no revision afterwards.",
        "frozen": "2026-08-26",
        "n0_source": "bench/prompts/quality-bank-1/calibration-disjoint.json (existing, unchanged, 458 real positions)",
        "extension_sources": [{**SOURCES["walden"], "license": LICENSE}, {**SOURCES["origin_of_species"], "license": LICENSE}],
        "tokenizer": {"path": "granite-4.1-3b tokenizer.json", "sha256": r["tokenizer_sha256"]},
        "nested_prefix_targets": LADDER_TARGETS,
        "nested_prefixes": r["nested"],
        "disjoint_from_qbank1": True,
        "disjoint_from_validation": True,
        "text_digest": cal_text_digest,
        "token_digest": cal_token_digest,
        "prompts": r["full_cal_pool"],
    }
    validation_out = {
        "note": "R4.0-CAL-A frozen validation pool for ENCODER-R4 / nvfp4-gptq-v1 — disjoint from BOTH the calibration pool and Q-BANK, used only in R4.0-CAL-B to measure whether GPTQ's fit has converged. Not touched until then.",
        "frozen": "2026-08-26",
        "sources": [{**SOURCES["pride_and_prejudice"], "license": LICENSE}, {**SOURCES["twenty_thousand_leagues"], "license": LICENSE}],
        "tokenizer": {"path": "granite-4.1-3b tokenizer.json", "sha256": r["tokenizer_sha256"]},
        "n_prompts": len(r["val_prompts"]),
        "total_positions": sum(p["n_positions"] for p in r["val_prompts"]),
        "disjoint_from_qbank1": True,
        "disjoint_from_calibration": True,
        "text_digest": val_text_digest,
        "token_digest": val_token_digest,
        "prompts": r["val_prompts"],
    }

    existing_cal_digest = None
    if os.path.exists(CAL_OUT):
        existing_cal_digest = json.load(open(CAL_OUT)).get("token_digest")
    if existing_cal_digest and existing_cal_digest != cal_token_digest:
        raise SystemExit(f"REFUSED: a calibration pool is already frozen and rebuilding changed it.\n  frozen      {existing_cal_digest}\n  regenerated {cal_token_digest}")

    json.dump(calibration_out, open(CAL_OUT, "w"), indent=1, ensure_ascii=False)
    json.dump(validation_out, open(VAL_OUT, "w"), indent=1, ensure_ascii=False)
    print()
    print("wrote", CAL_OUT)
    print("wrote", VAL_OUT)
    print("calibration token_digest:", cal_token_digest)
    print("validation  token_digest:", val_token_digest)


def cmd_verify(container):
    if not os.path.exists(CAL_OUT) or not os.path.exists(VAL_OUT):
        raise SystemExit("REFUSED: nothing frozen yet — run `build` first.")
    frozen_cal = json.load(open(CAL_OUT))
    frozen_val = json.load(open(VAL_OUT))
    r = build(container)
    cal_token_digest = sha256_text(canonical_json([{"id": p["id"], "ids": p["ids"]} for p in r["full_cal_pool"]]))
    val_token_digest = sha256_text(canonical_json([{"id": p["id"], "ids": p["ids"]} for p in r["val_prompts"]]))
    if cal_token_digest != frozen_cal["token_digest"]:
        raise SystemExit(f"REFUSED: calibration pool does not reproduce.\n  frozen      {frozen_cal['token_digest']}\n  regenerated {cal_token_digest}")
    if val_token_digest != frozen_val["token_digest"]:
        raise SystemExit(f"REFUSED: validation pool does not reproduce.\n  frozen      {frozen_val['token_digest']}\n  regenerated {val_token_digest}")
    print(f"OK  calibration {cal_token_digest}  ({len(r['full_cal_pool'])} prompts)")
    print(f"OK  validation  {val_token_digest}  ({len(r['val_prompts'])} prompts)")
    print("both pools reproduce exactly from pinned sources")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("build", "verify"):
        raise SystemExit(__doc__)
    (cmd_build if sys.argv[1] == "build" else cmd_verify)(sys.argv[2])
