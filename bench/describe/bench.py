#!/usr/bin/env python3
"""DESCRIBE bench: does an entity's browse surface facts about the entity?

Entities: bench/describe/entities.tsv (query, Wikidata id, kind).
Targets:  bench/describe/targets.json, derived from Wikidata claims with
          `--refresh` — the labels, in a dozen languages, of the values of
          properties that state a fact about the entity (country, capital,
          citizenship, occupation, field, genre, industry, continent,
          language, demonym, parent taxon, …). Nothing is typed by hand.
Match:    an edge is on-target when any word (≥4 letters, stop-list
          removed) of any target shares a 5-letter stem with the edge's
          label or one of its neighbour tokens. Conservative: "science"
          does not match "physics"; "French" matches "France".

Endpoints (pick one):
  --larql URL             larql directly: GET {URL}/v1/describe?entity=…
  --divinci URL --workspace ID   Divinci server-api white-label route;
                          bearer token from --bearer-env (default TOKEN)
Extra DESCRIBE query params: --param background=corpus --param limit=20 …

Output: a table, and --out FILE writes the full JSON (per-entity ranks,
metrics, params, endpoint, time). --baseline FILE diffs against a stored
run: metric deltas and every entity whose first on-target rank moved.

  bench/describe/bench.py --refresh
  bench/describe/bench.py --divinci https://api.example --workspace WS \\
      --out bench/describe/baselines/2026-09-04-prod.json
  bench/describe/bench.py --divinci … --param background=corpus \\
      --baseline bench/describe/baselines/2026-09-04-prod.json
"""
import argparse, json, os, re, sys, time, urllib.parse, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENTITIES = HERE / "entities.tsv"
TARGETS = HERE / "targets.json"
UA = "larql describe-bench (https://github.com/Divinci-AI/larql)"
WD = "https://www.wikidata.org/w/api.php"
LANGS = ["en", "fr", "de", "es", "pt", "it", "ja", "zh", "ko", "id", "vi", "ar", "hi", "ru", "tr"]
# Properties whose values state a fact about the entity.
PROPS = {
    "P31": "instance of", "P279": "subclass of", "P171": "parent taxon",
    "P17": "country", "P27": "citizenship", "P495": "country of origin", "P30": "continent",
    "P1376": "capital of", "P36": "capital", "P37": "official language", "P159": "headquarters",
    "P106": "occupation", "P101": "field of work", "P641": "sport", "P140": "religion",
    "P136": "genre", "P452": "industry", "P176": "manufacturer",
    "P50": "author", "P57": "director", "P170": "creator",
}
# Deliberately NOT: member of, owner of, part of, notable work, influenced by,
# founded by, birthplace — true, but hundreds of labels per entity, and a
# stem matcher over hundreds of words hits on anything.
STRING_PROPS = {"P1549": "demonym"}  # monolingual text values
# Words too common to count as a hit on their own.
STOP = {"of", "the", "and", "states", "united", "republic", "kingdom", "city", "country",
        "human", "taxon", "people", "person", "group", "organization", "organisation",
        "company", "business", "work", "entity", "state", "sovereign", "nation", "island",
        "landlocked", "member", "national", "professional", "association", "film", "series"}
# Heuristic labels of features that fire for everything / for a language, kept
# from the 2026-09-04 measurements. Reported as shares, not used for hits.
GENERIC = re.compile(r"^(especially|either|role|mode|all|untuk|hangi|Real|Above|true|These|Whoever|Purpose|shine|Chal)$", re.I)
FUNC = re.compile(r"^(kita|tôi|và|bintang|untuk|dengan|yang|của|những|các|hangi|için)$", re.I)


def get_json(url, params=None, headers=None, timeout=60):
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def read_entities():
    out = []
    for line in ENTITIES.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        q, qid, kind = line.split("\t")
        out.append({"query": q, "qid": qid, "kind": kind})
    return out


def chunks(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def refresh_targets(entities):
    ents = {}
    for batch in chunks([e["qid"] for e in entities], 50):
        d = get_json(WD, {"action": "wbgetentities", "ids": "|".join(batch), "props": "claims|labels",
                          "languages": "en", "format": "json"})
        ents.update(d["entities"])
    values = {}   # entity qid -> {prop: [value qids]}
    strings = {}  # entity qid -> [strings]
    # Labels separately: a claims-heavy batch can come back with labels
    # missing, and an empty label passed the sanity check below unnoticed.
    # Some items (Q937, Q615 on 2026-09-04) carry no plain `en` label at all
    # among a hundred others, so fall back through the English variants,
    # `mul`, and finally the first Latin-script label.
    en_labels = {}
    for batch in chunks([e["qid"] for e in entities], 50):
        d = get_json(WD, {"action": "wbgetentities", "ids": "|".join(batch), "props": "labels", "format": "json"})
        for qid, ent in d["entities"].items():
            labs = ent.get("labels", {})
            pick = ""
            for lang in ("en", "en-gb", "en-ca", "en-us", "mul", "fr", "de", "es"):
                if labs.get(lang, {}).get("value"):
                    pick = labs[lang]["value"]
                    break
            if not pick:
                latin = [l["value"] for l in labs.values() if re.fullmatch(r"[A-Za-z .,'\-]+", l["value"])]
                pick = latin[0] if latin else ""
            en_labels[qid] = pick
    for e in entities:
        ent = ents[e["qid"]]
        label = en_labels.get(e["qid"], "")
        # The id must be the thing we meant; a wrong id would score nonsense.
        if not label or (e["query"].split()[-1].lower() not in label.lower() and label.lower() not in e["query"].lower()):
            sys.exit(f"{e['query']} -> {e['qid']} is labelled {label!r}; fix entities.tsv")
        e["label"] = label
        vals, strs = {}, []
        for prop, claims in ent.get("claims", {}).items():
            if prop in PROPS and PROPS[prop]:
                for c in claims:
                    v = c.get("mainsnak", {}).get("datavalue", {}).get("value")
                    if isinstance(v, dict) and v.get("id"):
                        vals.setdefault(prop, []).append(v["id"])
            elif prop in STRING_PROPS:
                for c in claims:
                    v = c.get("mainsnak", {}).get("datavalue", {}).get("value")
                    if isinstance(v, dict) and v.get("text"):
                        strs.append(v["text"])
        values[e["qid"]] = vals
        strings[e["qid"]] = strs
    all_vals = sorted({v for vs in values.values() for l in vs.values() for v in l})
    labels = {}
    for batch in chunks(all_vals, 50):
        d = get_json(WD, {"action": "wbgetentities", "ids": "|".join(batch), "props": "labels",
                          "languages": "|".join(LANGS), "format": "json"})
        for qid, ent in d["entities"].items():
            labels[qid] = sorted({l["value"] for l in ent.get("labels", {}).values()})
        time.sleep(0.1)
    out = {}
    for e in entities:
        targets = {}
        for prop, vs in values[e["qid"]].items():
            for v in vs:
                for l in labels.get(v, []):
                    targets.setdefault(l, []).append(PROPS[prop])
        for s in strings[e["qid"]]:
            targets.setdefault(s, []).append("demonym")
        # The entity's own name is not a fact about it.
        own = target_words([e["query"], e["label"]])
        targets = {t: p for t, p in targets.items() if not target_words([t]) <= own or not target_words([t])}
        out[e["query"]] = {"qid": e["qid"], "label": e["label"], "kind": e["kind"],
                           "targets": {t: sorted(set(p)) for t, p in sorted(targets.items())}}
    TARGETS.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    n = sum(len(v["targets"]) for v in out.values())
    w = sum(len(target_words(v["targets"])) for v in out.values())
    print(f"targets.json: {len(out)} entities, {n} target labels, {w} match words")


def norm(s):
    return re.sub(r"^[▁\s]+", "", str(s)).strip().lower()


def stem_match(w, t):
    if w == t:
        return True
    if len(t) >= 5 and w.startswith(t[:5]):
        return True
    if len(w) >= 5 and t.startswith(w[:5]):
        return True
    return False


def target_words(targets):
    words = set()
    for t in targets:
        for w in re.findall(r"[^\W\d_]+", t.lower()):
            if len(w) >= 4 and w not in STOP:
                words.add(w)
        # CJK / no-space scripts: the whole label is a word.
        if not re.search(r"\s", t) and re.search(r"[぀-ヿ一-鿿가-힯]", t):
            words.add(t.lower())
    return words


def is_hit(edge, words):
    toks = [norm(edge["target"])] + [norm(x) for x in edge.get("neighbors", [])]
    for w in toks:
        if not w:
            continue
        for t in words:
            if stem_match(w, t):
                return t
    return None


def describe(args, query):
    params = dict(kv.split("=", 1) for kv in args.param)
    if args.larql:
        url = args.larql.rstrip("/") + "/v1/describe?" + urllib.parse.urlencode({"entity": query, **params})
        j = get_json(url, timeout=args.timeout)
        edges = [{"target": e["target"], "layer": e.get("layer"), "neighbors": e.get("also", []),
                  "relevance": e.get("relevance"), "gate": e.get("gate_score")} for e in j.get("edges", [])]
        meta = {k: j.get(k) for k in ("relevance_background", "relevance_panel")}
    else:
        tok = os.environ.get(args.bearer_env, "")
        url = f"{args.divinci.rstrip('/')}/white-label/{args.workspace}/larql/describe?" + \
            urllib.parse.urlencode({"entity": query, **params})
        j = get_json(url, headers={"Authorization": f"Bearer {tok}"}, timeout=args.timeout)
        edges = [{"target": a["target"], "layer": a.get("layer"), "neighbors": a.get("neighbors", []),
                  "relevance": a.get("relevance"), "gate": a.get("rawScore")} for a in j.get("associations", [])]
        meta = {"relevance_background": j.get("relevanceBackground"), "relevance_panel": j.get("relevancePanel")}
    return edges, meta


# A word that is a target for this share of the bench or more says nothing
# about any one of them ("capital", "amerika", "besar") and is dropped. A
# share, not a count: at four of 36, "Japan" (six entities) was dropped and
# Nintendo's Jepang/Japan hits went unscored (research log §22).
COMMON_SHARE = 0.25


def words_per_entity(entities, targets):
    per = {}
    count = {}
    for e in entities:
        t = targets.get(e["query"])
        if not t:
            sys.exit(f"no targets for {e['query']}: run --refresh")
        w = target_words(t["targets"]) - target_words([e["query"], t["label"]])
        per[e["query"]] = w
        for x in w:
            count[x] = count.get(x, 0) + 1
    common = {w for w, c in count.items() if c >= max(2, COMMON_SHARE * len(entities))}
    return {q: w - common for q, w in per.items()}, common


def run(args):
    entities = read_entities()
    targets = json.loads(TARGETS.read_text())
    per_words, common = words_per_entity(entities, targets)
    print(f"{len(common)} words shared by ≥{COMMON_SHARE:.0%} of entities dropped; "
          f"median words per entity {sorted(len(w) for w in per_words.values())[len(per_words)//2]}")
    rows = []
    all_edges = {}
    for e in entities:
        words = per_words[e["query"]]
        t0 = time.time()
        edges = meta = None
        for attempt in range(1 + args.retry_errors):
            try:
                edges, meta = describe(args, e["query"])
                break
            except Exception as ex:  # noqa: BLE001
                err = str(ex)
                if attempt < args.retry_errors:
                    # A gateway timeout on a cold instance (a residual panel
                    # being built) resolves itself; wait, then ask again.
                    print(f"{e['query']:18} retry after {err}")
                    time.sleep(args.retry_wait)
        if edges is None:
            rows.append({"query": e["query"], "kind": e["kind"], "error": err})
            print(f"{e['query']:18} ERROR {err}")
            continue
        dt = time.time() - t0
        all_edges[e["query"]] = edges
        hits = [(i + 1, edge["target"], is_hit(edge, words)) for i, edge in enumerate(edges)]
        hits = [(r, l, m) for r, l, m in hits if m]
        top10 = edges[:10]
        row = {
            "query": e["query"], "kind": e["kind"], "n": len(edges), "latency_s": round(dt, 1),
            "first_rank": hits[0][0] if hits else None,
            "first_label": hits[0][1] if hits else None, "first_matched": hits[0][2] if hits else None,
            "hits_top10": sum(1 for r, _, _ in hits if r <= 10),
            "generic_top10": sum(1 for x in top10 if GENERIC.match(str(x["target"]))),
            "func_top10": sum(1 for x in top10 if FUNC.match(str(x["target"]))),
            "top8": [x["target"] for x in edges[:8]],
            "meta": meta,
        }
        rows.append(row)
        fr = f"#{row['first_rank']} {row['first_label']}←{row['first_matched']}" if hits else "—"
        print(f"{e['query']:18} n={len(edges):2} {dt:4.0f}s hits@10={row['hits_top10']} first {fr:34} | {', '.join(row['top8'][:6])}")
    ok = [r for r in rows if "error" not in r]
    n = len(ok) or 1
    # Chance control: each entity's edges against every OTHER entity's
    # targets. A matcher that hits here hits on anything.
    chance_hits, chance_pairs = 0, 0
    for q, edges in all_edges.items():
        for q2, w2 in per_words.items():
            if q2 == q:
                continue
            chance_pairs += 1
            if any(is_hit(edge, w2) for edge in edges[:10]):
                chance_hits += 1
    metrics = {
        "chance_hit@10": chance_hits / (chance_pairs or 1),
        "entities": len(ok), "errors": len(rows) - len(ok),
        "hit@1": sum(1 for r in ok if r["first_rank"] == 1) / n,
        "hit@5": sum(1 for r in ok if r["first_rank"] and r["first_rank"] <= 5) / n,
        "hit@10": sum(1 for r in ok if r["first_rank"] and r["first_rank"] <= 10) / n,
        "mrr": sum(1 / r["first_rank"] for r in ok if r["first_rank"]) / n,
        "hits_top10_mean": sum(r["hits_top10"] for r in ok) / n,
        "generic_share_top10": sum(r["generic_top10"] for r in ok) / (10 * n),
        "func_share_top10": sum(r["func_top10"] for r in ok) / (10 * n),
        "latency_s_median": sorted(r["latency_s"] for r in ok)[len(ok) // 2] if ok else None,
    }
    by_kind = {}
    for r in ok:
        k = by_kind.setdefault(r["kind"], {"n": 0, "hit@10": 0, "mrr": 0.0})
        k["n"] += 1
        k["hit@10"] += 1 if r["first_rank"] and r["first_rank"] <= 10 else 0
        k["mrr"] += 1 / r["first_rank"] if r["first_rank"] else 0
    for k in by_kind.values():
        k["hit@10"] = round(k["hit@10"] / k["n"], 2)
        k["mrr"] = round(k["mrr"] / k["n"], 2)
    print("\n" + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()))
    print("by kind: " + "  ".join(f"{k}: hit@10 {v['hit@10']} mrr {v['mrr']} (n={v['n']})" for k, v in by_kind.items()))
    result = {"when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "endpoint": args.larql or f"{args.divinci} ws={args.workspace}", "params": args.param,
              "metrics": metrics, "by_kind": by_kind, "rows": rows}
    if args.out:
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
        print(f"wrote {args.out}")
    if args.baseline:
        compare(json.loads(Path(args.baseline).read_text()), result)
    return result


def compare(base, cur):
    print(f"\n== vs baseline {base['when']} params={base['params']}")
    for k, v in cur["metrics"].items():
        b = base["metrics"].get(k)
        if isinstance(v, float) and isinstance(b, (int, float)):
            d = v - b
            flag = "  " if abs(d) < 1e-9 else ("▲ " if d > 0 else "▼ ")
            print(f"  {flag}{k:22} {b:.3f} → {v:.3f} ({d:+.3f})")
    bi = {r["query"]: r for r in base["rows"]}
    moved = []
    for r in cur["rows"]:
        b = bi.get(r["query"])
        if not b or "error" in r or "error" in b:
            continue
        if b.get("first_rank") != r.get("first_rank"):
            moved.append(f"  {r['query']:18} first {b.get('first_rank') or '—'} → {r.get('first_rank') or '—'}"
                         f"  ({b.get('first_label')} → {r.get('first_label')})")
    print("moved:" if moved else "no entity's first on-target rank moved")
    for m in moved:
        print(m)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="rebuild targets.json from Wikidata")
    ap.add_argument("--larql", help="larql base URL")
    ap.add_argument("--divinci", help="Divinci API base URL")
    ap.add_argument("--workspace", help="Divinci workspace id (with --divinci)")
    ap.add_argument("--bearer-env", default="TOKEN")
    ap.add_argument("--param", action="append", default=[], help="extra DESCRIBE query param k=v")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--retry-errors", type=int, default=2, help="re-ask an entity that errored, this many times")
    ap.add_argument("--retry-wait", type=int, default=60, help="seconds between retries")
    ap.add_argument("--out")
    ap.add_argument("--baseline")
    args = ap.parse_args()
    if args.refresh:
        refresh_targets(read_entities())
        return
    if not (args.larql or (args.divinci and args.workspace)):
        ap.error("need --larql URL or --divinci URL --workspace ID")
    run(args)


if __name__ == "__main__":
    main()
