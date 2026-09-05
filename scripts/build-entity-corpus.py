#!/usr/bin/env python3
"""Build crates/larql-server/assets/entity-corpus.txt: the relevance `corpus` panel.

Source: English Wikipedia's daily top-1000 most-viewed articles (Wikimedia
pageviews REST API), sampled on the 1st and 15th of every month over the
window below. A title is kept by how many sampled days it appears in, so the
corpus is the durably looked-up population of things — not one week's news.

Filters: namespaces (anything with ':'), "List of …", "Deaths in …",
year/date-shaped titles, and titles under three characters are dropped;
disambiguators like " (film)" are stripped; underscores become spaces.

Stratified mode (`--stratify`, the shipped default since 2026-09-04): the
top `--pool` titles by days present are classified by their Wikidata
"instance of" class labels into kinds, and fixed per-kind quotas are filled
in days-present order. An unstratified corpus (2026-09-04, §16 of the
research log) was people/film heavy — Wikipedia's viewing population —
and that composition, not the size, decided which features read as
background.

Usage:  scripts/build-entity-corpus.py [--stratify] [--out PATH]
"""
import argparse, collections, json, re, sys, time, urllib.request, urllib.parse

API = "https://wikimedia.org/api/rest_v1/metrics/pageviews/top/en.wikipedia/all-access/{y}/{m:02d}/{d:02d}"
UA = "larql entity-corpus builder (https://github.com/Divinci-AI/larql)"
DROP = re.compile(r"^(List_of|Lists_of|Deaths_in|Main_Page$|\d{1,4}(_in_|$|_BC)|\d{4}[_-]\d)|[:]")
# Search-box noise and adult-site titles that top the daily list: not the
# population a browse is drawn from, and not something to ship in a data file.
NOISE = re.compile(r"^(X+|XXX.*|XNXX|XVideos|Pornhub|.*[Pp]orn.*|Sex|Sexual_intercourse)$")
# The bench entities (bench/describe/entities.tsv) are held out so no
# measurement is against its own background. Their Wikidata labels count too:
# the bench queries "Einstein" but the title is "Albert Einstein".
def held_out():
    import pathlib
    here = pathlib.Path(__file__).resolve().parent.parent / "bench" / "describe"
    names = set()
    for line in (here / "entities.tsv").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            names.add(line.split("\t")[0].strip())
    tj = here / "targets.json"
    if tj.exists():
        for v in json.loads(tj.read_text()).values():
            names.add(v["label"])
    return names


HELD_OUT = held_out()
HELD_OUT_LOWER = {n.lower() for n in HELD_OUT}


def clean(title: str):
    if DROP.search(title) or NOISE.fullmatch(title):
        return None
    t = re.sub(r"_\([^)]*\)$", "", title).replace("_", " ").strip()
    if len(t) < 3 or re.fullmatch(r"[\d\s.,-]+", t) or t.lower() in HELD_OUT_LOWER:
        return None
    return t


WP_API = "https://en.wikipedia.org/w/api.php"
WD_API = "https://www.wikidata.org/w/api.php"

# Kind by keywords in the label of a Wikidata class ("instance of" value).
# Order matters: the first kind whose pattern matches any class label wins.
KINDS = [
    ("person", re.compile(r"^human$|fictional (human|character)|^character$", re.I)),
    ("work", re.compile(r"\b(film|series|album|song|novel|book|video game|game|franchise|anime|manga|play|musical|comic|episode|season|single|miniseries|soundtrack|poem|opera|painting|sculpture|magazine|newspaper|literary work|written work|artwork)\b", re.I)),
    ("event", re.compile(r"\b(war|battle|election|championship|tournament|olympic|festival|attack|disaster|pandemic|earthquake|conflict|invasion|protest|ceremony|award|competition|cup|match|crisis|revolution|massacre|shooting|accident|incident|referendum|campaign|hurricane|tour)\b", re.I)),
    ("organisation", re.compile(r"\b(company|business|enterprise|organi[sz]ation|band|club|team|university|college|party|agency|league|network|studio|label|manufacturer|corporation|airline|website|platform|service|brand|institution|ministry|army|navy|force|school|conglomerate|chain|retailer|bank|federation|association|council|group|duo|orchestra|choir)\b", re.I)),
    ("place", re.compile(r"\b(city|town|country|state|province|region|island|river|mountain|capital|municipality|village|county|continent|ocean|sea|lake|district|settlement|territory|kingdom|empire|nation|borough|prefecture|republic|archipelago|peninsula|desert|park|building|stadium|airport|bridge|tower|palace|castle|temple|church|cathedral|monument|street|neighbourhood|neighborhood|planet|moon|galaxy|star)\b", re.I)),
]
OTHER = "other"
# Not entities: a disambiguation page is a name shared by several things.
SKIP_CLASS = re.compile(r"disambiguation|Wikimedia (list|category|template)", re.I)
# Share of the corpus each kind gets. Shortfalls are redistributed in
# days-present order across the kinds that still have candidates.
QUOTA = {"place": 0.20, "person": 0.25, "organisation": 0.15, "work": 0.15, "event": 0.10, OTHER: 0.15}


def get_json(url, params):
    req = urllib.request.Request(url + "?" + urllib.parse.urlencode(params), headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def chunks(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def wikibase_items(titles):
    """title -> Qid, following redirects; titles without an item are absent."""
    out = {}
    for batch in chunks(titles, 50):
        try:
            d = get_json(WP_API, {"action": "query", "prop": "pageprops", "ppprop": "wikibase_item",
                                  "titles": "|".join(batch), "redirects": 1, "format": "json"})
        except Exception as e:  # noqa: BLE001
            print(f"skip wp batch: {e}", file=sys.stderr); continue
        q = d.get("query", {})
        back = {r["to"]: r["from"] for r in q.get("redirects", [])}
        back.update({r["to"]: r["from"] for r in q.get("normalized", [])})
        for pg in q.get("pages", {}).values():
            qid = pg.get("pageprops", {}).get("wikibase_item")
            if qid:
                out[back.get(pg["title"], pg["title"])] = qid
        time.sleep(0.1)
    return out


def instance_classes(qids):
    """Qid -> list of 'instance of' class Qids."""
    out = {}
    for batch in chunks(qids, 50):
        try:
            d = get_json(WD_API, {"action": "wbgetentities", "ids": "|".join(batch), "props": "claims", "format": "json"})
        except Exception as e:  # noqa: BLE001
            print(f"skip wd batch: {e}", file=sys.stderr); continue
        for qid, ent in d.get("entities", {}).items():
            cls = []
            for c in ent.get("claims", {}).get("P31", []):
                v = c.get("mainsnak", {}).get("datavalue", {}).get("value", {})
                if isinstance(v, dict) and v.get("id"):
                    cls.append(v["id"])
            out[qid] = cls
        time.sleep(0.1)
    return out


def labels(qids):
    out = {}
    for batch in chunks(sorted(qids), 50):
        try:
            d = get_json(WD_API, {"action": "wbgetentities", "ids": "|".join(batch), "props": "labels", "languages": "en", "format": "json"})
        except Exception as e:  # noqa: BLE001
            print(f"skip label batch: {e}", file=sys.stderr); continue
        for qid, ent in d.get("entities", {}).items():
            out[qid] = ent.get("labels", {}).get("en", {}).get("value", "")
        time.sleep(0.1)
    return out


def kind_of(class_labels):
    for kind, pat in KINDS:
        if any(pat.search(l) for l in class_labels):
            return kind
    return OTHER


def stratify(ranked, keep, pool, dump=None):
    """ranked: [(title, days)] best first. Returns [(title, days, kind)]."""
    cand = ranked[:pool]
    titles = [t for t, _ in cand]
    items = wikibase_items(titles)
    classes = instance_classes(sorted(set(items.values())))
    lab = labels({c for cs in classes.values() for c in cs})
    kinds = {}
    for t, _ in cand:
        qid = items.get(t)
        if not qid:
            continue  # no Wikidata item: not a thing we can place, skip
        cls = [lab.get(c, "") for c in classes.get(qid, [])]
        if any(SKIP_CLASS.search(l) for l in cls):
            continue
        kinds[t] = kind_of(cls)
    if dump:
        with open(dump, "w") as f:
            for t, d in cand:
                qid = items.get(t)
                cls = "; ".join(lab.get(c, c) for c in classes.get(qid, [])) if qid else ""
                f.write(f"{t}\t{d}\t{kinds.get(t, '-')}\t{cls}\n")
    by_kind = collections.defaultdict(list)
    for t, d in cand:
        if t in kinds:
            by_kind[kinds[t]].append((t, d))
    want = {k: int(keep * q) for k, q in QUOTA.items()}
    chosen = []
    for k, n in want.items():
        chosen += [(t, d, k) for t, d in by_kind[k][:n]]
        by_kind[k] = by_kind[k][n:]
    # Redistribute any shortfall in days-present order over what is left.
    left = sorted(((t, d, k) for k, rest in by_kind.items() for t, d in rest), key=lambda x: (-x[1], x[0]))
    chosen += left[: keep - len(chosen)]
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-01")
    ap.add_argument("--end", default="2026-08")
    ap.add_argument("--keep", type=int, default=2048)
    ap.add_argument("--out", default="crates/larql-server/assets/entity-corpus.txt")
    ap.add_argument("--stratify", action="store_true", help="fill per-kind quotas (see QUOTA) from the top --pool titles")
    ap.add_argument("--pool", type=int, default=20000)
    ap.add_argument("--dump", help="write title/days/kind/classes for the pool (diagnostic)")
    a = ap.parse_args()
    sy, sm = map(int, a.start.split("-")); ey, em = map(int, a.end.split("-"))
    days = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        days += [(y, m, 1), (y, m, 15)]
        m += 1
        if m > 12: y, m = y + 1, 1
    seen = collections.Counter()
    for (y, m, d) in days:
        req = urllib.request.Request(API.format(y=y, m=m, d=d), headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                arts = json.load(r)["items"][0]["articles"]
        except Exception as e:  # noqa: BLE001
            print(f"skip {y}-{m:02d}-{d:02d}: {e}", file=sys.stderr)
            continue
        for x in arts:
            t = clean(x["article"])
            if t:
                seen[t] += 1
        time.sleep(0.2)
    ranked = sorted(seen.items(), key=lambda kv: (-kv[1], kv[0]))
    if a.stratify:
        chosen = stratify(ranked, a.keep, a.pool, a.dump)
        comp = collections.Counter(k for _, _, k in chosen)
        chosen.sort(key=lambda x: (-x[1], x[0]))
        with open(a.out, "w") as f:
            f.write(f"# {len(chosen)} titles; en.wikipedia daily top-1000, 1st+15th of each month {a.start}..{a.end},\n")
            f.write(f"# top {a.pool} by days present (max {len(days)}), classified by Wikidata instance-of and filled to\n")
            f.write("# per-kind quotas: " + ", ".join(f"{k} {comp[k]}" for k in QUOTA) + ". Built by scripts/build-entity-corpus.py --stratify.\n")
            for t, d, k in chosen:
                f.write(t + "\n")
        print(f"{len(days)} days sampled, {len(seen)} distinct, pool {a.pool}, kept {len(chosen)}: " + ", ".join(f"{k}={comp[k]}" for k in QUOTA))
        return
    ranked = ranked[: a.keep]
    with open(a.out, "w") as f:
        f.write(f"# {len(ranked)} titles; en.wikipedia daily top-1000, 1st+15th of each month {a.start}..{a.end},\n")
        f.write(f"# ranked by days present (max {len(days)}). Built by scripts/build-entity-corpus.py.\n")
        for t, _ in ranked:
            f.write(t + "\n")
    print(f"{len(days)} days sampled, {len(seen)} distinct titles, kept {len(ranked)}; min days-present {ranked[-1][1]}")


if __name__ == "__main__":
    main()
