#!/usr/bin/env python3
"""Build crates/larql-server/assets/entity-corpus.txt: the relevance `corpus` panel.

Source: English Wikipedia's daily top-1000 most-viewed articles (Wikimedia
pageviews REST API), sampled on the 1st and 15th of every month over the
window below. A title is kept by how many sampled days it appears in, so the
corpus is the durably looked-up population of things — not one week's news.

Filters: namespaces (anything with ':'), "List of …", "Deaths in …",
year/date-shaped titles, and titles under three characters are dropped;
disambiguators like " (film)" are stripped; underscores become spaces.

Usage:  scripts/build-entity-corpus.py [--days N] [--out PATH]
"""
import argparse, collections, json, re, sys, time, urllib.request

API = "https://wikimedia.org/api/rest_v1/metrics/pageviews/top/en.wikipedia/all-access/{y}/{m:02d}/{d:02d}"
UA = "larql entity-corpus builder (https://github.com/Divinci-AI/larql)"
DROP = re.compile(r"^(List_of|Lists_of|Deaths_in|Main_Page$|\d{1,4}(_in_|$|_BC)|\d{4}[_-]\d)|[:]")
# Search-box noise and adult-site titles that top the daily list: not the
# population a browse is drawn from, and not something to ship in a data file.
NOISE = re.compile(r"^(X+|XXX.*|XNXX|XVideos|Pornhub|.*[Pp]orn.*|Sex|Sexual_intercourse)$")
# The entities DESCRIBE quality is measured against (notes/research, Divinci
# server repo). Held out so no measurement is against its own background.
HELD_OUT = {"Paris", "France", "Tokyo", "Einstein", "Albert Einstein", "Amazon",
            "Amazon (company)", "Beethoven", "Ludwig van Beethoven"}


def clean(title: str):
    if DROP.search(title) or NOISE.fullmatch(title):
        return None
    t = re.sub(r"_\([^)]*\)$", "", title).replace("_", " ").strip()
    if len(t) < 3 or re.fullmatch(r"[\d\s.,-]+", t) or t in HELD_OUT:
        return None
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-01")
    ap.add_argument("--end", default="2026-08")
    ap.add_argument("--keep", type=int, default=2048)
    ap.add_argument("--out", default="crates/larql-server/assets/entity-corpus.txt")
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
    ranked = sorted(seen.items(), key=lambda kv: (-kv[1], kv[0]))[: a.keep]
    with open(a.out, "w") as f:
        f.write(f"# {len(ranked)} titles; en.wikipedia daily top-1000, 1st+15th of each month {a.start}..{a.end},\n")
        f.write(f"# ranked by days present (max {len(days)}). Built by scripts/build-entity-corpus.py.\n")
        for t, _ in ranked:
            f.write(t + "\n")
    print(f"{len(days)} days sampled, {len(seen)} distinct titles, kept {len(ranked)}; min days-present {ranked[-1][1]}")


if __name__ == "__main__":
    main()
