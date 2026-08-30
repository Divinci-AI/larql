#!/usr/bin/env python3
"""Probes for the local walk-vs-dense environment. See local-infer-env.sh.

Prompts are written in the fixture's own vocabulary: the tiny-vindex tokenizer
is WordLevel over the literal tokens "[0]".."[510]", so the prompt "[12] [34]"
encodes to exactly token ids [12, 34]. That exactness is the point -- it lets a
test drive a precise token sequence and get input-dependent output. (Before the
tokenizer was fixed, every prompt encoded to a run of zeros and the model
returned an identical distribution for every input, which made any agreement
between the two engines vacuous.)
"""
import base64, json, random, struct, sys, urllib.error, urllib.request

BASE = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:18791"


def post(path, obj):
    r = urllib.request.Request(BASE + path, data=json.dumps(obj).encode(),
                               headers={"content-type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(r))
    except urllib.error.HTTPError as e:
        return {"_err": e.code, "_body": e.read().decode()[:300]}


def unpatch(name):
    try:
        urllib.request.urlopen(urllib.request.Request(f"{BASE}/v1/patches/{name}", method="DELETE"))
    except Exception:
        pass


def preds(p, mode="walk", top=10):
    d = post("/v1/infer", {"prompt": p, "top": top, "mode": mode})
    return [(x["token"], x["probability"]) for x in d["predictions"]]


def rand_prompts(n, seed, lo=1, hi=8):
    rng = random.Random(seed)
    return [" ".join(f"[{rng.randint(0, 510)}]" for _ in range(rng.randint(lo, hi)))
            for _ in range(n)]


def compare(n=200):
    """Walk vs dense on an UNPATCHED vindex: they must agree exactly."""
    mism = 0
    tops = set()
    for p in rand_prompts(n, seed=42):
        d = post("/v1/infer", {"prompt": p, "top": 10, "mode": "compare"})
        w = [(x["token"], x["probability"]) for x in d["walk"]]
        de = [(x["token"], x["probability"]) for x in d["dense"]]
        tops.add(w[0][0])
        if w != de:
            mism += 1
            if mism <= 3:
                print(f"  DIVERGE {p}\n    walk {w[:3]}\n    dense{de[:3]}")
    print(f"prompts                : {n}")
    print(f"exact top-10 agreement : {n - mism}/{n}")
    print(f"distinct top-1 tokens  : {len(tops)}   (1 would mean the fixture ignores its input)")
    return mism == 0


def patch():
    """DELETE vs UPDATE: does a patch actually move the forward pass?

    DELETE is the op the Divinci 'suppress' model-edit emits. UPDATE (a new
    gate vector) is the control -- it is known to be honoured, so if DELETE
    shows no effect while UPDATE does, the difference is in patch handling
    and not in the measurement.
    """
    unpatch("t"); unpatch("u")
    prompts = rand_prompts(10, seed=5, lo=3, hi=3)
    base = {p: preds(p) for p in prompts}

    def apply(name, ops):
        post("/v1/patches/apply", {"name": name, "patch": {
            "version": 1, "base_model": "test/tiny-vindex",
            "created_at": "2026-08-30T00:00:00Z", "description": name,
            "operations": ops}})

    def changed():
        return sum(1 for p in prompts if preds(p) != base[p])

    print("DELETE — suppress 255 of 256 features (leaving one, so this is NOT")
    print("         the num_features==0 dense-fallback rung):")
    for layer in (0, 2, 4, 7):
        apply("t", [{"op": "delete", "layer": layer, "feature": f, "reason": "probe"}
                    for f in range(255)])
        print(f"  layer {layer}: walk output changed on {changed()}/{len(prompts)} prompts")
        unpatch("t")

    rng = random.Random(3)
    vec = [rng.gauss(0, 1) * 5 for _ in range(128)]
    apply("u", [{"op": "update", "layer": 4, "feature": 174,
                 "gate_vector_b64": base64.b64encode(struct.pack("128f", *vec)).decode()}])
    print(f"\nUPDATE — one new gate vector @ layer 4: changed on {changed()}/{len(prompts)} prompts")
    unpatch("u")


if __name__ == "__main__":
    {"compare": lambda: compare(), "patch": patch}[sys.argv[1]]()
