"""Exploratory camera-ready controls from saved outputs, without model inference.

Exp(1) weight family: https://arxiv.org/html/1808.08199 section 2.2.
Bootstrap null centering: https://www150.statcan.gc.ca/n1/pub/12-001-x/2009001/article/10882-eng.pdf
Our prompt/stratum combination is a superpopulation sensitivity, not design-exact inference.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from wsev.data.multipref import COARSE, load  # noqa: E402
from wsev.judges.preference import combine_orders, unswap  # noqa: E402
from wsev.stats.judge_controls import (  # noqa: E402
    agreement_features, holm, marginal_baselines, pool_label,
    poststratified_weights, summarize,
)

TAGS = ("qwen36-35b-a3b", "qwen36-27b-dense", "gemma4-26b-a4b")
ORIGINAL = ("A", "T", "B")


def read_calls(path, sample):
    """Match historical last-record-wins behavior and expose its missingness."""
    calls = collections.defaultdict(dict)
    duplicates = 0
    for line in path.read_text().splitlines():
        r = json.loads(line)
        cid, order = r["comparison_id"], r["order"]
        if cid not in sample:
            continue
        if order not in ("ab", "ba"):
            raise ValueError(f"unknown presentation order {order}")
        duplicates += order in calls[cid]
        lab = r.get("label_raw")
        calls[cid][order] = unswap(lab, order) if lab else None
    rows, judges, ab, ba = [], [], [], []
    for cid, orders in calls.items():
        if "ab" not in orders or "ba" not in orders:
            continue
        lab, _ = combine_orders(orders["ab"], orders["ba"])
        if lab is None:
            continue
        rows.append(sample[cid])
        judges.append(COARSE[lab])
        ab.append(COARSE[orders["ab"]] if orders["ab"] else "missing")
        ba.append(COARSE[orders["ba"]] if orders["ba"] else "missing")
    return rows, np.array(judges), np.array(ab), np.array(ba), duplicates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=pathlib.Path, default=REPO / "runs/c8/labels.jsonl")
    ap.add_argument("--out", type=pathlib.Path, default=REPO / "runs/c8/eval.json")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20261002)
    args = ap.parse_args()
    start = time.monotonic()
    if args.labels.exists():
        labels = [json.loads(x) for x in args.labels.read_text().splitlines()]
    else:
        labels = [{"comparison_id": c.comparison_id, "prompt_id": c.prompt_id,
                   "expert": c.coarse_labels("expert"), "normal": c.coarse_labels("normal")}
                  for c in load() if c.labels("expert") and c.labels("normal")]
        args.labels.parent.mkdir(parents=True, exist_ok=True)
        args.labels.write_text("".join(json.dumps(r) + "\n" for r in labels))
    for r in labels:
        r["stratum"] = int(pool_label(r["expert"], ORIGINAL) != pool_label(r["normal"], ORIGINAL))
    pop = collections.Counter(r["stratum"] for r in labels)
    masses = {s: n / len(labels) for s, n in pop.items()}
    rng = np.random.default_rng(17)
    selected = []
    for s, n in ((1, 1500), (0, 1000)):
        group = [r for r in labels if r["stratum"] == s]
        selected.extend(group[i] for i in rng.choice(len(group), n, replace=False))
    sample = {r["comparison_id"]: r for r in selected}
    prompts = sorted({r["prompt_id"] for r in selected})
    prompt_index = {p: i for i, p in enumerate(prompts)}
    mult = np.random.default_rng(args.seed).exponential(size=(args.n_boot, len(prompts)))
    out = {"milestone": "c8-camera-ready-exploratory-controls", "n_boot": args.n_boot,
           "seed": args.seed, "sample_seed": 17, "sample_n": len(sample),
           "population_strata": dict(pop), "resampling_unit": "prompt cluster",
           "method": "shared Exp(1) prompt multipliers; fixed original-stratum masses",
           "inference": "exploratory independent-prompt superpopulation sensitivity; no finite-population correction",
           "p_method": "two-sided unstudentized centered-multiplier approximate p; plus-one Monte Carlo adjustment",
           "judges": {}}
    c7 = json.loads((REPO / "runs/c7/eval.json").read_text())
    input_paths = [args.labels, REPO / "runs/c7/eval.json", pathlib.Path(__file__),
                   REPO / "src/wsev/stats/judge_controls.py"]
    for tag in TAGS:
        raw = REPO / "runs/c3" / tag / "raw.jsonl"
        input_paths.append(raw)
        rows, judge, ab, ba, duplicates = read_calls(raw, sample)
        strata = np.array([r["stratum"] for r in rows])
        pids = np.array([prompt_index[r["prompt_id"]] for r in rows])
        multipliers = np.vstack([np.ones(len(rows)), mult[:, pids]])
        w = poststratified_weights(strata, masses, multipliers)
        del multipliers
        e, n = [r["expert"] for r in rows], [r["normal"] for r in rows]
        feats = agreement_features(judge, e, n)
        names = list(feats)
        estimates = w @ np.column_stack([feats[x] for x in names])
        stats = {name: summarize(estimates[0, i], estimates[1:, i]) for i, name in enumerate(names)}
        assert round(stats["A_T_B"]["estimate"], 5) == c7["judges"][tag]["corpus_ipw"]["agree_diff"]
        assert round(stats["erasure"]["estimate"], 5) == c7["judges"][tag]["corpus_ipw"]["erasure_diff"]
        unanimous = np.array([len(set(ex)) == len(set(no)) == 1 for ex, no in zip(e, n)])
        complete = (ab != "missing") & (ba != "missing")
        for name, mask, feature in (("unanimous", unanimous, feats["A_T_B"]),
                                    ("complete_orders", complete, feats["A_T_B"])):
            vals = (w @ (mask * feature)) / (w @ mask)
            stats[name] = summarize(vals[0], vals[1:])
        for order, labs in (("ab", ab), ("ba", ba)):
            valid = labs != "missing"
            order_features = agreement_features(labs, e, n)
            for name in ("A_T_B", "per_annotator"):
                vals = (w @ (order_features[name] * valid)) / (w @ valid)
                stats[f"{order}_{name}"] = summarize(vals[0], vals[1:])
        me = np.array([pool_label(x, ORIGINAL) for x in e])
        mn = np.array([pool_label(x, ORIGINAL) for x in n])
        baselines = marginal_baselines(w, strata, masses, judge, me, mn)
        for name, baseline in zip(("global_marginal", "stratified_marginal"), baselines):
            stats[name] = summarize(baseline[0], baseline[1:])
            excess = estimates[:, names.index("A_T_B")] - baseline
            stats[f"excess_{name}"] = summarize(excess[0], excess[1:])
        prompt_counts = collections.Counter(r["prompt_id"] for r in rows)
        split_un = unanimous & (me != mn)
        rec = {"n_scored": len(rows), "n_prompts": len(prompt_counts),
               "rows_in_repeated_prompts": sum(v for v in prompt_counts.values() if v > 1),
               "sample_strata": dict(collections.Counter(strata.tolist())),
               "missingness": {"unscored": len(sample) - len(rows),
                               "one_valid_order": int((~complete).sum()),
                               "duplicate_records_in_sample": duplicates,
                               "selection": "last record per comparison/order, as in C7"},
               "n_unanimous": int(unanimous.sum()), "n_unanimous_split": int(split_un.sum()),
               "unanimous_split_matches": {"expert": int(((judge == me) & split_un).sum()),
                                           "normal": int(((judge == mn) & split_un).sum()),
                                           "neither": int(((judge != me) & (judge != mn) & split_un).sum())},
               "statistics": stats}
        out["judges"][tag] = rec
        print(tag, json.dumps({k: stats[k] for k in ("A_T_B", "per_annotator", "excess_stratified_marginal")}))
        del w
    family = [(tag, metric) for tag in TAGS for metric in ("A_T_B", "erasure")]
    for (tag, metric), p in zip(family, holm([out["judges"][t]["statistics"][m]["p_centered_approx"] for t, m in family])):
        out["judges"][tag]["statistics"][metric]["p_holm_six"] = p
    for tag, p in zip(TAGS, holm([out["judges"][t]["statistics"]["per_annotator"]["p_centered_approx"] for t in TAGS])):
        out["judges"][tag]["statistics"]["per_annotator"]["p_holm_three"] = p
    out["input_sha256"] = {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in input_paths}
    out["wall_s"] = time.monotonic() - start
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
