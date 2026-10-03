"""C7: what the judge-pool effect is on the CORPUS, not on the sample we drew.

C3 judged a deliberately stratified sample: 60% pool-split items, 40% pool-agree, because
power to separate the two pools lives almost entirely in the split items. That is the right
way to spend inference, and it makes every agreement number in C3 a statement about the
SAMPLE rather than about MultiPref. The sample overweights pool-split items relative to
their corpus fraction, which this script computes directly from the sampling frame.

This script reweights. Each scored item carries w = N_stratum / n_stratum, the inverse of
its sampling probability (using scored counts, assuming ignorable missingness within
strata), estimating a corpus-weighted mean. The CI is a stratified paired row bootstrap:
resample within each stratum, keeping the same items for the expert and crowd arms, since
the two arms differ only in whose labels the judge is scored against.

Reports both estimators side by side. The sample-level one is what C3 reported and remains
correct as a statement about the sample. C8 preserves these historical outputs and adds
prompt-cluster uncertainty and label-convention controls for the camera-ready.

Writes runs/c7/eval.json.

    ~/projects/moe_pruning/.venv/bin/python experiments/haic/c7_ipw_judge_pools.py
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from wsev.data.multipref import COARSE, load  # noqa: E402
from wsev.judges.preference import combine_orders, unswap  # noqa: E402
from wsev.stats.disagreement import defensible_set, majority_vote  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from c3_table2 import sample_ids  # noqa: E402

TIE_BREAK = ("A", "T", "B")
TAGS = ["qwen36-35b-a3b", "qwen36-27b-dense", "gemma4-26b-a4b"]


def stratified_paired_ci(x, y, strata, weights, *, n_boot=5000, alpha=0.05, seed=0):
    """Paired bootstrap on the weighted mean difference, resampling within strata."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])
        w = weights[idx]
        diffs[b] = np.average(x[idx], weights=w) - np.average(y[idx], weights=w)
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    est = float(np.average(x, weights=weights) - np.average(y, weights=weights))
    return est, float(lo), float(hi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=2500)
    ap.add_argument("--split-frac", type=float, default=0.6)
    ap.add_argument("--sample-seed", type=int, default=17)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=pathlib.Path, default=REPO / "runs" / "c7" / "eval.json")
    args = ap.parse_args()
    t0 = time.time()

    comps = [c for c in load() if c.labels("expert") and c.labels("normal")]
    # Restrict to the pre-registered stratified sample, exactly as c3_table2 reconstructs
    # it. Reading raw.jsonl unfiltered would fold in the under-powered pilot run of
    # qwen36-27b-dense, whose items were drawn from a different stratified draw.
    by_id = sample_ids(args.sample, args.split_frac, args.sample_seed)

    # Population strata: the sampling frame C3 drew from, under the same tie-break rule.
    pop = collections.Counter()
    for c in comps:
        me = majority_vote([COARSE[x] for x in c.labels("expert")], tie_break=TIE_BREAK)
        mc = majority_vote([COARSE[x] for x in c.labels("normal")], tie_break=TIE_BREAK)
        pop["pool_split" if me != mc else "pool_agree"] += 1
    n_pop = sum(pop.values())

    out = {
        "milestone": "c7-ipw-judge-pools",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "population": {"n": n_pop, **dict(pop),
                       "pool_split_frac": round(pop["pool_split"] / n_pop, 5)},
        "judges": {},
    }
    print(f"population: {n_pop} comparisons, {pop['pool_split']} pool-split "
          f"({pop['pool_split'] / n_pop:.1%}), {pop['pool_agree']} pool-agree\n")

    for tag in TAGS:
        raw = REPO / "runs" / "c3" / tag / "raw.jsonl"
        if not raw.exists():
            print(f"{tag}: no raw.jsonl, skipping")
            continue

        per_item: dict[str, dict[str, str | None]] = collections.defaultdict(dict)
        with raw.open() as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                lab = r.get("label_raw")
                per_item[r["comparison_id"]][r["order"]] = (
                    None if lab is None else unswap(lab, r["order"]))

        ae, ac, ee, ec, st = [], [], [], [], []
        for cid, orders in per_item.items():
            c = by_id.get(cid)
            if c is None or "ab" not in orders or "ba" not in orders:
                continue
            label, _ = combine_orders(orders["ab"], orders["ba"])
            if label is None:
                continue
            jc = COARSE[label]
            exp = [COARSE[x] for x in c.labels("expert")]
            crd = [COARSE[x] for x in c.labels("normal")]
            me = majority_vote(exp, tie_break=TIE_BREAK)
            mc = majority_vote(crd, tie_break=TIE_BREAK)
            ae.append(1.0 if jc == me else 0.0)
            ac.append(1.0 if jc == mc else 0.0)
            ee.append(1.0 if (jc in defensible_set(exp) and jc != me) else 0.0)
            ec.append(1.0 if (jc in defensible_set(crd) and jc != mc) else 0.0)
            st.append("pool_split" if me != mc else "pool_agree")

        ae, ac = np.array(ae), np.array(ac)
        ee, ec = np.array(ee), np.array(ec)
        st = np.array(st)
        n_s = collections.Counter(st.tolist())
        w = np.array([pop[s] / n_s[s] for s in st])

        # Unweighted, i.e. the sample-level estimator C3 committed to and reported.
        flat = np.ones_like(w)
        u_a = stratified_paired_ci(ae, ac, st, flat, n_boot=args.n_boot, seed=args.seed)
        u_e = stratified_paired_ci(ee, ec, st, flat, n_boot=args.n_boot, seed=args.seed)
        # Reweighted to the corpus.
        r_a = stratified_paired_ci(ae, ac, st, w, n_boot=args.n_boot, seed=args.seed)
        r_e = stratified_paired_ci(ee, ec, st, w, n_boot=args.n_boot, seed=args.seed)

        rec = {
            "n_scored": int(len(ae)),
            "sample_strata": dict(n_s),
            "sample_pool_split_frac": round(n_s["pool_split"] / len(ae), 5),
            "sample": {
                "agree_expert": round(float(ae.mean()), 5),
                "agree_crowd": round(float(ac.mean()), 5),
                "agree_diff": round(u_a[0], 5), "agree_ci95": [round(u_a[1], 5), round(u_a[2], 5)],
                "erasure_diff": round(u_e[0], 5), "erasure_ci95": [round(u_e[1], 5), round(u_e[2], 5)],
            },
            "corpus_ipw": {
                "agree_expert": round(float(np.average(ae, weights=w)), 5),
                "agree_crowd": round(float(np.average(ac, weights=w)), 5),
                "agree_diff": round(r_a[0], 5), "agree_ci95": [round(r_a[1], 5), round(r_a[2], 5)],
                "agree_excludes_zero": bool(r_a[1] > 0 or r_a[2] < 0),
                "erasure_diff": round(r_e[0], 5), "erasure_ci95": [round(r_e[1], 5), round(r_e[2], 5)],
                "erasure_excludes_zero": bool(r_e[1] > 0 or r_e[2] < 0),
            },
        }
        out["judges"][tag] = rec
        print(f"{tag}")
        print(f"  sample  ({len(ae)} items, {rec['sample_pool_split_frac']:.1%} split): "
              f"agree {u_a[0]:+.4f} [{u_a[1]:+.4f}, {u_a[2]:+.4f}]   "
              f"erasure {u_e[0]:+.4f} [{u_e[1]:+.4f}, {u_e[2]:+.4f}]")
        print(f"  corpus  (IPW to {rec['sample_pool_split_frac']:.1%} -> "
              f"{pop['pool_split'] / n_pop:.1%} split): "
              f"agree {r_a[0]:+.4f} [{r_a[1]:+.4f}, {r_a[2]:+.4f}]"
              f"{' *' if (r_a[1] > 0 or r_a[2] < 0) else ''}   "
              f"erasure {r_e[0]:+.4f} [{r_e[1]:+.4f}, {r_e[2]:+.4f}]"
              f"{' *' if (r_e[1] > 0 or r_e[2] < 0) else ''}")
        print()

    out["wall_s"] = round(time.time() - t0, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
