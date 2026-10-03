"""Build the judge table (paper Table 3) from the per-call outputs of every C3 run.

This closes a reproducibility hole. runs/c3/table2.json is quoted directly in the paper
and no script in the repository wrote it: it was maintained by hand, so it could not pick
up a new judge and could silently disagree with runs/c3/<tag>/eval.json. It now derives
from raw.jsonl for every tag present, and asserts it reproduces the two values that were
already published before accepting a third.

The table reports the SELF-CONSISTENT subset: items where the judge gives the same coarse
direction in both presentation orders. That is a deliberate, pre-registered restriction -
an order-inconsistent judgement carries no information about which pool the judge tracks,
and including it only adds noise in a known direction. The all-items estimator is reported
separately in each run's eval.json and in the paper's table caption.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from wsev.data.multipref import COARSE, load  # noqa: E402
from wsev.judges.preference import combine_orders, unswap  # noqa: E402
from wsev.stats.disagreement import majority_vote  # noqa: E402

TIE_BREAK = ("A", "T", "B")
# Values published before this script existed; reproducing them is the correctness gate.
PUBLISHED = {"qwen36-35b-a3b": (1383, -0.0687), "qwen36-27b-dense": (1884, -0.0520)}
# Not an independent judge: the same weights as qwen36-35b-a3b served in a different output
# mode, run for the format-sensitivity check. Listing it as a third row would double-count
# one model and inflate the apparent replication.
NOT_A_JUDGE = {"qwen36-35b-a3b-jsonmode"}


def sample_ids(sample: int, split_frac: float, seed: int):
    """Reconstruct the stratified sample c3_judge_pools drew, so tags stay comparable."""
    comps = list(load())
    split, agree = [], []
    for c in comps:
        me = majority_vote([COARSE[x] for x in c.labels("expert")], tie_break=TIE_BREAK)
        mc = majority_vote([COARSE[x] for x in c.labels("normal")], tie_break=TIE_BREAK)
        (split if me != mc else agree).append(c)
    rng = np.random.default_rng(seed)
    n_split = min(len(split), int(round(sample * split_frac)))
    n_agree = min(len(agree), sample - n_split)
    pick = ([split[i] for i in rng.choice(len(split), n_split, replace=False)]
            + [agree[i] for i in rng.choice(len(agree), n_agree, replace=False)])
    return {c.comparison_id: c for c in pick}


def analyse(tag: str, by_id: dict, n_boot: int, seed: int) -> dict | None:
    raw = REPO / "runs" / "c3" / tag / "raw.jsonl"
    if not raw.exists():
        return None
    per_item: dict[str, dict] = collections.defaultdict(dict)
    with raw.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            lab = r.get("label_raw")
            per_item[r["comparison_id"]][r["order"]] = unswap(lab, r["order"]) if lab else None

    ae, ac, n_scored, n_incons = [], [], 0, 0
    for cid, orders in per_item.items():
        c = by_id.get(cid)
        if c is None or "ab" not in orders or "ba" not in orders:
            continue
        label, _ = combine_orders(orders["ab"], orders["ba"])
        if label is None:
            continue
        n_scored += 1
        # "Self-consistent" means the two presentation orders agree on the DIRECTION.
        # combine_orders' own flag is stricter: it also fires when the orders differ only
        # in strength (clearly- vs slightly-better), which is not a position artefact and
        # would discard roughly a third of the usable items.
        ab, ba = orders["ab"], orders["ba"]
        if ab is None or ba is None or COARSE[ab] != COARSE[ba]:
            n_incons += 1
            continue
        jc = COARSE[label]
        me = majority_vote([COARSE[x] for x in c.labels("expert")], tie_break=TIE_BREAK)
        mc = majority_vote([COARSE[x] for x in c.labels("normal")], tie_break=TIE_BREAK)
        ae.append(1.0 if jc == me else 0.0)
        ac.append(1.0 if jc == mc else 0.0)

    ae_a, ac_a = np.asarray(ae), np.asarray(ac)
    d = ae_a - ac_a  # paired: same items, two references
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {
        "n_items_scored": n_scored,
        "direction_inconsistency": round(n_incons / n_scored, 6) if n_scored else None,
        "n_clean": len(ae_a),
        "agree_expert": round(float(ae_a.mean()), 6),
        "agree_crowd": round(float(ac_a.mean()), 6),
        "diff": round(float(d.mean()), 6),
        "ci95": [round(float(lo), 6), round(float(hi), 6)],
        "replicates": bool(hi < 0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=2500)
    ap.add_argument("--split-frac", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", type=pathlib.Path, default=REPO / "runs/c3/table2.json")
    args = ap.parse_args()

    by_id = sample_ids(args.sample, args.split_frac, args.seed)
    tags = sorted(p.name for p in (REPO / "runs/c3").iterdir()
                  if p.is_dir() and (p / "raw.jsonl").exists() and p.name not in NOT_A_JUDGE)

    out = {}
    for tag in tags:
        r = analyse(tag, by_id, args.n_boot, args.seed)
        if r is None:
            continue
        r["preregistered_sample"] = args.sample
        out[tag] = r
        print(f"{tag:22s} n_clean={r['n_clean']:5d} expert={r['agree_expert']:.4f} "
              f"crowd={r['agree_crowd']:.4f} diff={r['diff']:+.4f} "
              f"ci={[round(c,4) for c in r['ci95']]} replicates={r['replicates']}")

    for tag, (n_clean, diff) in PUBLISHED.items():
        if tag not in out:
            raise SystemExit(f"published tag {tag} missing; refusing to overwrite the table")
        got = out[tag]
        assert got["n_clean"] == n_clean, f"{tag}: n_clean {got['n_clean']} != published {n_clean}"
        assert abs(got["diff"] - diff) < 5e-4, f"{tag}: diff {got['diff']} != published {diff}"
    print("\nreproduced both published rows; writing table")
    args.out.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
