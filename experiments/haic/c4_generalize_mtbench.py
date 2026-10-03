"""C4: does the item-level-divergence / leaderboard-invariance pattern generalise?

C2 found, on MultiPref, that expert and crowd annotators reverse the winner on
16.8% of items while producing a bit-identical leaderboard (Kendall tau = 1.0000).
A single dataset with six widely-spaced models is a weak basis for that claim.

MT-Bench human judgments is an independent test with a different pool contrast
that occurs naturally in the data: some judgments come from the benchmark's own
authors (`author_*`) and some from recruited expert annotators (`expert_*`).
Different corpus, different task (multi-turn), different models, different notion
of "which pool" - so agreement between the two studies is evidence about the
phenomenon rather than about MultiPref.

PRISM-alignment was considered and rejected: it assigns one participant per
conversation, so no two annotators ever rate the same item and the contrast is
undefined. HelpSteer2 was rejected: it ships aggregated per-response attribute
scores, with no per-annotator identity and no pairwise preferences.

Writes runs/c4/eval.json.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from datasets import load_dataset  # noqa: E402

from wsev.stats.disagreement import majority_vote  # noqa: E402
from wsev.stats.ranking import Battle, paired_bootstrap_tau  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
TIE_BREAK = ("A", "T", "B")
WINNER = {"model_a": "A", "model_b": "B", "tie": "T"}
# MT-Bench records a 3-way outcome; map onto the 5-point scale the shared ranking
# code expects, using only the unambiguous endpoints.
TO_SCALE = {"A": "A-is-clearly-better", "T": "Tie", "B": "B-is-clearly-better"}


def judge_pool(judge) -> str:
    """MT-Bench judge ids are ('<id>', '<role>')-ish; authors vs recruited experts."""
    s = judge[0] if isinstance(judge, (list, tuple)) else str(judge)
    return "author" if str(s).startswith("author") else "expert"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-battles", type=int, default=20)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", type=pathlib.Path, default=REPO / "runs" / "c4" / "eval.json")
    args = ap.parse_args()

    t0 = time.time()
    ds = load_dataset("lmsys/mt_bench_human_judgments", split="human")

    cells: dict[tuple, dict[str, list[str]]] = collections.defaultdict(
        lambda: collections.defaultdict(list))
    pair: dict[tuple, tuple[str, str]] = {}
    for x in ds:
        key = (x["question_id"], x["model_a"], x["model_b"], x["turn"])
        cells[key][judge_pool(x["judge"])].append(WINNER[x["winner"]])
        pair[key] = (x["model_a"], x["model_b"])

    both = {k: v for k, v in cells.items() if v.get("author") and v.get("expert")}
    n = len(both)
    if n < 100:
        raise RuntimeError(f"insufficient pool overlap: only {n} items rated by both pools")

    # Convention-free subset: both pools internally unanimous, so no tie-break is
    # consulted. Mirrors the MultiPref primary analysis (see c5_robustness.py); the
    # paper argues tie-breaking manufactures disagreement, so the replication corpus
    # must not silently rely on it.
    unan = {k: v for k, v in both.items()
            if len(set(v["author"])) == 1 and len(set(v["expert"])) == 1}
    u_diff = u_rev = 0
    for v in unan.values():
        ma_u, me_u = v["author"][0], v["expert"][0]
        u_diff += ma_u != me_u
        u_rev += (ma_u in ("A", "B") and me_u in ("A", "B") and ma_u != me_u)

    n_diff = n_rev = 0
    battles: list[Battle] = []
    labels_expert: list[tuple[str, ...]] = []
    labels_author: list[tuple[str, ...]] = []
    for key, v in both.items():
        ma_, me_ = majority_vote(v["author"], tie_break=TIE_BREAK), majority_vote(v["expert"], tie_break=TIE_BREAK)
        n_diff += ma_ != me_
        n_rev += (ma_ in ("A", "B") and me_ in ("A", "B") and ma_ != me_)
        a, b = pair[key]
        battles.append(Battle(a, b, tuple(TO_SCALE[x] for x in v["author"] + v["expert"])))
        labels_author.append(tuple(TO_SCALE[x] for x in v["author"]))
        labels_expert.append(tuple(TO_SCALE[x] for x in v["expert"]))

    # The CI must describe the population Table 1 reports (the convention-free subset) and
    # resample the unit the design has. MT-Bench reuses each question across several model
    # pairs and turns: the 246 unanimous cells sit in 69 questions, 98% of them repeated.
    # Bootstrapping all 292 cells as independent rows answered a different question, with a
    # tie-break in it. The interval is unchanged; the estimand is the fix.
    unan_keys = list(unan)
    b_u = [Battle(*pair[k], tuple(TO_SCALE[x] for x in both[k]["author"] + both[k]["expert"]))
           for k in unan_keys]
    le_u = [tuple(TO_SCALE[x] for x in both[k]["expert"]) for k in unan_keys]
    la_u = [tuple(TO_SCALE[x] for x in both[k]["author"]) for k in unan_keys]
    _q: dict = collections.defaultdict(list)
    for _i, _k in enumerate(unan_keys):
        _q[_k[0]].append(_i)
    res = paired_bootstrap_tau(b_u, le_u, la_u,
                               hard=True, n_boot=args.n_boot, min_battles=args.min_battles,
                               clusters=[v for v in _q.values()])

    out = {
        "milestone": "c4-generalize-mtbench",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_s": round(time.time() - t0, 1),
        "corpus": "lmsys/mt_bench_human_judgments (split=human)",
        "convention_free_subset": {
            "n_items": len(unan),
            "majority_differs_rate": u_diff / max(len(unan), 1),
            "winner_reversal_rate_excl_ties": u_rev / max(len(unan), 1),
            "note": "both pools internally unanimous; no tie-break consulted",
        },
        "tie_break_used_for_full_set": list(TIE_BREAK),
        "pool_contrast": "benchmark authors vs recruited expert annotators",
        "n_judgments": len(ds),
        "n_item_cells": len(cells),
        "n_items_both_pools": n,
        "majority_differs_rate": n_diff / n,
        "winner_reversal_rate_excl_ties": n_rev / n,
        "n_models_ranked": res.n_models,
        "kendall_tau_expert_vs_author": res.tau,
        "kendall_tau_ci95": [res.ci_low, res.ci_high],
        "models_displaced": res.displaced_models,
        "max_rank_shift": res.max_rank_shift,
        "multipref_reference": {
            "majority_differs_rate": 0.498,
            "winner_reversal_rate_excl_ties": 0.168,
            "kendall_tau": 1.0,
            "models_displaced": 0,
        },
        "resampling_unit": "question cluster (246 convention-free cells in 69 questions)",
        "limitation": (
            f"the tau CI is computed on the {len(unan)} convention-free cells, the population "
            f"Table 1 reports, resampling the {len(_q)} question clusters they sit in; it is "
            f"wide ([{res.ci_low:.4f}, {res.ci_high:.4f}]) because six models admit few "
            "discordant pairs, so modest rank changes are not excluded, though the point "
            "estimate shows none."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))

    print(f"items rated by both pools : {n:,} of {len(cells):,} cells")
    print(f"majority differs          : {n_diff}/{n} = {n_diff/n:.1%}   (MultiPref 49.8%)")
    print(f"winner reversal, no ties  : {n_rev}/{n} = {n_rev/n:.1%}   (MultiPref 16.8%)")
    print(f"Kendall tau expert/author : {res.tau:.4f}  95% CI [{res.ci_low:.4f}, {res.ci_high:.4f}]")
    print(f"models displaced          : {res.displaced_models}/{res.n_models}   (MultiPref 0/6)")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
