"""C2: does the choice of annotator pool change the model leaderboard?

The MultiPref paper (arXiv 2410.19133, Limitations, "Intra-group variability of
annotators") states: "One of our key assumptions is that there is no variability
in intra-group annotators ... the dataset disambiguates between normal and expert
crowdworker annotations. We leave this exploration for future work."

This is that exploration. Zero GPU: the comparison is a re-analysis of labels that
already exist.

PRE-REGISTERED (runs/plan.json, milestone c2-whose-gold):
    KILL if Kendall tau between the expert-gold and crowd-gold leaderboards is
    > 0.9 AND its bootstrap CI lower bound is > 0.9. Report the null and stop;
    do not enlarge the model set hunting for a flip.

Writes runs/c2/eval.json.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from wsev.data.multipref import COARSE, load  # noqa: E402
from wsev.stats.disagreement import majority_vote  # noqa: E402
from wsev.stats.ranking import (  # noqa: E402
    Battle,
    kendall_tau,
    paired_bootstrap_tau,
    ranking,
    win_rates,
)

REPO = pathlib.Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-battles", type=int, default=50,
                    help="minimum battles for a model to enter the leaderboard")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", type=pathlib.Path, default=REPO / "runs" / "c2" / "eval.json")
    args = ap.parse_args()

    t0 = time.time()
    battles: list[Battle] = []
    expert_labels: list[tuple[str, ...]] = []
    crowd_labels: list[tuple[str, ...]] = []

    comps_kept = []
    for comp in load():
        exp = tuple(comp.labels("expert"))
        cro = tuple(comp.labels("normal"))
        if not exp or not cro:
            continue
        battles.append(Battle(comp.model_a, comp.model_b, tuple(comp.labels())))
        comps_kept.append(comp)
        expert_labels.append(exp)
        crowd_labels.append(cro)

    n = len(battles)
    if n == 0:
        raise RuntimeError("no battles with both pools")

    appearances: collections.Counter[str] = collections.Counter()
    for b in battles:
        appearances[b.model_a] += 1
        appearances[b.model_b] += 1

    be = [Battle(b.model_a, b.model_b, lx) for b, lx in zip(battles, expert_labels)]
    bc = [Battle(b.model_a, b.model_b, ly) for b, ly in zip(battles, crowd_labels)]

    wr_e = win_rates(be, hard=True, min_battles=args.min_battles)
    wr_c = win_rates(bc, hard=True, min_battles=args.min_battles)
    common = sorted(set(wr_e) & set(wr_c))
    wr_e = {m: wr_e[m] for m in common}
    wr_c = {m: wr_c[m] for m in common}
    rank_e, rank_c = ranking(wr_e), ranking(wr_c)

    # MultiPref reuses each prompt for two comparisons against different model pairs, so
    # rows are not independent units. Cluster by prompt; a whole cluster travels together.
    _g: dict = collections.defaultdict(list)
    for _i, _c in enumerate(comps_kept):
        _g[_c.prompt_id].append(_i)
    res = paired_bootstrap_tau(
        battles, expert_labels, crowd_labels,
        hard=True, n_boot=args.n_boot, min_battles=args.min_battles,
        clusters=[v for v in _g.values()],
    )

    # Soft (disagreement-aware) scoring on the pooled annotators, for the H1 contrast.
    wr_soft = win_rates(battles, hard=False, min_battles=args.min_battles)
    wr_hard = win_rates(battles, hard=True, min_battles=args.min_battles)
    wr_soft = {m: v for m, v in wr_soft.items() if m in common}
    wr_hard = {m: v for m, v in wr_hard.items() if m in common}
    tau_soft_hard = kendall_tau(ranking(wr_soft), ranking(wr_hard))

    ie = {m: i for i, m in enumerate(rank_e)}
    ic = {m: i for i, m in enumerate(rank_c)}
    table = [
        {
            "model": m,
            "battles": appearances[m],
            "expert_winrate": wr_e[m],
            "crowd_winrate": wr_c[m],
            "expert_rank": ie[m] + 1,
            "crowd_rank": ic[m] + 1,
            "rank_shift": ic[m] - ie[m],
        }
        for m in sorted(common, key=lambda x: ie[x])
    ]

    # Winner reversal: both pools name a decisive winner and they disagree. This is
    # the paper's headline item-level number, so it is computed here and persisted
    # rather than derived ad hoc at figure time.
    tb = ("A", "T", "B")
    n_major_diff = n_reversal = 0
    for comp, exp, cro in zip(comps_kept, expert_labels, crowd_labels):
        me = majority_vote([COARSE[x] for x in exp], tie_break=tb)
        mc = majority_vote([COARSE[x] for x in cro], tie_break=tb)
        n_major_diff += me != mc
        n_reversal += (me in ("A", "B") and mc in ("A", "B") and me != mc)

    bar = 0.9
    killed = res.tau > bar and res.ci_low > bar

    out = {
        "milestone": "c2-whose-gold",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_s": round(time.time() - t0, 1),
        "corpus": "allenai/multipref",
        "prior_art_note": (
            "arXiv 2410.19133 Limitations: 'One of our key assumptions is that there is no "
            "variability in intra-group annotators ... We leave this exploration for future work.'"
        ),
        "preregistered_kill_bar": {"kendall_tau_gt": bar, "and_ci_low_gt": bar},
        "n_battles": n,
        "majority_differs_rate": n_major_diff / n,
        "winner_reversal_rate_excl_ties": n_reversal / n,
        "n_winner_reversals": n_reversal,
        "min_battles_for_leaderboard": args.min_battles,
        "n_models_ranked": len(common),
        "kendall_tau_expert_vs_crowd": res.tau,
        "kendall_tau_ci95": [res.ci_low, res.ci_high],
        "n_bootstrap": res.n_boot,
        "resampling_unit": "prompt cluster (MultiPref doubles every prompt across model pairs)",
        "models_displaced": res.displaced_models,
        "max_rank_shift": res.max_rank_shift,
        "kendall_tau_soft_vs_hard_pooled": tau_soft_hard,
        "leaderboard": table,
        "verdict": {"killed_by_prereg_bar": killed},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))

    print(f"battles={n:,}  models ranked={len(common)}  (min {args.min_battles} battles)")
    print(f"Kendall tau (expert-gold vs crowd-gold) = {res.tau:.4f}"
          f"   95% CI [{res.ci_low:.4f}, {res.ci_high:.4f}]  boot={res.n_boot}")
    print(f"  models displaced = {res.displaced_models}/{len(common)}   max rank shift = {res.max_rank_shift}")
    print(f"  tau(soft vs hard, pooled) = {tau_soft_hard:.4f}")
    print(f"  pre-registered kill (tau>0.9 AND CI_low>0.9): {'KILLED' if killed else 'SURVIVES'}")
    print()
    print(f"{'model':52s} {'exp':>5s} {'crd':>5s} {'shift':>6s}")
    for r in table:
        print(f"{r['model'][:52]:52s} {r['expert_rank']:5d} {r['crowd_rank']:5d} {r['rank_shift']:+6d}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
