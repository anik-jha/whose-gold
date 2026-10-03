"""C5: robustness of every headline to the conventions it depends on.

Written in response to a PC-grade review that found three reporting defects:

  F1  The paper claims ties are broken toward the tie category. The code broke
      toward A. With exactly two annotators per pool and a 50.1% within-pool split
      rate, that convention decides the "majority" on roughly half the corpus, and
      MultiPref's label distribution is B-skewed 1.78:1, so an A-first rule
      manufactures opposite-winner pairs. This emits the full sensitivity table so
      the headline can be quoted under the rule the paper actually states.

  F6  "Sides with the crowd on 57.6%" was crowd/(crowd+expert), silently dropping
      the items where the judge matched neither pool. All three shares are emitted.

  F8  The cross-vendor judge (gemma4-26b-a4b) was added to runs/c3/ after this script
      was last run, so the paper quoted hand-estimated pool-split numbers for it that
      do not match a recomputation from raw.jsonl. Added to the tag loop below.

  F7  The quality-gap buckets existed only as prose in plan.json with no generator
      and did not reproduce. This computes them.

Zero GPU. Writes runs/c5/robustness.json.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from wsev.data.multipref import COARSE, load  # noqa: E402
from wsev.judges.preference import unswap  # noqa: E402
from wsev.stats.disagreement import majority_vote  # noqa: E402
from wsev.stats.ranking import Battle, ranking, win_rates  # noqa: E402
from wsev.stats.ranking import paired_bootstrap_tau  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]

# "as the paper states" is a FORCE rule, not an ordering - see Battle.a_credit_hard.
# NOTE (PC review F9): the leaderboard in runs/c2/eval.json ran the DEFAULT branch of
# a_credit_hard, which is alphabetical over coarse classes, i.e. A > B > Tie. It is NOT
# ("A","T","B"). Those two differ on Tie-vs-B splits and the corpus is 1.78:1 B-skewed,
# so they diverge badly: the default gives tau=1.0000, ("A","T","B") gives tau=0.4667.
# Both are reported so the paper can name what it actually ran.
RULES = {
    "default_alphabetical_what_c2_ran": ("A", "B", "T"),
    "force_tie_as_paper_states": "T",
    "B_first_mirror_check": ("B", "T", "A"),
    "A_over_tie_hypothetical": ("A", "T", "B"),
}


def _pool_majority(coarse: list[str], rule) -> str:
    """Coarse majority for one pool under either rule type (see Battle.a_credit_hard)."""
    counts = collections.Counter(coarse)
    top = max(counts.values())
    winners = [k for k, v in counts.items() if v == top]
    if len(winners) == 1:
        return winners[0]
    if isinstance(rule, str):
        return rule
    order = {c: i for i, c in enumerate(rule)}
    return min(winners, key=lambda w: order[w])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=REPO / "runs" / "c5" / "robustness.json")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    t0 = time.time()

    comps = [c for c in load() if c.labels("expert") and c.labels("normal")]
    n = len(comps)
    out: dict = {
        "milestone": "c5-robustness",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_comparisons": n,
        "why": "PC review F1/F2/F6/F7: headlines are convention-dependent; emit the conventions.",
    }

    # ---- F1: tie-break sensitivity of the two item-level headlines ----------
    tb_rows = {}
    for name, tb in RULES.items():
        differs = reverses = 0
        for c in comps:
            me = _pool_majority([COARSE[x] for x in c.labels("expert")], tb)
            mc = _pool_majority([COARSE[x] for x in c.labels("normal")], tb)
            differs += me != mc
            reverses += me in ("A", "B") and mc in ("A", "B") and me != mc
        tb_rows[name] = {"majority_differs": differs / n, "winner_reverses": reverses / n}

    # Tie-break-free: keep only items where BOTH pools are internally unanimous,
    # so no convention is consulted at all.
    unan = [c for c in comps
            if len({COARSE[x] for x in c.labels("expert")}) == 1
            and len({COARSE[x] for x in c.labels("normal")}) == 1]
    du = ru = 0
    for c in unan:
        me = COARSE[c.labels("expert")[0]]
        mc = COARSE[c.labels("normal")[0]]
        du += me != mc
        ru += me in ("A", "B") and mc in ("A", "B") and me != mc
    tb_rows["both_pools_unanimous_no_tiebreak"] = {
        "n_items": len(unan), "majority_differs": du / len(unan), "winner_reverses": ru / len(unan),
    }
    out["tiebreak_sensitivity"] = tb_rows

    # Label skew is the mechanism behind the A-first/B-first asymmetry.
    skew: collections.Counter[str] = collections.Counter()
    for c in comps:
        for x in c.labels():
            skew[COARSE[x]] += 1
    out["label_skew_coarse"] = dict(skew)
    out["B_to_A_vote_ratio"] = skew["B"] / max(skew["A"], 1)

    # ---- F2: leaderboard tau under the rule the paper states ----------------
    tau_by_rule = {}
    # Same prompt clustering as c2: MultiPref doubles every prompt across model pairs, so
    # the prompt is the resampling unit for these CIs too.
    _g: dict = collections.defaultdict(list)
    for _i, _c in enumerate(comps):
        _g[_c.prompt_id].append(_i)
    _clusters = [v for v in _g.values()]
    for name, tb in RULES.items():
        battles, el, cl = [], [], []
        for c in comps:
            battles.append(Battle(c.model_a, c.model_b, tuple(c.labels())))
            el.append(tuple(c.labels("expert")))
            cl.append(tuple(c.labels("normal")))
        res = paired_bootstrap_tau(battles, el, cl, hard=True, n_boot=args.n_boot,
                                   min_battles=50, tie_break=tb, clusters=_clusters)
        tau_by_rule[name] = {"tau": res.tau, "ci95": [res.ci_low, res.ci_high],
                             "models_displaced": res.displaced_models,
                             "n_models": res.n_models}
    out["leaderboard_tau_by_rule"] = tau_by_rule
    out["resampling_unit"] = "prompt cluster (MultiPref doubles every prompt across model pairs)"

    # ---- F7: does divergence depend on the model-quality gap? --------------
    # This ran under one tie-break rule, ("A","T","B"), and reported "flat". That rule is
    # the hypothetical outlier of F1 above: it is the only one of the four that moves the
    # leaderboard, and the paper names it an artifact of the corpus B-skew. Under every
    # other convention, and under the convention-free subset the paper's headline actually
    # uses, the sign is negative - divergence is HIGHER on closely matched pairs. Emit the
    # whole family so the paper cannot quote the one convention that flatters it.
    wr = win_rates([Battle(c.model_a, c.model_b, tuple(c.labels())) for c in comps],
                   hard=False, min_battles=50)

    def _tertiles(items, maj):
        gaps, divs = [], []
        for c in items:
            if c.model_a not in wr or c.model_b not in wr:
                continue
            gaps.append(abs(wr[c.model_a] - wr[c.model_b]))
            divs.append(float(maj(c, "expert") != maj(c, "normal")))
        gaps_a, divs_a = np.array(gaps), np.array(divs)
        qs = np.quantile(gaps_a, [1 / 3, 2 / 3])
        buckets = {}
        for label, mask in (("narrow", gaps_a <= qs[0]),
                            ("middle", (gaps_a > qs[0]) & (gaps_a <= qs[1])),
                            ("wide", gaps_a > qs[1])):
            buckets[label] = {"n": int(mask.sum()), "divergence": float(divs_a[mask].mean()),
                              "gap_range": [float(gaps_a[mask].min()), float(gaps_a[mask].max())]}
        r = float(np.corrcoef(gaps_a, divs_a)[0, 1])
        return {"n_items": int(len(gaps_a)), "tertiles": buckets,
                "pearson_r_gap_vs_divergence": r,
                "reading": ("flat" if abs(r) < 0.05
                            else ("rises with gap" if r > 0 else "falls with gap"))}

    qg = {"unanimous_subset_primary":
          _tertiles(unan, lambda c, pool: COARSE[c.labels(pool)[0]])}
    for name, tb in RULES.items():
        qg[name] = _tertiles(
            comps, lambda c, pool, r=tb: _pool_majority([COARSE[x] for x in c.labels(pool)], r))
    qg["note"] = ("Supersedes the 48.1%/51.9% figures in plan.json, which had no generator "
                  "and did not reproduce, AND the single-convention version of this block, "
                  "which reported only A_over_tie_hypothetical and read it as flat.")
    out["quality_gap_buckets"] = qg

    # ---- F6: three-way judge share on pool-split items ---------------------
    judges = {}
    for tag in ("qwen36-35b-a3b", "qwen36-27b-dense", "gemma4-26b-a4b"):
        raw = REPO / "runs" / "c3" / tag / "raw.jsonl"
        if not raw.exists():
            continue
        by_id = {c.comparison_id: c for c in comps}
        per: dict[str, dict[str, str]] = {}
        for line in raw.open():
            rr = json.loads(line)
            if rr.get("label_raw"):
                per.setdefault(rr["comparison_id"], {})[rr["order"]] = unswap(
                    rr["label_raw"], rr["order"])
        exp = crd = nei = 0
        for cid, o in per.items():
            if "ab" not in o or "ba" not in o or cid not in by_id:
                continue
            if COARSE[o["ab"]] != COARSE[o["ba"]]:
                continue
            c = by_id[cid]
            jc = COARSE[o["ab"]]
            me = majority_vote([COARSE[x] for x in c.labels("expert")], tie_break=("A", "T", "B"))
            mc = majority_vote([COARSE[x] for x in c.labels("normal")], tie_break=("A", "T", "B"))
            if me == mc:
                continue
            if jc == me:
                exp += 1
            elif jc == mc:
                crd += 1
            else:
                nei += 1
        tot = exp + crd + nei
        judges[tag] = {
            "n_pool_split_clean": tot,
            "expert": exp, "crowd": crd, "neither": nei,
            "expert_share_of_all": exp / max(tot, 1),
            "crowd_share_of_all": crd / max(tot, 1),
            "neither_share_of_all": nei / max(tot, 1),
            "crowd_share_of_decided": crd / max(exp + crd, 1),
        }
    out["judge_pool_split_three_way"] = judges
    out["wall_s"] = round(time.time() - t0, 1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))

    print(f"n={n:,}  label skew B:A = {out['B_to_A_vote_ratio']:.2f}:1\n")
    print(f"{'tie-break rule':36s} {'majority differs':>17s} {'winner reverses':>16s}")
    for k, v in tb_rows.items():
        extra = f"  (n={v['n_items']:,})" if "n_items" in v else ""
        print(f"{k:36s} {v['majority_differs']:16.2%} {v['winner_reverses']:15.2%}{extra}")
    print(f"\n{'rule':36s} {'tau':>6s} {'CI95':>22s} {'displaced':>10s}")
    for k, v in tau_by_rule.items():
        print(f"{k:36s} {v['tau']:6.4f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]{'':6s} "
              f"{v['models_displaced']:>3d}/{v['n_models']}")
    print(f"\n{'quality-gap tertiles, by convention':36s} {'narrow':>8s} {'middle':>8s} "
          f"{'wide':>8s} {'r':>8s}  reading")
    for k, v in out["quality_gap_buckets"].items():
        if not isinstance(v, dict):
            continue
        t = v["tertiles"]
        print(f"{k:36s} {t['narrow']['divergence']:8.1%} {t['middle']['divergence']:8.1%} "
              f"{t['wide']['divergence']:8.1%} {v['pearson_r_gap_vs_divergence']:+8.4f}  "
              f"{v['reading']} (n={v['n_items']:,})")
    print()
    for tag, v in judges.items():
        print(f"{tag:18s} pool-split n={v['n_pool_split_clean']:4d}  "
              f"expert {v['expert_share_of_all']:.1%} / crowd {v['crowd_share_of_all']:.1%} / "
              f"neither {v['neither_share_of_all']:.1%}   (crowd-of-decided {v['crowd_share_of_decided']:.1%})")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
