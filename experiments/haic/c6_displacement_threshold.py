"""C6: when DOES annotator-pool choice reach the leaderboard?

C2 found Kendall tau = 1.0000 with zero of six models displaced, and the paper reports
that as "aggregate rankings are robust to who annotates". That claim is weaker than it
looks, and this script measures how much weaker.

Six widely spaced models is a low-power test: with few adjacent pairs and large gaps
between them, a leaderboard can be invariant simply because nothing was close enough to
swap. A null obtained that way says little about the fifty-model arenas the field actually
publishes. So rather than leave the reader to guess, we turn the null into a threshold.

Three measurements, all grounded in the observed corpus rather than assumed:

  1. PERTURBATION SIZE. Switching pools shifts each model's win rate by some amount. We
     measure the distribution of those shifts directly. This is the effect size that any
     leaderboard has to survive.

  2. HOW LUCKY WAS THE NULL. Item-level bootstrap of the real corpus, recomputing BOTH
     leaderboards on each resample and counting how often any model is displaced. If the
     observed 0/6 is atypical of the resampling distribution, the invariance is partly
     luck and must be reported as such.

  3. THE THRESHOLD. Given the measured perturbation, the probability that two adjacent
     models swap is a function of the gap between them. We report the gap at which a swap
     becomes likely.

     The K-model column is an ILLUSTRATIVE SCENARIO and is labelled as one everywhere it
     appears: it extrapolates from six observed shifts under assumptions (iid Gaussian
     shifts, uniform spacing over the observed span) that six models cannot check. It is
     not a measurement and it is not a bound. What it does do correctly, since 2026-08-21,
     is share models between adjacent pairs -- the earlier closed form 1-(1-p)^(K-1)
     treated the K-1 swap events as independent, which they are not.

Writes runs/c6/eval.json.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from wsev.data.multipref import load  # noqa: E402
from wsev.stats.ranking import Battle, kendall_tau, ranking, win_rates  # noqa: E402


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-battles", type=int, default=50)
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--n-mc", type=int, default=200000,
                    help="Monte Carlo draws for the illustrative K-model scenario.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=pathlib.Path, default=REPO / "runs" / "c6" / "eval.json")
    args = ap.parse_args()
    t0 = time.time()

    battles, exp_lab, cro_lab, unanimous, prompt_ids = [], [], [], [], []
    for comp in load():
        e, c = tuple(comp.labels("expert")), tuple(comp.labels("normal"))
        if not e or not c:
            continue
        battles.append(Battle(comp.model_a, comp.model_b, tuple(comp.labels())))
        exp_lab.append(e)
        cro_lab.append(c)
        prompt_ids.append(comp.prompt_id)
        ce, cc = comp.coarse_labels("expert"), comp.coarse_labels("normal")
        unanimous.append(len(set(ce)) == 1 and len(set(cc)) == 1)
    n = len(battles)
    unanimous = np.asarray(unanimous)

    # MultiPref reuses each prompt for two comparisons against DIFFERENT model pairs:
    # 10,461 rows over 5,323 prompts, 5,138 of them doubled, so 98.2% of rows sit in a
    # repeated prompt. Resampling rows independently would treat those two as independent
    # draws. The resampling unit is therefore the prompt, and a whole cluster travels
    # together. It changes little here, because the two rows of a cluster inform different
    # cells of the leaderboard, but it is the unit the design actually has.
    _groups = {}
    for i, pid in enumerate(prompt_ids):
        _groups.setdefault(pid, []).append(i)
    clusters = [np.asarray(v) for v in _groups.values()]

    def draw(cl):
        """One cluster bootstrap resample: row indices, whole prompts kept together."""
        pick = rng.integers(0, len(cl), len(cl))
        return np.concatenate([cl[j] for j in pick])

    be = [Battle(b.model_a, b.model_b, l) for b, l in zip(battles, exp_lab)]
    bc = [Battle(b.model_a, b.model_b, l) for b, l in zip(battles, cro_lab)]
    wr_e = win_rates(be, hard=True, min_battles=args.min_battles)
    wr_c = win_rates(bc, hard=True, min_battles=args.min_battles)
    models = sorted(set(wr_e) & set(wr_c), key=lambda m: -wr_e[m])

    # --- 1. perturbation size -------------------------------------------------------
    shifts = {m: wr_c[m] - wr_e[m] for m in models}
    sd_shift = float(np.std(list(shifts.values()), ddof=1))
    # A swap between two models depends on the DIFFERENCE of their two shifts.
    sd_pair = sd_shift * math.sqrt(2.0)

    ordered = [wr_e[m] for m in models]
    gaps = [ordered[i] - ordered[i + 1] for i in range(len(ordered) - 1)]

    # --- 2. how lucky was the observed null ------------------------------------------
    rng = np.random.default_rng(args.seed)
    taus, disp = [], []
    # The paper used to quote a Gaussian p_swap for the closest adjacent pair. That
    # approximation is available here as a direct measurement instead: in each resample,
    # do the expert-gold and normal-gold rankings disagree on this pair? Recorded per pair,
    # alongside the bootstrap sd of each adjacent pair's shift DIFFERENCE, which is the
    # quantity the Gaussian route has to assume.
    pair_disagree = [[] for _ in range(len(models) - 1)]
    pair_shift_diff = [[] for _ in range(len(models) - 1)]
    for _ in range(args.n_boot):
        idx = draw(clusters)
        sx = [be[i] for i in idx]
        sy = [bc[i] for i in idx]
        rx = ranking(win_rates(sx, hard=True, min_battles=args.min_battles))
        ry = ranking(win_rates(sy, hard=True, min_battles=args.min_battles))
        common = set(rx) & set(ry)
        if len(common) < 3:
            continue
        rx = [m for m in rx if m in common]
        ry = [m for m in ry if m in common]
        taus.append(kendall_tau(rx, ry))
        ix = {m: i for i, m in enumerate(rx)}
        iy = {m: i for i, m in enumerate(ry)}
        disp.append(sum(1 for m in common if ix[m] != iy[m]))
        if set(models) <= common:
            we_b = win_rates(sx, hard=True, min_battles=args.min_battles)
            wc_b = win_rates(sy, hard=True, min_battles=args.min_battles)
            for j in range(len(models) - 1):
                hi, lo = models[j], models[j + 1]
                pair_disagree[j].append((ix[hi] < ix[lo]) != (iy[hi] < iy[lo]))
                pair_shift_diff[j].append((wc_b[hi] - we_b[hi]) - (wc_b[lo] - we_b[lo]))
    taus_a, disp_a = np.asarray(taus), np.asarray(disp)
    measured_pair = [
        {"above": models[j], "below": models[j + 1],
         "gap": round(gaps[j], 5),
         "p_swap_gaussian": round(_norm_cdf(-gaps[j] / sd_pair), 4),
         "p_disagree_measured": round(float(np.mean(pair_disagree[j])), 4),
         "sd_shift_difference_measured": round(float(np.std(pair_shift_diff[j], ddof=1)), 5)}
        for j in range(len(models) - 1)]

    # The same item bootstrap on the convention-free subset the headline uses, so the
    # displacement rate can be quoted for the population the headline table describes
    # rather than only for the full corpus under the default tie-break.
    un_clusters = [c[unanimous[c]] for c in clusters]
    un_clusters = [c for c in un_clusters if len(c)]
    un_disp = []
    for _ in range(args.n_boot):
        idx = draw(un_clusters)
        we_u = win_rates([be[i] for i in idx], hard=True, min_battles=args.min_battles)
        wc_u = win_rates([bc[i] for i in idx], hard=True, min_battles=args.min_battles)
        common = set(we_u) & set(wc_u)
        if len(common) < 3:
            continue
        rx = [m for m in ranking(we_u) if m in common]
        ry = [m for m in ranking(wc_u) if m in common]
        ix = {m: i for i, m in enumerate(rx)}
        iy = {m: i for i, m in enumerate(ry)}
        un_disp.append(sum(1 for m in common if ix[m] != iy[m]))
    un_disp_a = np.asarray(un_disp)

    # --- 3. the threshold --------------------------------------------------------------
    # P(adjacent pair swaps) = P(their shift difference exceeds the gap between them).
    def p_swap(gap: float) -> float:
        return _norm_cdf(-gap / sd_pair)

    gap_grid = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
    # For a K-model leaderboard spanning the observed win-rate range with uniform spacing.
    span = max(ordered) - min(ordered)
    k_grid = [6, 10, 20, 30, 50]
    k_rows = []

    # Adjacent swap events are NOT independent: consecutive pairs share a model, so the
    # closed form 1-(1-p)^(K-1) is wrong in both directions depending on the sign of the
    # shared shift. Draw one shift PER MODEL instead and let the sharing happen, which
    # removes that assumption entirely. What remains assumed is stated in the output:
    # shifts iid Gaussian with the measured SD, K models uniform over the observed span.
    mc = np.random.default_rng(args.seed + 1)
    for k in k_grid:
        g = span / (k - 1)
        base = np.array([max(ordered) - g * i for i in range(k)])
        draws = mc.normal(0.0, sd_shift, size=(args.n_mc, k))
        # Displacement = the perturbed ordering differs from the base ordering anywhere.
        order = np.argsort(-(base[None, :] + draws), axis=1, kind="stable")
        any_disp = (order != np.arange(k)[None, :]).any(axis=1)
        k_rows.append({"n_models": k, "uniform_gap": round(g, 5),
                       "p_adjacent_swap_marginal": round(p_swap(g), 4),
                       "p_any_displacement": round(float(any_disp.mean()), 4),
                       "p_any_displacement_indep_product": round(
                           1.0 - (1.0 - p_swap(g)) ** (k - 1), 4)})

    # Does the illustrative model reproduce the bootstrap on THIS leaderboard? The K=6 row
    # uses uniform spacing; the observed board is not uniform. Running the same simulation
    # on the observed gaps is the only version of that check that tests the model rather
    # than a differently-shaped hypothetical.
    obs_base = np.asarray(ordered)
    draws_obs = mc.normal(0.0, sd_shift, size=(args.n_mc, len(obs_base)))
    order_obs = np.argsort(-(obs_base[None, :] + draws_obs), axis=1, kind="stable")
    sim_on_observed = float((order_obs != np.arange(len(obs_base))[None, :]).any(axis=1).mean())

    out = {
        "milestone": "c6-displacement-threshold",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_s": round(time.time() - t0, 1),
        "n_battles": n,
        "n_models": len(models),
        "perturbation": {
            "per_model_shift_crowd_minus_expert": {m: round(shifts[m], 5) for m in models},
            "sd_of_shift": round(sd_shift, 5),
            "sd_of_pairwise_shift_difference": round(sd_pair, 5),
            "note": "The pool choice moves a model's win rate by this much. A swap between "
                    "two adjacent models needs the DIFFERENCE of their shifts to exceed the "
                    "gap between them, hence sd*sqrt(2).",
        },
        "observed_leaderboard": {
            "models_by_expert_winrate": models,
            "expert_winrates": [round(wr_e[m], 5) for m in models],
            "adjacent_gaps": [round(g, 5) for g in gaps],
            "smallest_adjacent_gap": round(min(gaps), 5),
            "p_swap_at_smallest_gap": round(p_swap(min(gaps)), 4),
        },
        "gaussian_route": {
            "sd_pair_assumed_independent": round(sd_pair, 5),
            "sd_pair_measured_bootstrap": round(
                float(np.mean([r["sd_shift_difference_measured"] for r in measured_pair])), 5),
            "union_with_assumed_sd": round(
                1.0 - float(np.prod([1.0 - _norm_cdf(-g / sd_pair) for g in gaps])), 4),
            "union_with_measured_sd": round(
                1.0 - float(np.prod([
                    1.0 - _norm_cdf(-gaps[j] / measured_pair[j]["sd_shift_difference_measured"])
                    for j in range(len(gaps))])), 4),
            "note": "The Gaussian swap route assumes the two models' shifts are independent "
                    "draws from the CROSS-MODEL dispersion, giving sd*sqrt(2). The bootstrap "
                    "measures the sampling sd of each adjacent pair's shift DIFFERENCE "
                    "directly, and it is about half that. Substituting the measured value "
                    "brings the independent-union estimate back to the bootstrap's "
                    "displacement rate; the assumed value roughly doubles it.",
        },
        "calibration_of_the_illustrative_model": {
            "sim_uniform_K6": round([r for r in k_rows if r["n_models"] == 6][0]["p_any_displacement"], 4),
            "sim_on_observed_gaps": round(sim_on_observed, 4),
            "bootstrap_measured": round(float((disp_a > 0).mean()), 4),
            "note": "The K=6 row of by_leaderboard_size agrees with the bootstrap only "
                    "under UNIFORM spacing. Run the identical simulation on the observed "
                    "gaps and it disagrees badly, because sd_of_shift is the dispersion of "
                    "six models' realised shifts, not the sampling sd of one model's shift "
                    "(measured at sd_shift_difference_measured, roughly half the sqrt(2)x "
                    "value the Gaussian route assumes). The agreement at K=6 uniform is "
                    "therefore a coincidence of two compensating errors and must not be "
                    "reported as a validation of the model.",
        },
        "unanimous_subset": {
            "n_items": int(unanimous.sum()),
            "n_boot": len(un_disp_a),
            "p_any_displacement": round(float((un_disp_a > 0).mean()), 4),
            "note": "Same item bootstrap on the 2,885 comparisons where BOTH pools are "
                    "internally unanimous - the population the headline divergence rates "
                    "describe - so the displacement rate can be quoted for that population "
                    "rather than only for the full corpus under the default tie-break.",
        },
        "bootstrap": {
            "n_boot": len(taus_a),
            "resampling_unit": "prompt cluster (98.2% of rows sit in a doubled prompt)",
            "measured_adjacent_pairs": measured_pair,
            "tau_mean": round(float(taus_a.mean()), 4),
            "tau_ci95": [round(float(q), 4) for q in np.quantile(taus_a, [0.025, 0.975])],
            "p_any_displacement": round(float((disp_a > 0).mean()), 4),
            "mean_models_displaced": round(float(disp_a.mean()), 4),
            "note": "Resamples items and recomputes BOTH leaderboards. The rate at which "
                    "displacement occurs here is the honest read on how decisive the "
                    "observed zero-displacement result is.",
        },
        "threshold": {
            "p_swap_by_gap": {str(g): round(p_swap(g), 4) for g in gap_grid},
            "gap_for_p_swap_0.05": round(-sd_pair * 1.6449, 5),
            "by_leaderboard_size": k_rows,
            "note": "by_leaderboard_size is an ILLUSTRATIVE SCENARIO, not a measurement. "
                    "It holds the measured per-model shift SD fixed, assumes those shifts "
                    "are iid Gaussian across models, and spreads K models uniformly across "
                    "the observed win-rate span. p_any_displacement is Monte Carlo over "
                    "per-model shifts, so adjacent pairs correctly share a model; the "
                    "indep_product column is the naive closed form kept for comparison. "
                    "None of these is a lower bound: real arenas cluster more tightly than "
                    "uniform (raising displacement) but correlated shifts across models "
                    "lower it, and we have not measured that correlation.",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "observed_leaderboard"}, indent=2)[:2400])
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
