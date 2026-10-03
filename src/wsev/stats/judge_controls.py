"""Paired, poststratified prompt-multiplier sensitivity for saved judge labels.

Inference targets variation across prompts conditional on the corpus stratum masses.
It is not an exact finite-population survey bootstrap. See the frozen specification
in final_plan/CAMERA_READY_PLAN.md for the exploratory test families.
"""

from __future__ import annotations

import numpy as np

LABELS = ("A", "T", "B")
RULES = {"A_T_B": ("A", "T", "B"), "A_B_T": ("A", "B", "T"),
         "B_T_A": ("B", "T", "A"), "force_tie": "T"}


def pool_label(labels, rule):
    counts = {x: labels.count(x) for x in set(labels)}
    winners = {x for x, n in counts.items() if n == max(counts.values())}
    if len(winners) == 1:
        return next(iter(winners))
    if isinstance(rule, str):
        return rule
    return next(x for x in rule if x in winners)


def poststratified_weights(strata, masses, multipliers):
    """Normalize positive row multipliers to fixed stratum masses, including no-op."""
    strata = np.asarray(strata)
    weights = np.array(multipliers, dtype=float, copy=True)
    if weights.ndim != 2 or weights.shape[1] != len(strata):
        raise ValueError("multipliers must have shape (replicates, rows)")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("multipliers must be finite and nonnegative")
    if set(np.unique(strata)) != set(masses) or not np.isclose(sum(masses.values()), 1):
        raise ValueError("every population stratum must be observed; masses must sum to one")
    for s, mass in masses.items():
        mask = strata == s
        totals = weights[:, mask].sum(axis=1)
        if (totals <= 0).any() or mass <= 0:
            raise ValueError("empty bootstrap stratum or nonpositive population mass")
        weights[:, mask] *= (mass / totals)[:, None]
    return weights


def summarize(estimate, replicates):
    reps = np.asarray(replicates)
    if not np.isfinite(reps).all() or not np.isfinite(estimate):
        raise ValueError("nonfinite statistic")
    p = (1 + np.count_nonzero(np.abs(reps - estimate) >= abs(estimate))) / (len(reps) + 1)
    return {"estimate": float(estimate),
            "ci95": np.quantile(reps, [.025, .975]).tolist(),
            "p_centered_approx": float(p)}


def holm(pvalues):
    """Holm adjusted p-values in original order, valid under arbitrary dependence."""
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("invalid p-values")
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted.tolist()


def agreement_features(judge, expert, normal):
    """Linear item outcomes; each difference pairs the same judge with both pools."""
    j = np.asarray(judge)
    e, n = list(expert), list(normal)
    out = {}
    for name, rule in RULES.items():
        me = np.array([pool_label(x, rule) for x in e])
        mn = np.array([pool_label(x, rule) for x in n])
        out[name] = (j == me).astype(float) - (j == mn)
        if name == "A_T_B":
            out["erasure"] = np.array([
                float(a in ex and a != exm) - float(a in no and a != nom)
                for a, ex, no, exm, nom in zip(j, e, n, me, mn)])
    out["per_annotator"] = np.array([
        sum(a == x for x in ex) / len(ex) - sum(a == x for x in no) / len(no)
        for a, ex, no in zip(j, e, n)])
    return out


def marginal_baselines(weights, strata, masses, judge, expert_majority, normal_majority):
    """Global and within-stratum expected differences under label independence.

    The latter is the analytic mean of within-stratum judge-label permutations.
    Bootstrap weights recompute all marginals jointly with observed agreement.
    """
    j, e, n = map(np.asarray, (judge, expert_majority, normal_majority))
    global_null = np.zeros(len(weights))
    stratified_null = np.zeros(len(weights))
    for label in LABELS:
        ji = (j == label).astype(float)
        diff = (e == label).astype(float) - (n == label)
        global_null += (weights @ ji) * (weights @ diff)
        for s, mass in masses.items():
            mask = np.asarray(strata) == s
            stratified_null += (weights[:, mask] @ ji[mask]) * (weights[:, mask] @ diff[mask]) / mass
    return global_null, stratified_null
