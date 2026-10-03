"""Leaderboards from pairwise preferences, and rank-agreement with uncertainty.

The paper's central question is not "which model is best" but "does the answer
change when you change whose labels count as gold". So every leaderboard here is
parameterised by an annotator pool, and the comparison between two leaderboards
always carries a bootstrap CI - a point estimate of Kendall tau would let a
rank flip look decisive when it is noise, which is exactly the error this paper
accuses single-label evaluation of making.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Sequence

import numpy as np

# Preference label -> credit awarded to model A. Ties split.
_A_CREDIT = {
    "A-is-clearly-better": 1.0,
    "A-is-slightly-better": 1.0,
    "Tie": 0.5,
    "B-is-slightly-better": 0.0,
    "B-is-clearly-better": 0.0,
}

# Coarse direction of each five-point label, and the credit each direction earns.
# Derived from _A_CREDIT so the two can never drift apart.
_COARSE_OF = {lab: ("A" if v > 0.5 else "B" if v < 0.5 else "T") for lab, v in _A_CREDIT.items()}
_COARSE_CREDIT = {"A": 1.0, "T": 0.5, "B": 0.0}


@dataclass(frozen=True)
class Battle:
    """One pairwise comparison with the per-annotator labels kept intact."""
    model_a: str
    model_b: str
    labels: tuple[str, ...]

    def a_credit(self) -> float:
        """Mean credit to A across this item's annotators (the soft, disagreement-aware view)."""
        return float(np.mean([_A_CREDIT[lab] for lab in self.labels]))

    def a_credit_hard(self, *, tie_break: Sequence[str] | str | None = None) -> float:
        """Credit to A after collapsing annotators to a majority label first.

        Tie handling is expressed over the COARSE classes "A"/"T"/"B", not the
        five-point labels: with two annotators per pool a split is between
        directions, not between shades of one direction. (Passing five-point labels
        would silently make every rule equivalent, since no candidate would ever
        match the ordering.)

        Two genuinely different rule types are supported, because a PC review found
        this paper describing one and implementing the other:

        * ``tie_break="T"`` (a bare class) -- FORCE. Any unresolved split becomes
          that class. This is what "break ties toward the tie category" means, and
          it is not expressible as an ordering: when annotators split A-vs-B, "T"
          is not among the tied candidates at all.
        * ``tie_break=("A", "T", "B")`` (a sequence) -- PRIORITY among the classes
          that actually tied. An A-first rule converts A-vs-B splits into A wins.

        The two disagree on exactly the items that decide this paper's headline.
        """
        counts: collections.Counter[str] = collections.Counter(
            _COARSE_OF[lab] for lab in self.labels)
        top = max(counts.values())
        winners = [k for k, v in counts.items() if v == top]
        if len(winners) == 1:
            return _COARSE_CREDIT[winners[0]]
        if isinstance(tie_break, str):
            if tie_break not in _COARSE_CREDIT:
                raise ValueError(f"forced tie class {tie_break!r} is not one of A/T/B")
            return _COARSE_CREDIT[tie_break]
        if tie_break:
            order = {cls: i for i, cls in enumerate(tie_break)}
            if not set(winners) <= set(order):
                raise ValueError(
                    f"tie_break {tuple(tie_break)} must cover coarse classes {sorted(winners)}")
            return _COARSE_CREDIT[min(winners, key=lambda w: order[w])]
        return _COARSE_CREDIT[min(winners)]


def win_rates(battles: Sequence[Battle], *, hard: bool = False,
              min_battles: int = 1,
              tie_break: Sequence[str] | None = None) -> dict[str, float]:
    """Per-model mean credit over all battles it appears in.

    `hard=True` collapses each item to a majority label before scoring, which is
    what a conventional leaderboard does. `hard=False` keeps the annotator
    distribution.
    """
    total: dict[str, float] = collections.defaultdict(float)
    count: dict[str, int] = collections.defaultdict(int)
    for b in battles:
        ca = b.a_credit_hard(tie_break=tie_break) if hard else b.a_credit()
        total[b.model_a] += ca
        count[b.model_a] += 1
        total[b.model_b] += 1.0 - ca
        count[b.model_b] += 1
    return {m: total[m] / count[m] for m in total if count[m] >= min_battles}


def ranking(scores: dict[str, float]) -> list[str]:
    """Models best-first. Ties broken by name so the order is deterministic."""
    return [m for m, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


def kendall_tau(rank_a: Sequence[str], rank_b: Sequence[str]) -> float:
    """Kendall tau-b over the models present in both rankings."""
    common = sorted(set(rank_a) & set(rank_b))
    if len(common) < 2:
        raise ValueError("need >= 2 shared models")
    ia = {m: i for i, m in enumerate(rank_a)}
    ib = {m: i for i, m in enumerate(rank_b)}
    conc = disc = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            x, y = common[i], common[j]
            s = (ia[x] - ia[y]) * (ib[x] - ib[y])
            if s > 0:
                conc += 1
            elif s < 0:
                disc += 1
    n = conc + disc
    if n == 0:
        return 1.0
    return (conc - disc) / n


@dataclass(frozen=True)
class TauResult:
    tau: float
    ci_low: float
    ci_high: float
    n_boot: int
    n_models: int
    n_battles: int
    displaced_models: int
    max_rank_shift: int


def paired_bootstrap_tau(
    battles: Sequence[Battle],
    labels_x: Sequence[tuple[str, ...]],
    labels_y: Sequence[tuple[str, ...]],
    *,
    hard: bool = True,
    n_boot: int = 2000,
    alpha: float = 0.05,
    min_battles: int = 20,
    seed: int = 0,
    tie_break: Sequence[str] | None = None,
    clusters: Sequence[Sequence[int]] | None = None,
) -> TauResult:
    """Kendall tau between two label-source rankings, with a paired bootstrap CI.

    Paired: each resample draws the SAME battle indices for both label sources,
    because the two rankings are computed over identical items and differ only in
    whose labels are used. An unpaired bootstrap would inflate the CI and hide a
    real rank difference (playbook: use the paired test when the design is paired).

    `clusters` is a partition of battle indices into independent units. Both corpora
    reuse one prompt (MultiPref) or one question (MT-Bench) across several model pairs,
    so rows are not independent and the cluster is the resampling unit; a whole cluster
    travels together. Passing None resamples rows, which is only correct when no such
    reuse exists.
    """
    if not (len(battles) == len(labels_x) == len(labels_y)):
        raise ValueError("battles and both label sequences must align")

    bx = [Battle(b.model_a, b.model_b, lx) for b, lx in zip(battles, labels_x)]
    by = [Battle(b.model_a, b.model_b, ly) for b, ly in zip(battles, labels_y)]

    rx = ranking(win_rates(bx, hard=hard, min_battles=min_battles, tie_break=tie_break))
    ry = ranking(win_rates(by, hard=hard, min_battles=min_battles, tie_break=tie_break))
    point = kendall_tau(rx, ry)

    common = set(rx) & set(ry)
    ix = {m: i for i, m in enumerate([m for m in rx if m in common])}
    iy = {m: i for i, m in enumerate([m for m in ry if m in common])}
    shifts = {m: abs(ix[m] - iy[m]) for m in common}
    displaced = sum(1 for v in shifts.values() if v > 0)
    max_shift = max(shifts.values()) if shifts else 0

    rng = np.random.default_rng(seed)
    n = len(battles)
    cl = [np.asarray(c, dtype=np.int64) for c in clusters] if clusters else None
    if cl is not None and sum(len(c) for c in cl) != n:
        raise ValueError("clusters must partition every battle index exactly once")
    taus = np.empty(n_boot, dtype=np.float64)
    got = 0
    for _ in range(n_boot):
        if cl is None:
            idx = rng.integers(0, n, size=n)
        else:
            idx = np.concatenate([cl[j] for j in rng.integers(0, len(cl), len(cl))])
        sx = [bx[i] for i in idx]
        sy = [by[i] for i in idx]
        try:
            taus[got] = kendall_tau(
                ranking(win_rates(sx, hard=hard, min_battles=min_battles, tie_break=tie_break)),
                ranking(win_rates(sy, hard=hard, min_battles=min_battles, tie_break=tie_break)),
            )
        except ValueError:
            continue
        got += 1

    taus = taus[:got]
    lo, hi = np.quantile(taus, [alpha / 2, 1 - alpha / 2]) if got else (float("nan"),) * 2
    return TauResult(
        tau=point, ci_low=float(lo), ci_high=float(hi), n_boot=got,
        n_models=len(common), n_battles=n,
        displaced_models=displaced, max_rank_shift=int(max_shift),
    )
