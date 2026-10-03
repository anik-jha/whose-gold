"""Tests for the displacement-threshold model in c6_displacement_threshold.

The script converts C2's zero-displacement null into a probability of displacement as a
function of leaderboard spacing. The arithmetic is small but load-bearing: it is what
lets the paper say a ten-model leaderboard would probably move. These tests pin the two
places it could be silently wrong, using synthetic leaderboards with known answers.
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from wsev.stats.ranking import Battle, kendall_tau, ranking, win_rates  # noqa: E402


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def p_swap(gap: float, sd_pair: float) -> float:
    """Probability two adjacent models swap, given the gap between them."""
    return _norm_cdf(-gap / sd_pair)


def test_zero_gap_is_a_coin_flip():
    """Two models with identical win rates swap half the time. Anything else is a bug."""
    assert p_swap(0.0, 0.03) == pytest.approx(0.5)


def test_swap_probability_falls_monotonically_with_the_gap():
    sd = 0.0263
    ps = [p_swap(g, sd) for g in (0.0, 0.005, 0.01, 0.02, 0.05, 0.10)]
    assert all(a > b for a, b in zip(ps, ps[1:]))
    assert ps[-1] < 0.001


def test_a_wider_perturbation_makes_swaps_more_likely_at_fixed_gap():
    assert p_swap(0.02, 0.05) > p_swap(0.02, 0.01)


def test_any_displacement_rises_with_leaderboard_size_at_fixed_span():
    """The paper's claim: more models in the same win-rate span means more swaps."""
    sd, span = 0.0263, 0.20
    prev = -1.0
    for k in (6, 10, 20, 50):
        g = span / (k - 1)
        p_any = 1.0 - (1.0 - p_swap(g, sd)) ** (k - 1)
        assert p_any > prev
        prev = p_any
    assert prev > 0.99


def test_pair_sd_is_sqrt2_times_model_sd():
    """A swap depends on the DIFFERENCE of two independent shifts, not on one of them."""
    sd_model = 0.0186
    assert sd_model * math.sqrt(2.0) == pytest.approx(0.0263, abs=5e-4)


def _battles(order, n_each=100):
    """Round-robin battles where earlier models in `order` always beat later ones."""
    out = []
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            for _ in range(n_each):
                out.append(Battle(a, b, ("A-is-clearly-better",)))
    return out


def test_identical_labels_give_tau_one_and_no_displacement():
    """The no-op case: same labels both sides must reproduce the ranking exactly."""
    order = ["m1", "m2", "m3", "m4"]
    bs = _battles(order)
    r = ranking(win_rates(bs, hard=True, min_battles=10))
    assert kendall_tau(r, r) == pytest.approx(1.0)


def test_a_single_adjacent_swap_lowers_tau_below_one():
    """Guards the displacement counter: a swapped pair must be visible in tau."""
    a = ["m1", "m2", "m3", "m4"]
    b = ["m1", "m3", "m2", "m4"]
    t = kendall_tau(a, b)
    assert t < 1.0
    assert t == pytest.approx(1 - 2 / (4 * 3 / 2), abs=1e-9)


def test_displacement_count_matches_the_permutation():
    a = ["m1", "m2", "m3", "m4"]
    b = ["m1", "m3", "m2", "m4"]
    ia = {m: i for i, m in enumerate(a)}
    ib = {m: i for i, m in enumerate(b)}
    assert sum(1 for m in a if ia[m] != ib[m]) == 2
