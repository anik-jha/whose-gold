"""Synthetic validity checks independent of the real camera-ready outcomes."""

import itertools

import numpy as np
import pytest

from wsev.stats.judge_controls import (
    RULES, agreement_features, holm, marginal_baselines, pool_label,
    poststratified_weights, summarize,
)


def test_unanimous_noop_and_same_pools():
    labels = [[x, x] for x in ("A", "T", "B")]
    for rule in RULES.values():
        assert [pool_label(x, rule) for x in labels] == ["A", "T", "B"]
    assert all(np.array_equal(x, np.zeros(3)) for x in
               agreement_features(["A", "B", "T"], labels, labels).values())


def test_poststratification_recovers_population_not_sample_fraction():
    st = np.array([0, 0, 0, 1])
    w = poststratified_weights(st, {0: .25, 1: .75}, np.ones((1, 4)))
    assert w @ np.array([1., 1., 1., 0.]) == pytest.approx([.25])
    assert w.sum() == pytest.approx(1)


def test_mirror_relabeling_preserves_per_annotator_and_mirrored_rule():
    mirror = {"A": "B", "B": "A", "T": "T"}
    j, e, n = ["A", "B"], [["A", "T"], ["B", "B"]], [["B", "T"], ["A", "B"]]
    f = agreement_features(j, e, n)
    g = agreement_features([mirror[x] for x in j],
                           [[mirror[x] for x in y] for y in e],
                           [[mirror[x] for x in y] for y in n])
    np.testing.assert_array_equal(f["A_T_B"], g["B_T_A"])
    np.testing.assert_array_equal(f["per_annotator"], g["per_annotator"])


def test_cluster_duplicates_do_not_manufacture_precision():
    rng = np.random.default_rng(9)
    multipliers = rng.exponential(size=(1500, 80))
    y = np.tile([-1., 1.], 40)
    w = poststratified_weights(np.zeros(80), {0: 1.}, multipliers)
    duplicated = poststratified_weights(np.zeros(160), {0: 1.}, np.repeat(multipliers, 2, axis=1))
    np.testing.assert_allclose(w @ y, duplicated @ np.repeat(y, 2), atol=1e-15)
    independent = poststratified_weights(np.zeros(160), {0: 1.}, rng.exponential(size=(1500, 160)))
    assert (w @ y).std() > 1.25 * (independent @ np.repeat(y, 2)).std()


def test_shared_prompt_across_strata_keeps_pair_cancellation():
    mult = np.random.default_rng(4).exponential(size=(300, 20))
    w = poststratified_weights(np.repeat([0, 1], 20), {0: .5, 1: .5}, np.tile(mult, (1, 2)))
    y = np.tile([-1., 1.], 10)
    np.testing.assert_allclose(w @ np.r_[y, -y], 0, atol=1e-15)


def test_null_mean_matches_enumerated_within_stratum_permutations():
    j = np.array(["A", "B", "T", "B"])
    e = np.array(["A", "A", "B", "T"])
    n = np.array(["T", "B", "T", "T"])
    st = np.array([0, 0, 1, 1])
    masses = {0: .4, 1: .6}
    w = poststratified_weights(st, masses, np.ones((1, 4)))
    _, expected = marginal_baselines(w, st, masses, j, e, n)
    permutations = []
    for a in itertools.permutations(j[:2]):
        for b in itertools.permutations(j[2:]):
            pj = np.array(a + b)
            permutations.append(float((w @ ((pj == e).astype(float) - (pj == n)))[0]))
    assert expected[0] == pytest.approx(np.mean(permutations))


def test_holm_known_family_and_zero_paired_effect():
    assert holm([.01, .04, .03]) == pytest.approx([.03, .06, .06])
    assert summarize(0., np.zeros(500))["p_centered_approx"] == 1


def test_degenerate_stratum_rejected():
    with pytest.raises(ValueError):
        poststratified_weights([0, 0], {0: .5, 1: .5}, np.ones((5, 2)))
