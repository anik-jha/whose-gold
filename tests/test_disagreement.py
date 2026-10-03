"""Invariant tests for the aggregators. Run in seconds, no downloads.

The central invariant: when annotators are unanimous, every aggregator must
collapse to the same answer as majority vote. If that ever breaks, every
downstream ranking claim in the paper is meaningless.
"""

from __future__ import annotations

import numpy as np
import pytest

from wsev.stats.disagreement import (
    dawid_skene,
    defensible_set,
    erasure,
    hard_score,
    label_distribution,
    majority_vote,
    normalized_entropy,
    soft_score,
    unanimity,
)

CLASSES = ["A", "T", "B"]


# --- invariant: unanimity ---------------------------------------------------

@pytest.mark.parametrize("lab", CLASSES)
@pytest.mark.parametrize("n", [1, 2, 4, 8])
def test_unanimous_all_aggregators_agree(lab, n):
    labels = [lab] * n
    assert unanimity(labels)
    assert majority_vote(labels) == lab
    assert soft_score(lab, labels) == 1.0
    assert hard_score(lab, labels) == 1.0
    assert normalized_entropy(labels, n_classes=len(CLASSES)) == 0.0
    assert defensible_set(labels) == {lab}
    assert not erasure(lab, labels)


def test_unanimous_dawid_skene_matches_majority():
    """Unanimous items must give DS posterior mass to the majority class."""
    n_items, n_annot, n_cls = 12, 4, 3
    votes, gold = [], []
    rng = np.random.default_rng(0)
    for i in range(n_items):
        c = int(rng.integers(n_cls))
        gold.append(c)
        votes.extend((i, a, c) for a in range(n_annot))
    res = dawid_skene(votes, n_items, n_annot, n_cls)
    assert res.item_posteriors.shape == (n_items, n_cls)
    np.testing.assert_allclose(res.item_posteriors.sum(axis=1), 1.0, atol=1e-9)
    assert res.item_posteriors.argmax(axis=1).tolist() == gold


# --- disagreement behaviour -------------------------------------------------

def test_split_vote_entropy_is_maximal():
    labels = ["A", "T", "B"]
    assert normalized_entropy(labels, n_classes=3) == pytest.approx(1.0)


def test_soft_score_is_vote_share():
    labels = ["A", "A", "B", "T"]
    assert soft_score("A", labels) == pytest.approx(0.5)
    assert soft_score("B", labels) == pytest.approx(0.25)
    assert soft_score("Z", labels) == 0.0


def test_distribution_sums_to_one():
    labels = ["A", "A", "B", "T"]
    assert sum(label_distribution(labels).values()) == pytest.approx(1.0)


# --- erasure: the headline metric's definition ------------------------------

def test_erasure_fires_on_defensible_minority():
    """2 of 4 annotators say B, majority tie-breaks to A: predicting B is erased."""
    labels = ["A", "A", "B", "B"]
    maj = majority_vote(labels, tie_break=["A", "T", "B"])
    assert maj == "A"
    assert "B" in defensible_set(labels)
    assert hard_score("B", labels, tie_break=["A", "T", "B"]) == 0.0
    assert erasure("B", labels, tie_break=["A", "T", "B"])


def test_no_erasure_for_undefended_label():
    """A label nobody chose is simply wrong, not erased."""
    labels = ["A", "A", "A", "B"]
    assert not erasure("T", labels)


def test_no_erasure_when_prediction_is_majority():
    labels = ["A", "A", "A", "B"]
    assert not erasure("A", labels)


def test_min_support_gates_erasure():
    """A single dissenting vote counts only when min_support allows it."""
    labels = ["A", "A", "A", "B"]
    assert erasure("B", labels, min_support=1)
    assert not erasure("B", labels, min_support=2)


# --- guards -----------------------------------------------------------------

def test_empty_labels_raise():
    for fn in (label_distribution, majority_vote):
        with pytest.raises(ValueError):
            fn([])


def test_majority_tie_break_is_deterministic():
    labels = ["B", "A"]
    assert majority_vote(labels) == majority_vote(labels)
    assert majority_vote(labels, tie_break=["A", "B"]) == "A"
    assert majority_vote(labels, tie_break=["B", "A"]) == "B"
