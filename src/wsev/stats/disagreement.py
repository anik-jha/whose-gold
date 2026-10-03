"""Aggregators and disagreement measures over per-annotator labels.

Every function here takes *per-annotator* labels and never a pre-aggregated gold
label. Collapsing to a single label is exactly the operation this paper argues
distorts system ranking, so the collapse must always be an explicit, named choice
at the call site.

Invariant enforced by tests/test_disagreement.py: when annotators are unanimous,
every aggregator must return the majority label exactly.
"""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np

Label = Hashable


def label_distribution(labels: Sequence[Label]) -> dict[Label, float]:
    """Empirical distribution over labels for one item."""
    if not labels:
        raise ValueError("no labels")
    c = collections.Counter(labels)
    n = len(labels)
    return {k: v / n for k, v in c.items()}


def majority_vote(labels: Sequence[Label], *, tie_break: Sequence[Label] | None = None) -> Label:
    """Modal label. Ties broken by `tie_break` order, else by sorted repr for determinism."""
    if not labels:
        raise ValueError("no labels")
    c = collections.Counter(labels)
    top = max(c.values())
    winners = [k for k, v in c.items() if v == top]
    if len(winners) == 1:
        return winners[0]
    if tie_break:
        order = {lab: i for i, lab in enumerate(tie_break)}
        return min(winners, key=lambda w: order.get(w, len(order)))
    return min(winners, key=repr)


def unanimity(labels: Sequence[Label]) -> bool:
    return len(set(labels)) == 1


def normalized_entropy(labels: Sequence[Label], *, n_classes: int | None = None) -> float:
    """Shannon entropy of the label distribution, normalized to [0, 1].

    0.0 = unanimous. 1.0 = uniform over `n_classes`. Returns 0.0 when only one
    class is possible, since there is then nothing to disagree about.
    """
    dist = label_distribution(labels)
    k = n_classes if n_classes is not None else len(dist)
    if k <= 1:
        return 0.0
    h = -sum(p * math.log(p) for p in dist.values() if p > 0)
    return h / math.log(k)


def defensible_set(labels: Sequence[Label], *, min_support: int = 1) -> set[Label]:
    """Labels defensible under the annotator pool: those with >= min_support votes.

    This is the set a scoring rule must respect. Marking a member of this set
    "wrong" is what we call erasure.
    """
    if min_support < 1:
        raise ValueError("min_support must be >= 1")
    c = collections.Counter(labels)
    return {k for k, v in c.items() if v >= min_support}


@dataclass(frozen=True)
class DawidSkeneResult:
    item_posteriors: np.ndarray  # [n_items, n_classes]
    confusion: np.ndarray  # [n_annotators, n_classes, n_classes]
    class_prior: np.ndarray  # [n_classes]
    n_iter: int
    converged: bool


def dawid_skene(
    votes: Sequence[tuple[int, int, int]],
    n_items: int,
    n_annotators: int,
    n_classes: int,
    *,
    max_iter: int = 200,
    tol: float = 1e-6,
    smoothing: float = 1e-9,
) -> DawidSkeneResult:
    """Classic Dawid-Skene EM over (item, annotator, label) triples.

    Reviewers will ask for this as the standard latent-truth baseline alongside
    majority vote. Implemented directly rather than pulled in as a dependency so
    the smoothing and convergence behaviour are inspectable.
    """
    counts = np.zeros((n_items, n_annotators, n_classes), dtype=np.float64)
    for item, annot, lab in votes:
        counts[item, annot, lab] += 1.0

    # Init posteriors from per-item vote shares.
    post = counts.sum(axis=1)
    row = post.sum(axis=1, keepdims=True)
    post = np.where(row > 0, post / np.maximum(row, smoothing), 1.0 / n_classes)

    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        # M-step.
        prior = post.mean(axis=0)
        prior = np.maximum(prior, smoothing)
        prior /= prior.sum()

        conf = np.einsum("ik,ijl->jkl", post, counts) + smoothing
        conf /= conf.sum(axis=2, keepdims=True)

        # E-step in log space (guards underflow with many annotators).
        loglik = np.log(prior)[None, :] + np.einsum("ijl,jkl->ik", counts, np.log(conf))
        loglik -= loglik.max(axis=1, keepdims=True)
        new_post = np.exp(loglik)
        new_post /= new_post.sum(axis=1, keepdims=True)

        delta = float(np.abs(new_post - post).max())
        post = new_post
        if delta < tol:
            converged = True
            break

    return DawidSkeneResult(post, conf, prior, it, converged)


def soft_score(prediction: Label, labels: Sequence[Label]) -> float:
    """Credit = fraction of annotators who chose `prediction`.

    The disagreement-aware counterpart to 0/1 accuracy against a collapsed gold
    label. A proper scoring rule under the empirical annotator distribution.
    """
    return label_distribution(labels).get(prediction, 0.0)


def hard_score(prediction: Label, labels: Sequence[Label], *,
               tie_break: Sequence[Label] | None = None) -> float:
    """Standard 0/1 accuracy against the majority-collapsed gold label."""
    return 1.0 if prediction == majority_vote(labels, tie_break=tie_break) else 0.0


def erasure(prediction: Label, labels: Sequence[Label], *, min_support: int = 1,
            tie_break: Sequence[Label] | None = None) -> bool:
    """True when a defensible prediction is scored wrong by majority collapse.

    This is the event whose rate is the paper's headline erasure metric.
    """
    return (prediction in defensible_set(labels, min_support=min_support)
            and hard_score(prediction, labels, tie_break=tie_break) == 0.0)
