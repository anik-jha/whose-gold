"""H0 gate applied to MultiPref, the multi-annotator corpus adopted 2026-08-09.

Same pre-registered bars as the HMDA gate (runs/plan.json, milestone h0-gate):
    KILL if the contested subset is < 5% of items, or unanimity exceeds 85%.

Also records the expert-vs-normal divergence rate, which is the quantity HAIC's
second theme ("expert disagreement and the limits of human feedback") is about,
and which HMDA structurally cannot supply.

Writes runs/h0/eval_multipref.json.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from wsev.data.multipref import COARSE, PREF_SCALE, load  # noqa: E402
from wsev.stats.disagreement import (  # noqa: E402
    defensible_set,
    majority_vote,
    normalized_entropy,
    unanimity,
)

REPO = pathlib.Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", type=pathlib.Path, default=REPO / "runs" / "h0" / "eval_multipref.json")
    args = ap.parse_args()

    t0 = time.time()
    n = 0
    n_unanimous = 0
    n_unanimous_coarse = 0
    n_expert_split = 0          # the two experts disagree with each other
    n_expert_vs_normal = 0      # expert majority != normal majority (coarse)
    ann_per_item: collections.Counter[int] = collections.Counter()
    pool_counts: collections.Counter[str] = collections.Counter()
    label_freq: collections.Counter[str] = collections.Counter()
    defensible_sizes: collections.Counter[int] = collections.Counter()
    entropies: list[float] = []
    by_category: collections.Counter[str] = collections.Counter()
    contested_by_category: collections.Counter[str] = collections.Counter()

    for comp in load(split=args.split):
        n += 1
        labels = comp.labels()
        coarse = comp.coarse_labels()
        ann_per_item[len(labels)] += 1
        for a in comp.annotations:
            pool_counts[a.pool] += 1
        for lab in labels:
            label_freq[lab] += 1

        if unanimity(labels):
            n_unanimous += 1
        if unanimity(coarse):
            n_unanimous_coarse += 1
        else:
            contested_by_category[comp.category] += 1
        by_category[comp.category] += 1

        entropies.append(normalized_entropy(labels, n_classes=len(PREF_SCALE)))
        defensible_sizes[len(defensible_set(coarse))] += 1

        exp = comp.coarse_labels("expert")
        nor = comp.coarse_labels("normal")
        if len(exp) >= 2 and not unanimity(exp):
            n_expert_split += 1
        if exp and nor:
            tb = ("A", "T", "B")
            if majority_vote(exp, tie_break=tb) != majority_vote(nor, tie_break=tb):
                n_expert_vs_normal += 1

    if n == 0:
        raise RuntimeError("no comparisons loaded")

    contested = 1.0 - n_unanimous / n
    contested_coarse = 1.0 - n_unanimous_coarse / n
    bar_min_contested, bar_max_unanimous = 0.05, 0.85
    passed = contested >= bar_min_contested and (1.0 - contested) <= bar_max_unanimous

    result = {
        "milestone": "h0-gate",
        "corpus": "allenai/multipref",
        "corpus_url": "https://huggingface.co/datasets/allenai/multipref",
        "schema_read_utc": "2026-08-09",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_s": round(time.time() - t0, 1),
        "preregistered_bars": {
            "min_contested_fraction": bar_min_contested,
            "max_unanimous_fraction": bar_max_unanimous,
        },
        "n_comparisons": n,
        "annotators_per_item": dict(sorted(ann_per_item.items())),
        "annotations_by_pool": dict(pool_counts),
        "label_frequency_5pt": dict(label_freq.most_common()),
        "coarse_map": COARSE,
        "defensible_set_sizes_coarse": dict(sorted(defensible_sizes.items())),
        "mean_normalized_entropy_5pt": sum(entropies) / len(entropies),
        "contested_fraction_5pt": contested,
        "contested_fraction_coarse": contested_coarse,
        "expert_pair_split_fraction": n_expert_split / n,
        "expert_vs_normal_majority_divergence": n_expert_vs_normal / n,
        "top_categories_by_contested_rate": sorted(
            ((c, contested_by_category[c] / v, v) for c, v in by_category.items() if v >= 50),
            key=lambda t: -t[1],
        )[:10],
        "verdict": {"pass": passed, "contested_fraction": contested},
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))

    print(f"MultiPref n={n:,}  annotators/item={dict(sorted(ann_per_item.items()))}")
    print(f"  contested (5pt)   = {contested:.3%}   (bar >= {bar_min_contested:.0%}) -> {'PASS' if passed else 'FAIL'}")
    print(f"  contested (coarse)= {contested_coarse:.3%}")
    print(f"  expert pair split = {n_expert_split / n:.3%}")
    print(f"  expert vs normal  = {n_expert_vs_normal / n:.3%}  <- HAIC theme, measurable directly")
    print(f"  mean norm entropy = {result['mean_normalized_entropy_5pt']:.4f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
