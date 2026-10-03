"""MultiPref loader: per-annotator preference labels with an expert/normal split.

Source: https://huggingface.co/datasets/allenai/multipref  (schema read from the
live dataset on 2026-08-09, not from documentation).

Why this corpus. HAIC's stated theme is "expert disagreement and the limits of
human feedback". MultiPref gives exactly 4 annotators per comparison, split 2
expert / 2 normal, on a 5-point ordinal preference scale - so "expert vs normal
disagreement" is measurable directly rather than by proxy. The task is preference
judging, which is the same task an LLM judge performs, making the judge-erasure
claim a like-for-like comparison instead of an analogy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal

from datasets import load_dataset

DATASET_ID = "allenai/multipref"

# Ordinal scale, worst-for-A to best-for-A. Order matters: it defines the
# tie-break precedence and the ordinal distance used by disagreement measures.
PREF_SCALE: tuple[str, ...] = (
    "A-is-clearly-better",
    "A-is-slightly-better",
    "Tie",
    "B-is-slightly-better",
    "B-is-clearly-better",
)
PREF_INDEX = {lab: i for i, lab in enumerate(PREF_SCALE)}

# Coarse 3-way collapse, used when a task needs A/Tie/B rather than the 5-point scale.
COARSE: dict[str, str] = {
    "A-is-clearly-better": "A",
    "A-is-slightly-better": "A",
    "Tie": "T",
    "B-is-slightly-better": "B",
    "B-is-clearly-better": "B",
}

Pool = Literal["expert", "normal", "all"]


@dataclass(frozen=True)
class Annotation:
    evaluator: str
    pool: Literal["expert", "normal"]
    overall_pref: str
    overall_confidence: str | None
    helpful_pref: str | None
    truthful_pref: str | None
    harmless_pref: str | None


@dataclass(frozen=True)
class Comparison:
    comparison_id: str
    prompt_id: str
    text: str
    completion_a: str
    completion_b: str
    model_a: str
    model_b: str
    category: str
    annotations: tuple[Annotation, ...]

    def labels(self, pool: Pool = "all") -> list[str]:
        return [a.overall_pref for a in self.annotations
                if pool == "all" or a.pool == pool]

    def coarse_labels(self, pool: Pool = "all") -> list[str]:
        return [COARSE[lab] for lab in self.labels(pool)]


def _annotations(row: dict) -> tuple[Annotation, ...]:
    out: list[Annotation] = []
    for key, pool in (("expert_worker_annotations", "expert"),
                      ("normal_worker_annotations", "normal")):
        for a in row.get(key) or []:
            pref = a.get("overall_pref")
            if pref is None:
                continue
            if pref not in PREF_INDEX:
                raise ValueError(f"unknown preference label {pref!r}; schema changed")
            out.append(Annotation(
                evaluator=a.get("evaluator", ""),
                pool=pool,  # type: ignore[arg-type]
                overall_pref=pref,
                overall_confidence=a.get("overall_confidence"),
                helpful_pref=a.get("helpful_pref"),
                truthful_pref=a.get("truthful_pref"),
                harmless_pref=a.get("harmless_pref"),
            ))
    return tuple(out)


def load(split: str = "train", *, min_annotators: int = 2) -> Iterator[Comparison]:
    """Yield comparisons carrying at least `min_annotators` usable annotations."""
    ds = load_dataset(DATASET_ID, split=split)
    for row in ds:
        anns = _annotations(row)
        if len(anns) < min_annotators:
            continue
        yield Comparison(
            comparison_id=row["comparison_id"],
            prompt_id=row["prompt_id"],
            text=row["text"],
            completion_a=row["completion_a"],
            completion_b=row["completion_b"],
            model_a=row["model_a"],
            model_b=row["model_b"],
            category=row.get("category", ""),
            annotations=anns,
        )
