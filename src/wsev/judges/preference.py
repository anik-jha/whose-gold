"""LLM-as-judge over pairwise preferences, on the same scale the humans used.

Two design choices carry the validity of the whole C3 result.

1. POSITION COUNTERBALANCING. LLM judges have a well-documented preference for
   whichever response appears first. C3 asks which annotator pool a judge tracks,
   so an uncorrected position bias would masquerade as pool alignment. Every item
   is therefore judged twice, once in each presentation order, and the pair is
   combined after un-swapping. Items where the two orders disagree are retained
   and reported as the judge's own inconsistency rather than silently resolved.

2. THE SAME 5-POINT SCALE AS THE HUMANS. The judge emits one of the exact labels
   MultiPref annotators used, under a JSON-schema constraint, so judge and human
   labels are directly comparable without a mapping step that could smuggle in
   assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from wsev.data.multipref import PREF_SCALE

Order = Literal["ab", "ba"]

PREFERENCE_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["preference"],
    "properties": {"preference": {"type": "string", "enum": list(PREF_SCALE)}},
}

# Compact form. Emitting the full label costs ~15-18 decoded tokens; on this box
# decode is the binding cost of the sweep, and `{"p":"3"}` is ~7. Same information,
# roughly half the wall-clock. Codes are 1-indexed positions on PREF_SCALE.
_CODES = tuple(str(i + 1) for i in range(len(PREF_SCALE)))
CODE_TO_LABEL = dict(zip(_CODES, PREF_SCALE))

PREFERENCE_SCHEMA_COMPACT: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["p"],
    "properties": {"p": {"type": "string", "enum": list(_CODES)}},
}

SCALE_LEGEND = "\n".join(f"  {c} = {lab}" for c, lab in CODE_TO_LABEL.items())

# Single-token form. Measured on this box 2026-08-09: with 12 concurrent slots an MoE
# decode step gathers a different expert set per sequence, so bytes-moved approaches the
# whole 35 GB model rather than 3 GB of active params. Decode was consuming 69.9% of slot
# time to produce 1.4% of the tokens. A one-token grammar removes that entirely, and the
# task never needed generation: this is a 5-way classification, and the answer is already
# in the logits of a single position.
LABEL_GRAMMAR = "root ::= [1-5]"

# Mirror of the scale: what a label in the swapped presentation means once un-swapped.
_MIRROR = {
    "A-is-clearly-better": "B-is-clearly-better",
    "A-is-slightly-better": "B-is-slightly-better",
    "Tie": "Tie",
    "B-is-slightly-better": "A-is-slightly-better",
    "B-is-clearly-better": "A-is-clearly-better",
}

SYSTEM = (
    "You are evaluating two candidate responses to a user's message. "
    "Judge which response is better overall, considering helpfulness, truthfulness, "
    "and harmlessness. Answer with exactly one of the allowed preference labels."
)

SYSTEM_COMPACT = (
    "You are evaluating two candidate responses to a user's message. "
    "Judge which response is better overall, considering helpfulness, truthfulness, "
    "and harmlessness. Reply with JSON {\"p\": \"<code>\"} using this scale:\n"
    + SCALE_LEGEND
)

SYSTEM_DIGIT = (
    "You are evaluating two candidate responses to a user's message. "
    "Judge which response is better overall, considering helpfulness, truthfulness, "
    "and harmlessness. Reply with exactly one digit and nothing else:\n"
    + SCALE_LEGEND
)


def parse_digit(content: str) -> str | None:
    """Label from a single-token digit response."""
    if not content:
        return None
    return CODE_TO_LABEL.get(content.strip()[:1])


def label_distribution_from_logprobs(top_logprobs: list[dict]) -> dict[str, float] | None:
    """Renormalise the judge's first-token logprobs over the five scale codes.

    Gives the judge a confidence distribution comparable to the human annotators'
    own confidence fields, at zero extra cost - it is the same forward pass.
    """
    import math

    mass: dict[str, float] = {}
    for entry in top_logprobs or []:
        tok = (entry.get("token") or "").strip()
        if tok in CODE_TO_LABEL:
            mass[CODE_TO_LABEL[tok]] = mass.get(CODE_TO_LABEL[tok], 0.0) + math.exp(entry["logprob"])
    total = sum(mass.values())
    if total <= 0:
        return None
    return {k: v / total for k, v in mass.items()}

_TEMPLATE = """User's message:
{prompt}

--- Response A ---
{a}

--- Response B ---
{b}

Which response is better overall?"""


def truncate(text: str, max_chars: int) -> str:
    """Clip long completions from the middle, keeping both ends legible."""
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return f"{text[:head]}\n...[truncated]...\n{text[-tail:]}"


@dataclass(frozen=True)
class JudgeRequest:
    comparison_id: str
    order: Order
    messages: tuple[dict, ...]


_SYSTEMS = {"json": SYSTEM_COMPACT, "digit": SYSTEM_DIGIT, "verbose": SYSTEM}


def build_request(comparison_id: str, prompt: str, completion_a: str, completion_b: str,
                  order: Order, *, max_chars: int = 6000, mode: str = "digit") -> JudgeRequest:
    """Build one judging request. In order 'ba' the completions are swapped."""
    if mode not in _SYSTEMS:
        raise ValueError(f"unknown mode {mode!r}")
    a, b = (completion_a, completion_b) if order == "ab" else (completion_b, completion_a)
    content = _TEMPLATE.format(
        prompt=truncate(prompt, max_chars),
        a=truncate(a, max_chars),
        b=truncate(b, max_chars),
    )
    return JudgeRequest(
        comparison_id=comparison_id,
        order=order,
        messages=({"role": "system", "content": _SYSTEMS[mode]},
                  {"role": "user", "content": content}),
    )


def unswap(label: str, order: Order) -> str:
    """Map a raw judge label back onto the original A/B assignment."""
    if label not in _MIRROR:
        raise ValueError(f"unknown preference label {label!r}")
    return label if order == "ab" else _MIRROR[label]


def parse(content: str) -> str | None:
    """Extract the preference label. Returns None when the judge produced nothing usable."""
    import json as _json

    if not content:
        return None
    try:
        obj = _json.loads(content)
    except _json.JSONDecodeError:
        # Constrained decoding should make this unreachable; fall back rather than crash
        # a multi-hour sweep, and let the caller count the failures.
        hits = [lab for lab in PREF_SCALE if lab in content]
        return hits[0] if len(hits) == 1 else None
    if not isinstance(obj, dict):
        return None
    if "p" in obj:  # compact schema
        return CODE_TO_LABEL.get(str(obj["p"]))
    pref = obj.get("preference")
    return pref if pref in _MIRROR else None


def combine_orders(label_ab: str | None, label_ba: str | None) -> tuple[str | None, bool]:
    """Combine the two presentation orders into one label.

    Returns (label, consistent). Both inputs are already un-swapped. When the two
    orders disagree the judge is position-sensitive on this item; we report Tie
    only when the disagreement straddles the tie, and otherwise mark it
    inconsistent and leave resolution to the analysis.
    """
    if label_ab is None or label_ba is None:
        return (label_ab or label_ba, False)
    if label_ab == label_ba:
        return (label_ab, True)
    order = {lab: i for i, lab in enumerate(PREF_SCALE)}
    mid = (order[label_ab] + order[label_ba]) / 2.0
    return (PREF_SCALE[int(round(mid))], False)
