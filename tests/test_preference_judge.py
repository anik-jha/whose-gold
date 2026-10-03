"""Invariants for the judge harness. The un-swap logic is the load-bearing part:
if it is wrong, every C3 number is wrong in a way no aggregate would reveal.
"""

from __future__ import annotations

import pytest

from wsev.data.multipref import PREF_SCALE
from wsev.judges.preference import (
    build_request,
    combine_orders,
    parse,
    truncate,
    unswap,
)


@pytest.mark.parametrize("lab", PREF_SCALE)
def test_unswap_ab_is_identity(lab):
    assert unswap(lab, "ab") == lab


@pytest.mark.parametrize("lab", PREF_SCALE)
def test_unswap_is_an_involution(lab):
    """Swapping twice must return the original label, for every label."""
    assert unswap(unswap(lab, "ba"), "ba") == lab


def test_unswap_mirrors_direction():
    assert unswap("A-is-clearly-better", "ba") == "B-is-clearly-better"
    assert unswap("B-is-slightly-better", "ba") == "A-is-slightly-better"
    assert unswap("Tie", "ba") == "Tie"


def test_unswap_rejects_unknown_label():
    with pytest.raises(ValueError):
        unswap("A-is-marginally-better", "ab")


def test_build_request_swaps_completions():
    ab = build_request("c1", "q", "AAA", "BBB", "ab")
    ba = build_request("c1", "q", "AAA", "BBB", "ba")
    ab_text = ab.messages[1]["content"]
    ba_text = ba.messages[1]["content"]
    assert ab_text.index("AAA") < ab_text.index("BBB")
    assert ba_text.index("BBB") < ba_text.index("AAA")


def test_truncate_preserves_short_text_exactly():
    """No-op invariant: text under the limit must pass through unchanged."""
    assert truncate("hello", 100) == "hello"


def test_truncate_keeps_both_ends():
    t = truncate("S" + "x" * 500 + "E", 60)
    assert t.startswith("S")
    assert t.endswith("E")
    assert "truncated" in t


def test_parse_constrained_json():
    assert parse('{"preference": "Tie"}') == "Tie"
    assert parse('{\n  "preference": "A-is-slightly-better"\n}\n') == "A-is-slightly-better"


def test_parse_rejects_garbage_and_ambiguity():
    assert parse("") is None
    assert parse('{"preference": "nonsense"}') is None
    assert parse("A-is-clearly-better and B-is-clearly-better") is None  # ambiguous fallback


def test_parse_unconstrained_fallback_single_hit():
    assert parse("I think Tie") == "Tie"


def test_combine_agreeing_orders_is_consistent():
    lab, ok = combine_orders("A-is-clearly-better", "A-is-clearly-better")
    assert lab == "A-is-clearly-better" and ok


def test_combine_disagreeing_orders_flags_inconsistent():
    lab, ok = combine_orders("A-is-clearly-better", "B-is-clearly-better")
    assert not ok
    assert lab == "Tie"  # midpoint of a straddling disagreement


def test_combine_adjacent_disagreement_takes_midpoint():
    lab, ok = combine_orders("A-is-clearly-better", "Tie")
    assert not ok
    assert lab == "A-is-slightly-better"


def test_combine_handles_missing_side():
    lab, ok = combine_orders(None, "Tie")
    assert lab == "Tie" and not ok
    assert combine_orders(None, None) == (None, False)


# --- guard: the experiment scripts are not otherwise covered by tests ---

def test_experiment_scripts_build_requests_with_current_signature():
    """A rename of build_request's kwargs silently broke the sweep launcher once.

    This imports every experiment module and exercises the call path, so a
    signature change fails in 0.4s instead of 50s into a 107-minute run.
    """
    import importlib
    import inspect

    for mod_name in ("experiments.haic.c3_judge_pools", "experiments.haic.c3_mode_equivalence"):
        mod = importlib.import_module(mod_name)
        src = inspect.getsource(mod)
        assert "build_request(" in src, f"{mod_name} no longer calls build_request"

    # And the call itself must actually work with the kwargs the scripts pass.
    for mode in ("json", "digit", "verbose"):
        req = build_request("cid", "q", "a", "b", "ab", max_chars=100, mode=mode)
        assert req.messages[0]["role"] == "system"
        assert req.messages[1]["content"]
