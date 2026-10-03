

# --- tie-break is a real modelling choice, so it must actually bite ---

def test_tiebreak_operates_on_coarse_classes_and_changes_the_answer():
    """A PC review found the paper described one tie-break and the code used another.

    With two annotators split across directions, the rule must decide the item -
    and different rules must give different answers. A tie_break expressed over
    five-point labels silently degenerates to "no rule at all", which is the bug
    this guards.
    """
    from wsev.stats.ranking import Battle

    split = Battle("m_a", "m_b", ("A-is-clearly-better", "B-is-clearly-better"))
    assert split.a_credit_hard(tie_break=("A", "T", "B")) == 1.0
    assert split.a_credit_hard(tie_break=("B", "T", "A")) == 0.0
    # "toward the tie category" is a FORCE, not an ordering: T is not among the
    # tied candidates when annotators split A-vs-B.
    assert split.a_credit_hard(tie_break="T") == 0.5


def test_tiebreak_is_inert_when_there_is_a_real_majority():
    from wsev.stats.ranking import Battle

    clear = Battle("m_a", "m_b", ("A-is-clearly-better", "A-is-slightly-better"))
    for rule in (("A", "T", "B"), ("B", "T", "A"), ("T", "A", "B")):
        assert clear.a_credit_hard(tie_break=rule) == 1.0


def test_tiebreak_rejects_a_rule_that_cannot_decide():
    from wsev.stats.ranking import Battle

    import pytest as _pytest
    split = Battle("m_a", "m_b", ("A-is-clearly-better", "Tie"))
    with _pytest.raises(ValueError):
        split.a_credit_hard(tie_break=("B",))
    with _pytest.raises(ValueError):
        split.a_credit_hard(tie_break="Z")


def test_force_and_priority_rules_disagree_on_ab_splits():
    """The two rule types differ on exactly the items that carry the headline."""
    from wsev.stats.ranking import Battle

    ab_split = Battle("m_a", "m_b", ("A-is-clearly-better", "B-is-clearly-better"))
    assert ab_split.a_credit_hard(tie_break=("A", "T", "B")) == 1.0   # A wins
    assert ab_split.a_credit_hard(tie_break="T") == 0.5               # recorded a tie

    at_split = Battle("m_a", "m_b", ("A-is-clearly-better", "Tie"))
    assert at_split.a_credit_hard(tie_break=("A", "T", "B")) == 1.0
    assert at_split.a_credit_hard(tie_break="T") == 0.5
