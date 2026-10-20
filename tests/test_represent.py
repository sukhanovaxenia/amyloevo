"""Choosing display representatives: the rank must come from the row budget.

The case these are built around is the one that makes a declared rank wrong.
On an orthologue set a genus contributes about one sequence per paralogue, so
one representative per genus reduces the pooled set a great deal and a single
paralogue group hardly at all -- and a per-group figure only ever sees the
second number. The fixture below reproduces that asymmetry, and the test that
matters is that the same metadata yields a different rank depending on the
scope it is asked about.
"""

from __future__ import annotations

import pandas as pd
import pytest

from amyloevo.viz import represent as P

#: Twelve genera in four families in two orders. Group A carries g1-g8, group B
#: carries g1-g4 and g9-g12, so pooled there are 12 genera while each group has
#: only 8 -- the asymmetry an orthologue set produces.
GENUS_FAMILY = {
    "g1": "f1", "g2": "f1", "g3": "f1",
    "g4": "f2", "g5": "f2", "g6": "f2",
    "g7": "f3", "g8": "f3", "g9": "f3",
    "g10": "f4", "g11": "f4", "g12": "f4",
}
FAMILY_ORDER = {"f1": "o1", "f2": "o1", "f3": "o2", "f4": "o2"}
GROUPS = {
    "A": [f"g{i}" for i in range(1, 9)],
    "B": [f"g{i}" for i in (1, 2, 3, 4, 9, 10, 11, 12)],
}


def _metadata() -> pd.DataFrame:
    rows = []
    for group, genera in GROUPS.items():
        for n, genus in enumerate(genera):
            rows.append({
                "id": f"{group}_{genus}",
                "genus": genus,
                "family": GENUS_FAMILY[genus],
                "order": FAMILY_ORDER[GENUS_FAMILY[genus]],
                "class": "c1",
                "paralogue": group,
                # distinct per row, except the deliberate tie below
                "core_coverage": 0.50 + n / 100,
            })
    frame = pd.DataFrame(rows)
    # two members of g1 tie on coverage, so the tie-break is exercised
    frame.loc[frame["id"] == "B_g1", "core_coverage"] = \
        float(frame.loc[frame["id"] == "A_g1", "core_coverage"].iloc[0])
    return frame


# --- the rank comes from the budget ---------------------------------------


def test_the_finest_rank_that_fits_is_chosen():
    frame = _metadata()
    choice = P.choose_rank(frame, target=10)
    assert choice.rank == "family"      # genus is 12, over the budget
    assert choice.n_levels == 4
    assert choice.fits
    assert ("genus", 12) in choice.rejected


def test_the_same_metadata_gives_a_different_rank_for_a_single_group():
    """The point of the module. Pooled, genus is over the budget and family
    wins; inside one paralogue group genus fits, and a figure of that group
    should use it."""
    frame = _metadata()
    pooled = P.choose_rank(frame, target=10)
    group = P.choose_rank(frame, target=10,
                          members=frame.loc[frame["paralogue"] == "A", "id"])
    assert pooled.rank == "family"
    assert group.rank == "genus"
    assert group.n_levels == 8


def test_a_rank_that_already_fits_is_not_coarsened():
    frame = _metadata()
    choice = P.choose_rank(frame, target=50)
    assert choice.rank == "genus"
    assert choice.rejected == []


def test_nothing_is_dropped_to_meet_an_impossible_budget():
    """Reaching a row count by discarding levels would mean choosing which
    clades the figure omits, and a row budget does not decide that."""
    frame = _metadata()
    choice = P.choose_rank(frame, target=1, ranks=("genus", "family", "order"))
    assert not choice.fits
    assert choice.rank == "order"
    assert choice.n_levels == 2


def test_a_rank_whose_levels_do_not_nest_is_refused():
    """More families than genera means one genus is filed under two families.
    Grouping by it would merge sequences that are not related the way the
    figure claims, so the rank is reported and not used."""
    frame = pd.DataFrame([
        {"id": "x1", "genus": "ga", "family": "f1", "core_coverage": 0.9},
        {"id": "x2", "genus": "ga", "family": "f2", "core_coverage": 0.8},
        {"id": "x3", "genus": "gb", "family": "f3", "core_coverage": 0.7},
    ])
    choice = P.choose_rank(frame, target=2, ranks=("genus", "family"))
    assert "family" in choice.non_monotonic
    assert choice.rank == "genus"


# --- picking the representative -------------------------------------------


def test_the_representative_is_the_highest_scoring_member():
    frame = _metadata()
    # a unique maximum inside f1, so the assertion tests the max and not the
    # tie-break: with the fixture's default coverages A_g3 and B_g3 are equal
    frame.loc[frame["id"] == "A_g3", "core_coverage"] = 0.99
    chosen, _ = P.representatives(frame, "family")
    assert chosen["f1"] == "A_g3"


def test_ties_are_broken_by_sorted_id_and_counted():
    """A level whose members score identically is represented by whichever id
    sorts first. That is arbitrary, so the count is reported rather than the
    choice being hidden."""
    frame = _metadata()
    chosen, ties = P.representatives(
        frame, "genus", members=["A_g1", "B_g1"])
    assert chosen["g1"] == "A_g1"
    assert ties == 1


def test_sequences_with_no_value_at_the_rank_are_pooled_not_dropped():
    """Dropping them would quietly remove whichever clade the taxonomy is
    worst for, which on this family is the invertebrates."""
    frame = _metadata()
    frame.loc[frame["id"] == "A_g1", "genus"] = ""
    frame.loc[frame["id"] == "A_g2", "genus"] = None
    chosen, _ = P.representatives(frame, "genus")
    assert P.UNRESOLVED in chosen
    assert chosen[P.UNRESOLVED] in {"A_g1", "A_g2"}


def test_selection_is_deterministic():
    frame = _metadata()
    first = P.select_for_display(frame, target=10).ids
    second = P.select_for_display(frame, target=10).ids
    assert first == second


def test_one_row_per_level_and_the_count_is_reported():
    frame = _metadata()
    selection = P.select_for_display(frame, target=10)
    assert len(selection.ids) == selection.n_levels == 4
    assert selection.n_input == 16
    assert selection.rank == "family"


# --- the report, which is what ends up in a caption -----------------------


def test_the_report_names_the_column_the_choice_was_made_on():
    frame = _metadata()
    text = P.format_selection_report(P.select_for_display(frame, target=10))
    assert "core_coverage" in text
    assert "caption:" in text
    assert "select on its own variable" in text


def test_a_missing_score_column_is_named_rather_than_silently_ignored():
    """Without the column every member scores the same, so every level is a
    tie and the panel is chosen entirely by identifier order. That is a
    legitimate fallback and an illegitimate thing to leave unsaid."""
    frame = _metadata().drop(columns=["core_coverage"])
    selection = P.select_for_display(frame, target=10)
    assert "ABSENT" in selection.score_column
    assert "ABSENT" in P.format_selection_report(selection)


def test_the_report_warns_when_the_budget_cannot_be_met():
    frame = _metadata()
    text = P.format_selection_report(
        P.select_for_display(frame, target=1, ranks=("genus", "order")))
    assert "over the budget" in text
    assert "not a decision a row count can make" in text


def test_the_report_notes_a_panel_outside_the_readable_range():
    frame = _metadata()
    text = P.format_selection_report(
        P.select_for_display(frame, target=10, ranks=("order",)))
    assert "reads well at" in text
