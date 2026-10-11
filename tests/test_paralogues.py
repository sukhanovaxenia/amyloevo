"""Paralogue assignment: the method, its confidence, and what it refuses."""

from __future__ import annotations

import numpy as np
import pytest

from amyloevo.group.paralogues import (
    OUTGROUP,
    assign,
    assignment_table,
    copy_number,
    core_columns,
    summarise,
)

# Two reference "paralogues" differing at the second half, plus members closer
# to one or the other. Short enough to reason about by hand.
REF_A = "MKLVFFAEDVGSNKGA"
REF_B = "MKLVFFAEDWWWWWGA"

ALIGNMENT = {
    "refA": REF_A,
    "refB": REF_B,
    "a1": "MKLVFFAEDVGSNKGA",      # identical to A
    "a2": "MKLVFFAEDVGSNKGC",      # one off A
    "b1": "MKLVFFAEDWWWWWGA",      # identical to B
    "b2": "MKLVFFAEDWWWWWGC",      # one off B
}
REFERENCES = {"refA": "A", "refB": "B"}


def test_core_columns_are_where_every_reference_has_a_residue() -> None:
    aln = dict(ALIGNMENT)
    aln["refA"] = "MKLV--AEDVGSNKGA"
    cols = core_columns(aln, REFERENCES)
    assert 4 not in cols and 5 not in cols
    assert 0 in cols and 15 in cols


def test_reference_not_in_alignment_is_refused() -> None:
    with pytest.raises(KeyError, match="not in the alignment"):
        core_columns(ALIGNMENT, {"absent": "X", "refB": "B"})


def test_assignment_follows_identity() -> None:
    by_id = {a.sequence_id: a for a in assign(ALIGNMENT, REFERENCES)}
    assert by_id["a1"].group == "A"
    assert by_id["a2"].group == "A"
    assert by_id["b1"].group == "B"
    assert by_id["b2"].group == "B"


def test_a_reference_is_assigned_to_itself() -> None:
    by_id = {a.sequence_id: a for a in assign(ALIGNMENT, REFERENCES)}
    assert by_id["refA"].group == "A"
    assert by_id["refA"].best_identity == pytest.approx(1.0)


def test_margin_separates_a_clear_call_from_a_near_tie() -> None:
    by_id = {a.sequence_id: a for a in assign(ALIGNMENT, REFERENCES)}
    assert by_id["a1"].margin > 0.2

    # Equidistant between the two references: a coin flip, and it must say so.
    # The two references differ at four columns (9-12); this sequence takes two
    # from each, so the identities are equal by construction rather than by
    # happening to land close.
    tied = {
        "refA": "MKLVFFAEDVGSNKGA",
        "refB": "MKLVFFAEDWWWWKGA",
        "t":    "MKLVFFAEDVGWWKGA",
    }
    by_id = {a.sequence_id: a for a in assign(tied, REFERENCES)}
    assert by_id["t"].identities["A"] == pytest.approx(by_id["t"].identities["B"])
    assert by_id["t"].margin == pytest.approx(0.0)
    assert not by_id["t"].is_confident()


def test_fragment_is_not_penalised_on_identity_but_is_on_coverage() -> None:
    """Identity over shared columns; completeness reported separately."""
    aln = dict(ALIGNMENT)
    aln["frag"] = "MKLVFFAED" + "-" * 7
    by_id = {a.sequence_id: a for a in assign(aln, REFERENCES)}
    assert by_id["frag"].best_identity == pytest.approx(1.0)
    assert by_id["frag"].core_coverage < 0.6


def test_one_reference_is_refused() -> None:
    with pytest.raises(ValueError, match="at least two references"):
        assign(ALIGNMENT, {"refA": "A"})


def test_references_with_no_shared_core_are_refused() -> None:
    aln = {"r1": "MKLV----", "r2": "----MKLV", "x": "MKLVMKLV"}
    with pytest.raises(ValueError, match="no column"):
        assign(aln, {"r1": "A", "r2": "B"})


def test_any_number_of_groups_works() -> None:
    """Nothing here is specific to three paralogues."""
    aln = {
        "r1": "AAAAAAAA", "r2": "CCCCCCCC", "r3": "DDDDDDDD", "r4": "EEEEEEEE",
        "x": "AAAAAAAC",
    }
    refs = {"r1": "G1", "r2": "G2", "r3": "G3", "r4": "G4"}
    by_id = {a.sequence_id: a for a in assign(aln, refs)}
    assert by_id["x"].group == "G1"
    assert len(by_id["x"].identities) == 4


# --------------------------------------------------------------------------- #
# Outgroup -- declared, never inferred
# --------------------------------------------------------------------------- #
def test_outgroup_members_get_no_paralogue_call() -> None:
    by_id = {
        a.sequence_id: a
        for a in assign(ALIGNMENT, REFERENCES, outgroup={"a2", "b2"})
    }
    assert by_id["a2"].group == OUTGROUP
    assert by_id["b2"].group == OUTGROUP
    assert by_id["a1"].group == "A"


def test_outgroup_keeps_its_nearest_reference_as_an_observation() -> None:
    """Which paralogue a pre-duplication gene most resembles is a real fact.

    It just is not an assignment, so it lives in a different field.
    """
    a2 = next(a for a in assign(ALIGNMENT, REFERENCES, outgroup={"a2"})
              if a.sequence_id == "a2")
    assert a2.group == OUTGROUP
    assert a2.nearest == "A"
    assert np.isfinite(a2.best_identity)


def test_outgroup_accepts_a_predicate() -> None:
    assigned = assign(ALIGNMENT, REFERENCES, outgroup=lambda sid: sid.startswith("b"))
    out = {a.sequence_id for a in assigned if a.is_outgroup}
    assert out == {"b1", "b2"}


def test_an_outgroup_sequence_is_never_confident() -> None:
    """Even at a wide margin: there is no group for the margin to be about."""
    a1 = next(a for a in assign(ALIGNMENT, REFERENCES, outgroup={"a1"})
              if a.sequence_id == "a1")
    assert a1.margin > 0.2
    assert not a1.is_confident()


def test_a_custom_outgroup_label_is_still_recognised_as_the_outgroup() -> None:
    """A family naming its outgroup "Invertebrate" must behave identically.

    Inferring outgroup status from the label string works only for the default
    and silently puts a custom label back in the group count -- which it did,
    on the real FXR1 set, turning three paralogues into four.
    """
    assignments = assign(
        ALIGNMENT, REFERENCES, outgroup={"a2"}, outgroup_label="Invertebrate"
    )
    a2 = next(a for a in assignments if a.sequence_id == "a2")
    assert a2.group == "Invertebrate"
    assert a2.is_outgroup
    assert copy_number(assignments, {"a1": 1, "a2": 1})["n_groups"] == 2


def test_outgroup_does_not_raise_the_copy_number_bar() -> None:
    """A family of two groups plus an outgroup still expects two copies.

    Counting the outgroup label as a third group would mark every complete
    genome deficient.
    """
    assignments = assign(ALIGNMENT, REFERENCES, outgroup={"a2"})
    taxid = {"refA": 1, "refB": 1, "a1": 2, "a2": 2, "b1": 2, "b2": 3}
    result = copy_number(assignments, taxid)
    assert result["n_groups"] == 2
    assert result["deficient"] == [3]


def test_summary_separates_outgroup_from_ambiguous() -> None:
    """Both are excluded from a between-group contrast, for different reasons."""
    text = summarise(assign(ALIGNMENT, REFERENCES, outgroup={"a2", "b2"}))
    assert "outside the reference groups" in text
    assert "of 4 assigned" in text


# --------------------------------------------------------------------------- #
# Copy number -- an observation, not an inference
# --------------------------------------------------------------------------- #
def test_copy_number_counts_per_taxon() -> None:
    assignments = assign(ALIGNMENT, REFERENCES)
    taxid = {"refA": 9606, "refB": 9606, "a1": 10090, "a2": 10090,
             "b1": 10090, "b2": 7227}
    result = copy_number(assignments, taxid, n_groups=2)
    assert result["per_taxon"] == {9606: 2, 10090: 3, 7227: 1}
    assert result["deficient"] == [7227]
    assert set(result["complete"]) == {9606, 10090}


def test_copy_number_accepts_a_callable() -> None:
    assignments = assign(ALIGNMENT, REFERENCES)
    result = copy_number(assignments, lambda sid: sid[0], n_groups=2)
    assert result["per_taxon"]["a"] == 2


def test_deficient_is_not_called_pre_duplication() -> None:
    """The word is deliberately absent: it is an inference a tree must make."""
    assignments = assign(ALIGNMENT, REFERENCES)
    result = copy_number(assignments, {"refA": 1, "refB": 1}, n_groups=2)
    assert "pre_duplication" not in result
    assert "deficient" in result


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def test_table_carries_every_identity_column() -> None:
    table = assignment_table(assign(ALIGNMENT, REFERENCES))
    assert {"identity_A", "identity_B"} <= set(table.columns)
    assert set(table["group"]) == {"A", "B"}


def test_summary_states_the_ambiguous_count() -> None:
    text = summarise(assign(ALIGNMENT, REFERENCES))
    assert "margin" in text
    assert "BETWEEN groups" in text


def test_summary_of_nothing() -> None:
    assert "No sequences" in summarise([])
