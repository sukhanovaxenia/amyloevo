"""The coordinate spine: a wrong column here corrupts every downstream measure.

These tests use alignments whose gaps are placed so that an off-by-one produces
a DIFFERENT PEPTIDE rather than a neighbouring index -- a wrong residue letter
is visible in a failure message, a wrong integer is not.
"""

from __future__ import annotations

import pytest

from amyloevo.project.columns import (
    AlignmentIndex,
    build_mapping,
    index_alignment,
    is_gap,
)

#: Abeta42 with KLVFFA at residues 16-21.
ABETA42 = "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA"


def test_ungapped_sequence_is_the_identity():
    mapping = build_mapping("plain", ABETA42)
    assert mapping.column_of(1) == 1
    assert mapping.column_of(16) == 16
    assert mapping.residue_at(21) == 21
    assert mapping.ungapped_length == len(ABETA42)


def test_a_leading_gap_shifts_every_residue():
    mapping = build_mapping("shifted", "---" + ABETA42)
    assert mapping.column_of(1) == 4
    assert mapping.column_of(16) == 19
    assert mapping.residue_character(19) == "K"


def test_the_offset_is_not_constant_across_a_sequence():
    """The failure this module exists to prevent: treating the gap count at the
    start of a region as the offset for the whole region."""
    aligned = ABETA42[:16] + "----" + ABETA42[16:]   # a gap INSIDE the peptide
    mapping = build_mapping("interrupted", aligned)
    assert mapping.column_of(16) == 16, "before the gap, offset 0"
    assert mapping.column_of(17) == 21, "after the gap, offset 4"
    assert mapping.column_of(21) == 25


def test_region_columns_exclude_interior_gaps_but_the_span_includes_them():
    """Two different questions with two different answers, neither the default."""
    aligned = ABETA42[:18] + "--" + ABETA42[18:]
    mapping = build_mapping("split", aligned)
    columns = mapping.columns_for_region(16, 21)
    assert len(columns) == 6, "six residues, whatever lies between them"
    assert mapping.region_span(16, 21) == (16, 23)
    assert 19 not in columns and 20 not in columns


def test_the_recovered_peptide_is_the_one_we_meant():
    aligned = "--" + ABETA42[:18] + "---" + ABETA42[18:]
    mapping = build_mapping("gappy", aligned)
    recovered = "".join(
        mapping.residue_character(c) for c in mapping.columns_for_region(16, 21)
    )
    assert recovered == "KLVFFA"


def test_a_gapped_column_has_no_residue_number():
    mapping = build_mapping("gapped", "AB--CD")
    assert mapping.residue_at(3) is None
    assert mapping.residue_at(5) == 3


def test_out_of_range_residue_returns_none_not_a_sentinel():
    mapping = build_mapping("short", "ACDE")
    assert mapping.column_of(99) is None
    assert mapping.column_of(0) is None


def test_a_column_outside_the_alignment_raises():
    mapping = build_mapping("short", "ACDE")
    with pytest.raises(IndexError, match="outside 1..4"):
        mapping.residue_character(5)


def test_a_reversed_region_is_rejected():
    mapping = build_mapping("x", ABETA42)
    with pytest.raises(ValueError, match="start 21 > end 16"):
        mapping.columns_for_region(21, 16)


# --- whole-alignment behaviour ---------------------------------------------


def _alignment() -> AlignmentIndex:
    return index_alignment(
        {
            "human": "MKLVFFAEDVG",
            "mouse": "MKLVYFAEDVG",
            "fly":   "MK--FFAE--G",
        }
    )


def test_unaligned_input_is_refused_with_the_lengths():
    with pytest.raises(ValueError, match="common length"):
        index_alignment({"a": "MKLV", "b": "MKLVFF"})


def test_empty_alignment_is_refused():
    with pytest.raises(ValueError, match="empty alignment"):
        index_alignment({})


def test_occupancy_counts_residues_not_sequences():
    index = _alignment()
    assert index.occupancy(1) == 3
    assert index.occupancy(3) == 2, "fly is gapped at column 3"
    assert index.occupancy(9) == 2


def test_column_residues_drop_gaps_unless_asked():
    index = _alignment()
    assert sorted(index.column_residues(3)) == ["L", "L"]
    assert sorted(index.column_residues(3, include_gaps=True)) == ["-", "L", "L"]


def test_transfer_finds_the_aligned_residue_in_another_species():
    index = _alignment()
    # human residue 5 is the first F of KLVFFA; fly has a residue there.
    assert index["human"].residue_character(index["human"].column_of(5)) == "F"
    assert index.transfer("human", 5, "mouse") == 5
    assert index.transfer("human", 5, "fly") == 3, "fly is two residues shorter"


def test_transfer_returns_none_where_the_orthologue_is_gapped():
    """Not a failure: the orthologue has no residue there, and an APR cannot be
    conserved at a position that does not exist."""
    index = _alignment()
    assert index.transfer("human", 3, "fly") is None


@pytest.mark.parametrize("character,expected", [("-", True), (".", True), ("~", True), ("A", False)])
def test_gap_characters(character, expected):
    assert is_gap(character) is expected
