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


# --------------------------------------------------------------------------- #
# Array projections
# --------------------------------------------------------------------------- #
class TestArrayProjections:
    """The whole-alignment forms, used by the block and occupancy work.

    These are 0-based on the column axis while the mapping classes above are
    1-based, so the tests pin the convention rather than trusting the names.
    """

    def test_residue_to_column_skips_gaps(self) -> None:
        import numpy as np

        from amyloevo.project.columns import residue_to_column

        np.testing.assert_array_equal(
            residue_to_column("MK--LV"), np.array([0, 1, 4, 5])
        )

    def test_column_to_residue_marks_gaps_with_zero(self) -> None:
        import numpy as np

        from amyloevo.project.columns import column_to_residue

        np.testing.assert_array_equal(
            column_to_residue("MK--LV"), np.array([1, 2, 0, 0, 3, 4])
        )

    def test_the_two_projections_are_inverse_where_defined(self) -> None:
        from amyloevo.project.columns import column_to_residue, residue_to_column

        aligned = "M-KL--VFF-A"
        cols = residue_to_column(aligned)
        back = column_to_residue(aligned)
        for residue_index, column in enumerate(cols):
            assert back[column] == residue_index + 1

    def test_projecting_a_mask_leaves_gap_columns_false(self) -> None:
        """A sequence that lacks the residue cannot have an APR there.

        Filling the interior gaps of a region would inflate it by whatever
        another species inserted in the middle of it.
        """
        import numpy as np

        from amyloevo.project.columns import project_mask

        # Residues M K L V; a mask over K..L spans a two-column gap.
        mask = np.array([False, True, True, False])
        out = project_mask(mask, "MK--LV", 6)
        np.testing.assert_array_equal(
            out, np.array([False, True, False, False, True, False])
        )

    def test_projection_handles_all_gap_characters(self) -> None:
        import numpy as np

        from amyloevo.project.columns import project_mask

        out = project_mask(np.array([True, True]), "M.~K", 4)
        np.testing.assert_array_equal(out, np.array([True, False, False, True]))

    def test_a_short_mask_projects_its_overlap_rather_than_failing(self) -> None:
        """A predictor run on a slightly different sequence version happens."""
        import numpy as np

        from amyloevo.project.columns import project_mask

        out = project_mask(np.array([True]), "MKLV", 4)
        np.testing.assert_array_equal(out, np.array([True, False, False, False]))

    def test_an_empty_mask_projects_to_nothing(self) -> None:
        import numpy as np

        from amyloevo.project.columns import project_mask

        assert not project_mask(np.array([], dtype=bool), "MKLV", 4).any()

    def test_mask_from_regions_is_one_based_inclusive(self) -> None:
        import numpy as np

        from amyloevo.project.columns import mask_from_regions

        np.testing.assert_array_equal(
            mask_from_regions([(2, 4)], 6),
            np.array([False, True, True, True, False, False]),
        )

    def test_mask_from_regions_accepts_objects_with_start_and_end(self) -> None:
        from dataclasses import dataclass

        from amyloevo.project.columns import mask_from_regions

        @dataclass
        class Region:
            start: int
            end: int

        assert mask_from_regions([Region(1, 2)], 4).tolist() == [
            True, True, False, False
        ]

    def test_a_region_past_the_end_is_clipped_not_an_error(self) -> None:
        from amyloevo.project.columns import mask_from_regions

        assert mask_from_regions([(3, 99)], 4).tolist() == [
            False, False, True, True
        ]

    def test_overlapping_regions_union(self) -> None:
        from amyloevo.project.columns import mask_from_regions

        assert mask_from_regions([(1, 3), (2, 5)], 6).tolist() == [
            True, True, True, True, True, False
        ]
