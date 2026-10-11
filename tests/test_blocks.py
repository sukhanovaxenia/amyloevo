"""APR blocks: definition, merging, and the per-sequence feature matrix."""

from __future__ import annotations

import numpy as np
import pytest

from amyloevo.conserve.blocks import (
    Block,
    block_profiles,
    blocks_from_occupancy,
    define_blocks,
    group_occupancy,
    merge_intervals,
    occupancy,
)


def mask(n: int, *spans) -> np.ndarray:
    out = np.zeros(n, dtype=bool)
    for a, b in spans:
        out[a : b + 1] = True
    return out


# --------------------------------------------------------------------------- #
# Block
# --------------------------------------------------------------------------- #
def test_width_is_inclusive() -> None:
    assert Block(10, 10).width == 1
    assert Block(10, 14).width == 5


def test_label_uses_reference_residues_when_it_can() -> None:
    col2res = np.zeros(20, dtype=int)
    col2res[5:10] = [91, 92, 93, 94, 95]
    assert Block(5, 9).label(col2res) == "91-95"


def test_label_falls_back_to_columns_in_an_insertion() -> None:
    """A block the reference does not have must not borrow its numbering.

    The ``col`` prefix is what keeps an alignment column from being read as a
    residue number four figures later.
    """
    col2res = np.zeros(20, dtype=int)
    assert Block(5, 9).label(col2res) == "col5-9"
    assert Block(5, 9).label() == "col5-9"


# --------------------------------------------------------------------------- #
# Merging
# --------------------------------------------------------------------------- #
def test_overlapping_blocks_merge() -> None:
    assert merge_intervals([(0, 10), (5, 15)]) == [Block(0, 15)]


def test_abutting_blocks_merge() -> None:
    """A single column between two blocks is a threshold crossing, not a gap."""
    assert merge_intervals([(0, 10), (11, 15)]) == [Block(0, 15)]


def test_a_real_gap_is_kept() -> None:
    assert merge_intervals([(0, 10), (13, 15)]) == [Block(0, 10), Block(13, 15)]


def test_merge_is_order_independent() -> None:
    assert merge_intervals([(13, 15), (0, 10), (5, 12)]) == [Block(0, 15)]


def test_merging_nothing() -> None:
    assert merge_intervals([]) == []


# --------------------------------------------------------------------------- #
# Occupancy
# --------------------------------------------------------------------------- #
def test_occupancy_is_over_members_present_not_all_members() -> None:
    """A column in an insertion is not diluted by sequences that lack it.

    Which is why coverage is returned alongside and has to be gated on: high
    occupancy over two present sequences is not evidence.
    """
    masks = {
        "a": mask(10, (0, 4)),
        "b": mask(10, (0, 4)),
        "c": np.zeros(6, dtype=bool),  # shorter: absent from columns 6-9
    }
    occ, cov = occupancy(masks, ["a", "b", "c"], 10)
    assert occ[0] == pytest.approx(2 / 3)
    assert cov[0] == pytest.approx(1.0)
    assert cov[8] == pytest.approx(2 / 3)


def test_occupancy_of_no_members_is_zero_not_nan() -> None:
    occ, cov = occupancy({}, [], 5)
    assert np.all(occ == 0) and np.all(cov == 0)


def test_unknown_members_are_ignored() -> None:
    occ, _ = occupancy({"a": mask(5, (0, 2))}, ["a", "ghost"], 5)
    assert occ[0] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Calling blocks
# --------------------------------------------------------------------------- #
def test_a_run_clearing_both_thresholds_becomes_a_block() -> None:
    occ = np.zeros(20)
    occ[5:12] = 0.8
    cov = np.ones(20)
    assert blocks_from_occupancy(occ, cov) == [Block(5, 11)]


def test_low_coverage_blocks_a_call_however_high_the_occupancy() -> None:
    occ = np.ones(20)
    cov = np.full(20, 0.2)
    assert blocks_from_occupancy(occ, cov) == []


def test_a_short_dip_is_bridged_rather_than_splitting_the_block() -> None:
    """One column below threshold is one indel in one sequence, not a boundary."""
    occ = np.zeros(30)
    occ[5:20] = 0.8
    occ[11] = 0.1
    assert blocks_from_occupancy(occ, np.ones(30)) == [Block(5, 19)]


def test_a_dip_wider_than_max_gap_does_split() -> None:
    occ = np.zeros(30)
    occ[5:20] = 0.8
    occ[11:16] = 0.1
    assert blocks_from_occupancy(occ, np.ones(30)) == [Block(5, 10), Block(16, 19)]


def test_bridging_happens_before_the_width_filter() -> None:
    """Otherwise a dip destroys the block instead of nicking it.

    Two four-column halves split by one dip are each below a min_width of 5 and
    both get dropped, so the region vanishes rather than shifting -- which is
    how a real block (FXR1 columns 1229-1239) came out five columns short.
    """
    occ = np.zeros(30)
    occ[5:14] = 0.8
    occ[9] = 0.1
    assert blocks_from_occupancy(occ, np.ones(30), min_width=5) == [Block(5, 13)]
    assert blocks_from_occupancy(occ, np.ones(30), min_width=5, max_gap=0) == []


def test_coverage_is_compared_as_a_fraction() -> None:
    """A count here against a fractional threshold passes every column.

    The old signature took a member count for exactly this reason; dropping it
    means the two sides have to be in the same units, so this pins them.
    """
    occ = np.ones(20)
    counts = np.full(20, 3.0)  # a count, as if 3 of 100 members were present
    assert blocks_from_occupancy(occ, counts, min_coverage=0.4) != []
    assert blocks_from_occupancy(occ, counts / 100.0, min_coverage=0.4) == []


def test_a_run_narrower_than_min_width_is_dropped() -> None:
    occ = np.zeros(20)
    occ[5:7] = 0.9
    assert blocks_from_occupancy(occ, np.ones(20), min_width=3) == []


def test_a_block_running_to_the_last_column_is_not_lost() -> None:
    """The loop closes after the final element, or the C-terminal block vanishes."""
    occ = np.zeros(20)
    occ[15:] = 0.9
    assert blocks_from_occupancy(occ, np.ones(20)) == [Block(15, 19)]


def test_a_block_starting_at_column_zero_is_not_lost() -> None:
    occ = np.zeros(20)
    occ[:5] = 0.9
    assert blocks_from_occupancy(occ, np.ones(20)) == [Block(0, 4)]


# --------------------------------------------------------------------------- #
# define_blocks -- per group, then merged
# --------------------------------------------------------------------------- #
def test_a_block_private_to_one_group_is_still_defined() -> None:
    """The point of defining per group: a minority block is a discriminative
    feature, and the pooled definition would miss it."""
    masks = {}
    for i in range(6):
        masks[f"g1_{i}"] = mask(60, (10, 20))
    for i in range(6):
        masks[f"g2_{i}"] = mask(60, (10, 20), (40, 50))
    groups = {"g1": [f"g1_{i}" for i in range(6)],
              "g2": [f"g2_{i}" for i in range(6)]}

    per_group = define_blocks(masks, groups, 60)
    pooled = define_blocks(masks, {"all": list(masks)}, 60)

    assert Block(40, 50) in per_group
    assert Block(40, 50) in pooled  # 50% pooled occupancy still clears 0.35
    # Tighten the threshold past the pooled fraction and only the per-group
    # definition keeps it, which is the asymmetry that matters.
    strict_groups = define_blocks(masks, groups, 60, min_occupancy=0.8)
    strict_pooled = define_blocks(masks, {"all": list(masks)}, 60, min_occupancy=0.8)
    assert Block(40, 50) in strict_groups
    assert Block(40, 50) not in strict_pooled


def test_a_group_below_min_group_does_not_define_blocks() -> None:
    """With four sequences every threshold lands on a count of one or two."""
    masks = {f"tiny_{i}": mask(60, (40, 50)) for i in range(3)}
    masks |= {f"big_{i}": mask(60, (10, 20)) for i in range(6)}
    groups = {"tiny": [f"tiny_{i}" for i in range(3)],
              "big": [f"big_{i}" for i in range(6)]}
    found = define_blocks(masks, groups, 60)
    assert Block(10, 20) in found
    assert all(b.start != 40 for b in found)


def test_overlapping_group_blocks_come_back_merged() -> None:
    masks = {f"g1_{i}": mask(60, (10, 25)) for i in range(6)}
    masks |= {f"g2_{i}": mask(60, (20, 35)) for i in range(6)}
    groups = {"g1": [f"g1_{i}" for i in range(6)],
              "g2": [f"g2_{i}" for i in range(6)]}
    assert define_blocks(masks, groups, 60) == [Block(10, 35)]


# --------------------------------------------------------------------------- #
# Feature matrix
# --------------------------------------------------------------------------- #
def test_profile_is_the_fraction_of_each_block_covered() -> None:
    masks = {"a": mask(60, (10, 14)), "b": mask(60, (10, 19))}
    profiles = block_profiles(masks, [Block(10, 19)], ["a", "b"])
    assert profiles[0, 0] == pytest.approx(0.5)
    assert profiles[1, 0] == pytest.approx(1.0)


def test_profile_rows_follow_the_member_order_given() -> None:
    masks = {"a": mask(20, (0, 9)), "b": np.zeros(20, dtype=bool)}
    assert block_profiles(masks, [Block(0, 9)], ["b", "a"])[0, 0] == 0.0


def test_profile_shape_is_members_by_blocks() -> None:
    masks = {f"s{i}": mask(60, (10, 19)) for i in range(7)}
    blocks = [Block(10, 19), Block(30, 39), Block(50, 59)]
    assert block_profiles(masks, blocks, list(masks)).shape == (7, 3)


def test_group_occupancy_is_one_row_per_block_one_column_per_group() -> None:
    masks = {f"g1_{i}": mask(60, (10, 19)) for i in range(4)}
    masks |= {f"g2_{i}": mask(60, (10, 14)) for i in range(4)}
    groups = {"g1": list(masks)[:4], "g2": list(masks)[4:]}
    table = group_occupancy(masks, [Block(10, 19)], groups)
    assert list(table["block"]) == ["10-19"]
    assert table["occ_g1"].iloc[0] == pytest.approx(1.0)
    assert table["occ_g2"].iloc[0] == pytest.approx(0.5)


def test_an_empty_group_is_nan_not_zero() -> None:
    """An unscored group and a group carrying nothing are different claims."""
    masks = {"a": mask(20, (0, 9))}
    table = group_occupancy(masks, [Block(0, 9)], {"has": ["a"], "empty": []})
    assert np.isnan(table["occ_empty"].iloc[0])
    assert table["occ_has"].iloc[0] == pytest.approx(1.0)
