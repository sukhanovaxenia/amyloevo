"""Sequence weights, and what they are allowed to change.

The point of these is the boundary: weighting must change the per-column sums
and nothing else. An unweighted run has to stay bit-for-bit what it was, or the
weighted and unweighted analyses are not comparable and the whole exercise --
run both, report what moved -- collapses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from amyloevo.conserve import blocks, weights


def _triplicate():
    """Three identical sequences and one distinct one."""
    return {"a": "MKVLLAA", "b": "MKVLLAA", "c": "MKVLLAA", "d": "QQWEERT"}


# --- the schemes -----------------------------------------------------------


def test_uniform_is_the_uncorrected_baseline():
    assert weights.uniform(["a", "b"]) == {"a": 1.0, "b": 1.0}


def test_every_scheme_has_mean_one():
    """So a threshold keeps the meaning it has unweighted, and the two runs are
    comparable without rescaling."""
    frame = pd.DataFrame({"id": list("abcd"),
                          "order": ["Rodentia"] * 3 + ["Diptera"]})
    for scheme in (weights.uniform(list("abcd")),
                   weights.by_rank(frame),
                   weights.henikoff(_triplicate())):
        values = np.array(list(scheme.values()))
        assert values.mean() == pytest.approx(1.0)


def test_henikoff_splits_one_sequence_worth_between_identical_rows():
    w = weights.henikoff(_triplicate())
    assert w["a"] == pytest.approx(w["b"]) == pytest.approx(w["c"])
    # the triplicate collectively weighs what the singleton does
    assert w["a"] + w["b"] + w["c"] == pytest.approx(w["d"])


def test_henikoff_does_not_reward_a_gappy_row_for_being_unlike_the_rest():
    """Treating gap as a residue would make a mostly-gap sequence the most
    distinctive thing in the alignment and hand it the largest weight."""
    aln = {"a": "MKVLLAA", "b": "MKVLLAA", "short": "MK-----"}
    w = weights.henikoff(aln)
    assert w["short"] < max(w["a"], w["b"]) * 2


def test_by_rank_gives_each_level_the_same_total():
    frame = pd.DataFrame({
        "id": ["r1", "r2", "r3", "r4", "d1"],
        "order": ["Rodentia"] * 4 + ["Diptera"],
    })
    w = weights.by_rank(frame, "order")
    assert sum(w[i] for i in ["r1", "r2", "r3", "r4"]) == pytest.approx(w["d1"])


def test_by_rank_pools_unresolved_taxa_instead_of_dropping_them():
    """Giving an unresolved taxon weight 1 would make missing taxonomy the
    heaviest category in the set."""
    frame = pd.DataFrame({"id": ["a", "b", "c"],
                          "order": ["Rodentia", "", None]})
    w = weights.by_rank(frame, "order")
    assert set(w) == {"a", "b", "c"}
    assert w["b"] == pytest.approx(w["c"])


def test_summarise_reports_the_effective_sample_size():
    text = weights.summarise(weights.henikoff(_triplicate()))
    assert "effective sample size" in text
    assert "4 sequence(s)" in text


# --- what weighting may and may not change ---------------------------------


def _masks(n_columns=10):
    """Three identical mammals carrying a block, one outgroup without it."""
    carry = np.zeros(n_columns, dtype=bool)
    carry[2:7] = True
    return {
        "m1": carry.copy(), "m2": carry.copy(), "m3": carry.copy(),
        "o1": np.zeros(n_columns, dtype=bool),
    }


def test_no_weights_is_identical_to_the_previous_arithmetic():
    """Not merely equivalent: the unweighted run must stay exactly
    reproducible, or the comparison has two moving parts."""
    masks = _masks()
    occ_a, cov_a = blocks.occupancy(masks, list(masks), 10)
    occ_b, cov_b = blocks.occupancy(masks, list(masks), 10, weights=None)
    assert np.array_equal(occ_a, occ_b)
    assert np.array_equal(cov_a, cov_b)


def test_uniform_weights_reproduce_the_unweighted_result():
    masks = _masks()
    plain = blocks.occupancy(masks, list(masks), 10)[0]
    weighted = blocks.occupancy(masks, list(masks), 10,
                                weights=weights.uniform(masks))[0]
    assert weighted == pytest.approx(plain)


def test_weighting_moves_occupancy_toward_the_down_weighted_majority():
    """The correction working: three identical mammals stop counting as three
    observations, so a block only they carry loses occupancy."""
    masks = _masks()
    plain = blocks.occupancy(masks, list(masks), 10)[0]
    w = {"m1": 1 / 3, "m2": 1 / 3, "m3": 1 / 3, "o1": 1.0}
    weighted = blocks.occupancy(masks, list(masks), 10, weights=w)[0]
    assert plain[4] == pytest.approx(0.75)      # 3 of 4 sequences
    assert weighted[4] == pytest.approx(0.5)    # 1 unit of 2
    assert weighted[4] < plain[4]


def test_a_member_missing_from_the_weights_counts_as_one():
    """A partial mapping under-corrects rather than silently dropping
    sequences from the denominator."""
    masks = _masks()
    full = blocks.occupancy(masks, list(masks), 10,
                            weights={"m1": 1.0, "m2": 1.0, "m3": 1.0,
                                     "o1": 1.0})[0]
    partial = blocks.occupancy(masks, list(masks), 10, weights={"m1": 1.0})[0]
    assert partial == pytest.approx(full)


def test_coverage_is_a_fraction_of_total_weight_not_of_the_count():
    """blocks_from_occupancy compares coverage against a fractional threshold,
    so the denominator has to be the weighted total or the gate stops meaning
    what it says."""
    masks = _masks()
    w = {"m1": 1 / 3, "m2": 1 / 3, "m3": 1 / 3, "o1": 1.0}
    cov = blocks.occupancy(masks, list(masks), 10, weights=w)[1]
    assert cov.max() <= 1.0 + 1e-9
    assert cov[0] == pytest.approx(1.0)      # every sequence spans the columns


def test_weighting_can_drop_a_block_carried_only_by_the_dense_clade():
    """The behaviour the whole exercise is about: a block present in a densely
    sampled clade and nowhere else stops clearing a family-level threshold."""
    masks = _masks()
    members = {"all": list(masks)}
    plain = blocks.define_blocks(masks, members, 10, min_group=1,
                                 min_occupancy=0.6)
    w = {"m1": 1 / 3, "m2": 1 / 3, "m3": 1 / 3, "o1": 1.0}
    weighted = blocks.define_blocks(masks, members, 10, min_group=1,
                                    min_occupancy=0.6, weights=w)
    assert plain and not weighted


def test_min_group_still_counts_sequences_not_weight():
    """A group of five near-identical sequences is still five sequences;
    whether it is enough to define blocks from is a sample-size question."""
    masks = _masks()
    tiny = {"g": ["m1", "m2"]}
    w = {"m1": 0.01, "m2": 0.01}
    assert blocks.define_blocks(masks, tiny, 10, min_group=2, weights=w)
    assert not blocks.define_blocks(masks, tiny, 10, min_group=3, weights=w)


def test_group_occupancy_takes_a_weighted_mean():
    masks = _masks()
    found = blocks.define_blocks(masks, {"all": list(masks)}, 10, min_group=1)
    members = {"mammal": ["m1", "m2", "m3"], "out": ["o1"]}
    w = {"m1": 1 / 3, "m2": 1 / 3, "m3": 1 / 3, "o1": 1.0}
    plain = blocks.group_occupancy(masks, found, members)
    weighted = blocks.group_occupancy(masks, found, members, weights=w)
    # within a group of identical sequences the mean is unchanged
    assert weighted["occ_mammal"].tolist() == pytest.approx(
        plain["occ_mammal"].tolist())


def test_block_profiles_are_per_sequence_and_weights_never_touch_them():
    """The feature matrix is one row per sequence. Weighting an individual's
    own profile would be meaningless, and the per-clade analyses read it."""
    masks = _masks()
    found = blocks.define_blocks(masks, {"all": list(masks)}, 10, min_group=1)
    profiles = blocks.block_profiles(masks, found, list(masks))
    assert profiles.shape == (4, len(found))
    assert profiles[0].tolist() == profiles[1].tolist()


@pytest.mark.parametrize("scale", [0.01, 0.5, 1.0, 7.0, 1000.0])
def test_occupancy_is_invariant_to_the_scale_of_the_weights(scale):
    """Occupancy is a ratio, so multiplying every weight by a constant must
    not move it. The first implementation failed this: a divide-by-zero guard
    written as max(present, 1.0) was safe for an integer count of sequences and
    clamped every fractional denominator below 1, so weights summing to 0.02
    gave occupancy 0.02 instead of 1.0. Blocks then vanished under weighting --
    which is what a working correction also does, so it would have read as a
    result."""
    masks = _masks()
    members = list(masks)
    plain = blocks.occupancy(masks, members, 10)[0]
    scaled = blocks.occupancy(
        masks, members, 10, weights={m: scale for m in members})[0]
    assert scaled == pytest.approx(plain)


def test_coverage_is_invariant_to_the_scale_of_the_weights():
    masks = _masks()
    members = list(masks)
    plain = blocks.occupancy(masks, members, 10)[1]
    scaled = blocks.occupancy(
        masks, members, 10, weights={m: 0.01 for m in members})[1]
    assert scaled == pytest.approx(plain)


def test_only_the_relative_weights_matter():
    """Two mappings differing by a constant factor must give the same blocks."""
    masks = _masks()
    members = {"all": list(masks)}
    a = blocks.define_blocks(masks, members, 10, min_group=1,
                             weights={"m1": 1, "m2": 1, "m3": 1, "o1": 3})
    b = blocks.define_blocks(masks, members, 10, min_group=1,
                             weights={"m1": 10, "m2": 10, "m3": 10, "o1": 30})
    assert a == b
