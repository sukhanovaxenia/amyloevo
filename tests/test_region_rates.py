"""Tests for amyloevo.phylo.region_rates."""

from __future__ import annotations

import numpy as np
import pytest

from amyloevo.phylo import region_rates as rr


# --------------------------------------------------------------------------- #
# masks and spans

def test_mask_is_inclusive_on_both_ends():
    mask = rr.column_mask([(2, 4)], 10)
    assert mask.tolist() == [False, False, True, True, True,
                             False, False, False, False, False]


def test_overlapping_blocks_count_a_column_once():
    mask = rr.column_mask([(2, 5), (4, 7)], 10)
    assert mask.sum() == 6


def test_a_label_is_refused_rather_than_parsed():
    # '91-99' is reference residues and 'col1615-1620' is columns; one parser
    # cannot be right for both, so neither is attempted.
    with pytest.raises(TypeError, match="label"):
        rr.column_mask(["91-99"], 2000)


def test_block_outside_the_alignment_is_refused():
    with pytest.raises(ValueError, match="outside an alignment"):
        rr.column_mask([(5, 12)], 10)


def test_block_objects_are_accepted():
    class Block:
        def __init__(self, a, b):
            self.start, self.stop = a, b

    assert rr.column_mask([Block(1, 3)], 8).sum() == 3


# --------------------------------------------------------------------------- #
# reading IQ-TREE rate files

RATE_FILE = """# Site-specific rates determined by empirical Bayes
# Columns are tab-separated
Site\tRate\tCat\tC_Rate
1\t0.5\t1\t0.4
2\t1.5\t3\t1.6
3\t1.0\t2\t1.0
"""


def test_rates_are_read_by_column_name(tmp_path):
    p = tmp_path / "x.rate"
    p.write_text(RATE_FILE)
    assert rr.read_site_rates(p).tolist() == [0.5, 1.5, 1.0]


def test_rates_are_read_even_when_the_rate_column_moves(tmp_path):
    # the same data with Rate in a different position: reading the second
    # field positionally would return the category index instead
    p = tmp_path / "y.rate"
    p.write_text("Site\tCat\tRate\n1\t1\t0.5\n2\t3\t1.5\n3\t2\t1.0\n")
    assert rr.read_site_rates(p).tolist() == [0.5, 1.5, 1.0]


def test_rows_out_of_order_are_sorted_by_site(tmp_path):
    p = tmp_path / "z.rate"
    p.write_text("Site\tRate\n3\t1.0\n1\t0.5\n2\t1.5\n")
    assert rr.read_site_rates(p).tolist() == [0.5, 1.5, 1.0]


def test_a_rate_file_from_another_alignment_is_refused(tmp_path):
    p = tmp_path / "x.rate"
    p.write_text(RATE_FILE)
    with pytest.raises(ValueError, match="different alignments"):
        rr.read_site_rates(p, n_columns=400)


def test_a_file_without_a_rate_column_is_refused(tmp_path):
    p = tmp_path / "x.rate"
    p.write_text("Site\tCat\n1\t1\n")
    with pytest.raises(ValueError, match="no 'Rate' column"):
        rr.read_site_rates(p)


def test_missing_sites_are_refused(tmp_path):
    p = tmp_path / "x.rate"
    p.write_text("Site\tRate\n1\t0.5\n3\t1.0\n")
    with pytest.raises(ValueError, match="contiguous"):
        rr.read_site_rates(p)


# --------------------------------------------------------------------------- #
# the contrast

def test_contrast_arithmetic():
    rates = np.array([1.0, 1.0, 3.0, 3.0])
    summary = rr.contrast(rates, np.array([True, True, False, False]))
    assert summary.mean_region == 1.0
    assert summary.mean_background == 3.0
    assert summary.ratio == pytest.approx(1 / 3)
    assert (summary.n_region, summary.n_background) == (2, 2)


def test_contrast_needs_both_sides():
    with pytest.raises(ValueError, match="needs both sides"):
        rr.contrast(np.ones(4), np.ones(4, dtype=bool))


# --------------------------------------------------------------------------- #
# placement

def test_placed_blocks_keep_their_widths_and_do_not_overlap():
    rng = np.random.default_rng(0)
    widths = [3, 5, 2]
    for _ in range(300):
        spans = rr.place_blocks(widths, 40, rng)
        assert sorted(b - a + 1 for a, b in spans) == sorted(widths)
        assert all(0 <= a <= b < 40 for a, b in spans)
        flat = sorted(spans)
        for (a1, b1), (a2, _) in zip(flat, flat[1:]):
            assert b1 < a2


def test_blocks_that_cannot_fit_are_refused():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="cannot be placed"):
        rr.place_blocks([10, 10], 15, rng)


def test_placement_reaches_both_ends_of_the_alignment():
    # a null that never puts a block at column 0 or at the last column would
    # be comparing against arrangements the data could not take either
    rng = np.random.default_rng(1)
    first, last = False, False
    for _ in range(2000):
        spans = rr.place_blocks([4], 20, rng)
        first |= spans[0][0] == 0
        last |= spans[0][1] == 19
    assert first and last


# --------------------------------------------------------------------------- #
# the test itself

def test_uniform_rates_give_a_ratio_of_one_and_no_signal():
    rates = np.ones(200)
    result = rr.placement_test(rates, [(10, 19), (50, 59)],
                               permutations=200,
                               rng=np.random.default_rng(2))
    assert result.observed == pytest.approx(1.0)
    assert result.p_value > 0.05


def test_a_genuinely_slow_region_is_detected():
    rates = np.ones(300)
    rates[100:140] = 0.05            # one slow block, far below everything
    result = rr.placement_test(rates, [(100, 139)], permutations=500,
                               rng=np.random.default_rng(3))
    assert result.observed < 0.2
    assert result.p_value < 0.01


def test_p_value_is_never_zero():
    rates = np.ones(200)
    rates[0:20] = 0.001
    result = rr.placement_test(rates, [(0, 19)], permutations=100,
                               rng=np.random.default_rng(4))
    assert result.p_value >= 1 / 101


def test_null_p_values_are_not_systematically_small():
    """Under the null the test must not manufacture significance.

    This is the property the block-relocation design exists for: with rates
    autocorrelated along the alignment, a column-shuffling null would reject
    almost every time here.
    """
    rng = np.random.default_rng(5)
    small = 0
    trials = 60
    for _ in range(trials):
        # autocorrelated rates: a random walk smoothed over 10 columns
        noise = rng.normal(size=400)
        rates = np.convolve(noise, np.ones(10) / 10, mode="same")
        rates = rates - rates.min() + 0.1
        spans = rr.place_blocks([12, 9, 15], 400, rng)   # blocks placed at random
        result = rr.placement_test(rates, spans, permutations=200, rng=rng)
        small += result.p_value < 0.05
    assert small <= trials * 0.20, f"{small}/{trials} false positives"


# --------------------------------------------------------------------------- #
# the partition file

def test_partition_is_one_based_and_covers_every_column():
    text = rr.partition_text([(0, 2), (5, 6)], 10, model="LG+G4")
    assert "charset apr = 1-3 6-7;" in text
    assert "charset background = 4-5 8-10;" in text
    assert "LG+G4:apr" in text


def test_partition_needs_both_sides():
    with pytest.raises(ValueError, match="both sides"):
        rr.partition_text([(0, 9)], 10)


# --------------------------------------------------------------------------- #
# the report

def test_report_names_the_direction():
    rates = np.ones(100)
    rates[0:10] = 0.2
    summary = rr.contrast(rates, rr.column_mask([(0, 9)], 100))
    text = rr.format_region_report(summary, name="APR")
    assert "SLOWER" in text
    rates2 = np.ones(100)
    rates2[0:10] = 5.0
    summary2 = rr.contrast(rates2, rr.column_mask([(0, 9)], 100))
    assert "FASTER" in rr.format_region_report(summary2, name="APR")


# --------------------------------------------------------------------------- #
# occupancy and the symmetric filter

def test_occupancy_counts_residues_per_column(tmp_path):
    p = tmp_path / "a.faa"
    p.write_text(">x\nAB-D\n>y\nA--D\n>z\nAB-D\n")
    assert rr.occupancy(p).tolist() == pytest.approx([1.0, 2 / 3, 0.0, 1.0])


def test_occupancy_refuses_an_unaligned_file(tmp_path):
    p = tmp_path / "b.faa"
    p.write_text(">x\nABCD\n>y\nAB\n")
    with pytest.raises(ValueError, match="not an alignment"):
        rr.occupancy(p)


def test_occupancy_accepts_a_mapping():
    assert rr.occupancy({"x": "A-C", "y": "ABC"}).tolist() == pytest.approx(
        [1.0, 0.5, 1.0])


def test_usable_is_applied_to_both_sides():
    rates = np.array([1.0, 99.0, 1.0, 3.0, 99.0, 3.0])
    mask = np.array([True, True, True, False, False, False])
    keep = np.array([True, False, True, True, False, True])
    summary = rr.contrast(rates, mask, usable=keep)
    assert summary.mean_region == 1.0        # the 99 inside is excluded
    assert summary.mean_background == 3.0    # and so is the 99 outside
    assert summary.n_region == 2 and summary.n_background == 2


def test_filtering_one_side_only_would_change_the_answer():
    # guards the symmetry: if `usable` were applied to the region alone the
    # background would keep its outlier and the ratio would collapse
    rates = np.array([1.0, 99.0, 1.0, 3.0, 99.0, 3.0])
    mask = np.array([True, True, True, False, False, False])
    keep = np.array([True, False, True, True, False, True])
    symmetric = rr.contrast(rates, mask, usable=keep).ratio
    lopsided = rates[mask & keep].mean() / rates[~mask].mean()
    assert symmetric == pytest.approx(1 / 3)
    assert lopsided != pytest.approx(symmetric)


def test_placement_test_honours_usable():
    rates = np.ones(300)
    rates[100:140] = 0.05
    keep = np.zeros(300, dtype=bool)
    keep[20:260] = True                   # a contiguous usable core
    result = rr.placement_test(rates, [(100, 139)], permutations=300,
                               rng=np.random.default_rng(11), usable=keep)
    assert result.observed < 0.2
    assert result.p_value < 0.01


def test_a_width_that_fits_in_no_usable_run_is_refused():
    """Not silently widened to an unrestricted null: if a block of this width
    could not have been defined anywhere the mask allows, the mask and the
    blocks disagree and the user has to say which one is wrong."""
    rates = np.ones(300)
    keep = np.zeros(300, dtype=bool)
    keep[::7] = True                      # no run longer than one column
    with pytest.raises(ValueError, match="could not have been defined"):
        rr.placement_test(rates, [(100, 139)], permutations=50,
                          rng=np.random.default_rng(14), usable=keep)


def test_the_null_rests_on_as_much_data_as_the_observation():
    """The flaw this guards: blocks sit on well-covered columns by definition,
    so placements scattered over a mostly-sparse alignment carry a fraction of
    the usable columns, their ratios are noisy, and the null fans out until
    nothing can reach significance. Matching on the usable count removes that
    and the spread collapses."""
    rng = np.random.default_rng(31)
    rates = rng.gamma(2.0, 0.4, size=2000)
    keep = np.zeros(2000, dtype=bool)
    keep[:400] = True                     # only a fifth of the alignment usable
    blocks = [(10, 23), (60, 71), (120, 133), (200, 213)]
    confined = rr.placement_test(rates, blocks, permutations=600,
                                 rng=np.random.default_rng(32), usable=keep,
                                 restrict=True)
    loose = rr.placement_test(rates, blocks, permutations=600,
                              rng=np.random.default_rng(32), usable=keep,
                              restrict=False, min_usable=1)
    assert (confined.null_hi - confined.null_lo) < (loose.null_hi - loose.null_lo)
    assert confined.observed_usable == confined.n_block_columns
    assert confined.outside_allowed == 0


def test_blocks_with_no_usable_column_are_refused():
    """The observed contrast fails first, and says so in its own terms."""
    rates = np.ones(200)
    keep = np.zeros(200, dtype=bool)
    keep[:12] = True                      # the blocks sit outside all of it
    with pytest.raises(ValueError, match="needs both sides"):
        rr.placement_test(rates, [(150, 179)], permutations=100,
                          rng=np.random.default_rng(12), usable=keep)


def test_a_null_that_mostly_lands_outside_the_usable_columns_is_refused():
    """The observed blocks are fine; the PLACEMENTS are not, and that is a
    different failure. Averaging the few placements that happened to land on
    usable columns would answer a question nobody asked."""
    rates = np.ones(4000)
    rates[5:35] = 0.2
    keep = np.zeros(4000, dtype=bool)
    keep[:40] = True                      # observed block usable, most placements not
    with pytest.raises(ValueError, match="too sparse|too fragmented"):
        rr.placement_test(rates, [(5, 34)], permutations=200,
                          rng=np.random.default_rng(13), usable=keep,
                          restrict=False)


def test_min_usable_excludes_placements_resting_on_a_handful_of_columns():
    """A placement overlapping two usable columns yields a ratio computed from
    two numbers. Keeping those widens the null with noise and makes the test
    look more conservative than it is, so they are dropped rather than
    averaged in -- and the threshold has to be seen to do something."""
    rates = np.ones(400)
    rates[20:50] = 0.3
    keep = np.zeros(400, dtype=bool)
    keep[:300] = True
    lax = rr.placement_test(rates, [(20, 49)], permutations=400,
                            rng=np.random.default_rng(21), usable=keep,
                            min_usable=1, restrict=False)
    strict = rr.placement_test(rates, [(20, 49)], permutations=400,
                               rng=np.random.default_rng(21), usable=keep,
                               min_usable=25, restrict=False)
    assert strict.permutations < lax.permutations
    assert lax.observed == pytest.approx(strict.observed)


def test_two_sided_p_is_calibrated_on_a_skewed_null():
    """A ratio's null is right-skewed: bounded by zero below, unbounded above.
    Measured as |observed - median| a halving counts as milder than a doubling,
    so an observation sitting at the 2.5th percentile of the null can come back
    around p = 0.14 while the correct answer is near 0.05. On the log scale the
    two directions are the same distance, which is what two-sided has to mean
    for a ratio.
    """
    rates = np.ones(400)
    rates[100:300] = 0.02             # half the protein slow -> skewed null
    blocks = [(110, 129), (160, 179), (240, 259)]
    result = rr.placement_test(rates, blocks, permutations=3000,
                               rng=np.random.default_rng(77), restrict=False)
    # the observation lands on the lower 2.5% edge of the null
    assert result.observed == pytest.approx(result.null_lo, rel=0.05)
    assert result.p_value < 0.10, (
        f"p={result.p_value:.4f} for an observation at the 2.5th percentile; "
        f"the statistic is not measuring both directions alike")


def test_partition_excludes_the_same_columns_as_the_contrast():
    """The background must not quietly take the sparse columns the contrast
    threw out: those sit in indel-rich, badly alignable stretches and are
    genuinely fast, so keeping them inflates the background and the region
    looks more constrained than it is."""
    keep = np.array([True] * 6 + [False] * 4)
    text = rr.partition_text([(0, 2)], 10, model="LG+G4", usable=keep)
    assert "charset apr = 1-3;" in text
    assert "charset background = 4-6;" in text
    assert "charset unusable = 7-10;" in text
    # every site belongs to some partition: none dropped without saying so.
    # The excluded set carries a plain model, so count the assignments rather
    # than the model string.
    charpartition = [ln for ln in text.splitlines() if "charpartition" in ln][0]
    assert charpartition.count(":") == 3
    assert "LG+G4:apr" in text and "LG+G4:background" in text


def test_partition_without_a_mask_keeps_every_column_in_two_sets():
    text = rr.partition_text([(0, 2)], 10)
    assert "charset apr = 1-3;" in text
    assert "charset background = 4-10;" in text
    assert "unusable" not in text


def test_partition_refuses_when_a_side_has_no_usable_column():
    keep = np.array([True] * 3 + [False] * 7)
    with pytest.raises(ValueError, match="both sides"):
        rr.partition_text([(0, 2)], 10, usable=keep)


def test_the_excluded_set_gets_a_model_without_rate_heterogeneity():
    """Its speed is never read, it carries almost no information, and it is
    where IQ-TREE's free-rate optimiser aborts. Four rate categories there buy
    nothing and cost the whole run."""
    keep = np.array([True] * 6 + [False] * 4)
    text = rr.partition_text([(0, 2)], 10, model="JTT+R4", usable=keep)
    assert "JTT+R4:apr" in text
    assert "JTT+R4:background" in text
    assert "JTT:unusable" in text
    assert "JTT+R4:unusable" not in text


def test_the_excluded_model_can_be_named_outright():
    keep = np.array([True] * 6 + [False] * 4)
    text = rr.partition_text([(0, 2)], 10, model="JTT+G4", usable=keep,
                             excluded_model="LG")
    assert "LG:unusable" in text
