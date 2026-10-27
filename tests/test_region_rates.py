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
