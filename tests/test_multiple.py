"""Multiple-comparison adjustment: correctness against a reference, and edges.

The implementation is validated against ``scipy.stats.false_discovery_control``
rather than against values quoted from memory. A hand-copied "expected" column
is exactly as likely to be wrong as the code it checks, and a wrong reference
makes a correct implementation look broken -- which happened once while this was
being written.
"""

from __future__ import annotations

import numpy as np
import pytest

from amyloevo.stats.multiple import benjamini_hochberg, holm, n_discoveries


# --------------------------------------------------------------------------- #
# Benjamini-Hochberg
# --------------------------------------------------------------------------- #
def test_matches_scipy_over_many_random_families() -> None:
    scipy_stats = pytest.importorskip("scipy.stats")
    if not hasattr(scipy_stats, "false_discovery_control"):
        pytest.skip("scipy too old for false_discovery_control")
    rng = np.random.default_rng(0)
    for _ in range(200):
        m = int(rng.integers(1, 60))
        p = rng.random(m)
        np.testing.assert_allclose(
            benjamini_hochberg(p),
            scipy_stats.false_discovery_control(p, method="bh"),
            rtol=0, atol=1e-12,
        )


def test_q_values_are_monotone_in_p() -> None:
    """The step-up correction must not reorder the family."""
    rng = np.random.default_rng(1)
    p = rng.random(50)
    q = benjamini_hochberg(p)
    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= -1e-12)


def test_a_single_test_is_unadjusted() -> None:
    assert benjamini_hochberg([0.037])[0] == pytest.approx(0.037)


def test_q_never_exceeds_one() -> None:
    assert np.all(benjamini_hochberg([0.9, 0.95, 0.99, 1.0]) <= 1.0)


def test_input_order_is_preserved() -> None:
    p = [0.5, 0.001, 0.2]
    q = benjamini_hochberg(p)
    # The smallest p must keep its position, whatever the internal sort did.
    assert int(np.argmin(q)) == 1


def test_nans_pass_through_without_entering_the_denominator() -> None:
    """A test that could not be run is not a test that failed to reject.

    Counting it in ``m`` would penalise every other test in the family for a
    comparison that never happened -- e.g. a rank skipped for having no
    replicated levels.
    """
    with_nan = benjamini_hochberg([0.01, np.nan, 0.02])
    without = benjamini_hochberg([0.01, 0.02])
    assert np.isnan(with_nan[1])
    np.testing.assert_allclose(with_nan[[0, 2]], without)


def test_empty_and_all_nan() -> None:
    assert benjamini_hochberg([]).size == 0
    assert np.all(np.isnan(benjamini_hochberg([np.nan, np.nan])))


def test_ties_get_the_same_q() -> None:
    q = benjamini_hochberg([0.04, 0.04, 0.9])
    assert q[0] == pytest.approx(q[1])


# --------------------------------------------------------------------------- #
# Holm
# --------------------------------------------------------------------------- #
def test_holm_is_never_less_conservative_than_bh() -> None:
    """FWER control costs power; if it ever looked cheaper, something is wrong."""
    rng = np.random.default_rng(2)
    for _ in range(100):
        p = rng.random(int(rng.integers(2, 40)))
        assert np.all(holm(p) >= benjamini_hochberg(p) - 1e-12)


def test_holm_smallest_p_is_bonferroni() -> None:
    p = [0.001, 0.4, 0.6, 0.8]
    assert holm(p)[0] == pytest.approx(0.004)


def test_holm_is_monotone() -> None:
    rng = np.random.default_rng(3)
    p = rng.random(30)
    q = holm(p)
    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= -1e-12)


# --------------------------------------------------------------------------- #
# Counting
# --------------------------------------------------------------------------- #
def test_n_discoveries_counts_at_the_threshold() -> None:
    assert n_discoveries([0.01, 0.05, 0.051, np.nan], alpha=0.05) == 2


def test_n_discoveries_of_nothing_is_zero() -> None:
    assert n_discoveries([]) == 0
    assert n_discoveries([np.nan, np.nan]) == 0


def test_the_families_this_package_actually_builds() -> None:
    """A realistic shape: 91 per-protein tests, each on ~4 regions.

    With four observations the smallest attainable permutation p is bounded well
    above any FDR threshold, so the per-protein family can be entirely null
    while the pooled direction is unanimous. The adjustment must not hide that;
    it must make it visible.
    """
    floor = 0.0625  # the smallest p a 4-observation two-sided sign test can give
    q = benjamini_hochberg(np.full(91, floor))
    assert n_discoveries(q, alpha=0.05) == 0
    assert np.all(q >= floor)
