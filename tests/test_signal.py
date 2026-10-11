"""Taxonomic signal: what PERMANOVA says, and what it refuses to say.

Most of these tests exist because the unguarded version of this analysis gave a
confident wrong answer on real data -- FXR2 returned R^2 = 0.99 at order and
family level on a design where almost every level held one sequence. A test
suite for this module is mostly a set of designs it must decline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from amyloevo.phylo.signal import (
    MIN_LEVELS,
    MIN_PER_LEVEL,
    NEGLIGIBLE_WITHIN_SCATTER,
    SignalResult,
    format_signal_report,
    permanova,
    taxonomic_signal,
)


def _distances(profiles: np.ndarray) -> np.ndarray:
    from scipy.spatial.distance import pdist, squareform

    return squareform(pdist(profiles, metric="euclidean"))


# --------------------------------------------------------------------------- #
# PERMANOVA itself
# --------------------------------------------------------------------------- #
def test_separated_groups_are_detected() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 0.2, size=(12, 6))
    b = rng.normal(3.0, 0.2, size=(12, 6))
    labels = np.array(["a"] * 12 + ["b"] * 12)
    f, r2, p = permanova(_distances(np.vstack([a, b])), labels, draws=499)
    assert f > 10
    assert r2 > 0.8
    assert p == pytest.approx(1 / 500)


def test_no_group_effect_is_not_significant() -> None:
    """One draw of noise with labels that mean nothing."""
    rng = np.random.default_rng(1)
    profiles = rng.normal(size=(40, 8))
    labels = np.array(["a", "b", "c", "d"] * 10)
    _, _, p = permanova(_distances(profiles), labels, draws=499, seed=7)
    assert p > 0.05


def test_the_permutation_p_is_calibrated() -> None:
    """Under the null, p should be roughly uniform -- not merely non-significant.

    A sampler that is subtly biased still produces 'not significant' on a single
    null draw; it shows up as a rejection rate far from alpha over many draws.
    """
    rng = np.random.default_rng(2)
    pvals = []
    for i in range(60):
        profiles = rng.normal(size=(24, 5))
        labels = np.array(["a", "b", "c", "d"] * 6)
        pvals.append(permanova(_distances(profiles), labels, draws=199, seed=i)[2])
    rejected = sum(1 for p in pvals if p <= 0.05)
    # 60 draws at alpha=0.05: expect 3, and 9 is already a 1-in-1000 tail.
    assert rejected <= 8, f"rejected {rejected}/60 nulls at alpha=0.05"


def test_p_cannot_be_smaller_than_one_over_draws_plus_one() -> None:
    rng = np.random.default_rng(3)
    profiles = np.vstack([rng.normal(0, 0.01, (10, 4)), rng.normal(50, 0.01, (10, 4))])
    labels = np.array(["a"] * 10 + ["b"] * 10)
    assert permanova(_distances(profiles), labels, draws=99)[2] == pytest.approx(0.01)


def test_one_level_is_degenerate() -> None:
    rng = np.random.default_rng(4)
    profiles = rng.normal(size=(10, 3))
    f, r2, p = permanova(_distances(profiles), np.array(["a"] * 10))
    assert np.isnan(f) and np.isnan(r2) and np.isnan(p)


def test_all_singletons_is_refused_rather_than_reported_as_perfect() -> None:
    """The failure this guard exists for.

    With one observation per level there is no within-group scatter, the F ratio
    has a zero denominator and R^2 is 1 by construction. Reported as a number it
    is the strongest row in the table and means nothing.
    """
    rng = np.random.default_rng(5)
    profiles = rng.normal(size=(12, 4))
    labels = np.array([f"level{i}" for i in range(12)])
    f, r2, p = permanova(_distances(profiles), labels)
    assert np.isnan(f) and np.isnan(r2) and np.isnan(p)


def test_zero_within_group_scatter_is_refused() -> None:
    """Replicated levels, but every member of a level identical."""
    profiles = np.repeat(np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]), 4, axis=0)
    labels = np.repeat(["a", "b", "c"], 4)
    f, _, _ = permanova(_distances(profiles), labels)
    assert np.isnan(f)


def test_identical_profiles_give_no_statistic() -> None:
    profiles = np.zeros((12, 3))
    labels = np.repeat(["a", "b", "c"], 4)
    assert np.isnan(permanova(_distances(profiles), labels)[0])


# --------------------------------------------------------------------------- #
# The rank sweep
# --------------------------------------------------------------------------- #
def _frame(n: int, *, levels_per_rank: dict[str, int]) -> pd.DataFrame:
    data = {}
    for rank, k in levels_per_rank.items():
        data[rank] = [f"{rank}{i % k}" for i in range(n)]
    return pd.DataFrame(data)


def test_a_rank_with_real_structure_is_reported() -> None:
    rng = np.random.default_rng(6)
    n_levels, per = 5, 6
    centres = rng.normal(0, 5, size=(n_levels, 4))
    profiles = np.vstack([
        centres[i] + rng.normal(0, 0.3, size=(per, 4)) for i in range(n_levels)
    ])
    meta = pd.DataFrame({"order": np.repeat([f"o{i}" for i in range(n_levels)], per)})
    result = taxonomic_signal(profiles, meta, ranks=("order",), draws=299)
    row = result.ranks[0]
    assert row.tested
    assert row.levels == n_levels
    assert row.n == n_levels * per
    assert row.q_value <= 0.05


def test_a_design_of_mostly_singleton_levels_is_skipped() -> None:
    """77 sequences, 56 families: not enough replicated levels to test."""
    rng = np.random.default_rng(7)
    profiles = rng.normal(size=(77, 10))
    families = [f"f{i}" for i in range(56)] + [f"f{i % 56}" for i in range(21)]
    meta = pd.DataFrame({"family": families})
    result = taxonomic_signal(profiles, meta, ranks=("family",), draws=99)
    row = result.ranks[0]
    assert not row.tested
    assert "members" in row.skipped
    assert np.isnan(row.r2)


def test_degenerate_profiles_are_flagged_while_the_test_still_runs() -> None:
    """The real FXR2 shape: replicated levels, but almost no distinct profiles.

    7 orders over 48 sequences carrying 7 profiles between them, three orders
    internally identical. Every replication filter passes and R^2 reaches 0.99
    for arithmetic reasons, so the row must be reported WITH the reason rather
    than skipped: the permutation p-value is honest, the R^2 is not an effect
    size.
    """
    sizes = [12, 11, 7, 5, 5, 5, 3]
    rng = np.random.default_rng(70)
    centres = rng.normal(0, 3, size=(len(sizes), 8))
    rows, labels = [], []
    for i, size in enumerate(sizes):
        # One profile per order, repeated -- plus a whisper of noise in two of
        # them so within-group scatter is small but not exactly zero.
        jitter = 0.01 if i in (0, 1) else 0.0
        rows.append(np.tile(centres[i], (size, 1))
                    + rng.normal(0, jitter, size=(size, 8)))
        labels += [f"o{i}"] * size
    profiles = np.vstack(rows)
    meta = pd.DataFrame({"order": labels})

    row = taxonomic_signal(profiles, meta, ranks=("order",), draws=499).ranks[0]
    assert row.tested, "the test is valid and must not be suppressed"
    assert row.levels == 7 and row.n == 48
    assert row.r2 > 0.95
    assert row.p_value < 0.01
    assert row.scatter_negligible
    assert row.within_total == pytest.approx(1.0 - row.r2)


def test_the_flag_survives_jitter_that_makes_every_row_distinct() -> None:
    """Why the flag keys on scatter and not on a count of distinct rows.

    Adding noise far too small to matter makes every profile unique, so a
    distinct-row criterion stops firing on a design it was built to catch. The
    pathology is unchanged, and so is the flag.
    """
    rng = np.random.default_rng(73)
    centres = rng.normal(0, 3, size=(6, 8))
    profiles = np.vstack([np.tile(centres[i], (8, 1)) for i in range(6)])
    meta = pd.DataFrame({"order": np.repeat([f"o{i}" for i in range(6)], 8)})

    jittered = profiles + rng.normal(0, 1e-6, size=profiles.shape)
    row = taxonomic_signal(jittered, meta, ranks=("order",), draws=199).ranks[0]
    assert row.distinct_profiles == row.n, "every row is now numerically unique"
    assert row.scatter_negligible, "and the degeneracy is still there"


def test_a_healthy_design_is_not_flagged() -> None:
    """FXR1's shape: real within-order scatter, R^2 in the plausible range."""
    rng = np.random.default_rng(71)
    centres = rng.normal(0, 2, size=(6, 8))
    profiles = np.vstack([
        centres[i] + rng.normal(0, 1.4, size=(9, 8)) for i in range(6)
    ])
    meta = pd.DataFrame({"order": np.repeat([f"o{i}" for i in range(6)], 9)})
    row = taxonomic_signal(profiles, meta, ranks=("order",), draws=299).ranks[0]
    assert row.tested
    assert row.within_total > NEGLIGIBLE_WITHIN_SCATTER
    assert not row.scatter_negligible


def test_distinct_profiles_counts_the_forms_in_a_discretised_profile() -> None:
    """Block occupancy is discrete, and the count is the sentence to report.

    Four occupancy patterns spread over five orders, two of which mix a pair of
    patterns so there is real within-group scatter and the row gets tested.
    """
    forms = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 0.5, 0.0],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
    ])
    plan = {                      # order -> the form index of each member
        "o0": [0, 0, 0, 0, 1],    # mixed
        "o1": [1, 1, 1, 1],
        "o2": [2, 2, 2, 2, 3],    # mixed
        "o3": [3, 3, 3],
        "o4": [0, 0, 0],
    }
    profiles = np.vstack([forms[i] for members in plan.values() for i in members])
    labels = [name for name, members in plan.items() for _ in members]
    row = taxonomic_signal(
        profiles, pd.DataFrame({"order": labels}), ranks=("order",), draws=199
    ).ranks[0]
    assert row.tested
    assert row.n == len(labels) == 20
    assert row.levels == 5
    assert row.distinct_profiles == 4, "four occupancy patterns, 20 sequences"


def test_the_flag_is_in_the_report_with_what_to_say_instead() -> None:
    profiles = np.repeat(np.arange(7.0).reshape(7, 1) * np.ones((1, 4)),
                         [12, 11, 7, 5, 5, 5, 3], axis=0)
    profiles = profiles + np.random.default_rng(72).normal(0, 0.01, profiles.shape)
    meta = pd.DataFrame({"order": np.repeat([f"o{i}" for i in range(7)],
                                            [12, 11, 7, 5, 5, 5, 3])})
    text = format_signal_report(
        taxonomic_signal(profiles, meta, group="FXR2", ranks=("order",), draws=199)
    )
    assert "(!)" in text
    assert "NOT an effect size" in text
    assert "distinct" in text


def test_a_level_below_min_per_level_is_dropped_from_n() -> None:
    rng = np.random.default_rng(8)
    profiles = rng.normal(size=(26, 4))
    # Four levels of 6, plus two singletons that must not be tested.
    labels = [f"g{i // 6}" for i in range(24)] + ["lonely1", "lonely2"]
    meta = pd.DataFrame({"genus": labels})
    result = taxonomic_signal(profiles, meta, ranks=("genus",), draws=99)
    row = result.ranks[0]
    assert row.levels == 4
    assert row.n == 24


def test_too_few_sequences_skips_every_rank_with_a_reason() -> None:
    rng = np.random.default_rng(9)
    profiles = rng.normal(size=(6, 4))
    meta = _frame(6, levels_per_rank={"order": 2, "family": 3})
    result = taxonomic_signal(profiles, meta, ranks=("order", "family"), draws=99)
    assert [r.tested for r in result.ranks] == [False, False]
    assert all("sequences" in r.skipped for r in result.ranks)


def test_an_absent_rank_column_is_named_not_crashed_on() -> None:
    rng = np.random.default_rng(10)
    profiles = rng.normal(size=(24, 4))
    meta = pd.DataFrame({"order": [f"o{i % 4}" for i in range(24)]})
    result = taxonomic_signal(profiles, meta, ranks=("order", "subphylum"), draws=99)
    assert result.ranks[1].skipped == "column absent"


def test_p_values_are_adjusted_across_ranks() -> None:
    """Four nested ranks on one dataset is a family of four tests."""
    rng = np.random.default_rng(11)
    centres = rng.normal(0, 4, size=(5, 4))
    profiles = np.vstack([centres[i] + rng.normal(0, 0.4, (6, 4)) for i in range(5)])
    classes = np.repeat([f"c{i}" for i in range(5)], 6)
    meta = pd.DataFrame({"class": classes, "order": classes, "family": classes,
                         "genus": classes})
    result = taxonomic_signal(profiles, meta, draws=199)
    tested = [r for r in result.ranks if r.tested]
    assert len(tested) == 4
    assert all(r.q_value >= r.p_value - 1e-12 for r in tested)


def test_only_tested_ranks_enter_the_adjustment() -> None:
    """A skipped rank is not a test, so it must not inflate m."""
    rng = np.random.default_rng(12)
    centres = rng.normal(0, 4, size=(4, 4))
    profiles = np.vstack([centres[i] + rng.normal(0, 0.4, (6, 4)) for i in range(4)])
    good = np.repeat([f"o{i}" for i in range(4)], 6)
    meta = pd.DataFrame({"order": good, "genus": [f"g{i}" for i in range(24)]})
    result = taxonomic_signal(profiles, meta, ranks=("order", "genus"), draws=199)
    order = result.ranks[0]
    assert order.q_value == pytest.approx(order.p_value)
    assert not result.ranks[1].tested


def test_significant_lists_only_tested_ranks() -> None:
    result = SignalResult(group="g", n_sequences=0)
    assert result.significant() == []


def test_nan_profile_rows_are_blamed_on_the_input_not_the_design() -> None:
    """Both failures come back as nan; only one of them is the caller's data."""
    rng = np.random.default_rng(13)
    profiles = rng.normal(size=(24, 4))
    profiles[3, 0] = np.nan
    meta = pd.DataFrame({"order": [f"o{i % 4}" for i in range(24)]})
    result = taxonomic_signal(profiles, meta, ranks=("order",), draws=99)
    row = result.ranks[0]
    assert not row.tested
    assert "non-finite" in row.skipped
    assert "index 3" in row.skipped


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def test_report_states_the_skip_reason_and_the_caveat() -> None:
    rng = np.random.default_rng(14)
    profiles = rng.normal(size=(77, 8))
    meta = pd.DataFrame({"family": [f"f{i}" for i in range(77)]})
    text = format_signal_report(
        taxonomic_signal(profiles, meta, group="FXR2", ranks=("family",), draws=99)
    )
    assert "FXR2" in text
    assert "skipped" in text
    assert "CLUSTER by clade" in text
    assert "caused" in text


def test_the_guard_constants_are_not_accidentally_permissive() -> None:
    """These thresholds are the whole guard; a change to them is a decision."""
    assert MIN_PER_LEVEL >= 3
    assert MIN_LEVELS >= 3
