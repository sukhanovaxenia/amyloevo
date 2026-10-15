"""Binarising block occupancy: the threshold, and the prevalence that D needs.

The screen is the point of this module. A trait carried by three tips out of
372 produces a *D* with a sampling variance covering its whole interpretable
range, and nothing downstream distinguishes that from a measurement -- so the
refusal has to happen here, in the open, with the reason attached.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from amyloevo.phylo import traits as tr


def test_binarise_is_strictly_above_the_threshold():
    profiles = np.array([[0.0, 0.5], [0.51, 1.0]])
    binary = tr.binarise(profiles, ["a", "b"], ["blk1", "blk2"], threshold=0.5)
    assert binary.loc["a"].tolist() == [0, 0]       # 0.5 is not above 0.5
    assert binary.loc["b"].tolist() == [1, 1]


def test_binarise_keeps_ids_and_labels():
    binary = tr.binarise(np.zeros((2, 3)), ["x", "y"], [1, 2, 3])
    assert list(binary.index) == ["x", "y"]
    assert list(binary.columns) == ["1", "2", "3"]
    assert binary.index.name == "id"


def test_binarise_refuses_a_non_finite_occupancy():
    profiles = np.array([[0.9, np.nan]])
    with pytest.raises(ValueError, match="non-finite"):
        tr.binarise(profiles, ["a"], ["blk1", "blk2"])


def test_binarise_refuses_a_shape_mismatch():
    with pytest.raises(ValueError, match="profiles are"):
        tr.binarise(np.zeros((2, 2)), ["a", "b", "c"], ["blk1", "blk2"])


def test_binarise_refuses_a_threshold_outside_the_unit_interval():
    with pytest.raises(ValueError, match="threshold"):
        tr.binarise(np.zeros((1, 1)), ["a"], ["blk"], threshold=1.0)


# --- the screen ------------------------------------------------------------


def _binary(column):
    return pd.DataFrame({"blk": column}, index=[f"t{i}" for i in range(len(column))])


def test_a_balanced_column_is_testable():
    check = tr.screen(_binary([1] * 50 + [0] * 50))[0]
    assert check.usable
    assert check.n_present == 50
    assert check.prevalence == pytest.approx(0.5)


def test_a_constant_column_is_refused():
    for column in ([1] * 40, [0] * 40):
        check = tr.screen(_binary(column))[0]
        assert not check.usable
        assert "constant" in check.reason


def test_a_rare_trait_is_refused_on_the_count():
    check = tr.screen(_binary([1] * 3 + [0] * 97))[0]
    assert not check.usable
    assert "minority state on 3" in check.reason


def test_the_minority_is_counted_on_whichever_side_is_smaller():
    # 97 of 100 present: prevalence is high, but the minority is still 3
    check = tr.screen(_binary([1] * 97 + [0] * 3))[0]
    assert not check.usable
    assert check.minority == 3
    assert "minority state on 3" in check.reason


def test_the_count_and_the_fraction_catch_different_designs():
    # passes the count (10), fails the fraction (10/120 < 5% is false -- 8.3%);
    # so raise the fraction to show the two are independent gates
    column = [1] * 10 + [0] * 110
    assert tr.screen(_binary(column))[0].usable
    strict = tr.screen(_binary(column), min_prevalence=0.1)[0]
    assert not strict.usable
    assert "prevalence" in strict.reason


def test_a_small_but_balanced_column_still_fails_the_count():
    # 10 of 12 passes a 5% prevalence gate; the minority of 2 must still stop it
    check = tr.screen(_binary([1] * 10 + [0] * 2))[0]
    assert not check.usable
    assert check.minority == 2


def test_usable_lists_only_the_testable_columns():
    binary = pd.DataFrame(
        {
            "good": [1] * 50 + [0] * 50,
            "constant": [1] * 100,
            "rare": [1] * 2 + [0] * 98,
        },
        index=[f"t{i}" for i in range(100)],
    )
    assert tr.usable(tr.screen(binary)) == ["good"]


def test_the_report_states_the_threshold_and_every_refusal():
    binary = pd.DataFrame(
        {"good": [1] * 50 + [0] * 50, "rare": [1] * 2 + [0] * 98},
        index=[f"t{i}" for i in range(100)],
    )
    text = tr.format_trait_report(tr.screen(binary), threshold=0.5)
    assert "occupancy > 0.50" in text
    assert "1 of 2 block(s) will be tested" in text
    assert "minority state on 2" in text
    assert "0.4 and 0.6" in text            # the sensitivity instruction


# --- covariates ------------------------------------------------------------


def test_covariates_builds_only_the_columns_it_is_given():
    frame = tr.covariates(["a", "b"], groups={"a": "FXR1", "b": "FMR1"})
    assert list(frame.columns) == ["group"]
    assert frame.loc["b", "group"] == "FMR1"


def test_covariates_fills_a_missing_length_with_nan_rather_than_zero():
    frame = tr.covariates(["a", "b"], lengths={"a": 621})
    assert frame.loc["a", "length"] == 621
    assert np.isnan(frame.loc["b", "length"])


def test_continuous_passes_the_occupancies_through():
    profiles = np.array([[0.25, 0.75]])
    frame = tr.continuous(profiles, ["a"], ["blk1", "blk2"])
    assert frame.loc["a", "blk1"] == pytest.approx(0.25)
