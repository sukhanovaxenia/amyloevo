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
    assert "1 of 2 block-by-scope test(s) will be run" in text
    assert "minority state on 2" in text
    assert "0.4 and 0.6" in text            # the sensitivity instruction


# --- per-clade screening ---------------------------------------------------


def _clade_frame(n_per_clade=40):
    """Three clades; block A is clade-specific, block B is balanced everywhere."""
    ids, groups, a, b = [], {}, [], []
    for clade in ("FXR1", "FMR1", "FXR2"):
        for i in range(n_per_clade):
            pid = f"{clade}_{i}"
            ids.append(pid)
            groups[pid] = clade
            a.append(1 if clade == "FXR1" else 0)
            b.append(i % 2)
    return pd.DataFrame({"A": a, "B": b}, index=ids), groups


def test_screen_by_clade_covers_the_whole_tree_and_each_clade():
    binary, groups = _clade_frame()
    checks = tr.screen_by_clade(binary, groups)
    scopes = {c.scope for c in checks}
    assert scopes == {tr.WHOLE_TREE, "FXR1", "FMR1", "FXR2"}
    assert len(checks) == 4 * 2                     # four scopes, two blocks


def test_a_clade_specific_block_is_testable_overall_and_constant_inside():
    """The point of the per-clade pass. Block A is carried by exactly one clade:
    over the whole tree that is a third of the tips and perfectly testable,
    inside each clade there is nothing left to explain."""
    binary, groups = _clade_frame()
    by_key = {(c.scope, c.label): c for c in tr.screen_by_clade(binary, groups)}
    assert by_key[(tr.WHOLE_TREE, "A")].usable
    assert by_key[(tr.WHOLE_TREE, "A")].prevalence == pytest.approx(1 / 3)
    for clade in ("FXR1", "FMR1", "FXR2"):
        check = by_key[(clade, "A")]
        assert not check.usable
        assert "constant" in check.reason
        assert by_key[(clade, "B")].usable       # balanced inside each clade


def test_a_small_clade_is_declined_with_its_size_as_the_reason():
    binary, groups = _clade_frame(n_per_clade=40)
    for i in range(5):
        binary.loc[f"tiny_{i}"] = [1, 0]
        groups[f"tiny_{i}"] = "tiny"
    checks = [c for c in tr.screen_by_clade(binary, groups) if c.scope == "tiny"]
    assert checks and not any(c.usable for c in checks)
    assert "clade has 5 tip(s)" in checks[0].reason


def test_ids_with_no_clade_are_left_out_of_the_per_clade_pass():
    binary, groups = _clade_frame()
    binary.loc["orphan"] = [1, 1]
    checks = tr.screen_by_clade(binary, groups)
    assert {c.scope for c in checks} == {tr.WHOLE_TREE, "FXR1", "FMR1", "FXR2"}
    whole = next(c for c in checks if c.scope == tr.WHOLE_TREE)
    assert whole.n == 121                        # the orphan still has a tip


def test_whole_tree_can_be_switched_off():
    binary, groups = _clade_frame()
    checks = tr.screen_by_clade(binary, groups, whole_tree=False)
    assert tr.WHOLE_TREE not in {c.scope for c in checks}


def test_tasks_lists_only_the_usable_scope_block_pairs():
    binary, groups = _clade_frame()
    checks = tr.screen_by_clade(binary, groups)
    work = tr.tasks(checks)
    assert list(work.columns) == ["scope", "block"]
    assert len(work) == sum(1 for c in checks if c.usable)
    assert ("FXR1", "A") not in set(map(tuple, work.to_numpy()))
    assert (tr.WHOLE_TREE, "A") in set(map(tuple, work.to_numpy()))


def test_the_report_separates_the_scopes_and_says_they_differ():
    binary, groups = _clade_frame()
    text = tr.format_trait_report(tr.screen_by_clade(binary, groups))
    assert "whole tree" in text
    assert "within FXR1" in text
    assert "deepest split IS the duplication" in text


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
