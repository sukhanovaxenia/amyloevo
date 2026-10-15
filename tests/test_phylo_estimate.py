"""Running the R estimators, and what their p-values mean afterwards.

The floor tests are the reason this file exists. This project has already
shipped a table in which every per-protein test came back non-significant after
FDR adjustment, and the cause was not the data: the permutation count put the
smallest attainable p above alpha/m, so no q could ever clear the threshold. The
arithmetic is two lines and it was got wrong twice by reasoning about it instead
of computing it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from amyloevo.phylo import estimate


# --- the permutation floor -------------------------------------------------


def test_no_warning_when_the_floor_clears_alpha_over_m():
    # 15 blocks at alpha 0.05 need p <= 0.00333; 1000 draws floor at 0.000999
    assert estimate.floor_warning(1000, 15, 0.05) == ""


def test_warns_when_the_floor_sits_above_alpha_over_m():
    warning = estimate.floor_warning(100, 15, 0.05)
    assert warning
    assert "0.00990" in warning or "0.009901" in warning
    assert "0.003333" in warning


def test_the_warning_names_the_permutation_count_that_would_suffice():
    warning = estimate.floor_warning(100, 15, 0.05)
    # need 1/(k+1) <= 0.05/15 -> k >= 299
    assert "299" in warning
    assert estimate.floor_warning(299, 15, 0.05) == ""


def test_the_boundary_is_not_a_warning():
    # exactly at the boundary: 1/(k+1) == alpha/m
    assert estimate.floor_warning(19, 1, 0.05) == ""
    assert estimate.floor_warning(18, 1, 0.05) != ""


def test_more_blocks_demand_more_permutations():
    assert estimate.floor_warning(299, 15, 0.05) == ""
    assert estimate.floor_warning(299, 30, 0.05) != ""


def test_no_tests_means_nothing_to_warn_about():
    assert estimate.floor_warning(100, 0, 0.05) == ""


# --- adjustment ------------------------------------------------------------


def _table(**columns):
    n = len(next(iter(columns.values())))
    base = {
        "block": [f"blk{i}" for i in range(n)],
        "n": [100] * n,
        "n_present": [50] * n,
    }
    return pd.DataFrame({**base, **columns})


def test_adjust_adds_a_q_column_per_p_column():
    table = estimate.adjust(_table(
        d_p_random=[0.001, 0.02, 0.3],
        lambda_p=[0.04, 0.5, 0.9],
    ))
    assert "d_q_random" in table.columns
    assert "lambda_q" in table.columns
    assert (table["d_q_random"] >= table["d_p_random"]).all()


def test_each_p_column_is_its_own_family():
    """Two readings of one statistic are not two findings: adjusting d_p_random
    and d_p_brownian together would double m and weaken both."""
    table = estimate.adjust(_table(
        d_p_random=[0.01, 0.02, 0.03],
        d_p_brownian=[0.01, 0.02, 0.03],
    ))
    # m = 3 per family: the largest q is 0.03 * 3/3. Pooling the two columns
    # would make m = 6 and push it to 0.06.
    for column in ("d_q_random", "d_q_brownian"):
        assert table[column].max() == pytest.approx(0.03)
        assert table[column].max() != pytest.approx(0.06)


def test_a_missing_p_is_excluded_rather_than_counted_as_one():
    """An estimator that declined contributes no test. Filling its p with 1
    would inflate m and weaken every other block's q."""
    with_gap = estimate.adjust(_table(d_p_random=[0.01, np.nan, 0.01]))
    without = estimate.adjust(_table(d_p_random=[0.01, 0.01]))
    assert np.isnan(with_gap.loc[1, "d_q_random"])
    assert with_gap["d_q_random"].max() == pytest.approx(
        without["d_q_random"].max()
    )


def test_adjust_leaves_absent_columns_alone():
    table = estimate.adjust(_table(d_p_random=[0.01, 0.02]))
    assert "glm_q" not in table.columns


def test_adjust_does_not_mutate_its_input():
    original = _table(d_p_random=[0.01, 0.02])
    estimate.adjust(original)
    assert "d_q_random" not in original.columns


# --- the report ------------------------------------------------------------


def test_the_report_carries_the_floor_warning_when_it_applies():
    table = estimate.adjust(_table(d_p_random=[0.01] * 15))
    text = estimate.format_phylo_report(table, permutations=100)
    assert "!!" in text
    assert "permutations" in text


def test_the_report_says_so_when_nothing_was_tested():
    text = estimate.format_phylo_report(pd.DataFrame(columns=["block"]))
    assert "No block passed the trait screen" in text


def test_the_report_prints_dashes_not_nan_for_a_declined_estimator():
    table = _table(d_estimate=[np.nan], d_p_random=[np.nan],
                   d_p_brownian=[np.nan], lambda_p=[np.nan],
                   pagel_lambda=[np.nan])
    text = estimate.format_phylo_report(table)
    assert "nan" not in text
    assert "--" in text


def test_the_report_actually_prints_the_lambda_it_was_given():
    """The first version read every column off itertuples, which renames any
    column that is not a valid identifier. `lambda` is a Python keyword, so the
    whole lambda column printed as a dash in every row and nothing errored."""
    text = estimate.format_phylo_report(_table(pagel_lambda=[0.734]))
    assert "0.734" in text


def test_a_column_still_called_lambda_is_read_too():
    """R could be changed back, or an older table re-read. Falling back on the
    bare name costs one argument and removes a silent blanking."""
    table = _table(d_estimate=[0.4])
    table["lambda"] = [0.521]
    assert "0.521" in estimate.format_phylo_report(table)


def test_the_report_relays_estimator_notes():
    table = _table(d_estimate=[0.4], note=["phyloglm failed: singular"])
    text = estimate.format_phylo_report(table)
    assert "estimator notes" in text
    assert "singular" in text


def test_the_report_states_what_the_clade_term_would_have_meant():
    """The omission of the group predictor is a decision, and a reader of the
    table has to be told it was made and why."""
    text = estimate.format_phylo_report(_table(d_estimate=[0.4]))
    assert "paired within-taxon contrast" in text
    assert "two independent events" in text


def test_the_report_warns_that_branch_lengths_are_not_time():
    text = estimate.format_phylo_report(_table(d_estimate=[0.4]))
    assert "substitutions per site" in text


def test_the_report_includes_the_tree_diagnostics_it_is_given():
    text = estimate.format_phylo_report(
        _table(d_estimate=[0.4]),
        support="3 of 4 scored", rooting="rooted on 58 tips",
        conflicts="11 groups differ", clades=["FXR1: monophyletic"],
    )
    for fragment in ("3 of 4 scored", "rooted on 58 tips", "11 groups differ",
                     "FXR1: monophyletic"):
        assert fragment in text


# --- the R bridge ----------------------------------------------------------


def test_run_estimators_says_which_binary_is_missing(tmp_path):
    traits = pd.DataFrame({"blk": [1, 0]}, index=pd.Index(["a", "b"], name="id"))
    with pytest.raises(FileNotFoundError, match="not on PATH"):
        estimate.run_estimators(
            tmp_path / "t.nwk", traits, traits, ["a"],
            r_script=tmp_path / "x.R", workdir=tmp_path / "w",
            rscript="definitely-not-a-real-rscript",
        )
