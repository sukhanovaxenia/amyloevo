"""Conservation measures, and the residual the project actually turns on."""

from __future__ import annotations

import pytest

from amyloevo.conserve.scores import (
    aa_type,
    apr_fraction,
    apr_residual,
    jensen_shannon_divergence,
    measure_column,
    shannon_entropy,
)


def test_an_invariant_column_has_zero_entropy():
    assert shannon_entropy(["L"] * 20) == 0.0


def test_two_equally_frequent_residues_give_one_bit():
    assert shannon_entropy(["L"] * 10 + ["V"] * 10) == pytest.approx(1.0)


def test_entropy_is_case_insensitive():
    assert shannon_entropy(["l", "L", "l", "L"]) == 0.0


def test_rarity_registers_in_jsd_where_entropy_cannot():
    """A column of pure tryptophan is more surprising than one of pure leucine.
    Entropy says both are equally conserved; that is what the background is for.
    """
    assert shannon_entropy(["W"] * 20) == shannon_entropy(["L"] * 20) == 0.0
    assert jensen_shannon_divergence(["W"] * 20) > jensen_shannon_divergence(["L"] * 20)


def test_jsd_is_bounded_in_unit_interval():
    for residues in (["W"] * 20, ["L"] * 20, list("ACDEFGHIKLMNPQRSTVWY")):
        assert 0.0 <= jensen_shannon_divergence(residues) <= 1.0


def test_a_column_matching_the_background_has_near_zero_jsd():
    from amyloevo.conserve.scores import BACKGROUND_FREQUENCIES

    residues = []
    for residue, frequency in BACKGROUND_FREQUENCIES.items():
        residues += [residue] * int(round(frequency * 10000))
    assert jensen_shannon_divergence(residues) < 0.01


def test_apr_fraction_is_the_share_of_orthologues_calling_it():
    assert apr_fraction([True, True, False, False]) == 0.5
    assert apr_fraction([]) == 0.0


# --- occupancy gate ---------------------------------------------------------


def test_a_sparse_column_is_undefined_not_conserved():
    """Three residues out of two hundred give entropy ~0 whatever they are, so
    without this gate every insertion enters as a maximally conserved site."""
    result = measure_column(7, ["L", "L", "L"], [True] * 3, n_sequences=200)
    assert result.defined is False
    assert result.shannon is None and result.apr_fraction is None
    assert "3/200" in result.reason


def test_a_well_occupied_column_is_measured():
    result = measure_column(7, ["L"] * 180, [True] * 180, n_sequences=200)
    assert result.defined is True
    assert result.shannon == 0.0
    assert result.apr_fraction == 1.0
    assert result.occupancy_fraction == pytest.approx(0.9)


def test_the_occupancy_threshold_is_configurable():
    sparse = ["L"] * 60
    assert measure_column(1, sparse, [True] * 60, n_sequences=200).defined is False
    assert (
        measure_column(
            1, sparse, [True] * 60, n_sequences=200, min_occupancy=0.25
        ).defined
        is True
    )


def test_zero_sequences_is_an_error_not_a_division():
    with pytest.raises(ValueError, match="n_sequences must be positive"):
        measure_column(1, [], [], n_sequences=0)


# --- the residual -----------------------------------------------------------


def _column(index, residues, calls, n=10):
    return measure_column(index, residues, calls, n_sequences=n)


def _trend_columns(start=10):
    """Columns in which sequence conservation predicts APR conservation.

    The residual is defined against a fitted trend, so a test of it has to
    establish the trend first. A balanced 2x2 of (conserved/variable) x
    (APR/no APR) does NOT: it makes the two variables uncorrelated by
    construction, the slope collapses to whatever rounding is left, and the
    ordering of the residuals becomes an artefact of the background
    frequencies rather than a property of the metric. That mistake cost this
    test one rewrite.
    """
    columns = []
    for offset in range(6):
        variability = offset                       # 0 = invariant, 5 = diverse
        residues = list("LVIMFA"[: variability + 1])
        residues = (residues * 10)[:10]
        calls = [True] * (10 - offset) + [False] * offset
        columns.append(_column(start + offset, residues, calls))
    return columns


def test_the_residual_separates_two_columns_with_equal_apr_conservation():
    """The point of the whole package. Both probes are called APR by every
    orthologue; one is sequence-invariant, the other turns over. Under a trend
    where conservation predicts the APR call, only the second is surprising --
    and the residual is what says so."""
    conserved_probe = _column(1, ["L"] * 10, [True] * 10)
    variable_probe = _column(2, list("LVIMFAVLIM"), [True] * 10)
    columns = _trend_columns() + [conserved_probe, variable_probe]

    residuals = apr_residual(columns)
    assert residuals[2] > residuals[1], (
        "a column that stays amyloidogenic while its residues turn over must "
        "carry the larger residual"
    )


def test_with_no_trend_the_residual_is_only_a_centred_apr_fraction():
    """An honest limit of the metric, pinned so it is not over-read.

    When sequence conservation carries no information about the APR call, the
    slope is ~0 and the residual reduces to apr_fraction minus its mean. It is
    then NOT a conservation-corrected quantity, and a run whose fitted slope is
    near zero should be reported as such rather than interpreted.
    """
    columns = [
        _column(1, ["L"] * 10, [True] * 10),
        _column(2, list("LVIMFAVLIM"), [True] * 10),
        _column(3, ["G"] * 10, [False] * 10),
        _column(4, list("GSTNQPGSTN"), [False] * 10),
    ]
    residuals = apr_residual(columns)
    centred = {c.column: c.apr_fraction - 0.5 for c in columns}
    for column, value in residuals.items():
        assert value == pytest.approx(centred[column], abs=0.12)


def test_a_residual_is_not_reported_when_the_fit_is_undefined():
    """Undefined is not zero: zero reads as 'fully explained by sequence', which
    is the opposite of 'not measurable'."""
    assert apr_residual([_column(1, ["L"] * 10, [True] * 10)]) == {}
    identical = [_column(i, ["L"] * 10, [True] * 10) for i in range(1, 5)]
    assert apr_residual(identical) == {}, "no variation in sequence conservation"


def test_undefined_columns_are_excluded_from_the_fit():
    columns = [
        _column(1, ["L"] * 10, [True] * 10),
        _column(2, list("LVIMFAVLIM"), [True] * 10),
        _column(3, list("GSTNQPGSTN"), [False] * 10),
        _column(4, ["W"], [True], n=10),   # occupancy 1/10, undefined
    ]
    residuals = apr_residual(columns)
    assert 4 not in residuals
    assert set(residuals) == {1, 2, 3}


def test_residuals_of_a_fitted_set_sum_to_about_zero():
    """An OLS property, and the cheapest check that the fit is a fit."""
    columns = [
        _column(1, ["L"] * 10, [True] * 10),
        _column(2, list("LVIMFAVLIM"), [True] * 10),
        _column(3, ["G"] * 10, [False] * 10),
        _column(4, list("GSTNQPGSTN"), [False] * 10),
        _column(5, list("LLLLLVVVVV"), [True] * 5 + [False] * 5),
    ]
    assert sum(apr_residual(columns).values()) == pytest.approx(0.0, abs=1e-9)


# --- aa_type stays its own axis --------------------------------------------


@pytest.mark.parametrize(
    "residue,expected",
    [("L", "hydrophobic"), ("F", "aromatic"), ("S", "polar"),
     ("K", "positive"), ("D", "negative"), ("P", "special"), ("X", "unknown")],
)
def test_amino_acid_classes(residue, expected):
    assert aa_type(residue) == expected
