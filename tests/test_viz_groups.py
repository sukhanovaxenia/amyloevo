"""Group figures: that they draw, and that they stay a fixed size.

A plotting routine is mostly checked by eye, so these tests cover the two things
an eye does not catch: that the figure does not grow with the number of
sequences, and that axes are labelled by the metadata ``label`` column rather
than by accession.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from amyloevo.viz.groups import (  # noqa: E402
    GROUP_COLOURS,
    colour_for,
    plot_block_occupancy,
    plot_group_load,
    plot_occupancy_profiles,
)

GROUPS = ("FXR1", "FMR1", "FXR2", "outgroup")


@pytest.fixture
def occupancy() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {g: rng.random(620) for g in GROUPS}


@pytest.fixture(autouse=True)
def _close_figures():
    import matplotlib.pyplot as plt

    yield
    plt.close("all")


# --------------------------------------------------------------------------- #
# Colours
# --------------------------------------------------------------------------- #
def test_a_group_keeps_its_colour_when_another_is_filtered_out() -> None:
    """Otherwise two figures in one paper disagree about which line is FXR1."""
    full = colour_for(GROUPS)
    assert colour_for(("FMR1", "FXR1"))["FXR1"] == full["FXR1"]


def test_more_groups_than_colours_still_assigns_one() -> None:
    many = [f"g{i}" for i in range(len(GROUP_COLOURS) + 3)]
    assert len(colour_for(many)) == len(many)


# --------------------------------------------------------------------------- #
# Occupancy profiles
# --------------------------------------------------------------------------- #
def test_one_facet_per_group(occupancy) -> None:
    fig = plot_occupancy_profiles(occupancy, reference_label="human FXR1")
    assert len(fig.axes) == len(GROUPS)


def test_a_domain_track_adds_one_axis(occupancy) -> None:
    domains = [("Agenet", 1, 110, True), ("KH_1", 221, 277, True)]
    fig = plot_occupancy_profiles(occupancy, domains=domains)
    assert len(fig.axes) == len(GROUPS) + 1


def test_figure_height_does_not_follow_the_number_of_sequences(occupancy) -> None:
    """The failure mode this module was written around: a 378,000-pixel image.

    Height scales with GROUPS, which is small and bounded; the per-sequence view
    belongs in a paginated table.
    """
    counts = {g: 100_000 for g in GROUPS}
    small = plot_occupancy_profiles(occupancy)
    large = plot_occupancy_profiles(occupancy, counts=counts)
    assert large.get_size_inches()[1] == small.get_size_inches()[1]
    assert large.get_size_inches()[1] < 12


def test_profiles_of_unequal_length_do_not_crash() -> None:
    """A group whose reference projection is shorter than another's."""
    fig = plot_occupancy_profiles({"a": np.zeros(50), "b": np.zeros(30)})
    assert fig.axes[0].get_xlim()[1] == 50


def test_a_single_group_still_draws() -> None:
    assert len(plot_occupancy_profiles({"only": np.zeros(20)}).axes) == 1


# --------------------------------------------------------------------------- #
# Block heatmap
# --------------------------------------------------------------------------- #
def test_block_heatmap_is_group_by_block() -> None:
    table = pd.DataFrame({
        "block": [f"B{i}" for i in range(10)],
        "occ_FXR1": np.linspace(0, 1, 10),
        "occ_FMR1": np.linspace(1, 0, 10),
    })
    fig = plot_block_occupancy(table, block_labels=list(table["block"]))
    image = fig.axes[0].get_images()[0]
    assert image.get_array().shape == (2, 10)
    assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == ["FXR1", "FMR1"]


def test_block_labels_default_to_ordinals() -> None:
    table = pd.DataFrame({"occ_A": [0.5, 0.5], "occ_B": [0.1, 0.9]})
    fig = plot_block_occupancy(table)
    assert [t.get_text() for t in fig.axes[0].get_xticklabels()] == ["1", "2"]


def test_a_missing_occupancy_is_left_blank_not_drawn_as_zero() -> None:
    """An unscored block and an empty block are different claims."""
    table = pd.DataFrame({"occ_A": [np.nan, 0.4], "occ_B": [0.2, 0.6]})
    fig = plot_block_occupancy(table)
    annotations = [t.get_text() for t in fig.axes[0].texts]
    assert ".20" in annotations
    assert not any("nan" in a for a in annotations)


# --------------------------------------------------------------------------- #
# Paired load
# --------------------------------------------------------------------------- #
def test_group_load_states_the_number_of_pairs() -> None:
    rng = np.random.default_rng(1)
    paired = pd.DataFrame(
        rng.random((55, 3)) * 0.1,
        columns=["FXR1", "FMR1", "FXR2"],
        index=[f"taxon{i}" for i in range(55)],
    )
    fig = plot_group_load(paired)
    assert "n=55" in fig.axes[0].get_title(loc="left")


def test_group_load_respects_a_requested_order() -> None:
    paired = pd.DataFrame({"a": [0.1, 0.2], "b": [0.3, 0.4], "c": [0.5, 0.6]})
    fig = plot_group_load(paired, order=["c", "a"])
    labels = [t.get_text() for t in fig.axes[0].texts]
    assert labels == ["c", "a"]
