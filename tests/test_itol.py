"""iTOL annotation files: the formats, and the silent failures they invite.

iTOL does not validate uploads in any way a caller can see. A row whose label
is not a leaf is dropped, a malformed colour becomes black, and an unexpected
binary value becomes "absent". Every test here is about one of those three
turning a broken dataset into a plausible-looking figure.
"""

from __future__ import annotations

import pandas as pd
import pytest

from amyloevo.viz import itol


def _presence():
    return pd.DataFrame(
        {"blk1": [1, 0, -1], "blk2": [0, 1, 1]},
        index=["a", "b", "c"],
    )


def _fields(text, key):
    for line in text.splitlines():
        if line.startswith(key + "\t"):
            return line.split("\t")[1:]
    return None


# --- binary ----------------------------------------------------------------


def test_binary_dataset_has_the_header_itol_requires():
    text = itol.binary_dataset(_presence())
    lines = text.splitlines()
    assert lines[0] == "DATASET_BINARY"
    assert "SEPARATOR TAB" in lines
    assert "DATA" in lines
    assert _fields(text, "FIELD_LABELS") == ["blk1", "blk2"]


@pytest.mark.parametrize("text", [
    itol.binary_dataset(pd.DataFrame({"b": [1]}, index=["a"])),
    itol.tree_colors({"a": "FXR1"}),
    itol.labels_dataset({"a": "Homo sapiens"}),
    itol.gradient_dataset({"a": 0.5}),
    itol.simplebar_dataset({"a": 0.5}),
    itol.collapse_clades([("a", "b")]),
])
def test_the_separator_line_is_space_delimited_in_every_dataset(text):
    """The one line not written with the separator it declares. iTOL reads it
    before it knows what the separator is, so a tab there makes the whole
    upload fail with "separator needed" and nothing else in the file is ever
    reached. Every writer must get this right, hence the sweep."""
    line = next(l for l in text.splitlines() if l.startswith("SEPARATOR"))
    assert line == "SEPARATOR TAB"
    assert "\t" not in line


def test_binary_rows_follow_the_data_line_and_carry_every_column():
    text = itol.binary_dataset(_presence())
    body = text.splitlines()[text.splitlines().index("DATA") + 1:]
    assert len(body) == 3
    assert body[0].split("\t") == ["a", "1", "0"]
    assert body[2].split("\t") == ["c", "-1", "1"]


def test_minus_one_survives_as_unknown():
    """0 is a claim that the sequence lacks the block; -1 says nothing is
    known. 75 of the FXR1 tips have no block profile at all."""
    text = itol.binary_dataset(_presence())
    assert "\t-1\t" in text


def test_an_unexpected_binary_value_is_refused():
    frame = pd.DataFrame({"blk1": [2]}, index=["a"])
    with pytest.raises(ValueError, match="not -1, 0 or 1"):
        itol.binary_dataset(frame)


def test_a_missing_value_becomes_unknown_not_absent():
    frame = pd.DataFrame({"blk1": [float("nan")]}, index=["a"])
    text = itol.binary_dataset(frame)
    assert text.strip().endswith("a\t-1")


def test_field_colours_must_match_the_columns():
    with pytest.raises(ValueError, match="field colour"):
        itol.binary_dataset(_presence(), field_colours=["#D55E00"])


def test_per_field_colours_are_used_in_order():
    text = itol.binary_dataset(
        _presence(), field_colours=["#D55E00", "#0072B2"]
    )
    assert _fields(text, "FIELD_COLORS") == ["#D55E00", "#0072B2"]
    assert _fields(text, "LEGEND_COLORS") == ["#D55E00", "#0072B2"]


def test_the_legend_can_be_left_out():
    assert "LEGEND_TITLE" not in itol.binary_dataset(_presence(), legend=False)


def test_a_dataset_with_no_blocks_is_refused():
    with pytest.raises(ValueError, match="no block columns"):
        itol.binary_dataset(pd.DataFrame(index=["a"]))


# --- labels and colours ----------------------------------------------------


def test_tree_colors_emits_a_range_and_a_label_row_per_leaf():
    text = itol.tree_colors({"a": "FXR1", "b": "FMR1"})
    body = text.splitlines()[text.splitlines().index("DATA") + 1:]
    rows = [l.split("\t") for l in body]
    assert ["a", "range", "#D55E00", "FXR1"] in rows
    assert ["a", "label", "#D55E00"] in rows
    assert ["b", "range", "#0072B2", "FMR1"] in rows


def test_no_dataset_specific_palette_is_exported():
    """amyloevo is a general package, not one study's analysis.

    A module-level dict keyed on the group names of whichever dataset prompted
    the module would make it that study's package, and `palette_for` already
    serves every caller by computing the palette from the groups it is given.
    This asserts the constant stays gone, since a convenience constant is
    exactly the kind of thing that creeps back.
    """
    assert not hasattr(itol, "GROUP_PALETTE")
    assert "GROUP_PALETTE" not in itol.__all__


def test_an_unknown_group_gets_a_deterministic_colour():
    first = itol.palette_for(["FXR1", "NEW", "OTHER"])
    second = itol.palette_for(["OTHER", "NEW", "FXR1"])
    assert first["NEW"] == second["NEW"]
    assert first["NEW"] != first["OTHER"]


def test_leaves_with_no_group_are_skipped_rather_than_coloured_black():
    text = itol.tree_colors({"a": "FXR1", "b": "", "c": None})
    assert "\nb\t" not in text and "\nc\t" not in text


def test_either_colour_mode_can_be_switched_off():
    only_labels = itol.tree_colors({"a": "FXR1"}, colour_ranges=False)
    assert "range" not in only_labels
    assert "label" in only_labels


def test_labels_dataset_maps_ids_to_display_names():
    text = itol.labels_dataset({"9606.ENSP00000350170": "Homo sapiens FXR1"})
    assert text.splitlines()[0] == "LABELS"
    assert "9606.ENSP00000350170\tHomo sapiens FXR1" in text


def test_a_tab_in_a_label_is_refused():
    with pytest.raises(ValueError, match="tab or newline"):
        itol.labels_dataset({"a": "Homo\tsapiens"})


def test_a_newline_in_a_label_is_refused():
    with pytest.raises(ValueError, match="tab or newline"):
        itol.labels_dataset({"a": "Homo\nsapiens"})


def test_a_malformed_colour_is_refused():
    with pytest.raises(ValueError, match="hex colour"):
        itol.tree_colors({"a": "X"}, palette={"X": "red"})


# --- continuous ------------------------------------------------------------


def test_gradient_dataset_writes_the_values():
    text = itol.gradient_dataset({"a": 0.125, "b": 0.5})
    assert text.splitlines()[0] == "DATASET_GRADIENT"
    assert "a\t0.125" in text
    assert "COLOR_MIN\t#FFFFFF" in text


def test_a_non_finite_value_is_omitted_rather_than_written_as_zero():
    text = itol.gradient_dataset({"a": float("nan"), "b": 0.5, "c": None})
    assert "\na\t" not in text
    assert "\nc\t" not in text
    assert "b\t0.5" in text


def test_simplebar_carries_a_width():
    text = itol.simplebar_dataset({"a": 1.0}, width=150)
    assert text.splitlines()[0] == "DATASET_SIMPLEBAR"
    assert "WIDTH\t150" in text


# --- the silent-drop guard -------------------------------------------------


def test_check_tips_separates_the_three_outcomes():
    result = itol.check_tips(["a", "b", "zz"], ["a", "b", "c"])
    assert result["matched"] == ["a", "b"]
    assert result["not_on_tree"] == ["zz"]
    assert result["tips_without_data"] == ["c"]


def test_check_tips_is_the_only_warning_of_a_wrong_id_column():
    """Every row dropped: iTOL would render an empty track and say nothing."""
    result = itol.check_tips(["ENSP00000350170"], ["9606.ENSP00000350170"])
    assert result["matched"] == []
    assert len(result["not_on_tree"]) == 1


def test_collapse_takes_two_leaves_per_clade():
    text = itol.collapse_clades([("a", "b"), ("c", "d")])
    assert text.splitlines()[0] == "COLLAPSE"
    assert "a\tb" in text and "c\td" in text


def test_the_tree_palette_is_the_figure_palette():
    """Not two palettes that happen to agree: itol.palette_for delegates to
    viz.groups.colour_for, so a group cannot be one colour in a plot and
    another on the tree. Checked on names belonging to no particular study, so
    that the agreement is the mechanism and not a coincidence of one dataset.
    """
    from amyloevo.viz.groups import colour_for

    names = ["Alpha", "Beta", "Gamma", "Delta"]
    assert itol.palette_for(names) == colour_for(names)


def test_an_explicit_palette_overrides_the_shared_one():
    from amyloevo.viz.groups import colour_for

    resolved = itol.palette_for(["Alpha", "Beta"], palette={"Alpha": "#000000"})
    assert resolved["Alpha"] == "#000000"
    assert resolved["Beta"] == colour_for(["Alpha", "Beta"])["Beta"]


# --- categorical tracks ----------------------------------------------------


def test_category_styles_varies_shape_before_colour():
    """Shape survives greyscale and colour blindness; colour alone does not.
    The first six groups must differ in both."""
    styles = itol.category_styles([f"g{i}" for i in range(6)])
    assert len({s for s, _ in styles.values()}) == 6
    assert len({c for _, c in styles.values()}) == 6


def test_category_styles_is_stable_under_input_order():
    first = itol.category_styles(["Aves", "Mammalia", "Insecta"])
    second = itol.category_styles(["Insecta", "Aves", "Mammalia"])
    assert first == second


def test_category_styles_accepts_a_mapping_and_uses_its_values():
    """Iterating a mapping yields its KEYS. Passing an id->group mapping here
    would otherwise style the sequence ids and every later lookup would miss."""
    styles = itol.category_styles({"seq1": "Mammalia", "seq2": "Aves"})
    assert set(styles) == {"Mammalia", "Aves"}


def test_symbol_dataset_writes_every_field_itol_expects():
    text = itol.symbol_dataset({"a": "Mammalia"}, size=10)
    body = text.splitlines()[text.splitlines().index("DATA") + 1:]
    pid, shape, size, colour, fill, position, label = body[0].split("\t")
    assert pid == "a" and label == "Mammalia"
    assert 1 <= int(shape) <= 12
    assert size == "10" and fill == "1"
    assert int(position) < 0                 # an external column, not on the branch
    assert colour.startswith("#")
    assert "MAXIMUM_SIZE\t10" in text


def test_symbols_default_to_an_external_column():
    """At 357 leaves a symbol placed on the branch lands in the thicket near
    the tips, which is what 'not visible' means."""
    assert itol.EXTERNAL_COLUMN < 0
    text = itol.symbol_dataset({"a": "X"})
    assert text.strip().endswith(f"1\t{itol.EXTERNAL_COLUMN}\tX")


def test_a_leaf_with_no_group_gets_no_symbol():
    text = itol.symbol_dataset({"a": "Mammalia", "b": "", "c": None})
    body = text.splitlines()[text.splitlines().index("DATA") + 1:]
    assert len(body) == 1


def test_symbol_dataset_refuses_a_styles_mapping_that_misses_a_group():
    with pytest.raises(KeyError, match="no style for group"):
        itol.symbol_dataset({"a": "Mammalia", "b": "Aves"},
                            styles={"Mammalia": (2, "#D55E00")})


def test_colorstrip_carries_its_width_and_a_label_per_row():
    text = itol.colorstrip_dataset({"a": "Mammalia"}, width=30)
    assert text.splitlines()[0] == "DATASET_COLORSTRIP"
    assert "STRIP_WIDTH\t30" in text
    assert text.strip().endswith("Mammalia")


def test_both_categorical_tracks_agree_on_colour():
    groups = {"a": "Mammalia", "b": "Aves"}
    styles = itol.category_styles(groups.values())
    strip = itol.colorstrip_dataset(groups)
    for name, (_, colour) in styles.items():
        assert colour in strip


# --- clade ranges ----------------------------------------------------------


def test_tree_colors_takes_clade_ranges_as_leaf_pairs():
    groups = {"a": "FXR1", "b": "FXR1", "c": "FMR1", "d": "FXR2",
              "e": "pre-duplication"}
    text = itol.tree_colors(groups, colour_ranges=False,
                            clades=[("a|b", "FXR1")])
    body = text.splitlines()[text.splitlines().index("DATA") + 1:]
    assert body[0].split("\t") == ["a|b", "range", "#D55E00", "FXR1"]
    assert not any(r.split("\t")[1] == "range" for r in body[1:])


def test_a_colour_depends_on_the_whole_set_of_groups_passed():
    """viz.groups.colour_for assigns the palette by position in the SORTED
    group names, so a file built from a subset gets different colours from the
    figures. Sharing the function is necessary and not sufficient: the caller
    has to pass every group. Pinned here because the failure is silent -- the
    tree simply comes out in colours that do not match the plots beside it."""
    full = itol.palette_for(["FXR1", "FMR1", "FXR2", "pre-duplication"])
    subset = itol.palette_for(["FXR1"])
    assert full["FXR1"] == "#D55E00"
    assert subset["FXR1"] != full["FXR1"]


def test_clade_ranges_and_per_leaf_ranges_can_both_be_off():
    text = itol.tree_colors({"a": "FXR1"}, colour_ranges=False)
    assert "range" not in text
