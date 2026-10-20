"""iTOL annotation files for a tree this workflow built.

Each function returns the complete text of one iTOL dataset. Nothing here draws
anything: iTOL does, from these files, which is the point -- a 357-tip tree is
not a matplotlib figure anyone wants to read.

**The one failure mode that matters.** iTOL matches dataset rows to the tree by
leaf label, and a row whose label is not a leaf is *silently ignored*. Upload a
dataset built from the wrong id column and iTOL shows an empty track with no
error, which looks like "no APRs in this family" rather than "no rows matched".
So every writer here takes the ids it is given, :func:`check_tips` compares them
against the tree's actual tips, and the workflow script prints the comparison
instead of assuming it.

**Absent is not the same as unmeasured.** In ``DATASET_BINARY`` 0 means the
sequence does not carry the block and -1 means nothing is known about it. A
sequence that failed the completeness filter has no block profile at all, and
writing 0 for it would put 75 invented absences on the FXR1 tree.

**The group colours are shared with the figures.** :func:`palette_for`
delegates to ``viz.groups.colour_for``, the single place a group's colour is
decided, so a group is the same colour on the tree as in every plot beside it.
That is the Okabe-Ito set, which survives the common forms of colour blindness
and greyscale printing. No palette is held here as a constant: a module-level
dict keyed on one dataset's group names would both tie this package to that
dataset and drift from ``colour_for`` the first time either changed.
"""

from __future__ import annotations

from .groups import GROUP_COLOURS, colour_for

#: iTOL shape codes. 2 is a filled circle, which reads best at 300+ leaves.
SHAPE_SQUARE, SHAPE_CIRCLE, SHAPE_STAR = 1, 2, 3
SHAPE_TRIANGLE_RIGHT, SHAPE_TRIANGLE_LEFT, SHAPE_CHECK = 4, 5, 6

#: The shapes worth cycling for a categorical variable. Codes 7-12 exist but
#: are lines and arrows, which do not read as category markers at leaf size.
#: Six shapes crossed with the palette gives far more distinguishable
#: combinations than either alone -- and shape survives greyscale printing and
#: the common forms of colour blindness, which colour does not.
CATEGORY_SHAPES = (
    SHAPE_CIRCLE, SHAPE_SQUARE, SHAPE_STAR, SHAPE_TRIANGLE_RIGHT,
    SHAPE_TRIANGLE_LEFT, SHAPE_CHECK,
)

#: Where a symbol is drawn. A negative integer is an external column -- -1 is
#: the first one, outside the tree -- and a value in [0, 1] places it along the
#: branch. External is the default here: at 357 leaves a symbol on the branch
#: lands in the thicket near the tips and is exactly what "not visible" means.
EXTERNAL_COLUMN = -1

SEPARATOR = "\t"

#: The separator declaration, and the one line in the file that is NOT written
#: with the separator it declares. iTOL has to read this line before it knows
#: what the separator is, so the keyword and its value are always divided by a
#: single space -- which is also why ``SEPARATOR SPACE`` is a legal thing to
#: write. Emitting it as ``SEPARATOR<tab>TAB`` makes iTOL fail to recognise the
#: declaration at all and reject the upload with "separator needed", and every
#: other line in the file is then irrelevant because it never gets that far.
SEPARATOR_DECL = "SEPARATOR TAB"


def _check_label(value: str) -> str:
    """Reject a label that would break the file format."""
    text = str(value)
    if "\t" in text or "\n" in text or "\r" in text:
        raise ValueError(
            f"label {text!r} contains a tab or newline; iTOL datasets are "
            f"tab-separated and the row would be parsed as the wrong number "
            f"of fields"
        )
    return text


def _check_colour(value: str) -> str:
    text = str(value)
    if not (text.startswith("#") and len(text) in (4, 7)):
        raise ValueError(
            f"colour {text!r} is not a hex colour such as '#D55E00'; iTOL "
            f"ignores a malformed colour and falls back to black without "
            f"saying so"
        )
    int(text[1:], 16)
    return text


def palette_for(groups, palette=None) -> dict[str, str]:
    """A colour per group name, from the same function the figures use.

    Deterministic, and deliberately not independent: ``viz.groups.colour_for``
    is the single place a group's colour is decided, so the tree and every plot
    beside it agree by construction. ``palette`` overrides it entirely when a
    caller has its own scheme; names missing from an override fall back to
    ``colour_for``.

    **Pass every group, not the ones you are drawing.** ``colour_for`` assigns
    the palette by position in the sorted names, so a call with a subset shifts
    every colour: a group that is third of four takes the first colour when
    called on its own. Sharing the function is necessary and not sufficient,
    and the mismatch is silent -- the tree simply comes out in colours that do
    not match the plots beside it.
    """
    names = [str(g) for g in dict.fromkeys(groups) if g]
    resolved = colour_for(names)
    if palette:
        resolved.update({str(k): v for k, v in palette.items()})
    return resolved


def check_tips(ids, tips) -> dict[str, list[str]]:
    """Compare dataset ids against the tree's tips.

    Returns ``{"matched": [...], "not_on_tree": [...], "tips_without_data":
    [...]}``. The middle one is the dangerous list: iTOL drops those rows in
    silence.
    """
    ids = [str(i) for i in ids]
    tip_set = {str(t) for t in tips}
    id_set = set(ids)
    return {
        "matched": [i for i in ids if i in tip_set],
        "not_on_tree": [i for i in ids if i not in tip_set],
        "tips_without_data": sorted(t for t in tip_set if t not in id_set),
    }


def _header(kind: str, label: str, colour: str) -> list[str]:
    return [
        kind,
        SEPARATOR_DECL,
        f"DATASET_LABEL{SEPARATOR}{_check_label(label)}",
        f"COLOR{SEPARATOR}{_check_colour(colour)}",
    ]


def binary_dataset(
    presence,
    *,
    label: str = "APR blocks",
    colour: str = "#D55E00",
    field_colours=None,
    shape: int = SHAPE_CIRCLE,
    legend: bool = True,
) -> str:
    """``DATASET_BINARY``: one shape column per block, per leaf.

    ``presence`` is a DataFrame indexed by sequence id with one column per
    block, holding 1 (present), 0 (absent) or -1 (unknown). Any other value is
    refused: iTOL reads an unexpected value as 0, turning "unmeasured" into
    "absent" on the figure.
    """
    columns = [_check_label(c) for c in presence.columns]
    if not columns:
        raise ValueError("no block columns to draw")
    colours = (
        [colour] * len(columns) if field_colours is None
        else [_check_colour(c) for c in field_colours]
    )
    if len(colours) != len(columns):
        raise ValueError(
            f"{len(colours)} field colour(s) for {len(columns)} column(s); "
            f"iTOL pairs them positionally and a short list shifts every "
            f"colour after the gap"
        )

    lines = _header("DATASET_BINARY", label, colour)
    lines += [
        "FIELD_SHAPES" + SEPARATOR + SEPARATOR.join([str(int(shape))] * len(columns)),
        "FIELD_LABELS" + SEPARATOR + SEPARATOR.join(columns),
        "FIELD_COLORS" + SEPARATOR + SEPARATOR.join(colours),
    ]
    if legend:
        lines += [
            f"LEGEND_TITLE{SEPARATOR}{_check_label(label)}",
            "LEGEND_SHAPES" + SEPARATOR + SEPARATOR.join([str(int(shape))] * len(columns)),
            "LEGEND_COLORS" + SEPARATOR + SEPARATOR.join(colours),
            "LEGEND_LABELS" + SEPARATOR + SEPARATOR.join(columns),
        ]
    lines.append("DATA")

    allowed = {-1, 0, 1}
    for pid, row in presence.iterrows():
        values = []
        for value in row:
            try:
                integer = int(value)
            except (TypeError, ValueError):
                integer = -1
            if integer not in allowed:
                raise ValueError(
                    f"{pid}: binary value {value!r} is not -1, 0 or 1. iTOL "
                    f"reads anything unexpected as absent, which would show an "
                    f"unmeasured sequence as one that lacks the block."
                )
            values.append(str(integer))
        lines.append(_check_label(pid) + SEPARATOR + SEPARATOR.join(values))
    return "\n".join(lines) + "\n"


def tree_colors(
    groups,
    palette=None,
    *,
    colour_labels: bool = True,
    colour_ranges: bool = True,
    clades=None,
) -> str:
    """``TREE_COLORS``: a background and label colour per leaf, by group.

    ``groups`` maps sequence id to group name. ``label`` rows colour the leaf
    text; ``range`` rows give a coloured background.

    ``clades`` is the better way to draw those backgrounds when the groups are
    monophyletic: an iterable of ``(node, group)`` where ``node`` is a clade
    specification from ``phylo.tree.mrca_pair`` (``"LEAF1|LEAF2"``). One range
    per clade paints a single block, where 357 per-leaf ranges paint 357 thin
    stripes that read as noise and hide where a clade begins and ends. Pass
    ``colour_ranges=False`` alongside it so the two do not fight.
    """
    colours = palette_for(groups.values(), palette)
    lines = ["TREE_COLORS", SEPARATOR_DECL, "DATA"]
    for node, group in (clades or ()):
        lines.append(SEPARATOR.join([
            _check_label(node), "range",
            _check_colour(colours[str(group)]), _check_label(group),
        ]))
    for pid, group in groups.items():
        if not group:
            continue
        colour = _check_colour(colours[str(group)])
        name = _check_label(pid)
        if colour_ranges:
            lines.append(
                SEPARATOR.join([name, "range", colour, _check_label(group)])
            )
        if colour_labels:
            lines.append(SEPARATOR.join([name, "label", colour]))
    return "\n".join(lines) + "\n"


def category_styles(names, palette=None, shapes=CATEGORY_SHAPES):
    """A distinct ``(shape, colour)`` per group NAME, deterministic by name.

    Takes the names, not an id-to-name mapping: iterating a mapping yields its
    keys, so passing one here silently styles the sequence ids instead of the
    groups and every lookup afterwards misses. A mapping is accepted and its
    values used, rather than being left to fail somewhere else.

    Shape varies fastest and colour next, so the first six groups differ in
    shape *and* colour and only then do shapes repeat under a new colour. Sorted
    by name, so the same set of groups always gets the same styles and two runs
    produce comparable figures.
    """
    from collections.abc import Mapping

    if isinstance(names, Mapping):
        names = names.values()
    names = sorted({str(g) for g in names if g})
    colours = palette_for(names, palette)
    out = {}
    for i, name in enumerate(names):
        out[name] = (shapes[i % len(shapes)], colours[name])
    return out


def symbol_dataset(
    groups,
    *,
    label: str = "taxonomic group",
    palette=None,
    styles=None,
    size: int = 10,
    position: int = EXTERNAL_COLUMN,
    fill: int = 1,
    legend: bool = True,
) -> str:
    """``DATASET_SYMBOL``: one shape per leaf, by category.

    ``groups`` maps sequence id to a group name -- a taxonomic group, as
    produced by ``io.annotate.major_groups``, or anything else categorical.
    Leaves with no group are left out rather than given a default symbol: an
    unknown taxon and a taxon in the leftover bucket are different statements.

    This is the right track for a taxonomic group on a **gene** tree. A clade
    annotation is not: "Mammalia" appears once inside each paralogue clade and
    is therefore not a clade at all, so a range drawn at its MRCA would cover
    the whole tree. A per-leaf symbol carries the same information and cannot
    be wrong about the topology.
    """
    styles = styles or category_styles(groups.values(), palette)
    unstyled = sorted({str(g) for g in groups.values() if g} - set(styles))
    if unstyled:
        raise KeyError(
            f"no style for group(s) {unstyled}; a styles mapping passed in "
            f"must cover every group present, or iTOL would draw nothing for "
            f"those leaves and say nothing about it"
        )
    lines = _header("DATASET_SYMBOL", label, next(iter(styles.values()))[1]
                    if styles else "#1A1A1A")
    lines.append(f"MAXIMUM_SIZE{SEPARATOR}{int(size)}")
    if legend and styles:
        names = list(styles)
        lines += [
            f"LEGEND_TITLE{SEPARATOR}{_check_label(label)}",
            "LEGEND_SHAPES" + SEPARATOR
            + SEPARATOR.join(str(styles[n][0]) for n in names),
            "LEGEND_COLORS" + SEPARATOR
            + SEPARATOR.join(_check_colour(styles[n][1]) for n in names),
            "LEGEND_LABELS" + SEPARATOR
            + SEPARATOR.join(_check_label(n) for n in names),
        ]
    lines.append("DATA")
    for pid, group in groups.items():
        if not group:
            continue
        shape, colour = styles[str(group)]
        lines.append(SEPARATOR.join([
            _check_label(pid), str(int(shape)), str(int(size)),
            _check_colour(colour), str(int(fill)), str(int(position)),
            _check_label(group),
        ]))
    return "\n".join(lines) + "\n"


def colorstrip_dataset(
    groups,
    *,
    label: str = "taxonomic group",
    palette=None,
    width: int = 25,
    margin: int = 0,
    colour_branches: bool = False,
    legend: bool = True,
) -> str:
    """``DATASET_COLORSTRIP``: a colour band beside the leaves, by category.

    The same information as :func:`symbol_dataset` in a form that reads better
    at several hundred leaves -- a contiguous band shows where a group starts
    and stops, which separate symbols do not. Both are offered because they
    answer different questions: the band shows blocks, the symbols survive
    greyscale and sit outside the tree where they can be compared against other
    external columns.
    """
    colours = palette_for(groups.values(), palette)
    lines = _header("DATASET_COLORSTRIP", label,
                    next(iter(colours.values())) if colours else "#1A1A1A")
    lines += [
        f"STRIP_WIDTH{SEPARATOR}{int(width)}",
        f"MARGIN{SEPARATOR}{int(margin)}",
        f"COLOR_BRANCHES{SEPARATOR}{1 if colour_branches else 0}",
    ]
    if legend and colours:
        names = sorted(colours)
        lines += [
            f"LEGEND_TITLE{SEPARATOR}{_check_label(label)}",
            "LEGEND_SHAPES" + SEPARATOR + SEPARATOR.join(["1"] * len(names)),
            "LEGEND_COLORS" + SEPARATOR
            + SEPARATOR.join(_check_colour(colours[n]) for n in names),
            "LEGEND_LABELS" + SEPARATOR
            + SEPARATOR.join(_check_label(n) for n in names),
        ]
    lines.append("DATA")
    for pid, group in groups.items():
        if not group:
            continue
        lines.append(SEPARATOR.join([
            _check_label(pid), _check_colour(colours[str(group)]),
            _check_label(group),
        ]))
    return "\n".join(lines) + "\n"


def labels_dataset(names) -> str:
    """``LABELS``: replace each leaf's id with something readable.

    ``names`` maps sequence id to display label. The ids stay authoritative for
    joins -- this only changes what is drawn, which is why the mapping is built
    once in ``io.annotate`` and reused rather than re-derived here.
    """
    lines = ["LABELS", SEPARATOR_DECL, "DATA"]
    for pid, name in names.items():
        lines.append(_check_label(pid) + SEPARATOR + _check_label(name))
    return "\n".join(lines) + "\n"


def gradient_dataset(
    values,
    *,
    label: str = "APR load",
    colour: str = "#D55E00",
    colour_min: str = "#FFFFFF",
    colour_max: str | None = None,
) -> str:
    """``DATASET_GRADIENT``: one continuous value per leaf, as a heat strip.

    ``values`` maps sequence id to a number. Non-finite values are omitted
    rather than written as 0 -- see the module docstring.
    """
    import math

    lines = _header("DATASET_GRADIENT", label, colour)
    lines += [
        f"COLOR_MIN{SEPARATOR}{_check_colour(colour_min)}",
        f"COLOR_MAX{SEPARATOR}{_check_colour(colour_max or colour)}",
        "DATA",
    ]
    for pid, value in values.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(number):
            continue
        lines.append(f"{_check_label(pid)}{SEPARATOR}{number:.6g}")
    return "\n".join(lines) + "\n"


def simplebar_dataset(
    values,
    *,
    label: str = "APR load",
    colour: str = "#D55E00",
    width: int = 200,
) -> str:
    """``DATASET_SIMPLEBAR``: the same numbers as bars outside the leaves.

    Worth having alongside the gradient: a gradient shows where the extremes
    are and a bar chart shows how far apart they are, and on a tree this size
    the second question is usually the one being asked.
    """
    import math

    lines = _header("DATASET_SIMPLEBAR", label, colour)
    lines += [f"WIDTH{SEPARATOR}{int(width)}", "DATA"]
    for pid, value in values.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(number):
            continue
        lines.append(f"{_check_label(pid)}{SEPARATOR}{number:.6g}")
    return "\n".join(lines) + "\n"


def collapse_clades(nodes) -> str:
    """``COLLAPSE``: fold named clades so the tree fits on a screen.

    ``nodes`` is an iterable of leaf-label pairs; iTOL collapses the clade
    defined by each pair's MRCA. Two labels per clade rather than an internal
    node id, because IQ-TREE's internal node labels are support values and are
    not stable between runs.
    """
    lines = ["COLLAPSE", SEPARATOR_DECL, "DATA"]
    for pair in nodes:
        first, second = pair
        lines.append(_check_label(first) + SEPARATOR + _check_label(second))
    return "\n".join(lines) + "\n"


__all__ = [
    "CATEGORY_SHAPES",
    "EXTERNAL_COLUMN",
    "GROUP_COLOURS",
    "SEPARATOR_DECL",
    "SHAPE_CHECK",
    "SHAPE_CIRCLE",
    "SHAPE_SQUARE",
    "SHAPE_STAR",
    "SHAPE_TRIANGLE_LEFT",
    "SHAPE_TRIANGLE_RIGHT",
    "binary_dataset",
    "category_styles",
    "check_tips",
    "collapse_clades",
    "colorstrip_dataset",
    "gradient_dataset",
    "labels_dataset",
    "palette_for",
    "simplebar_dataset",
    "symbol_dataset",
    "tree_colors",
]
