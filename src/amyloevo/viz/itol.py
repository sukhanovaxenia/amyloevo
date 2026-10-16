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

**The group colours are shared with the figures.** ``GROUP_PALETTE`` is the
palette ``viz.groups`` and the FXR1 project's figures already use, so a
paralogue is the same colour on the tree as in every plot beside it. It is the
Okabe-Ito set, which survives the common forms of colour blindness and
greyscale printing.
"""

from __future__ import annotations

from .groups import GROUP_COLOURS, colour_for

#: The FXR1 family's colours, for reference and for tests. DERIVED from
#: ``viz.groups.colour_for`` rather than written out again: that function
#: assigns the Okabe-Ito cycle by sorted group name, and a second hand-keyed
#: dict here would agree with it today and drift the first time either changes.
#: A paralogue being one colour in a figure and another on the tree is the kind
#: of error a reader cannot see and cannot recover from.
GROUP_PALETTE = colour_for(["FXR1", "FMR1", "FXR2", "pre-duplication"])

#: iTOL shape codes. 2 is a filled circle, which reads best at 300+ leaves.
SHAPE_SQUARE, SHAPE_CIRCLE, SHAPE_STAR = 1, 2, 3

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
) -> str:
    """``TREE_COLORS``: a background and label colour per leaf, by group.

    ``groups`` maps sequence id to group name. ``range`` rows give the leaf a
    coloured background with the group name as its legend entry; ``label`` rows
    colour the leaf text. Both are emitted by default because either alone is
    hard to read on a large tree -- the background is invisible on a narrow
    leaf, and coloured text disappears when labels are hidden.
    """
    colours = palette_for(groups.values(), palette)
    lines = ["TREE_COLORS", SEPARATOR_DECL, "DATA"]
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
    "GROUP_COLOURS",
    "GROUP_PALETTE",
    "SHAPE_CIRCLE",
    "SHAPE_SQUARE",
    "SHAPE_STAR",
    "binary_dataset",
    "check_tips",
    "collapse_clades",
    "gradient_dataset",
    "labels_dataset",
    "palette_for",
    "simplebar_dataset",
    "tree_colors",
]
