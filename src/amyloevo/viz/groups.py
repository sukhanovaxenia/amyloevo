"""Figures for group-wise APR conservation.

Three figures, each answering one question a paralogue family raises:

``plot_occupancy_profiles``  Where along the reference does each group carry
    APRs? Faceted one row per group, because identity by facet survives
    greyscale printing and colour-vision deficiency in a way a four-colour
    overlay does not, and because overlaid occupancy curves for four groups are
    unreadable wherever they agree -- which is the part that matters.

``plot_block_occupancy``  Which blocks does each group carry? A small heatmap
    of group x block, annotated, which is the compact form of the question
    "what is shared and what is private".

``plot_group_load``  How much APR does each group carry, paired within taxon?
    A slopegraph rather than four box plots: the comparison is WITHIN a genome,
    and a box plot throws away the pairing that makes it a controlled contrast.

Every figure labels by the metadata table's ``label`` column, so an axis reads
``Homo sapiens FXR1`` and not ``9606.ENSP00000350170``. See
``amyloevo.io.annotate`` for how that label is built and disambiguated.

Figure height never scales with the number of sequences. A per-sequence figure
over an orthologue sweep is how a plotting routine produces a 378,000-pixel
image that no viewer will open; these summarise by group, and the per-sequence
view is a paginated table rather than a picture.
"""

from __future__ import annotations

import numpy as np

#: Okabe-Ito derived, checked for CVD separation. Assigned in fixed order so a
#: group keeps its colour when another is filtered out.
GROUP_COLOURS = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#E69F00", "#56B4E9", "#8C6D31", "#555555",
)
INK, MUTED, GRID = "#1a1a1a", "#595959", "#d8d8d8"


def _style():
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.size": 8,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
    })


def colour_for(groups) -> dict[str, str]:
    """Fixed colour per group, stable under filtering."""
    return {g: GROUP_COLOURS[i % len(GROUP_COLOURS)] for i, g in enumerate(sorted(groups))}


def plot_occupancy_profiles(
    occupancy: dict[str, np.ndarray],
    *,
    domains=None,
    reference_label: str = "reference",
    counts: dict[str, int] | None = None,
    title: str | None = None,
):
    """Per-group APR occupancy along the reference, one facet per group.

    ``occupancy`` maps group name -> array of per-residue occupancy in
    reference coordinates. ``domains`` is an optional sequence of
    ``(name, start, stop, is_primary)`` drawn as a track beneath.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    _style()
    groups = list(occupancy)
    colours = colour_for(groups)
    n = len(groups)
    height_ratios = [1] * n + ([0.5] if domains else [])
    fig, axes = plt.subplots(
        n + (1 if domains else 0), 1, figsize=(7.2, 1.3 * n + (0.8 if domains else 0)),
        sharex=True, gridspec_kw={"height_ratios": height_ratios, "hspace": 0.28},
    )
    axes = np.atleast_1d(axes)
    length = max(len(v) for v in occupancy.values())
    x = np.arange(1, length + 1)

    for ax, group in zip(axes[:n], groups):
        y = np.asarray(occupancy[group], dtype=float)
        ax.fill_between(x[: len(y)], 0, y, color=colours[group], lw=0, step="mid")
        if domains:
            for _, start, stop, _primary in domains:
                ax.axvspan(start, stop, color=GRID, alpha=0.35, lw=0, zorder=0)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_xlim(1, length)
        ax.grid(axis="y", color=GRID, lw=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        suffix = f"  (n={counts[group]})" if counts and group in counts else ""
        ax.text(0.995, 0.86, f"{group}{suffix}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color=INK, fontweight="bold")
        ax.set_ylabel("occupancy", fontsize=7)

    if domains:
        dax = axes[-1]
        dax.set_ylim(0, 1)
        dax.set_xlim(1, length)
        for name, start, stop, primary in domains:
            dax.add_patch(Rectangle(
                (start, 0.30), stop - start, 0.62,
                facecolor="#9ecae1" if primary else "#e0e0e0",
                edgecolor=MUTED, lw=0.4))
            dax.text((start + stop) / 2, 0.61, name, ha="center", va="center",
                     fontsize=5.5, color="#17405c" if primary else "#3d3d3d",
                     rotation=90 if (stop - start) < length / 10 else 0)
        dax.set_yticks([])
        for spine in ("left", "top", "right"):
            dax.spines[spine].set_visible(False)
        dax.set_xlabel(f"{reference_label} residue", fontsize=8)
    else:
        axes[-1].set_xlabel(f"{reference_label} residue", fontsize=8)

    if title:
        fig.suptitle(title, fontsize=10, y=0.98)
    return fig


def plot_block_occupancy(block_table, *, groups=None, block_labels=None, title=None):
    """Group x block occupancy heatmap, annotated.

    ``block_table`` is a DataFrame with one row per block and one
    ``occ_<group>`` column per group.
    """
    import matplotlib.pyplot as plt

    _style()
    groups = groups or [c[4:] for c in block_table.columns if c.startswith("occ_")]
    matrix = block_table[[f"occ_{g}" for g in groups]].to_numpy(dtype=float).T
    labels = block_labels or [str(i + 1) for i in range(matrix.shape[1])]

    fig, ax = plt.subplots(figsize=(0.42 * len(labels) + 2.2, 0.4 * len(groups) + 1.6))
    im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups, fontsize=7)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=6)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if not np.isfinite(value):
                continue
            ax.text(j, i, f"{value:.2f}".lstrip("0"), ha="center", va="center",
                    fontsize=5, color="white" if value > 0.55 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("occupancy", fontsize=6)
    cb.ax.tick_params(labelsize=6)
    if title:
        ax.set_title(title, fontsize=9, loc="left")
    return fig


def plot_group_load(paired, *, order=None, ylabel="APR residues / length", title=None):
    """Paired within-taxon load across groups, as a slopegraph.

    ``paired`` is a DataFrame indexed by taxon with one column per group and no
    missing values -- taxa carrying every group. The pairing is the point: it
    removes the between-species component, which is the one phylogeny lives in,
    so the contrast is within a genome and repeated once per taxon.
    """
    import matplotlib.pyplot as plt

    _style()
    order = list(order or paired.columns)
    colours = colour_for(order)
    fig, ax = plt.subplots(figsize=(1.1 * len(order) + 1.8, 3.4))

    for _, row in paired.iterrows():
        ax.plot(range(len(order)), [row[g] for g in order],
                color=MUTED, lw=0.4, alpha=0.3, zorder=1)
    top = float(np.nanmax(paired[order].to_numpy()))
    for i, group in enumerate(order):
        ax.scatter([i] * len(paired), paired[group], s=12, color=colours[group],
                   edgecolor="white", lw=0.3, zorder=3)
        ax.text(i, top * 1.08, group, ha="center", fontsize=8,
                color=INK, fontweight="bold")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([])
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.set_ylim(0, top * 1.18)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_title(title or f"paired within taxon (n={len(paired)})",
                 fontsize=8, loc="left")
    return fig
