"""Figures.

``itol`` writes the annotation files for a tree iTOL renders; ``groups`` draws
the group-wise APR comparisons: occupancy along the reference,
block occupancy as a group x block heatmap, and paired within-taxon load.

Two rules hold across every figure here. Axes are labelled from the metadata
table's ``label`` column, so a reader sees ``Homo sapiens FXR1`` rather than
``9606.ENSP00000350170``; see :mod:`amyloevo.io.annotate` for how that label is
built. And figure size never scales with the number of sequences -- a
per-sequence figure over an orthologue sweep is how a plotting routine produces
an image no viewer will open. Per-sequence views belong in a paginated table.

matplotlib is imported inside the functions, so importing this package does not
require it.
"""

from . import itol
from .groups import (
    GROUP_COLOURS,
    colour_for,
    plot_block_occupancy,
    plot_group_load,
    plot_occupancy_profiles,
)

__all__ = [
    "GROUP_COLOURS",
    "itol",
    "colour_for",
    "plot_block_occupancy",
    "plot_group_load",
    "plot_occupancy_profiles",
]
