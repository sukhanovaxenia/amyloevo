"""iTOL annotation files for a tree this workflow already built.

NOT YET IMPLEMENTED. This file exists so the DAG is complete and
`snakemake --dryrun` reports the whole graph rather than a missing input -- and so
that what remains is written down where the work happens, not only in a document.

What it must do:
    Two datasets: APR presence per species (binary, DATASET_BINARY) and clade colours from the taxid (TREE_COLORS). Leaf labels must match the tree's, which came from the alignment's headers -- reuse amyloevo.io.headers rather than re-parsing here.
"""

from __future__ import annotations

snakemake = globals()["snakemake"]

raise NotImplementedError(
    "itol_annotations.py is a stub. See the module docstring for what this step must produce. "
    "The logic belongs in src/amyloevo/, imported here -- a Snakemake script is "
    "not unit-testable, and this is a step where being quietly wrong is possible."
)
