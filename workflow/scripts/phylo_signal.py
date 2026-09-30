"""Phylogenetic signal in APR presence, and its dependency on divergence and clade.

NOT YET IMPLEMENTED. This file exists so the DAG is complete and
`snakemake --dryrun` reports the whole graph rather than a missing input -- and so
that what remains is written down where the work happens, not only in a document.

What it must do:
    Fritz & Purvis D or Pagel lambda per APR column, then phyloglm for the dependency. Needs the tree and a per-species binary trait. R via rpy2 or a subprocess to Rscript -- decide once and keep it in the package, not here. Do NOT substitute a correlation: species share ancestry, so that is pseudoreplication.
"""

from __future__ import annotations

snakemake = globals()["snakemake"]

raise NotImplementedError(
    "phylo_signal.py is a stub. See the module docstring for what this step must produce. "
    "The logic belongs in src/amyloevo/, imported here -- a Snakemake script is "
    "not unit-testable, and this is a step where being quietly wrong is possible."
)
