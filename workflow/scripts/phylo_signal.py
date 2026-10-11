"""Phylogenetic signal in APR presence, and its dependency on divergence and clade.

NOT YET IMPLEMENTED. This file exists so the DAG is complete and
`snakemake --dryrun` reports the whole graph rather than a missing input -- and so
that what remains is written down where the work happens, not only in a document.

What it must do:
    Fritz & Purvis D or Pagel lambda per APR column, then phyloglm for the dependency. Needs the tree and a per-species binary trait. R via rpy2 or a subprocess to Rscript -- decide once and keep it in the package, not here. Do NOT substitute a correlation: species share ancestry, so that is pseudoreplication.

    What exists instead, and does NOT discharge this: `rule taxonomic_signal`
    (scripts/taxonomic_signal.py) runs a PERMANOVA per taxonomic rank on the APR
    block profiles, inside each paralogue group, BH-adjusted across ranks. That is a
    distribution-free SCREEN: a rank effect there means profiles cluster by clade,
    which nested sampling alone produces, so it answers whether building the tree is
    worth the compute and never whether the clade explains the trait. If the screen
    is read as this step, the project has silently swapped the question it exists to
    ask for a weaker one it can already answer.
"""

from __future__ import annotations

snakemake = globals()["snakemake"]

raise NotImplementedError(
    "phylo_signal.py is a stub. See the module docstring for what this step must produce. "
    "The logic belongs in src/amyloevo/, imported here -- a Snakemake script is "
    "not unit-testable, and this is a step where being quietly wrong is possible."
)
