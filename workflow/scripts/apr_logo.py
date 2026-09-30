"""Sequence logo over the columns of each APR, weighted by occupancy.

NOT YET IMPLEMENTED. This file exists so the DAG is complete and
`snakemake --dryrun` reports the whole graph rather than a missing input -- and so
that what remains is written down where the work happens, not only in a document.

What it must do:
    Take columns where apr_fraction exceeds the tier threshold, extract their residues from the alignment, and draw with logomaker. Annotate each logo with occupancy, because a logo over a sparsely occupied column is a picture of missing data.
"""

from __future__ import annotations

snakemake = globals()["snakemake"]

raise NotImplementedError(
    "apr_logo.py is a stub. See the module docstring for what this step must produce. "
    "The logic belongs in src/amyloevo/, imported here -- a Snakemake script is "
    "not unit-testable, and this is a step where being quietly wrong is possible."
)
