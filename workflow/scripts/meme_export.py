"""APR segments in MEME minimal motif format, for motif discovery.

NOT YET IMPLEMENTED. This file exists so the DAG is complete and
`snakemake --dryrun` reports the whole graph rather than a missing input -- and so
that what remains is written down where the work happens, not only in a document.

What it must do:
    One motif block per contiguous APR run, its PWM from the aligned residues at those columns. Write the occupancy per column in the comment line so a downstream MEME run is not silently fitting gaps.
"""

from __future__ import annotations

snakemake = globals()["snakemake"]

raise NotImplementedError(
    "meme_export.py is a stub. See the module docstring for what this step must produce. "
    "The logic belongs in src/amyloevo/, imported here -- a Snakemake script is "
    "not unit-testable, and this is a step where being quietly wrong is possible."
)
