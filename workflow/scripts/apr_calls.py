"""Collapse each orthologue's merged predictor table into one APR call per residue.

NOT YET IMPLEMENTED. This file exists so the DAG is complete and
`snakemake --dryrun` reports the whole graph rather than a missing input -- and so
that what remains is written down where the work happens, not only in a document.

What it must do:
    Read the per-tool <predictor>_bin columns of every merged table under results/predict/<protein>/merged, count them fractionally, and emit sequence_id, residue_number, residue, apr_score, apr (bool). The counting rule already exists in aggressor-wrappers as fractional_consensus -- import it rather than reimplementing, so one consensus definition serves both packages.
"""

from __future__ import annotations

snakemake = globals()["snakemake"]

raise NotImplementedError(
    "apr_calls.py is a stub. See the module docstring for what this step must produce. "
    "The logic belongs in src/amyloevo/, imported here -- a Snakemake script is "
    "not unit-testable, and this is a step where being quietly wrong is possible."
)
