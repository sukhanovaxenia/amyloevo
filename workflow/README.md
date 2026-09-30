# workflow

Snakemake owns the running; the package owns the logic. Every rule is thin — read
inputs, call one function from `amyloevo`, write outputs. When a rule starts
holding logic worth testing, that logic belongs in `src/amyloevo/`, where a unit
test can reach it.

```bash
# What would run, and why
snakemake -s workflow/Snakefile --configfile config/config.yaml -n

# Locally, with per-rule conda environments
snakemake -s workflow/Snakefile --configfile config/config.yaml \
    --use-conda --cores 8

# On a cluster (SLURM)
snakemake -s workflow/Snakefile --configfile config/config.yaml \
    --use-conda --profile profiles/slurm --jobs 100
```

Input is one multifasta per protein under `orthologs_dir`; the filename stem is the
protein id, so adding a protein is adding a file.

## Implemented vs stubbed

| script | state |
|---|---|
| `find_reference.py` | done — locates the reference, fails when a group has no anchor |
| `align_metadata.py` | done — aligner, version, arguments, checksum beside the alignment |
| `conserve.py` | done — per-column occupancy, entropy, JSD, APR fraction |
| `summarise.py` | done — fits the residual trend once and reports the slope with it |
| `provenance.py` | done — panel, aligner, redundancy correction, thresholds |
| `apr_calls.py` | **stub** — fractional consensus over the merged tables |
| `phylo_signal.py` | **stub** — Fritz & Purvis *D* / Pagel's λ, then `phyloglm` |
| `apr_logo.py` | **stub** — logos over APR columns, annotated with occupancy |
| `meme_export.py` | **stub** — APR segments as MEME minimal motifs |
| `itol_annotations.py` | **stub** — DATASET_BINARY for APRs, TREE_COLORS for clades |

Each stub raises with what it must produce, so `--dryrun` reports the whole graph
rather than a missing input.

## Three config choices with scientific consequences

**`redundancy.method`** — `cluster` and `weight` correct the same bias (densely
sampled clades counting many times in a per-column fraction). The workflow refuses
to do both: `method: cluster` together with `cluster.cap_only: true` is rejected at
parse time, because clustering *to bound compute* and clustering *to fix sampling*
are different claims about the result and the provenance records which was made.

**`align.tool`** — the aligner and its arguments land in
`results/align/<protein>.aln.yaml` with a checksum. Column assignments move with
the aligner, and the residual is a difference of two column-dependent terms, so
two runs with different aligners are not comparable.

**`predict.from_intervals`** — set it and prediction is skipped entirely: sequence
conservation is measured at APR coordinates you supply. That drops `predict_panel`
and `apr_calls` from the DAG, and with them the residual, which needs a call per
orthologue.

## Read the slope before the residual

`results/summary/residual_fit.yaml` carries the fitted slope and says in words
whether the residual means anything. Where sequence conservation carries no
information about the APR call the slope is ~0, the residual collapses to a centred
APR fraction, and it is **not** conservation-corrected. The fit is run-level, not
per protein: a few hundred columns give a slope with no stability.
