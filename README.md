# amyloevo

Evolutionary conservatism of amyloidogenic regions (APRs) across orthologues.

Whether APR positions are conserved is not the question — conserved positions
are conserved for many reasons, and an APR inside a conserved domain reports on
the domain. The question is whether a position stays *amyloidogenic* while its
residues turn over, which is what selection on the property rather than on the
sequence would look like. That is a contrast between two measurements, so
amyloevo makes both and reports the residual between them.

See [`docs/DESIGN.md`](docs/DESIGN.md) for the reasoning, the cost model and the
two methodological traps this design is built to avoid.

## Status

Early. The modules with no external dependency are implemented and tested; the
ones that wrap an aligner, a tree program or the predictor panel are specified
but not yet written.

| module | state |
|---|---|
| `project` — residue ↔ alignment column, mask projection | **implemented**, 30 tests |
| `conserve.scores` — entropy, JSD, APR fraction, residual | **implemented**, 23 tests |
| `conserve.blocks` — APR blocks and per-sequence occupancy | **implemented**, 28 tests |
| `io.headers` — EggNOG / UniProt / NCBI headers | **implemented**, 12 tests |
| `io.taxonomy` — taxid → name and lineage (NCBI taxdump) | **implemented**, 18 tests |
| `io.annotate` — metadata table and display labels | **implemented**, with the above |
| `group` — paralogue assignment, margin, copy number | **implemented**, 21 tests |
| `stats` — Benjamini-Hochberg, Holm | **implemented**, 14 tests |
| `phylo.signal` — taxonomic signal screen (PERMANOVA) | **implemented**, 24 tests |
| `viz.groups` — occupancy, block heatmap, paired load | **implemented**, 12 tests |
| `align` — MAFFT / MUSCLE wrapper | specified |
| `predict` — aggressor-wrappers panel over orthologues | specified |
| `phylo` — tree-based signal (D, λ) and regression | specified |
| `viz` — logos, MEME export | specified |

`phylo.signal` is a **screen, not the phylogenetic test**. It says whether APR
profiles cluster by clade, which nested sampling alone produces; the tree-based
tests the design calls for are still specified and `rule phylogenetic_signal`
still raises. Reading the screen as that test swaps the question the project
exists to ask for a weaker one.

## Install

```bash
git clone <this repo>
cd amyloevo
python -m pip install -e ".[dev]"
pytest
```

The core is pure Python: `project`, `conserve`, `io`, `group` and `stats` import
without an aligner, a tree program or the predictor panel installed, which is
why they are the parts built first and the parts a contributor can test on any
machine. `phylo.signal` needs scipy; `viz` needs matplotlib, imported inside the
functions so the package still imports without it.

## Input

One multifasta of orthologues per protein of interest (EggNOG group), for the
372-protein FXR1 set. amyloevo aligns them itself — MAFFT or MUSCLE — rather
than expecting alignments, so the aligner and its parameters are recorded with
the result and two runs remain comparable.

Two inputs are worth configuring before the first run, both in
[`config/config.yaml`](config/config.yaml):

- **`taxonomy.taxdump`** — an unpacked NCBI taxdump. Without it the rank columns
  are empty and every clade-level test is skipped for want of levels. `merged.dmp`
  matters: a set assembled a few years ago carries taxids NCBI has since merged,
  and a taxon that fails to resolve leaves the analysis *silently*, dropped from
  every denominator without being counted as missing.
- **`group.references`** — one well-annotated species' paralogues, when the set is
  a paralogue family rather than an orthologue group. It usually is: the FXR1 set
  is 372 sequences described as FXR1 orthologues and is in fact the whole
  FMR1/FXR1/FXR2 family, where FXR1 and FMR1 differ roughly twofold in APR load.
  Conservation measured before the groups are known is a measure over a mixture.
  Leave it `null` for a genuine single-group set; inventing a split is worse than
  having none.

## A note on labels

`9606.ENSP00000350170` is unique, stable and unreadable; `Homo sapiens FXR1` is
readable and not guaranteed unique. So every table carries both: `id` for joins
and provenance, `label` for display, built from a configurable template
(default `{tax_name} {group}`), falling back down a chain for partly resolved
rows, and disambiguated with the accession on *every* member of a colliding set
rather than on the second onwards. Figures label from `label`; nothing joins on
it.

## What it does not do

It does not implement amyloid predictors. Predictions come from
`aggressor-wrappers`, which runs the panel and binarises each tool with its own
threshold and direction; amyloevo consumes the merged tables. It does not
implement consensus tiers either — that is `amyloscope`. Three packages, three
responsibilities: execution, analysis, evolution.
