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
| `project` — residue ↔ alignment column | **implemented**, 19 tests |
| `conserve` — entropy, JSD, APR fraction, residual | **implemented**, 23 tests |
| `io.headers` — EggNOG / UniProt / NCBI headers | **implemented**, 12 tests |
| `align` — MAFFT / MUSCLE wrapper | specified |
| `predict` — aggressor-wrappers panel over orthologues | specified |
| `phylo` — phylogenetic signal and regression | specified |
| `viz` — logos, MEME export | specified |

## Install

```bash
git clone <this repo>
cd amyloevo
python -m pip install -e ".[dev]"
pytest
```

The core is pure Python: `project`, `conserve` and `io` import without an
aligner, a tree program or the predictor panel installed, which is why they are
the parts built first and the parts a contributor can test on any machine.

## Input

One multifasta of orthologues per protein of interest (EggNOG group), for the
372-protein FXR1 set. amyloevo aligns them itself — MAFFT or MUSCLE — rather
than expecting alignments, so the aligner and its parameters are recorded with
the result and two runs remain comparable.

## What it does not do

It does not implement amyloid predictors. Predictions come from
`aggressor-wrappers`, which runs the panel and binarises each tool with its own
threshold and direction; amyloevo consumes the merged tables. It does not
implement consensus tiers either — that is `amyloscope`. Three packages, three
responsibilities: execution, analysis, evolution.
