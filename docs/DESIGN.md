# amyloevo — design

Evolutionary conservatism of amyloidogenic regions (APRs) across orthologues.

The question is not whether APR positions are conserved. Conserved protein
positions are conserved for many reasons, and an APR sitting in a conserved
domain tells us about the domain, not about amyloidogenicity. The question is
whether a position stays *amyloidogenic* while its residues turn over — whether
selection acts on the property rather than on the sequence that happens to
produce it. That is a contrast between two measurements, so both have to be
made, and the residual between them is the result.

## 1. The panel constraint decides the cost model

Five of the twelve predictors in `aggressor-wrappers` have no local arm:

```python
_WEB_ONLY = {"waltz", "pasta", "aggreprot", "aggrescan", "foldamyloid"}
```

The FXR1 set is 372 proteins; an EggNOG group runs to hundreds of sequences, so
the orthologue sweep is 10^4–10^5 sequences. The web arm cannot serve that —
not in wall-clock time, and not within what those services will tolerate. The
sweep therefore runs the **local panel**: AmyloDeep, AmyloGram, APPNN, PATH,
TANGO, and ArchCandy / Cross-Beta through their local arms.

One panel, one run, and **the reference sequence is inside it**. An EggNOG group
normally contains the reference member, so the reference is a row of the same
table scored by the same tools -- not a separate step to pay for. There is no
"reference panel" stage; an earlier draft of this document implied one and was
wrong.

What must not happen is comparing across panels. An APR called by twelve tools,
set against orthologues called by seven, is a panel-composition artefact wearing
the costume of a result: every orthologue appears to have lost APRs that were
never tested for. The twelve-tool calls from the FXR1 sweep are therefore an
*external comparison* -- how much does dropping the web arm change the call on
the same protein? -- and never an input to a conservation measure. The panel is
recorded per run, and two runs with different panels are not comparable.

If the reference sequence is absent from a group's multifasta, it is added before
alignment rather than scored separately.

### Running without the panel at all

A reduced mode is legitimate and cheap: given APR intervals already known for the
reference proteins, amyloevo can measure **sequence conservation at those columns**
and never call a predictor. That yields the baseline of section 3 and the
occupancy bookkeeping, but not APR conservation and therefore not the residual --
which needs a call per orthologue. Useful as a first pass and as a sanity check
on the alignment; not a substitute for the contrast the project turns on.

## 2. Data model — one table, separate columns

Every measurement lands in one long table keyed by the coordinate spine:

| column | meaning |
|---|---|
| `protein` | reference protein id (FXR1 set) |
| `species` / `taxid` | from the orthologue header |
| `alignment_column` | 1-based column of that protein's MSA |
| `residue_number` | 1-based position **in that species' sequence**, or null for a gap |
| `residue` | the amino acid, or `-` |
| `aa_type` | physicochemical class of that residue |
| `apr_score` | consensus score from the local panel |
| `apr_call` | binary, from the panel's fractional consensus |
| `coverage_depth`, `window_size` | window provenance, carried from the predictors |

`aa_type` and `apr_score` stay **separate columns**, never fused into one
composite. A single number cannot be decomposed when a reviewer asks which part
of it moved, and fusing them forecloses the one comparison this project exists
to make.

## 3. The two conservation measures, and their residual

Per alignment column, over the orthologues present at that column:

- **Sequence conservation** — Shannon entropy and Jensen–Shannon divergence
  against a background amino-acid distribution, gap-aware. Columns below a
  configurable occupancy are reported as undefined rather than as conserved: a
  column present in three species out of two hundred is not evidence.
- **APR conservation** — the fraction of orthologues whose `apr_call` is true at
  that column.
- **The residual** — APR conservation after regressing out sequence
  conservation. This is the headline metric. A high residual marks a column that
  stays amyloidogenic while its residues change; a column where both are high is
  simply a conserved column and says nothing specific about amyloid.

## 4. Two traps

**Pseudoreplication.** "Does APR presence depend on divergence or species group"
cannot be answered by correlation. Species share ancestry, so the observations
are not independent and any clade-level pattern will appear significant.
Phylogenetic signal in a binary trait: Fritz & Purvis's *D*, or Pagel's λ.
The dependency itself: phylogenetic logistic regression (`phyloglm`). The
species tree comes from NCBI taxonomy or EggNOG's own tree; a gene tree would be
the stricter choice and is a later option, not the default.

**Boundary inheritance.** Every APR here is a window-derived call and carries the
±(`window_size` − 1) boundary uncertainty the predictors impose. `coverage_depth`
and the window spread propagate into the alignment columns, because conservation
measured at a boundary residue rests on far weaker evidence than conservation
measured at a core one. Averaged together without that weighting, the two blur
exactly the contrast in §3.

## 5. Modules

| module | responsibility | external dependency |
|---|---|---|
| `io` | read the per-protein orthologue multifasta (EggNOG), parse taxid and species from headers | none |
| `align` | MAFFT / MUSCLE wrapper; records aligner, version and parameters | the aligner binary |
| `project` | `(species, residue_number) ↔ alignment_column`, gap-aware in both directions | none |
| `predict` | call `aggressor-wrappers` on every orthologue; consume its merged tables | aggressor-wrappers |
| `conserve` | the three measures of §3 | none |
| `phylo` | species tree, phylogenetic signal, phylogenetic regression | R `phyloglm` / `caper`, or a Python equivalent |
| `viz` | per-APR sequence logos; MEME-format export for motif discovery over APR segments | logomaker / MEME suite |

`predict` consumes `aggressor-wrappers`; it does not reimplement predictors.
`conserve` consumes `project`; it never touches an alignment directly. The
modules with no external dependency are testable without installing anything,
which is why they are built first.

## 6. Snakemake owns the running; the package owns the logic

Counting the external programs -- MAFFT or MUSCLE, CD-HIT or MMseqs2, IQ-TREE,
the predictor panel itself -- and multiplying by 372 proteins gives on the order
of a thousand jobs with a real dependency graph, partial failures, and a need to
re-run only what changed. That is a workflow manager's job, and a Python package
driving it with subprocess loops would be a worse Snakemake.

So the split is not either/or:

- **`workflow/` (Snakemake)** owns orchestration: the per-protein DAG, external
  tool invocation, resources, checkpointing, cluster or local parallelism, and
  the provenance of section 7. One rule per external program.
- **`src/amyloevo/` (the package)** stays the library those rules import: the
  coordinate spine, the conservation measures, header parsing, the phylogenetic
  test wrappers, the iTOL annotation writers. This is where the subtle
  correctness lives -- an off-by-one in column mapping, the occupancy gate, the
  residual's undefined case -- and it is unit-testable, which a Snakemake rule is
  not.

A rule stays thin: read inputs, call one library function, write outputs. When a
rule starts containing logic worth testing, that logic belongs in the package.

### Redundancy is corrected once, not twice

CD-HIT or MMseqs2 clustering and phylogenetic weighting address the *same*
problem -- densely sampled clades counting many times over in a per-column
fraction. Applying both double-corrects, and the result is a conservation
estimate whose denominator nobody can reconstruct. Pick one, record which:
clustering is cruder and cheaper and shrinks the alignment; phylogenetic
weighting is principled and keeps every sequence. Clustering purely to bound
compute is a separate decision from clustering to fix sampling, and should be
recorded as such.

## 7. Provenance

Each run records: the panel and each tool's version, the aligner and its
parameters, the occupancy threshold, the consensus rule, and the source of the
orthologue set. Alignment choice moves column assignments, so a result without
its aligner recorded is not reproducible — and the residual of §3 is a
difference of two quantities that both depend on that alignment.
