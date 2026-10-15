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

## 4. Four traps

**The set is not an orthologue set.** An "orthologue set" from EggNOG or STRING
is routinely a paralogue family. The FXR1 set that prompted `group/` is 372
sequences described as FXR1 orthologues and is in fact the whole FMR1/FXR1/FXR2
family: 116 FXR1, 101 FMR1, 97 FXR2 and 58 single-copy invertebrate genes, over
152 taxa of which 82 contribute exactly three. The paralogue differences are the
larger ones — FXR1 and FMR1 differ roughly twofold in APR load — so every
statement of the form "conserved across orthologues" computed on the pooled set
is a statement about a mixture.

So the group is established *before* conservation is measured, and from the data
rather than from the set's name: `group.assign` scores every sequence by identity
to caller-supplied references over the columns where all references carry a
residue. The best-vs-second **margin** is the only honest confidence available,
and a sequence 48% identical to one reference and 47% to another has not been
assigned, it has been rounded.

Which sequences sit *outside* the reference groups is a separate, declared input,
not an inference. On the FXR1 set every one of the 58 invertebrate genes falls
below the 0.05 margin — a perfectly sensitive screen — but so do 20 vertebrate
sequences that genuinely belong to a paralogue. A low margin means *ambiguous*;
it does not mean *pre-duplication*, and reading it that way moves those 20 out
of the group they belong to. `copy_number` reports sequences per taxon and marks
the deficient ones; it does not call them ancestral, because incomplete
sampling, gene loss and predating the duplication are different facts and only a
tree separates them.

**Pseudoreplication.** "Does APR presence depend on divergence or species group"
cannot be answered by correlation. Species share ancestry, so the observations
are not independent and any clade-level pattern will appear significant.
Phylogenetic signal in a binary trait: Fritz & Purvis's *D*, or Pagel's λ.
The dependency itself: phylogenetic logistic regression (`phyloglm`).

**The tree is a gene tree, and that was a decision.** An earlier version of this
section made the species tree the default and called a gene tree "the stricter
choice, a later option". It is now the default, inferred from the family
alignment itself, because on a paralogue family a species tree cannot hold the
thing being measured: the FXR1 set has three sequences per taxon and a species
tree has one tip per taxon, so the paralogues would have to be averaged away
before the test that is supposed to distinguish them. What the gene tree costs
is that the duplication is *inside* it, and three consequences follow, each now
a check rather than an assumption:

- **Monophyly is a property of the tree, not of the assignment.** `group.assign`
  calls a sequence FXR1 by identity to a reference; whether the FXR1 sequences
  form a clade is a separate claim that the tree either supports or refutes.
  `phylo.clade_checks` makes it and names the intruders. A group that comes out
  paraphyletic is a result — a misassignment, an incomplete lineage, or a
  duplication history with more than one event — and it also rules out taking
  that group's MRCA as its subtree, because the subtree holds other groups.
- **Rooting is a declaration, checked.** The outgroup is the declared
  pre-duplication set, for the reason already in this section: a low identity
  margin means *ambiguous*, not *ancestral*. What `phylo.root_on_outgroup` adds
  is the check that the declared outgroup is monophyletic on this tree; when it
  is not, its MRCA contains ingroup tips and rooting there moves them across the
  root. The default is to refuse rather than root anyway.
- **The clade cannot be a regression predictor.** Each paralogue group descends
  from one duplication, so a group contrast rests on two independent events
  however many tips it spans. `phyloglm` would return a coefficient computed as
  if all 282 tips were data — the same pseudoreplication this section exists to
  prevent, one level up. The group term is therefore absent from the fit, and
  the controlled instrument for the paralogue difference is the paired
  within-taxon contrast over the taxa carrying all three copies.

**Zero-length branches are where the trait has no history under it.** Identical
aligned rows sit at the ends of branches of length zero, so a trait that differs
between them changes with no branch length to attribute the change to.
Deduplicating the alignment looks like the fix and is worse: it keeps one member
and discards the others' traits. On the FXR1 trimmed alignment 33 groups of
identical rows cover 90 of 372 sequences — only 7 of those groups are identical
*before* trimming — and 11 contain members whose APR masks differ, by up to 84
residues. So every sequence is kept, the tree program collapses them for the
search and restores them in its output, and `phylo.trait_conflicts` measures how
much trait variance ends up unattributable instead of deleting it.

**A near-universal block cannot be tested, and that is not a threshold to
tune.** *D* needs both states present in quantity. Two of the fifteen FXR1
blocks are carried by 98.6% and 95.7% of sequences; there is no variation in
them for the tree to explain, and lowering the prevalence gate would only
replace a refusal with a number whose sampling variance covers the whole
interpretable scale. `phylo.screen` declines them and says why.

`phylo.signal` is the screen that comes before any of that: a PERMANOVA per rank
on the APR profiles, run **inside** a paralogue group, because every mammal in a
three-paralogue family contributes one of each and so "mammal" and "paralogue"
explain the same variance. It answers only whether profiles cluster by clade,
which nested sampling alone produces; it is what decides whether building the
tree is worth it, and never a substitute for the tree.

Two guards sit on it, both put there by wrong answers it gave first:

- **Replication.** Pseudo-F and R² rise mechanically as levels approach
  singletons: with 56 families among 77 sequences, within-group scatter is zero
  by construction. A rank is dropped unless enough of its levels are replicated,
  and the surviving level and member counts are reported.
- **Degenerate profiles.** Level replication is necessary and not sufficient.
  FXR2 at order level passes every replication filter — 7 orders, 48 sequences,
  no singletons — and still returns R² = 0.989 with pseudo-F = 591, because
  those 48 sequences carry **7 distinct block profiles** between them and three
  orders are internally identical. Within-group scatter is 1% of the total and
  R² follows arithmetically. FXR1 over the same ranks has 13 distinct profiles
  among 58 sequences and returns R² = 0.658. The permutation *p* is honest in
  both cases, so the row is reported rather than dropped, flagged, and with the
  distinct-profile count beside it. The sentence to write is "FXR2's APR
  architecture takes only seven forms and they are clade-specific", not "order
  explains 99% of the variance".

**Families of tests.** Four nested ranks on one dataset is four tests; one test
per protein across a sweep is hundreds. Both are adjusted with Benjamini-Hochberg
(`stats.multiple`), and the per-protein case is where the adjustment earns its
keep: on the FXR1 termini analysis the direction is unanimous (91/91, sign test
*p* = 8×10⁻²⁸) while **no single protein** is significant at FDR 0.05, because
each carries about four APRs and the smallest attainable *p* sits above the
threshold. Those answer different questions and both belong in the report; an
unadjusted per-protein table answers neither.

**Boundary inheritance.** Every APR here is a window-derived call and carries the
±(`window_size` − 1) boundary uncertainty the predictors impose. `coverage_depth`
and the window spread propagate into the alignment columns, because conservation
measured at a boundary residue rests on far weaker evidence than conservation
measured at a core one. Averaged together without that weighting, the two blur
exactly the contrast in §3.

## 5. Modules

| module | responsibility | external dependency |
|---|---|---|
| `io.headers` | read the per-protein orthologue multifasta (EggNOG), parse taxid and species from headers | none |
| `io.taxonomy` | taxid → name and lineage, from an NCBI taxdump or a caller's table; retired taxids resolved through `merged.dmp` | none |
| `io.annotate` | the metadata table (`taxid`, `tax_name`, `protein`, rank columns) and the display `label` every figure reads | none |
| `align` | MAFFT / MUSCLE wrapper; records aligner, version and parameters | the aligner binary |
| `project` | `(species, residue_number) ↔ alignment_column`, gap-aware in both directions | none |
| `group` | paralogue assignment from references, margin, copy number per taxon (§4) | none |
| `predict` | call `aggressor-wrappers` on every orthologue; consume its merged tables | aggressor-wrappers |
| `conserve` | the three measures of §3 | none |
| `stats` | multiple-comparison adjustment (BH, Holm) shared by every family of tests | none |
| `phylo` | taxonomic signal screen, gene-tree rooting and clade checks, trait binarisation and its prevalence screen, the R bridge (§4) | scipy, biopython; R `caper` / `phytools` / `phylolm` |
| `viz` | group-wise occupancy, block heatmap, paired load; later per-APR logos and MEME export | matplotlib; later logomaker / MEME |

`predict` consumes `aggressor-wrappers`; it does not reimplement predictors.
`conserve` consumes `project`; it never touches an alignment directly. `phylo`
and `viz` consume `group`, never the other way round: grouping is a property of
the sequences and must not depend on what is being plotted or tested. The
modules with no external dependency are testable without installing anything,
which is why they are built first.

Nothing in `group`, `phylo`, `stats` or `viz` is specific to FXR1, to
vertebrates, or to three paralogues. The references define the groups; the rank
columns define the ranks. FXR1 is the validation case, not the design: the
generalised code reproduces the FXR1-specific implementation's identities to
1.1×10⁻¹⁶, all 372 group calls, and all eight rows of its PERMANOVA table.

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
