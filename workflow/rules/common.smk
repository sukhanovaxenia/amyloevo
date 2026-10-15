"""Shared helpers: the protein list, target assembly, and the redundancy switch.

Kept in one place so a rule never has to reason about configuration. The two
switches with scientific consequences -- how redundancy is corrected, and whether
prediction runs at all -- are resolved here once and read as booleans elsewhere.
"""

from pathlib import Path

ORTHOLOG_DIR = Path(config["orthologs_dir"])
SUFFIX = config["orthologs_suffix"]

#: Protein ids come from the input filenames, so adding a protein is adding a
#: file. Sorted for a deterministic DAG.
PROTEINS = sorted(p.stem for p in ORTHOLOG_DIR.glob(f"*{SUFFIX}"))

if not PROTEINS:
    raise WorkflowError(
        f"No orthologue files matching *{SUFFIX} under {ORTHOLOG_DIR}. "
        f"Each file is one protein's EggNOG group and its stem is the protein id."
    )

# --- Redundancy: exactly one correction ------------------------------------
REDUNDANCY = config["redundancy"]["method"]
if REDUNDANCY not in ("cluster", "weight", "none"):
    raise WorkflowError(
        f"redundancy.method must be cluster, weight or none (got {REDUNDANCY!r})"
    )

CLUSTER_FOR_SAMPLING = REDUNDANCY == "cluster"
CAP_ONLY = bool(config["redundancy"]["cluster"].get("cap_only", False))
#: Clustering happens when it corrects sampling OR purely to bound compute; the
#: provenance records which, because they are different claims about the result.
DO_CLUSTER = CLUSTER_FOR_SAMPLING or CAP_ONLY
#: Tree weighting happens only as the sampling correction, never alongside
#: clustering-for-sampling: both correct the same bias and would double-correct.
DO_WEIGHT = REDUNDANCY == "weight"

if CLUSTER_FOR_SAMPLING and CAP_ONLY:
    raise WorkflowError(
        "redundancy: cluster.cap_only is for bounding COMPUTE while sampling is "
        "corrected another way. With method: cluster the clustering already is the "
        "sampling correction, so set cap_only: false."
    )

# --- Prediction, or intervals supplied ------------------------------------
FROM_INTERVALS = config["predict"].get("from_intervals")
DO_PREDICT = not FROM_INTERVALS

# --- Alignment source ------------------------------------------------------
def sequences_for_alignment(wildcards):
    """Clustered representatives when clustering runs, the raw group otherwise."""
    if DO_CLUSTER:
        return f"results/cluster/{wildcards.protein}.fasta"
    return str(ORTHOLOG_DIR / f"{wildcards.protein}{SUFFIX}")


def apr_calls(wildcards):
    """Per-orthologue APR calls, or the supplied intervals in reduced mode."""
    if DO_PREDICT:
        return f"results/predict/{wildcards.protein}/apr_calls.tsv"
    return FROM_INTERVALS


def phylo_targets():
    if config["phylo"]["tree"] == "none":
        return []
    targets = [*expand("results/phylo/{protein}.treefile", protein=PROTEINS)]
    # The phylogenetic test needs a ROOTED tree, and the root comes from the
    # declared outgroup, which only exists alongside a paralogue split. With no
    # references there is no outgroup to root on and no clade to contrast, so
    # the tree is still built -- it is useful on its own -- and the test is not
    # requested. Asking for it anyway would fail the run at the rooting step
    # with an error about data when the cause is configuration.
    if DO_PHYLO_TEST:
        targets += expand(
            "results/summary/{protein}.phylo_signal.tsv", protein=PROTEINS
        )
    return targets


def viz_targets():
    targets = []
    viz = config["viz"]
    if viz.get("logos"):
        targets += expand("results/viz/logos/{protein}.png", protein=PROTEINS)
    if viz.get("meme_export"):
        targets += expand("results/viz/meme/{protein}.meme", protein=PROTEINS)
    if viz.get("itol") and config["phylo"]["tree"] != "none":
        targets += expand("results/viz/itol/{protein}_apr.txt", protein=PROTEINS)
    return targets


# --- Groups, taxonomy and the signal screen --------------------------------
def ortholog_fasta(protein):
    """The raw orthologue group. Metadata is read from the headers of the FULL
    set, not of the clustered representatives: a species dropped by clustering
    still has a taxon, and losing it from the metadata would quietly shrink
    every rank-wise denominator."""
    return str(ORTHOLOG_DIR / f"{protein}{SUFFIX}")


#: Paralogue assignment runs only when references are configured. With fewer
#: than two the rule writes an empty table and says why -- the correct outcome
#: for a set that really is one orthologue group.
GROUP_REFERENCES = config["group"].get("references") or {}
DO_ASSIGN = len(GROUP_REFERENCES) >= 2

#: The phylogenetic test additionally needs an outgroup to root on. Both
#: conditions are configuration, not data, so they are read here once rather
#: than discovered by a rule failing halfway through a run.
DO_PHYLO_TEST = DO_ASSIGN and bool(config["group"].get("outgroup"))


def group_targets():
    targets = [
        *expand("results/meta/{protein}.tsv", protein=PROTEINS),
        *expand("results/blocks/{protein}.tsv", protein=PROTEINS),
        *expand(
            "results/summary/{protein}.taxonomic_signal.tsv", protein=PROTEINS
        ),
    ]
    if config["viz"].get("group_figures", True):
        targets += expand(
            "results/viz/groups/{protein}_occupancy.png", protein=PROTEINS
        )
    return targets
