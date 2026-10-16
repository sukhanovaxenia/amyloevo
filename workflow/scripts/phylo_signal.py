"""Phylogenetic signal in APR block presence, on the gene tree.

Thin: root the tree on the declared outgroup, reconcile the tips with the block
profiles, binarise the occupancies at one named threshold, screen each block for
the prevalence *D* needs, hand the survivors to R, adjust what comes back.

What it does NOT do, and what replaced the stub this file used to be:

* It does not substitute a correlation across tips. Species share ancestry, so
  the tips are not independent and any clade-level pattern is significant by
  construction. That is the reason the step exists.
* It does not take the PERMANOVA screen in ``rule taxonomic_signal`` as an
  answer. That screen says profiles cluster by clade, which nested sampling
  alone produces.
* It does not deduplicate the alignment to get rid of zero-length branches.
  Identical rows are collapsed by the tree program and restored in its output;
  deduplicating here would pick one member as the representative and discard
  the others' traits. On the FXR1 trimmed alignment 11 of 33 identical-row
  groups contain members whose APR masks differ -- by up to 84 residues -- so
  that discard is not hypothetical. :func:`trait_conflicts` measures it instead.
* It does not fit the paralogue group as a predictor. Each group is one
  monophyletic clade from one duplication; see ``phylo_signal.R``.
"""

from __future__ import annotations

import sys
from pathlib import Path

snakemake = globals()["snakemake"]
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from amyloevo.phylo import estimate, traits as tr, tree as T  # noqa: E402

R_SCRIPT = Path(__file__).resolve().parent / "phylo_signal.R"


def read_fasta(path) -> dict[str, str]:
    seqs: dict[str, str] = {}
    name, buf = None, []
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            if name is not None:
                seqs[name] = "".join(buf)
            name, buf = line[1:].split()[0], []
        else:
            buf.append(line.strip())
    if name is not None:
        seqs[name] = "".join(buf)
    return seqs


def main() -> None:
    params = snakemake.params
    out_table = Path(snakemake.output.table)
    out_report = Path(snakemake.output.report)
    workdir = out_table.parent / f"{out_table.stem}.rwork"

    profiles = np.load(snakemake.input.profiles, allow_pickle=True)
    matrix = profiles["profiles"]
    ids = [str(x) for x in profiles["ids"]]
    labels = [str(x) for x in profiles["labels"]]

    groups = pd.read_csv(snakemake.input.groups, sep="\t")
    meta = pd.read_csv(snakemake.input.meta, sep="\t", dtype={"taxid": "string"})
    alignment = read_fasta(snakemake.input.alignment)
    gene_tree = T.read_tree(snakemake.input.tree)

    # --- the tree ---------------------------------------------------------
    support = T.support_summary(gene_tree)

    outgroup_label = params.outgroup_label
    if len(groups) and "group" in groups.columns:
        outgroup_ids = sorted(groups.loc[groups["group"] == outgroup_label, "id"])
        group_of = dict(zip(groups["id"], groups["group"]))
    else:
        outgroup_ids, group_of = [], {}

    rooting = T.root_on_outgroup(
        gene_tree, outgroup_ids,
        require_monophyly=bool(params.require_outgroup_monophyly),
    )
    if rooting.tree is None:
        out_report.write_text(
            f"PHYLOGENETIC SIGNAL -- not run.\n\n{rooting}\n\n"
            f"The root is what D and lambda read ancestry off, so this is not a\n"
            f"detail to work around. Either the outgroup declaration is wrong, or\n"
            f"the tree disagrees with it and that disagreement is the result.\n"
            f"Set phylo.require_outgroup_monophyly to false to root anyway.\n"
        )
        pd.DataFrame(columns=["scope", "block"]).to_csv(out_table, sep="\t", index=False)
        raise SystemExit(f"outgroup rooting refused: {rooting.refused}")
    rooted = rooting.tree

    membership: dict[str, list[str]] = {}
    for pid, group in group_of.items():
        if group and group != outgroup_label:
            membership.setdefault(group, []).append(pid)
    clades = [str(c) for c in T.clade_checks(rooted, membership)]

    # --- the traits -------------------------------------------------------
    binary = tr.binarise(matrix, ids, labels, threshold=params.presence_threshold)
    occupancy = tr.continuous(matrix, ids, labels)

    reconciliation = T.reconcile(T.tip_labels(rooted), binary.index)
    shared = reconciliation.shared
    if len(shared) < 3:
        out_report.write_text(
            f"PHYLOGENETIC SIGNAL -- not run.\n\n{reconciliation}\n"
        )
        pd.DataFrame(columns=["scope", "block"]).to_csv(out_table, sep="\t", index=False)
        raise SystemExit("fewer than three sequences have both a tip and a trait")
    binary = binary.loc[shared]
    occupancy = occupancy.loc[shared]

    conflicts = T.trait_conflicts(T.identical_groups(alignment), binary)

    # Screened over the whole tree AND inside each clade. On a gene tree of a
    # paralogue family the whole-tree answer is close to foreordained -- the
    # deepest split is the duplication and the groups carry different profiles,
    # so "does this trait follow this tree" has to come back yes, at p-values
    # below what a double can hold. The per-clade rows are where it does not.
    checks = tr.screen_by_clade(
        binary,
        group_of,
        min_minority=params.min_minority,
        min_prevalence=params.min_prevalence,
    )
    work = tr.tasks(checks)
    testable = sorted(set(work["block"])) if len(work) else []
    trait_report = tr.format_trait_report(
        checks, threshold=params.presence_threshold,
        min_minority=params.min_minority, min_prevalence=params.min_prevalence,
    )

    # --- the estimators ---------------------------------------------------
    if not testable:
        report = "\n\n".join([
            trait_report,
            estimate.format_phylo_report(
                pd.DataFrame(columns=["scope", "block"]), permutations=params.permutations,
                support=support, rooting=rooting, conflicts=conflicts, clades=clades,
            ),
        ])
        out_report.write_text(report + "\n")
        pd.DataFrame(columns=["scope", "block"]).to_csv(out_table, sep="\t", index=False)
        print(report)
        return

    lengths = dict(zip(meta["id"], meta.get("length", pd.Series(dtype=float))))
    covariates = tr.covariates(
        shared,
        groups=group_of,
        lengths=lengths or None,
        root_distance=T.root_to_tip(rooted),
    )
    for label in testable:
        covariates[f"occupancy_{label}"] = occupancy[label].to_numpy()

    tree_path = workdir / "rooted.nwk"
    workdir.mkdir(parents=True, exist_ok=True)
    # Rooted in R as well, from the same outgroup: ape::root places the root on
    # the outgroup's stem, which Biopython does not, and phyloglm needs that.
    T.write_tree(rooted, tree_path)

    table = estimate.run_estimators(
        tree_path, binary[testable], covariates, outgroup_ids, tasks=work,
        r_script=R_SCRIPT, workdir=workdir,
        signal=params.signal, regression=params.regression,
        permutations=params.permutations, seed=params.seed,
    )
    table = estimate.adjust(table, alpha=params.alpha)
    table.to_csv(out_table, sep="\t", index=False)

    report = "\n\n".join([
        trait_report,
        estimate.format_phylo_report(
            table, alpha=params.alpha, permutations=params.permutations,
            support=support, rooting=rooting, conflicts=conflicts, clades=clades,
        ),
    ])
    out_report.write_text(report + "\n")
    print(report)


main()
