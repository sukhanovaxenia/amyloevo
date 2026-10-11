"""The distribution-free screen: do APR profiles cluster by clade?

Per paralogue group, one PERMANOVA per taxonomic rank on the block-occupancy
profiles, Benjamini-Hochberg adjusted across ranks.

This is NOT the phylogenetic test of `rule phylogenetic_signal` and does not
discharge it. A rank effect here means profiles cluster by clade, which nested
sampling alone produces. It is the cheap test that decides whether building the
species tree is worth the compute.

Run within a group on purpose: every member of a three-paralogue family
contributes one of each, so clade and paralogue explain the same variance in a
pooled test and whichever is fitted first wins.
"""

from __future__ import annotations

import sys
from pathlib import Path

snakemake = globals()["snakemake"]
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from amyloevo.phylo import format_signal_report, taxonomic_signal  # noqa: E402

#: A group smaller than this is not tested at all. Below it the rank filters
#: would do the refusing anyway, and this makes the reason legible.
MIN_GROUP = 20


def main() -> None:
    params = snakemake.params
    profiles = np.load(snakemake.input.profiles, allow_pickle=False)
    meta = pd.read_csv(snakemake.input.meta, sep="\t", dtype={"taxid": "string"})
    groups = pd.read_csv(snakemake.input.groups, sep="\t")

    matrix = profiles["profiles"]
    ids = [str(x) for x in profiles["ids"]]
    meta = meta.set_index("id").reindex(ids).reset_index()

    if len(groups):
        group_of = dict(zip(groups["id"], groups["group"]))
        meta["group"] = [group_of.get(i, "") for i in ids]
    else:
        # No paralogue split configured: the whole set is one group. Named so
        # the output does not imply a split that was never made.
        meta["group"] = "all"

    rows, reports = [], []
    for name, index in meta.groupby("group").groups.items():
        index = np.asarray(index)
        if len(index) < MIN_GROUP:
            reports.append(
                f"{name}: {len(index)} sequence(s) -- below {MIN_GROUP}, not "
                f"tested. Any rank inside it would be a handful of singletons."
            )
            continue
        result = taxonomic_signal(
            matrix[index],
            meta.iloc[index].reset_index(drop=True),
            group=str(name),
            ranks=tuple(params.ranks),
            draws=params.draws,
            seed=params.seed,
            min_per_level=params.min_per_level,
            min_levels=params.min_levels,
            min_members=params.min_members,
        )
        reports.append(format_signal_report(result))
        for rank in result.ranks:
            rows.append({
                "group": name,
                "rank": rank.rank,
                "levels": rank.levels,
                "n": rank.n,
                "distinct_profiles": rank.distinct_profiles,
                "pseudo_F": rank.pseudo_f,
                "R2": rank.r2,
                "within_total": rank.within_total,
                "p_perm": rank.p_value,
                "q_bh": rank.q_value,
                "scatter_negligible": rank.scatter_negligible,
                "skipped": rank.skipped,
            })

    pd.DataFrame(rows).to_csv(snakemake.output.table, sep="\t", index=False)
    text = "\n\n".join(reports) + "\n"
    Path(snakemake.output.report).write_text(text)
    print(text)


main()
