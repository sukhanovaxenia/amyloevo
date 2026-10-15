"""Decide which sequences the tree is inferred from, and record the decision.

One filter: a tip whose aligned row is mostly gaps. Nothing in the likelihood
objects to it -- absent characters simply do not constrain the topology -- so
such a tip lands wherever the remaining columns happen to point, with a branch
length that means nothing, and it drags its neighbours' branch lengths with it.
That last part is why the filter is here and not after inference: removing the
tip from a finished tree does not undo its effect on the rest.

What this does NOT do is deduplicate. Identical rows produce zero-length
branches, which look like something to clean up and are not: the tree program
collapses them for the search and restores them in the output, whereas
deduplicating here picks one member as the representative and discards the
others' traits. On the FXR1 trimmed alignment that discard is not hypothetical
-- 11 of 33 identical-row groups hold members whose APR masks differ, by up to
84 residues -- so the duplicates are kept and `phylo.trait_conflicts` measures
how much trait variance ends up with no branch length under it.
"""

from __future__ import annotations

import sys
from pathlib import Path

snakemake = globals()["snakemake"]
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd  # noqa: E402

from amyloevo.phylo.tree import identical_groups, tip_occupancy  # noqa: E402

#: Wrap width for the output FASTA. 60 is the convention every viewer expects.
WRAP = 60


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
    minimum = float(snakemake.params.min_occupancy)
    alignment = read_fasta(snakemake.input[0])
    occupancy = tip_occupancy(alignment)

    keep = [pid for pid in alignment if occupancy[pid] >= minimum]
    dropped = [pid for pid in alignment if occupancy[pid] < minimum]

    if len(keep) < 4:
        raise SystemExit(
            f"only {len(keep)} sequence(s) reach {minimum:.0%} occupancy out of "
            f"{len(alignment)}. Either the threshold is wrong for this alignment "
            f"or the alignment is mostly gaps -- check the trimming before "
            f"inferring a tree from it."
        )

    with open(snakemake.output.alignment, "w") as handle:
        for pid in keep:
            row = alignment[pid]
            handle.write(f">{pid}\n")
            for i in range(0, len(row), WRAP):
                handle.write(row[i : i + WRAP] + "\n")

    pd.DataFrame({
        "id": dropped,
        "occupancy": [occupancy[pid] for pid in dropped],
        "reason": f"below min_tip_occupancy={minimum}",
    }).sort_values("occupancy").to_csv(
        snakemake.output.dropped, sep="\t", index=False
    )

    groups = identical_groups({pid: alignment[pid] for pid in keep})
    n_redundant = sum(len(g) - 1 for g in groups)
    print(
        f"{len(keep)} of {len(alignment)} sequence(s) kept; {len(dropped)} below "
        f"{minimum:.0%} occupancy."
    )
    print(
        f"{len(groups)} group(s) of identical rows covering "
        f"{sum(len(g) for g in groups)} sequence(s) ({n_redundant} redundant) are "
        f"KEPT: the tree program collapses them for the search and restores them, "
        f"and removing them here would discard traits that differ between them."
    )


main()
