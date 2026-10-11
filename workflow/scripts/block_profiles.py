"""Define APR blocks and score every sequence's occupancy over them.

Thin: read the alignment and the APR calls, project the masks once, call
`amyloevo.conserve.blocks`, write the feature matrix and the block table.

The projection happens exactly once, here. Projecting a mask that is already in
alignment coordinates scatters the profile across the wrong columns, and the
result looks like a plausible table rather than an error -- it produced 5 blocks
instead of 15 and dropped cluster purity from 90% to 33% before it was caught.
So residue-space masks and column-space masks are kept in separately named
variables and never passed to each other's consumers.
"""

from __future__ import annotations

import sys
from pathlib import Path

snakemake = globals()["snakemake"]
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from amyloevo.conserve.blocks import (  # noqa: E402
    block_profiles,
    define_blocks,
    group_occupancy,
)
from amyloevo.project.columns import column_to_residue, project_mask  # noqa: E402


def read_fasta(path: str) -> dict[str, str]:
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


def residue_masks(calls: pd.DataFrame, lengths: dict[str, int]) -> dict[str, np.ndarray]:
    """Per-sequence boolean APR mask in RESIDUE space, from interval rows.

    Expects columns ``id, start, end`` with 1-based inclusive residue
    coordinates -- the shape `apr_calls.tsv` carries.
    """
    masks = {pid: np.zeros(n, dtype=bool) for pid, n in lengths.items()}
    for row in calls.itertuples(index=False):
        mask = masks.get(row.id)
        if mask is None:
            continue
        start = max(int(row.start) - 1, 0)
        stop = min(int(row.end), mask.size)
        mask[start:stop] = True
    return masks


def main() -> None:
    params = snakemake.params
    aligned = read_fasta(snakemake.input.alignment)
    n_col = len(next(iter(aligned.values())))
    calls = pd.read_csv(snakemake.input.calls, sep="\t")
    groups = pd.read_csv(snakemake.input.groups, sep="\t")
    reference = Path(snakemake.input.reference).read_text().strip()

    lengths = {pid: sum(1 for c in al if c != "-") for pid, al in aligned.items()}
    res_masks = residue_masks(calls, lengths)
    col_masks = {
        pid: project_mask(res_masks[pid], aligned[pid], n_col)
        for pid in aligned
        if pid in res_masks
    }

    ids = [p for p in aligned if p in col_masks and col_masks[p].any()]
    if not ids:
        raise ValueError(
            f"no sequence in {snakemake.input.alignment} carries an APR call, so "
            f"there is nothing to define blocks over. Check that the ids in "
            f"{snakemake.input.calls} match the alignment headers."
        )

    if len(groups):
        group_of = dict(zip(groups["id"], groups["group"]))
        members_by_group: dict[str, list[str]] = {}
        for pid in ids:
            members_by_group.setdefault(group_of.get(pid, "unassigned"), []).append(pid)
    else:
        members_by_group = {"all": ids}

    blocks = define_blocks(
        col_masks, members_by_group, n_col,
        min_occupancy=params.min_occupancy,
        min_coverage=params.min_coverage,
        min_width=params.min_width,
        max_gap=params.max_gap,
        min_group=params.min_group,
    )
    if not blocks:
        raise ValueError(
            "no APR block cleared the thresholds, so there are no features to "
            "test. Lower blocks.min_occupancy (currently "
            f"{params.min_occupancy}) or check that the groups in "
            f"{snakemake.input.groups} have at least blocks.min_group members."
        )

    col2res = column_to_residue(aligned[reference]) if reference in aligned else None
    labels = [b.label(col2res) for b in blocks]

    matrix = block_profiles(col_masks, blocks, ids)
    np.savez_compressed(
        snakemake.output.profiles,
        profiles=matrix,
        ids=np.array(ids, dtype=object),
        labels=np.array(labels, dtype=object),
        starts=np.array([b.start for b in blocks]),
        stops=np.array([b.stop for b in blocks]),
    )

    table = group_occupancy(col_masks, blocks, members_by_group)
    table.insert(0, "label", labels)
    table.to_csv(snakemake.output.blocks, sep="\t", index=False)

    print(f"{len(ids)} sequence(s), {len(blocks)} block(s): {', '.join(labels)}")


main()
