"""iTOL annotation files for the gene tree.

Thin: read the tree, the block profiles, the group calls and the metadata, call
``amyloevo.viz.itol``, write one file per dataset. Five of them, because iTOL
takes one upload per track:

    <protein>_apr.txt        block presence as a shape per block (DATASET_BINARY)
    <protein>_clades.txt     paralogue colours on leaves (TREE_COLORS)
    <protein>_labels.txt     species + paralogue instead of the id (LABELS)
    <protein>_load.txt       APR residues / length, as a heat strip (GRADIENT)
    <protein>_load_bars.txt  the same numbers as bars (SIMPLEBAR)

The check this script exists to print: iTOL matches rows to the tree by leaf
label and silently drops a row that matches nothing. A dataset built from the
wrong id column therefore uploads cleanly and renders an empty track, which
reads as "this family has no APRs" rather than "nothing matched". So every
dataset's ids are compared against the tree's actual tips and the counts go to
the log.
"""

from __future__ import annotations

import sys
from pathlib import Path

snakemake = globals()["snakemake"]
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from amyloevo.phylo.tree import read_tree, tip_labels  # noqa: E402
from amyloevo.viz import itol  # noqa: E402


def main() -> None:
    out = {k: Path(v) for k, v in dict(snakemake.output).items()}
    for path in out.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    tips = tip_labels(read_tree(snakemake.input.tree))
    meta = pd.read_csv(snakemake.input.meta, sep="\t", dtype={"taxid": "string"})
    groups = pd.read_csv(snakemake.input.groups, sep="\t")
    profiles = np.load(snakemake.input.profiles, allow_pickle=True)

    matrix = profiles["profiles"]
    ids = [str(x) for x in profiles["ids"]]
    labels = [str(x) for x in profiles["labels"]]
    threshold = float(snakemake.params.presence_threshold)

    group_of = (
        dict(zip(groups["id"], groups["group"])) if len(groups) else {}
    )
    label_of = (
        dict(zip(meta["id"], meta["label"])) if "label" in meta.columns else {}
    )

    # --- block presence ---------------------------------------------------
    # Every tip gets a row. A tip with no block profile gets -1 across the
    # board, not 0: it failed the completeness filter, so nothing is known
    # about its blocks, and writing 0 would draw 75 invented absences on the
    # FXR1 tree.
    present = (matrix > threshold).astype(int)
    frame = pd.DataFrame(present, index=ids, columns=labels)
    frame = frame.reindex([t for t in tips]).fillna(-1).astype(int)

    palette = itol.palette_for(group_of.values())
    out["apr"].write_text(
        itol.binary_dataset(
            frame,
            label=f"APR blocks (occupancy > {threshold:g})",
            colour=palette.get("FXR1", "#D55E00"),
        ),
        encoding="utf-8",
    )

    # --- clade colours and readable labels --------------------------------
    on_tree = {t: group_of.get(t, "") for t in tips}
    out["clades"].write_text(
        itol.tree_colors(on_tree, palette), encoding="utf-8"
    )
    out["labels"].write_text(
        itol.labels_dataset({t: label_of.get(t, t) for t in tips}),
        encoding="utf-8",
    )

    # --- APR load ---------------------------------------------------------
    # Mean block occupancy, not APR residues over length: the residue-level
    # load is computed from the predictor panel, which this rule does not read.
    # Named for what it is so the two are not confused in a figure legend.
    load = {pid: float(row.mean()) for pid, row in zip(ids, matrix)}
    out["load"].write_text(
        itol.gradient_dataset(
            load, label="mean block occupancy",
            colour=palette.get("FXR1", "#D55E00"),
        ),
        encoding="utf-8",
    )
    out["load_bars"].write_text(
        itol.simplebar_dataset(
            load, label="mean block occupancy",
            colour=palette.get("FXR1", "#D55E00"),
        ),
        encoding="utf-8",
    )

    # --- the check --------------------------------------------------------
    checks = {
        "block presence": list(frame.index),
        "clade colours": [k for k, v in on_tree.items() if v],
        "labels": list(tips),
        "load": list(load),
    }
    print(f"{len(tips)} tip(s) on the tree")
    for name, rows in checks.items():
        result = itol.check_tips(rows, tips)
        print(
            f"  {name:16s} {len(result['matched']):4d} matched, "
            f"{len(result['not_on_tree']):4d} would be DROPPED by iTOL, "
            f"{len(result['tips_without_data']):4d} tip(s) with no row"
        )
        if result["not_on_tree"]:
            print(
                f"    first unmatched id: {result['not_on_tree'][0]} -- iTOL "
                f"ignores these without reporting anything, so the track would "
                f"render short or empty"
            )
    missing_group = sum(1 for v in on_tree.values() if not v)
    if missing_group:
        print(
            f"  {missing_group} tip(s) have no group call and are left "
            f"uncoloured rather than given a default colour"
        )


main()
