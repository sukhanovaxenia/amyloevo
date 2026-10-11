"""The three group-wise figures.

Thin: read the feature matrix and the metadata, call `amyloevo.viz.groups`,
save. Every styling and layout decision lives in the package, where it is at
least importable and testable; a figure built inside a Snakemake script is a
figure nobody can regenerate without the workflow.

The paired-load figure needs taxa carrying every group, and there may be none --
a set with no paralogue split, or one where no genome is complete. That is not
an error, and the script writes a figure saying so rather than failing the DAG
or, worse, silently plotting an empty axis.
"""

from __future__ import annotations

import sys
from pathlib import Path

snakemake = globals()["snakemake"]
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from amyloevo.viz.groups import (  # noqa: E402
    plot_block_occupancy,
    plot_group_load,
    plot_occupancy_profiles,
)


def empty_figure(message: str):
    """A figure that states why it is empty, instead of an empty figure."""
    fig, ax = plt.subplots(figsize=(6.0, 1.6))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=8, wrap=True)
    return fig


def main() -> None:
    params = snakemake.params
    data = np.load(snakemake.input.profiles, allow_pickle=True)
    matrix = data["profiles"]
    ids = [str(x) for x in data["ids"]]
    labels = [str(x) for x in data["labels"]]
    starts = data["starts"]
    stops = data["stops"]

    blocks = pd.read_csv(snakemake.input.blocks, sep="\t")
    meta = pd.read_csv(snakemake.input.meta, sep="\t", dtype={"taxid": "string"})
    groups = pd.read_csv(snakemake.input.groups, sep="\t")

    group_of = (
        dict(zip(groups["id"], groups["group"])) if len(groups) else {}
    )
    assigned = [group_of.get(i, "all") for i in ids]
    names = sorted(set(assigned))
    counts = {g: assigned.count(g) for g in names}

    # --- occupancy along the reference ------------------------------------
    n_col = int(stops.max()) + 1 if len(stops) else 1
    occupancy: dict[str, np.ndarray] = {}
    for group in names:
        rows = [i for i, g in enumerate(assigned) if g == group]
        profile = np.zeros(n_col)
        for j, (a, b) in enumerate(zip(starts, stops)):
            profile[int(a) : int(b) + 1] = matrix[rows, j].mean()
        occupancy[group] = profile
    domains = [
        (label, int(a), int(b), True)
        for label, a, b in zip(labels, starts, stops)
    ]
    fig = plot_occupancy_profiles(
        occupancy, domains=domains, counts=counts,
        reference_label=params.reference_label,
    )
    fig.savefig(snakemake.output.occupancy, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- block heatmap -----------------------------------------------------
    fig = plot_block_occupancy(blocks, groups=names, block_labels=labels)
    fig.savefig(snakemake.output.heatmap, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- paired within-taxon load -----------------------------------------
    load = pd.DataFrame({
        "id": ids,
        "group": assigned,
        "load": matrix.mean(axis=1),
    }).merge(meta[["id", "taxid"]], on="id", how="left")
    paired = (
        load.dropna(subset=["taxid"])
        .pivot_table(index="taxid", columns="group", values="load", aggfunc="mean")
        .dropna()
    )
    if len(paired) >= 3 and paired.shape[1] >= 2:
        fig = plot_group_load(paired, order=names)
    else:
        fig = empty_figure(
            f"No paired within-taxon comparison: {len(paired)} taxon/taxa carry "
            f"all {len(names)} group(s).\nThe unpaired comparison is in "
            f"{Path(snakemake.output.heatmap).name}; pairing is what removes the "
            f"between-species component, so it is not substituted for here."
        )
    fig.savefig(snakemake.output.load, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(
        f"wrote 3 figure(s) for {len(ids)} sequence(s) in {len(names)} group(s); "
        f"paired taxa: {len(paired)}"
    )


main()
