"""APR blocks, and per-sequence occupancy over them.

This is the feature-construction step, and it matters more than any test applied
afterwards. A sequence's APR profile over a long alignment is sparse -- in the
FXR1 family each protein covers 40-90 of the 623 columns ever covered -- and
clustering or testing that vector directly measures mostly shared absence.
Average-linkage clustering on it produces one cluster holding everything plus a
few singletons, which is a chaining artefact wearing the costume of a result.

So columns are first grouped into the **blocks** the set actually uses, and each
sequence is scored by the fraction of each block it covers. That is both more
robust and directly interpretable: a cluster can then be described by which
blocks its members carry, which is a sentence about biology rather than about a
dendrogram.

**Blocks are defined per group and then merged.** Defining them on the pooled
set finds only what the majority carries, and a block present in a minority of
one group is exactly the discriminative feature a group contrast needs. The
threshold for *defining* a block is therefore deliberately lower than the one
for *calling* a conserved column -- a feature that separates groups is useful
even where it is not conserved in any of them. The two thresholds are separate
parameters for that reason, and collapsing them to one loses the distinction.

**Occupancy is computed in alignment space, from masks in residue space.** The
projection happens exactly once, here. Handing an already-projected mask to
:func:`occupancy` projects it twice and scatters the profile across the wrong
columns -- which produced 5 garbage blocks instead of 15 and dropped cluster
purity from 90% to 33% before it was caught.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Minimum fraction of a group that must carry a column for it to join a block.
#: Lower than a conservation threshold on purpose; see the module docstring.
DEFINE_MIN_OCCUPANCY = 0.35
#: Minimum fraction of a group that must have a residue at a column at all.
#: Below this the occupancy numerator rests on too few sequences to mean
#: anything, and an insertion carried by three sequences would define a block.
DEFINE_MIN_COVERAGE = 0.4
#: Blocks narrower than this are dropped. A predictor's window already blurs
#: boundaries by several residues, so a one- or two-column block is inside the
#: noise of its own call.
DEFINE_MIN_WIDTH = 3
#: Dips of up to this many columns are bridged rather than splitting a block.
#: A single column falling below threshold inside an otherwise solid run is
#: usually one indel in one sequence, not two separate regions -- and splitting
#: there produces two sub-width fragments that are then both dropped, so the
#: block disappears entirely rather than merely shifting.
DEFINE_MAX_GAP = 2
#: A group smaller than this does not get to define blocks: with four sequences
#: every threshold lands on a count of one or two.
DEFINE_MIN_GROUP = 5


@dataclass(frozen=True)
class Block:
    """One APR block in alignment coordinates, inclusive on both ends."""

    start: int
    stop: int

    @property
    def width(self) -> int:
        return self.stop - self.start + 1

    def label(self, column_to_residue=None) -> str:
        """``"91-98"`` in reference residue numbers, or ``"col1229-1239"``.

        The reference-residue form is what belongs on a figure axis and in a
        results table; the column form is the honest fallback for a block that
        falls in an insertion the reference does not have, and its ``col``
        prefix is there so the two are never confused for one another.
        """
        if column_to_residue is not None:
            a = int(column_to_residue[self.start])
            b = int(column_to_residue[self.stop])
            if a and b:
                return f"{a}-{b}"
        return f"col{self.start}-{self.stop}"


def merge_intervals(intervals) -> list[Block]:
    """Merge overlapping or abutting intervals into blocks.

    Abutting counts as overlapping: two blocks that touch were split by a
    threshold crossing a single column, which is not a boundary the data
    supports.
    """
    ordered = sorted(
        (Block(int(a), int(b)) for a, b in intervals), key=lambda x: (x.start, x.stop)
    )
    merged: list[Block] = []
    for block in ordered:
        if merged and block.start <= merged[-1].stop + 1:
            last = merged[-1]
            merged[-1] = Block(last.start, max(last.stop, block.stop))
        else:
            merged.append(block)
    return merged


def occupancy(
    column_masks: dict[str, np.ndarray], members, n_columns: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-column APR occupancy and coverage over ``members``.

    ``column_masks`` must already be in ALIGNMENT space, one boolean array of
    length ``n_columns`` per sequence id. Returns ``(occupancy, coverage)``:
    the fraction of members calling an APR at each column, and the fraction of
    members present at it. Occupancy is over members present, so a column in an
    insertion is not diluted by the sequences that lack it -- which is also why
    ``coverage`` has to be reported alongside and gated on.
    """
    members = [m for m in members if m in column_masks]
    if not members:
        return np.zeros(n_columns), np.zeros(n_columns)
    calls = np.zeros(n_columns, dtype=float)
    present = np.zeros(n_columns, dtype=float)
    for pid in members:
        mask = np.asarray(column_masks[pid], dtype=bool)
        calls[: mask.size] += mask
        present[: mask.size] += 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        occ = np.where(present > 0, calls / np.maximum(present, 1.0), 0.0)
    return occ, present / len(members)


def blocks_from_occupancy(
    occ: np.ndarray,
    coverage: np.ndarray,
    *,
    min_occupancy: float = DEFINE_MIN_OCCUPANCY,
    min_coverage: float = DEFINE_MIN_COVERAGE,
    min_width: int = DEFINE_MIN_WIDTH,
    max_gap: int = DEFINE_MAX_GAP,
) -> list[Block]:
    """Runs where occupancy and coverage both clear their thresholds.

    Dips of up to ``max_gap`` columns are bridged; see :data:`DEFINE_MAX_GAP`.
    ``min_width`` is applied after bridging, so a block is measured as the data
    present it rather than as the thresholds happened to chop it.

    ``coverage`` is a FRACTION of the members, matching :func:`occupancy`. An
    absolute count here with a fractional threshold silently passes every
    column, so the two must agree -- which is why no member count is taken as a
    parameter.
    """
    keep = (occ >= min_occupancy) & (coverage >= min_coverage)
    idx = np.flatnonzero(keep)
    if idx.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = prev = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i - prev <= max_gap + 1:
            prev = i
            continue
        runs.append((start, prev))
        start = prev = i
    runs.append((start, prev))
    return [Block(a, b) for a, b in runs if b - a + 1 >= min_width]


def define_blocks(
    column_masks: dict[str, np.ndarray],
    members_by_group: dict[str, list[str]],
    n_columns: int,
    *,
    min_occupancy: float = DEFINE_MIN_OCCUPANCY,
    min_coverage: float = DEFINE_MIN_COVERAGE,
    min_width: int = DEFINE_MIN_WIDTH,
    max_gap: int = DEFINE_MAX_GAP,
    min_group: int = DEFINE_MIN_GROUP,
) -> list[Block]:
    """Union of the blocks each group defines on its own.

    Pass ``{"all": [...]}`` for an ungrouped set; the behaviour is then the
    pooled definition, which is the right one when there are no groups and the
    wrong one when there are.
    """
    found: list[tuple[int, int]] = []
    for members in members_by_group.values():
        usable = [m for m in members if m in column_masks]
        if len(usable) < min_group:
            continue
        occ, cov = occupancy(column_masks, usable, n_columns)
        found += [
            (b.start, b.stop)
            for b in blocks_from_occupancy(
                occ, cov, min_occupancy=min_occupancy,
                min_coverage=min_coverage, min_width=min_width,
                max_gap=max_gap,
            )
        ]
    return merge_intervals(found)


def block_profiles(
    column_masks: dict[str, np.ndarray], blocks, members
) -> np.ndarray:
    """``(n_members, n_blocks)`` fraction of each block each member covers.

    This is the feature matrix for clustering and for
    :func:`amyloevo.phylo.signal.taxonomic_signal`.
    """
    blocks = list(blocks)
    out = np.zeros((len(members), len(blocks)), dtype=float)
    for i, pid in enumerate(members):
        mask = np.asarray(column_masks[pid], dtype=bool)
        for j, block in enumerate(blocks):
            window = mask[block.start : block.stop + 1]
            out[i, j] = float(window.mean()) if window.size else 0.0
    return out


def group_occupancy(
    column_masks: dict[str, np.ndarray],
    blocks,
    members_by_group: dict[str, list[str]],
) -> "object":
    """Per-group mean occupancy of each block, as a DataFrame.

    One row per block with an ``occ_<group>`` column per group -- the shape
    :func:`amyloevo.viz.groups.plot_block_occupancy` draws, and the table that
    answers "what is shared and what is private".
    """
    import pandas as pd

    blocks = list(blocks)
    data = {
        "block": [f"{b.start}-{b.stop}" for b in blocks],
        "col_start": [b.start for b in blocks],
        "col_stop": [b.stop for b in blocks],
        "width": [b.width for b in blocks],
    }
    for group, members in members_by_group.items():
        usable = [m for m in members if m in column_masks]
        if not usable:
            data[f"occ_{group}"] = [float("nan")] * len(blocks)
            continue
        profiles = block_profiles(column_masks, blocks, usable)
        data[f"occ_{group}"] = profiles.mean(axis=0).tolist()
    return pd.DataFrame(data)
