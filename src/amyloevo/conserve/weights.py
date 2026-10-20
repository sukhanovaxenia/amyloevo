"""Sequence weights, for the sums that treat near-identical sequences as data.

Every per-column quantity in this package -- occupancy, coverage, the block
definitions that follow from them -- is a sum over sequences, and an orthologue
set is not a sample of independent sequences. The FXR1 family is 372 sequences
over 66 orders, of which 217 are mammals; weight every order equally and the
effective sample size is 147. The unweighted numbers count those 217 as 217
observations, so a block carried by densely sampled mammals and nobody else
clears a family-level occupancy threshold it should not clear.

**Weighting is not clustering, and the difference is the whole point.**
Clustering removes sequences, and the sequences it removes are precisely the
densely sampled clades a gene tree needs in order to resolve them. Weighting
removes nothing: the alignment is untouched, every sequence keeps its row,
every tip keeps its branch length, and the only thing that changes is what each
sequence contributes to a per-column average. The design's rule that redundancy
is corrected ONCE applies to the pair -- cluster or weight, never both.

**What changes, and what should.** Weighting shifts which columns pass the
occupancy threshold, so the block list changes. That is the correction working.
A block present in eighty near-identical mammals and nowhere else was never a
property of the family; down-weighting makes it fall out of the family-level
list, where it belongs in the per-clade view instead. Nothing about the tree,
the divergences or the per-sequence profiles moves.

Three schemes, because none of them is objective and the choice has to be
stated:

:func:`uniform`
    No correction. The honest baseline to compare against, and the right
    choice for a set that really is one sequence per taxon.

:func:`by_rank`
    Equal total weight per taxonomic level -- every order counts once,
    however many sequences it contributed. Transparent and easy to state in a
    methods section, and crude: orders are not of equal depth, so this
    over-corrects a species-rich recent radiation and under-corrects an old
    sparse one.

:func:`henikoff`
    Position-based weights from the alignment itself (Henikoff & Henikoff
    1994), the scheme HMMER and the BLOSUM matrices use. It needs no taxonomy
    and no tree, which is its advantage here: a tree-based weight would be
    derived from the same gene tree that was inferred from the same biased
    sample, so the correction and the thing it corrects would share their
    error.

All three return weights with mean 1, so a threshold keeps the meaning it has
in the unweighted run and the two are comparable without rescaling.
"""

from __future__ import annotations

import numpy as np

GAPS = ("-", ".")


def _normalise(weights: dict[str, float]) -> dict[str, float]:
    """Rescale to mean 1, so thresholds mean the same as unweighted."""
    if not weights:
        return weights
    total = sum(weights.values())
    if total <= 0:
        return {k: 1.0 for k in weights}
    factor = len(weights) / total
    return {k: v * factor for k, v in weights.items()}


def uniform(ids) -> dict[str, float]:
    """Weight 1 for every sequence: the uncorrected baseline."""
    return {str(i): 1.0 for i in ids}


def by_rank(metadata, rank: str = "order", *, id_column: str = "id") -> dict[str, float]:
    """Equal total weight per level of ``rank``.

    A level with forty sequences gives each of them 1/40 before normalisation,
    so forty rodents count as much as one sponge. Sequences with no value at
    that rank are pooled into one level rather than dropped -- an unresolved
    taxon is still a sequence, and silently giving it weight 1 would make
    missing taxonomy the heaviest category in the set.
    """
    ids = [str(i) for i in metadata[id_column]]
    levels = [str(v) if v and str(v).strip() else "(unresolved)"
              for v in metadata[rank]]
    sizes: dict[str, int] = {}
    for level in levels:
        sizes[level] = sizes.get(level, 0) + 1
    return _normalise({i: 1.0 / sizes[l] for i, l in zip(ids, levels)})


def henikoff(alignment) -> dict[str, float]:
    """Position-based weights (Henikoff & Henikoff 1994).

    For each column, a residue shared by many sequences is worth little and a
    rare one is worth much: a sequence collects ``1 / (r * s)`` from each
    column, where ``r`` is the number of distinct residues in the column and
    ``s`` how many sequences carry that sequence's residue. Identical
    sequences therefore split one sequence's worth of weight between them,
    which is the behaviour wanted here.

    Gaps are not treated as a residue, and the total is divided by the number
    of columns where the sequence HAS a residue. Both choices separate
    redundancy from incompleteness: counting gap as a symbol would hand a
    mostly-gap row a large weight for being unlike everything else, and not
    dividing would hand it a small one for being short. Completeness is
    already handled, separately, by the coverage term in
    ``conserve.blocks.occupancy``.
    """
    ids = [str(k) for k in alignment]
    if not ids:
        return {}
    matrix = np.array([list(str(alignment[i]).upper()) for i in ids])
    is_gap = np.isin(matrix, [g.upper() for g in GAPS])

    totals = np.zeros(len(ids))
    for column in range(matrix.shape[1]):
        residues = matrix[:, column]
        present = ~is_gap[:, column]
        if not present.any():
            continue
        values, counts = np.unique(residues[present], return_counts=True)
        r = len(values)
        counts_by_value = dict(zip(values, counts))
        for row in np.flatnonzero(present):
            totals[row] += 1.0 / (r * counts_by_value[residues[row]])

    n_columns = (~is_gap).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        per_column = np.where(n_columns > 0, totals / np.maximum(n_columns, 1), 0.0)
    if not per_column.any():
        return uniform(ids)
    return _normalise(dict(zip(ids, per_column)))


def summarise(weights: dict[str, float], metadata=None, rank: str = "order") -> str:
    """What the weighting did, in the terms a methods section needs.

    The effective sample size is ``(sum w)^2 / sum w^2`` -- the number of
    equally weighted sequences carrying the same information. Quoting it beside
    the raw count is the honest way to report a weighted analysis: 372
    sequences at an effective 147 is a different claim from 372 sequences.

    The per-level totals need one guard. Under :func:`by_rank` at the same
    rank, every level sums to ``n / k`` by construction, so a list of the five
    heaviest levels is the same number printed five times and reads as a
    ranking that is not there. That case is named instead of listed.
    """
    if not weights:
        return "no weights"
    w = np.array(list(weights.values()), dtype=float)
    ess = w.sum() ** 2 / (w ** 2).sum()
    heaviest = max(weights.items(), key=lambda kv: kv[1])
    lines = [
        f"{len(w)} sequence(s); weights mean {w.mean():.3f}, "
        f"range {w.min():.3f}-{w.max():.3f}",
        f"effective sample size {ess:.1f} "
        f"({ess / len(w):.0%} of the raw count)",
        f"heaviest single sequence {heaviest[0]} at {heaviest[1]:.2f}, "
        f"{heaviest[1] / w.sum():.1%} of all weight",
    ]
    if metadata is not None and rank in getattr(metadata, "columns", []):
        by_level: dict[str, float] = {}
        for pid, level in zip(metadata["id"], metadata[rank]):
            key = str(level) if level and str(level).strip() else "(unresolved)"
            by_level[key] = by_level.get(key, 0.0) + weights.get(str(pid), 0.0)
        totals = np.array(list(by_level.values()), dtype=float)
        spread = (totals.max() - totals.min()) / totals.mean() if totals.mean() else 0.0
        if spread < 1e-6:
            lines.append(
                f"weight per {rank} is equal by construction: "
                f"{totals.mean():.2f} each across {len(by_level)} {rank}s, so "
                f"there is no ranking of {rank}s to read here"
            )
        else:
            top = sorted(by_level.items(), key=lambda kv: -kv[1])[:5]
            lines.append(
                f"heaviest {rank}s (share of total weight): " + ", ".join(
                    f"{k} {v / totals.sum():.1%}" for k, v in top)
            )
    return "\n".join(lines)


__all__ = ["by_rank", "henikoff", "summarise", "uniform"]
