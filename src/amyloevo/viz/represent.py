"""One representative per taxonomic level, at whichever rank fits the figure.

An alignment panel is read row by row, so its useful size is set by the reader
and not by the dataset: roughly 30 to 60 rows, below which the figure wastes
its space and above which nobody follows a column across it. Reducing 372
sequences to that range means picking one sequence per taxonomic level, and the
only real decision is WHICH RANK -- which is why this module takes a row budget
and finds the rank, rather than taking a rank and hoping.

**Why the rank has to be chosen from the data and not declared.** On an
orthologue set, a genus contributes about one sequence per paralogue, so one
representative per genus removes almost nothing from within a paralogue group
while removing two thirds of the pooled set. Measured on the FXR1 family:

    scope              sequences   genus   family   order   class
    all                      372     134      105      66      25
    FXR1                     116      96       72      42       7
    FMR1                     101      92       69      40       7
    FXR2                      97      85       61      35       5
    pre-duplication           58      38       33      24      18

Genus takes FXR1 from 116 to 96 -- still twice a readable panel -- while order
takes it to 42. The pooled 64% reduction at genus level is almost entirely the
same genus appearing once per paralogue, and a per-group figure never sees it.
So a rank that looks obviously right for the set is the wrong one for the
figure, and the budget is what tells them apart.

**The representative must not be chosen by anything the figure displays.**
Picking the sequence with the most predicted APRs, the best support, or the
longest run of the feature being drawn makes the panel select on its own
variable, and the result looks cleaner than the data in a way the reader cannot
see. The default here is alignment completeness, which is a property of the
record rather than of the biology, and :func:`format_selection_report` prints
the column that was used so the figure caption can carry it. Choosing by
completeness is not neutral either -- it prefers well-annotated genomes -- but
it is at least orthogonal to what an APR figure shows, and stating it lets a
reader discount it.

Ties are broken by sorted identifier, so the same metadata always yields the
same panel.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Ranks from finest to coarsest. The walk stops at the first one that fits the
#: budget, so the order is the whole algorithm.
RANKS = ("genus", "family", "order", "class", "phylum")

#: Rows an alignment panel can carry and still be read. The default target is
#: the middle of it rather than the top: a panel at the limit has no room for
#: the extra row a reviewer asks for.
READABLE_ROWS = (30, 60)
TARGET_ROWS = 45

#: Levels with no value at a rank are pooled under this label rather than
#: dropped. An unresolved taxon is still a sequence, and dropping it would
#: quietly remove whichever clade the taxonomy is worst for -- usually the
#: invertebrates, which is the group most of the interest is in.
UNRESOLVED = "(unresolved)"


@dataclass
class RankChoice:
    """Which rank was picked for a row budget, and what it beat."""

    rank: str
    n_levels: int
    fits: bool
    target: int
    counts: dict[str, int] = field(default_factory=dict)
    rejected: list[tuple[str, int]] = field(default_factory=list)
    non_monotonic: list[str] = field(default_factory=list)


@dataclass
class Selection:
    """The chosen representatives and how they were chosen."""

    ids: list[str] = field(default_factory=list)
    rank: str = ""
    n_levels: int = 0
    n_input: int = 0
    score_column: str = ""
    fits: bool = True
    target: int = TARGET_ROWS
    counts: dict[str, int] = field(default_factory=dict)
    rejected: list[tuple[str, int]] = field(default_factory=list)
    non_monotonic: list[str] = field(default_factory=list)
    unresolved_members: int = 0
    ties_broken: int = 0


def _rows(metadata, *, id_column: str, members=None) -> list[int]:
    """Positions of the rows to consider, in metadata order."""
    ids = [str(i) for i in metadata[id_column]]
    if members is None:
        return list(range(len(ids)))
    keep = {str(m) for m in members}
    return [i for i, pid in enumerate(ids) if pid in keep]


def _level(metadata, rank: str, position: int) -> str:
    value = metadata[rank]
    try:
        raw = value.iloc[position]
    except AttributeError:
        raw = value[position]
    text = "" if raw is None else str(raw).strip()
    return text if text and text.lower() != "nan" else UNRESOLVED


def level_counts(metadata, ranks=RANKS, *, id_column: str = "id",
                 members=None) -> dict[str, int]:
    """Distinct levels at each rank, over the rows considered."""
    positions = _rows(metadata, id_column=id_column, members=members)
    available = [r for r in ranks if r in getattr(metadata, "columns", [])]
    return {
        rank: len({_level(metadata, rank, p) for p in positions})
        for rank in available
    }


def choose_rank(metadata, *, target: int = TARGET_ROWS, ranks=RANKS,
                id_column: str = "id", members=None) -> RankChoice:
    """The finest rank whose level count fits ``target``.

    Walking finest to coarsest and stopping at the first fit gives maximum
    taxonomic resolution inside the budget, which is the right objective: a
    panel of 42 orders says more than the same panel built from 7 classes.

    When even the coarsest rank has more levels than the budget, the coarsest
    is returned with ``fits=False`` rather than some further reduction being
    invented. Dropping levels to reach a row count would mean choosing which
    clades the figure omits, and nothing in a row budget decides that.

    Level counts should fall as the rank coarsens, because taxonomy nests. A
    rank that breaks this is reported in ``non_monotonic`` instead of being
    used: it means a level at the finer rank maps to more than one level at the
    coarser one, which is a metadata error -- the same genus filed under two
    families, say -- and a figure built on it would group sequences wrongly.
    """
    counts = level_counts(metadata, ranks, id_column=id_column,
                          members=members)
    if not counts:
        return RankChoice(rank="", n_levels=0, fits=False, target=target)

    ordered = [r for r in ranks if r in counts]
    non_monotonic = [
        ordered[i] for i in range(1, len(ordered))
        if counts[ordered[i]] > counts[ordered[i - 1]]
    ]
    usable = [r for r in ordered if r not in non_monotonic]

    rejected = []
    for rank in usable:
        if counts[rank] <= target:
            return RankChoice(rank=rank, n_levels=counts[rank], fits=True,
                              target=target, counts=counts, rejected=rejected,
                              non_monotonic=non_monotonic)
        rejected.append((rank, counts[rank]))

    coarsest = usable[-1] if usable else ordered[-1]
    return RankChoice(rank=coarsest, n_levels=counts[coarsest], fits=False,
                      target=target, counts=counts,
                      rejected=rejected[:-1] if rejected else [],
                      non_monotonic=non_monotonic)


def representatives(metadata, rank: str, *, score_column: str = "core_coverage",
                    id_column: str = "id", members=None
                    ) -> tuple[dict[str, str], int]:
    """One id per level of ``rank``: the highest ``score_column``, ties by id.

    Returns the mapping and how many levels were decided by the tie-break,
    because that count is the honest measure of how arbitrary the panel is. A
    level whose members all score identically -- which happens when the score
    is a coarse completeness figure -- is represented by whichever id sorts
    first, and that is a choice the figure should admit to rather than hide.

    ``score_column`` must not be derived from what the figure displays; see the
    module docstring. It is a parameter and not a constant so that the caller
    states it, and :func:`format_selection_report` prints it back.
    """
    positions = _rows(metadata, id_column=id_column, members=members)
    ids = [str(i) for i in metadata[id_column]]

    scores: list[float] = []
    has_score = score_column in getattr(metadata, "columns", [])
    for position in positions:
        if not has_score:
            scores.append(0.0)
            continue
        column = metadata[score_column]
        try:
            raw = column.iloc[position]
        except AttributeError:
            raw = column[position]
        try:
            scores.append(float(raw))
        except (TypeError, ValueError):
            scores.append(float("-inf"))

    grouped: dict[str, list[tuple[float, str]]] = {}
    for score, position in zip(scores, positions):
        grouped.setdefault(_level(metadata, rank, position), []).append(
            (score, ids[position]))

    chosen, ties = {}, 0
    for level, candidates in grouped.items():
        best = max(score for score, _ in candidates)
        tied = sorted(pid for score, pid in candidates if score == best)
        if len(tied) > 1:
            ties += 1
        chosen[level] = tied[0]
    return chosen, ties


def select_for_display(metadata, *, target: int = TARGET_ROWS, ranks=RANKS,
                       score_column: str = "core_coverage",
                       id_column: str = "id", members=None) -> Selection:
    """Pick the rank for the budget, then one representative per level."""
    positions = _rows(metadata, id_column=id_column, members=members)
    choice = choose_rank(metadata, target=target, ranks=ranks,
                         id_column=id_column, members=members)
    if not choice.rank:
        return Selection(n_input=len(positions), target=target, fits=False)

    chosen, ties = representatives(
        metadata, choice.rank, score_column=score_column,
        id_column=id_column, members=members)
    unresolved = sum(
        1 for p in positions if _level(metadata, choice.rank, p) == UNRESOLVED
    )
    return Selection(
        ids=[chosen[level] for level in sorted(chosen)],
        rank=choice.rank,
        n_levels=choice.n_levels,
        n_input=len(positions),
        score_column=score_column if score_column in
        getattr(metadata, "columns", []) else f"{score_column} (ABSENT)",
        fits=choice.fits,
        target=target,
        counts=choice.counts,
        rejected=choice.rejected,
        non_monotonic=choice.non_monotonic,
        unresolved_members=unresolved,
        ties_broken=ties,
    )


def format_selection_report(selection: Selection) -> str:
    """The selection as text, including the line a figure caption needs."""
    if not selection.rank:
        return ("no usable taxonomic rank in the metadata, so no "
                "representatives were chosen")

    low, high = READABLE_ROWS
    lines = [
        f"{selection.n_input} sequence(s) -> {len(selection.ids)} row(s), "
        f"one per {selection.rank} "
        f"({1 - len(selection.ids) / selection.n_input:.0%} fewer)",
    ]
    if selection.counts:
        lines.append("  levels per rank: " + ", ".join(
            f"{rank} {count}" for rank, count in selection.counts.items()))
    for rank, count in selection.rejected:
        lines.append(f"  {rank} rejected: {count} levels, over the budget of "
                     f"{selection.target}")
    if not selection.fits:
        lines.append(
            f"  !! even {selection.rank} leaves {selection.n_levels} rows, "
            f"over the budget of {selection.target}. Nothing was dropped to "
            f"reach it:\n     choosing which clades a figure omits is not a "
            f"decision a row count can make.\n     Either raise the budget, "
            f"split the panel, or pass a coarser rank list."
        )
    elif not low <= len(selection.ids) <= high:
        lines.append(f"  note: {len(selection.ids)} rows is outside the "
                     f"{low}-{high} a panel reads well at")
    if selection.non_monotonic:
        lines.append(
            f"  !! {', '.join(selection.non_monotonic)} has MORE levels than "
            f"the finer rank above it, so the\n     taxonomy does not nest "
            f"there -- one finer level maps to several coarser ones.\n     "
            f"Those ranks were not used. Fix the metadata before trusting any "
            f"grouping by them."
        )
    if selection.unresolved_members:
        lines.append(f"  {selection.unresolved_members} sequence(s) have no "
                     f"{selection.rank}; they are pooled as {UNRESOLVED} and "
                     f"share one row")
    if selection.ties_broken:
        lines.append(f"  {selection.ties_broken} level(s) had a tie on "
                     f"{selection.score_column} and were decided by sorted id")
    lines += [
        "",
        f"  caption: one sequence per {selection.rank}, the one with the "
        f"highest {selection.score_column};",
        "  ties by identifier. The representative is NOT chosen by any "
        "quantity this figure",
        "  displays, which would make the panel select on its own variable.",
    ]
    return "\n".join(lines)


__all__ = [
    "RANKS",
    "READABLE_ROWS",
    "RankChoice",
    "Selection",
    "TARGET_ROWS",
    "UNRESOLVED",
    "choose_rank",
    "format_selection_report",
    "level_counts",
    "representatives",
    "select_for_display",
]
