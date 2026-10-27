"""Substitution rate inside a set of regions, against the rest of the alignment.

``rates`` answers which CLADE accumulated more substitutions. This answers a
different question on the same tree: whether a set of REGIONS -- APR blocks,
a domain, any column set -- evolves at a different rate from the rest of the
protein. The two are independent. A paralogue can be the slower clade overall
and still hold its amyloidogenic core more tightly than its own background.

**Why the comparison is within-sequence.** Every column is seen through the
same tree, the same taxa and the same elapsed time, so none of those can
explain a difference between columns. That is the whole reason to ask the
question this way round: a rate contrast between regions needs no calibration,
no contemporaneity argument and no outgroup. It is the one evolutionary
statement a gene family makes about its own parts without borrowing anything.

**Why the null relocates whole blocks.** Rates are autocorrelated along a
protein -- a buried helix is slow for its whole length -- so shuffling single
columns tests against a null that no real region could satisfy and returns a
significant result for almost any region set. Relocating the blocks intact,
keeping their widths and their number, tests the question actually being asked:
would regions of these sizes, placed anywhere in this protein, look like this?
The autocorrelation is then present under the null as well as in the data, and
only the POSITION is random.

**Blocks are alignment columns, and labels are not blocks.**
``conserve.blocks.Block`` is defined in alignment coordinates; its ``label()``
renders reference-residue numbers for display and falls back to a ``col``-
prefixed form for blocks the reference does not cover. A label set therefore
mixes two coordinate systems, and parsing it back into columns silently
misplaces every block of one kind or the other. Nothing here accepts a label:
the functions take ``Block`` objects or ``(start, stop)`` pairs, and a string
is refused rather than guessed at.

**No circularity.** APR blocks are defined by predictor occupancy, not by
conservation. Were they defined by conservation, a rate contrast would be a
restatement of the definition rather than a finding, and this module would be
measuring its own input.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Permutations for the placement test. Enough for a two-sided p-value to
#: resolve 0.001; the test is cheap, so the default is not stingy.
PERMUTATIONS = 10_000


@dataclass(frozen=True)
class RegionRates:
    """Mean and median rate inside the regions and outside them."""

    n_columns: int
    n_region: int
    n_background: int
    mean_region: float
    mean_background: float
    median_region: float
    median_background: float

    @property
    def ratio(self) -> float:
        """Region mean over background mean. Below 1 is the constrained case."""
        if self.mean_background == 0:
            return float("nan")
        return self.mean_region / self.mean_background


@dataclass(frozen=True)
class PlacementTest:
    """The observed ratio against blocks of the same widths placed at random."""

    observed: float
    p_value: float
    permutations: int
    null_median: float
    null_lo: float
    null_hi: float
    widths: tuple[int, ...]
    n_block_columns: int = 0
    observed_usable: int = 0
    outside_allowed: int = 0


def _as_spans(blocks) -> list[tuple[int, int]]:
    """Accept Block objects or (start, stop) pairs; refuse labels."""
    spans: list[tuple[int, int]] = []
    for item in blocks:
        if isinstance(item, str):
            raise TypeError(
                f"{item!r} is a label, not a block. Labels mix reference-residue "
                f"and column coordinates ('91-99' against 'col1615-1620'), so "
                f"parsing them back into columns misplaces one kind or the "
                f"other. Pass Block objects, or (start, stop) column pairs."
            )
        start = getattr(item, "start", None)
        stop = getattr(item, "stop", None)
        if start is None or stop is None:
            try:
                start, stop = item
            except (TypeError, ValueError):
                raise TypeError(
                    f"cannot read a column span from {item!r}"
                ) from None
        start, stop = int(start), int(stop)
        if stop < start:
            raise ValueError(f"block {start}-{stop} ends before it starts")
        spans.append((start, stop))
    return spans


def column_mask(blocks, n_columns: int) -> np.ndarray:
    """Boolean mask over alignment columns, True inside any block.

    Spans are inclusive on both ends, as ``Block`` is, and zero-based, as
    ``conserve.blocks`` builds them. Overlapping blocks are fine: the mask is a
    union, so an overlap counts a column once however many blocks cover it.
    """
    if n_columns <= 0:
        raise ValueError(f"n_columns={n_columns}")
    mask = np.zeros(int(n_columns), dtype=bool)
    for start, stop in _as_spans(blocks):
        if start < 0 or stop >= n_columns:
            raise ValueError(
                f"block {start}-{stop} falls outside an alignment of "
                f"{n_columns} column(s). A block set from a different "
                f"alignment than the rates is the one error this cannot "
                f"detect from the numbers alone, so it is refused here."
            )
        mask[start : stop + 1] = True
    return mask


def read_site_rates(path, *, n_columns: int | None = None) -> np.ndarray:
    """Per-site rates from an IQ-TREE ``.rate`` file, in column order.

    Columns are found BY NAME from the header line, not by position: the file
    carries ``Site``, ``Rate`` and sometimes ``Cat``/``C_Rate``, and the set
    varies with the model. Reading the second field positionally works until a
    run uses a model that writes a different column there.

    ``n_columns`` is checked when given, and that check is the point of the
    argument. A rate file from a different alignment parses perfectly and
    produces a confident, meaningless contrast; the only thing that catches it
    is the site count.
    """
    text = Path(path).read_text()
    header: list[str] | None = None
    sites: list[int] = []
    values: list[float] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = re.split(r"\s+", line)
        if header is None:
            if fields[0].lower() != "site":
                raise ValueError(
                    f"{path}: expected a header line beginning with 'Site', "
                    f"got {raw[:60]!r}"
                )
            header = [f.lower() for f in fields]
            if "rate" not in header:
                raise ValueError(f"{path}: no 'Rate' column; header is {fields}")
            continue
        if len(fields) < len(header):
            raise ValueError(f"{path}: short row {raw[:60]!r}")
        sites.append(int(fields[header.index("site")]))
        values.append(float(fields[header.index("rate")]))
    if header is None or not values:
        raise ValueError(f"{path}: no rate rows")

    order = np.argsort(np.asarray(sites))
    rates = np.asarray(values, dtype=float)[order]
    ordered_sites = np.asarray(sites)[order]
    expected = np.arange(ordered_sites[0], ordered_sites[0] + len(ordered_sites))
    if not np.array_equal(ordered_sites, expected):
        raise ValueError(
            f"{path}: site ids are not a contiguous run "
            f"({ordered_sites[0]}..{ordered_sites[-1]} with "
            f"{len(ordered_sites)} rows)"
        )
    if n_columns is not None and len(rates) != n_columns:
        raise ValueError(
            f"{path} has {len(rates)} site(s) but the alignment has "
            f"{n_columns} column(s). These are different alignments, and a "
            f"contrast computed across them would be meaningless while "
            f"looking entirely normal."
        )
    return rates


def occupancy(alignment) -> np.ndarray:
    """Fraction of sequences with a residue at each column.

    An untrimmed alignment carries columns held by a handful of sequences, and
    a site rate estimated from a handful of tips is noise wearing the same
    units as a rate estimated from hundreds. Those columns have to be excluded
    from BOTH sides of a contrast or from neither -- excluding them from one
    side is how a filter becomes a thumb on the scale -- which is what the
    ``usable`` argument below is for.

    ``alignment`` is a mapping of name to aligned sequence, or a FASTA path.
    """
    if isinstance(alignment, (str, Path)):
        seqs: list[str] = []
        current: list[str] = []
        for line in Path(alignment).read_text().splitlines():
            if line.startswith(">"):
                if current:
                    seqs.append("".join(current))
                current = []
            else:
                current.append(line.strip())
        if current:
            seqs.append("".join(current))
    else:
        seqs = [str(v) for v in alignment.values()]
    if not seqs:
        raise ValueError(f"{alignment}: no sequences")
    widths = {len(s) for s in seqs}
    if len(widths) != 1:
        raise ValueError(
            f"not an alignment: {len(widths)} different lengths, "
            f"{min(widths)}..{max(widths)}"
        )
    arr = np.frombuffer("".join(seqs).encode("ascii"), dtype="S1")
    arr = arr.reshape(len(seqs), widths.pop())
    filled = ~np.isin(arr, [b"-", b".", b"~", b" "])
    return filled.mean(axis=0)


def _usable_mask(usable, n: int) -> np.ndarray:
    if usable is None:
        return np.ones(n, dtype=bool)
    out = np.asarray(usable, dtype=bool)
    if out.shape != (n,):
        raise ValueError(f"usable has shape {out.shape}, expected ({n},)")
    return out


def contrast(rates, mask, *, usable=None) -> RegionRates:
    """Rate summary inside the mask and outside it.

    ``usable`` restricts the comparison to columns worth comparing -- those
    with enough occupancy for their rate to mean anything. It is applied to
    both sides, which is the only way a filter can be applied without
    deciding the answer.
    """
    rates = np.asarray(rates, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if rates.shape != mask.shape:
        raise ValueError(f"rates {rates.shape} and mask {mask.shape} differ")
    keep = _usable_mask(usable, rates.size)
    inside, outside = rates[mask & keep], rates[~mask & keep]
    if inside.size == 0 or outside.size == 0:
        raise ValueError(
            f"{inside.size} column(s) in the regions and {outside.size} "
            f"outside: a contrast needs both sides"
        )
    return RegionRates(
        n_columns=int(rates.size),
        n_region=int(inside.size),
        n_background=int(outside.size),
        mean_region=float(inside.mean()),
        mean_background=float(outside.mean()),
        median_region=float(np.median(inside)),
        median_background=float(np.median(outside)),
    )


def place_blocks(widths, n_columns: int, rng, *, within=None,
                 max_tries: int = 200) -> list[tuple[int, int]] | None:
    """Lay blocks of these widths at random, non-overlapping and in bounds.

    ``within`` is the set of columns a block could actually have been defined
    in, and it matters more than it looks. A block is only defined where
    coverage reaches a threshold (``conserve.blocks.DEFINE_MIN_COVERAGE``), so
    the real blocks sit in well-filled columns BY CONSTRUCTION. Relocating them
    across the whole alignment therefore compares them against arrangements the
    defining procedure could never have produced: most such placements land
    largely in sparse columns, their rate is averaged over a handful of
    estimable sites, and the null comes out far noisier than the observation it
    is being compared with. The test is then conservative for a reason that has
    nothing to do with the biology.

    Returns None when the blocks could not be placed within the allowed
    columns in ``max_tries`` attempts, which the caller counts rather than
    retries forever.

    Without ``within`` the slack is split at random into the gaps before,
    between and after the blocks. Blocks may abut; that is a legitimate
    placement, and forbidding it would bias the null towards spread-out
    arrangements.
    """
    widths = [int(w) for w in widths]
    if any(w <= 0 for w in widths):
        raise ValueError(f"widths must be positive: {widths}")
    total = sum(widths)
    n_columns = int(n_columns)
    if total > n_columns:
        raise ValueError(
            f"blocks total {total} column(s) but the alignment has "
            f"{n_columns}: they cannot be placed without overlapping"
        )

    if within is None:
        free = n_columns - total
        order = rng.permutation(len(widths))
        breaks = np.sort(rng.integers(0, free + 1, size=len(widths)))
        spans: list[tuple[int, int]] = []
        used = 0
        for i, idx in enumerate(order):
            w = widths[idx]
            start = int(breaks[i]) + used
            spans.append((start, start + w - 1))
            used += w
        return spans

    allowed = np.asarray(within, dtype=bool)
    if allowed.shape != (n_columns,):
        raise ValueError(f"within has shape {allowed.shape}, "
                         f"expected ({n_columns},)")
    # starts[w] lists every position where a run of w allowed columns begins
    runs = np.cumsum(allowed) - np.cumsum(np.concatenate(([0], allowed[:-1])))
    starts_for: dict[int, np.ndarray] = {}
    for w in set(widths):
        ok = np.ones(n_columns - w + 1, dtype=bool)
        for off in range(w):
            ok &= allowed[off : n_columns - w + 1 + off]
        starts_for[w] = np.flatnonzero(ok)
        if starts_for[w].size == 0:
            raise ValueError(
                f"no run of {w} allowed column(s) anywhere: a block of that "
                f"width could not have been defined, yet one was"
            )

    for _ in range(max_tries):
        taken = np.zeros(n_columns, dtype=bool)
        spans = []
        for idx in rng.permutation(len(widths)):
            w = widths[idx]
            cands = starts_for[w]
            placed = False
            for _ in range(32):
                s = int(cands[rng.integers(0, cands.size)])
                if not taken[s : s + w].any():
                    taken[s : s + w] = True
                    spans.append((s, s + w - 1))
                    placed = True
                    break
            if not placed:
                break
        if len(spans) == len(widths):
            return spans
    return None


def placement_test(rates, blocks, *, permutations: int = PERMUTATIONS,
                   rng=None, usable=None, min_usable: int = 5,
                   restrict: bool = True, within=None) -> PlacementTest:
    """Is the region/background ratio unusual for blocks of these widths?

    Two-sided: a region that evolves FASTER than its background is as much a
    finding as one that evolves slower, and deciding the direction after seeing
    the data is how a one-sided test becomes a one-and-a-half-sided one.

    The p-value counts the observed value among the null values, with the
    observation itself included on both sides -- so the smallest attainable
    p-value is ``1 / (permutations + 1)`` and never 0. A p of 0 would claim the
    test resolved something it did not.
    """
    rates = np.asarray(rates, dtype=float)
    spans = _as_spans(blocks)
    widths = [stop - start + 1 for start, stop in spans]
    mask = column_mask(spans, rates.size)
    keep = _usable_mask(usable, rates.size)
    observed = contrast(rates, mask, usable=keep).ratio
    rng = np.random.default_rng() if rng is None else rng

    # A placement has to rest on about as many usable columns as the real
    # blocks do, or the null is noisier than the observation it is compared
    # with and the test is conservative for a reason that has nothing to do
    # with biology. Blocks are defined only where coverage is sufficient, so
    # the real ones sit on well-filled columns; scattered at random over an
    # alignment that is mostly sparse, most placements rest on a quarter as
    # many estimable sites and the null fans out accordingly.
    #
    # Matching the usable COUNT instead was tried and is not available: random
    # placements cannot be as well covered as the real blocks, because the real
    # ones sit in the occupied core by construction, so the requirement rejects
    # essentially every placement. Confining the geometry to that core is the
    # comparison that exists -- "among regions of the protein that are
    # alignable across the family, are these the slow ones?".
    #
    # The observed blocks need not lie WHOLLY inside the allowed columns, and
    # how far they do is reported rather than enforced: shrinking the null to
    # match would answer a narrower question than the one asked.
    allowed = within if within is not None else (keep if restrict else None)
    observed_inside = int((mask & keep).sum())
    n_block_columns = int(mask.sum())
    outside_allowed = (0 if allowed is None
                       else int((mask & ~np.asarray(allowed, dtype=bool)).sum()))
    null = np.empty(permutations, dtype=float)
    thin = 0
    for i in range(permutations):
        placed = place_blocks(widths, rates.size, rng, within=allowed)
        if placed is None:
            null[i] = np.nan
            thin += 1
            continue
        m = column_mask(placed, rates.size)
        inside = rates[m & keep]
        outside = rates[~m & keep]
        if inside.size < min_usable or outside.size == 0 or outside.mean() == 0:
            null[i] = np.nan
            thin += 1
            continue
        null[i] = inside.mean() / outside.mean()

    good = null[~np.isnan(null)]
    if good.size < permutations // 2:
        raise ValueError(
            f"only {good.size} of {permutations} placements worked "
            f"({thin} rejected). The allowed columns are too sparse or too "
            f"fragmented to place blocks of these widths; lower "
            f"--min-occupancy, or pass restrict=False and read the much wider "
            f"null for what it is."
        )
    good = good[good > 0]

    # A ratio is bounded below by zero and unbounded above, so its null is
    # skewed and |observed - median| treats a halving as milder than a
    # doubling. On the log scale the two are the same distance, which is what
    # "two-sided" has to mean for a ratio. Measured on the raw scale an
    # observation sitting exactly at the 2.5th percentile of the null can come
    # back at p = 0.16, because the long upper tail supplies the comparisons.
    logs = np.log(good)
    centre_log = float(np.median(logs))
    obs_log = float(np.log(observed)) if observed > 0 else float("-inf")
    extreme = int(np.sum(np.abs(logs - centre_log) >= abs(obs_log - centre_log)))
    p = (extreme + 1) / (good.size + 1)
    return PlacementTest(
        observed=float(observed),
        p_value=float(p),
        permutations=int(good.size),
        null_median=float(np.exp(centre_log)),
        null_lo=float(np.percentile(good, 2.5)),
        null_hi=float(np.percentile(good, 97.5)),
        widths=tuple(widths),
        n_block_columns=n_block_columns,
        observed_usable=observed_inside,
        outside_allowed=outside_allowed,
    )


def partition_text(blocks, n_columns: int, *, model: str = "JTT+R4",
                   region_name: str = "apr",
                   background_name: str = "background") -> str:
    """A NEXUS partition putting the regions and the rest in separate sets.

    The permutation test asks whether the rates LOOK different; a partitioned
    model asks the likelihood to estimate how different, as one rate multiplier
    per set with the tree held fixed. They answer the same question from
    opposite directions, and agreement between them is worth more than either.

    Column numbers are written one-based and inclusive, as NEXUS wants, from
    the zero-based spans ``conserve.blocks`` produces. That conversion is the
    single place an off-by-one can enter this file, which is why it happens
    here and not in the caller.
    """
    mask = column_mask(blocks, n_columns)

    def runs(flag: bool) -> list[str]:
        out, start = [], None
        for i in range(n_columns):
            if mask[i] == flag and start is None:
                start = i
            elif mask[i] != flag and start is not None:
                out.append(f"{start + 1}-{i}")
                start = None
        if start is not None:
            out.append(f"{start + 1}-{n_columns}")
        return out

    region, background = runs(True), runs(False)
    if not region or not background:
        raise ValueError("a partition needs columns on both sides")
    return (
        "#nexus\n"
        "begin sets;\n"
        f"  charset {region_name} = {' '.join(region)};\n"
        f"  charset {background_name} = {' '.join(background)};\n"
        f"  charpartition regions = {model}:{region_name}, "
        f"{model}:{background_name};\n"
        "end;\n"
    )


def format_region_report(summary: RegionRates, test: PlacementTest | None = None,
                         *, name: str = "regions") -> str:
    """One readable block, with the direction spelled out rather than implied."""
    lines = [
        f"{name}: {summary.n_region} column(s) of {summary.n_columns}, "
        f"background {summary.n_background}",
        f"  mean rate   {summary.mean_region:.4f} in {name}, "
        f"{summary.mean_background:.4f} outside",
        f"  median rate {summary.median_region:.4f} in {name}, "
        f"{summary.median_background:.4f} outside",
        f"  ratio       {summary.ratio:.4f}",
    ]
    if summary.ratio < 1:
        lines.append(f"  -> {name} evolve SLOWER than the background")
    elif summary.ratio > 1:
        lines.append(f"  -> {name} evolve FASTER than the background")
    else:
        lines.append(f"  -> {name} and background are indistinguishable")
    if test is not None:
        lines += [
            f"  placement test over {test.permutations} relocation(s) of "
            f"{len(test.widths)} block(s)",
            f"    null ratio median {test.null_median:.4f}, "
            f"95% of null between {test.null_lo:.4f} and {test.null_hi:.4f}",
            f"    p = {test.p_value:.4g} (two-sided, on the log of the ratio)",
        ]
        if test.outside_allowed:
            lines.append(
                f"    {test.outside_allowed} of {test.n_block_columns} block "
                f"column(s) lie outside the columns placements were confined")
            lines.append(
                f"    to, so the observation sits slightly outside the class "
                f"it is compared with")
        if test.p_value > 0.05:
            lines.append(
                "    -> blocks of these widths placed anywhere would look like "
                "this; the")
            lines.append(
                "       contrast is not evidence that THESE regions are "
                "special.")
    return "\n".join(lines)