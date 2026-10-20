"""Relative divergence rates between clades, and what the data can support.

**What this module estimates is RELATIVE rate** -- which clade accumulated more
substitutions per unit of shared history. That is the part a gene-family
alignment determines on its own, with no external information at all.

Absolute ages are a separate question and they depend on something this module
cannot supply: a calibration. A gene family has no fossils of its own, so a
date has to come from outside. There is one case where it does, and it is worth
naming because it is common: when the paralogues are OHNOLOGS -- retained
duplicates from a whole-genome duplication -- the duplication node is that WGD
event, and the WGD is dated independently of the family. A single such
calibration is enough for least-squares dating, which is why
:func:`dating_command` takes a calibration file rather than refusing the job.
What the calibration cannot do is turn into certainty: a WGD's own date carries
the disagreement of the literature that dated it, and two rounds close together
are hard to separate, so an age from one WGD calibration inherits that spread.

**Why the comparison is paired.** Mean root-to-tip distance per clade looks
like the obvious statistic and is not a valid one: tips within a clade share
branches, so they are not independent observations, and a difference between
two clade means has no standard error that counts them as such. Pairing fixes
this at the source. Where one taxon carries a sequence in two clades -- the
normal case for a paralogue family -- the two paths differ only in which
paralogue they followed, since the species history below the duplication is the
same. The paired difference therefore isolates the paralogue, and over many
taxa the differences ARE comparable to each other.

**Why crown-to-tip and not root-to-tip.** Measured from the root, each path
includes its clade's stem branch, which is one number shared by every taxon in
that clade. That constant would be added to every paired difference and read as
a rate difference it is not. Measured from each clade's crown -- the MRCA of its
own members -- the comparison is between within-clade path lengths for the same
species, which is what a relative-rate statement is about. The stem lengths are
reported separately by :func:`stem_lengths`, since the duplication's own branch
is a quantity of interest and not a nuisance.

**Branch lengths here are substitutions per site.** A clade with the longer
paths either evolved faster or had more time, and on a gene tree whose tips are
contemporaneous the second is excluded only if the taxa are matched -- which is
what the pairing does. Unpaired, the two are not separable and this module says
so rather than reporting a number.

Tests are permutational rather than parametric, and dependency-free for the
same reason the rest of the package is: the paired differences are not
guaranteed normal, exact small-sample tables are not worth a dependency, and
the permutation floor is already a convention here (see
:mod:`amyloevo.phylo.estimate`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Permutations for the paired tests. The floor on the p-value is
#: ``1 / (permutations + 1)``; at 9999 that is 1e-4, which leaves room under a
#: Benjamini-Hochberg alpha/m for the handful of clade pairs a family has.
PERMUTATIONS = 9999

#: Fewer paired taxa than this and the test is not run. A paired permutation
#: over n taxa has 2**n sign assignments, so at n = 5 the smallest attainable
#: p is 1/32 and no result can clear a corrected threshold; the limit is about
#: attainable resolution, not about normality.
MIN_PAIRS = 10

#: Seed for the permutation draws, so a reported p is reproducible. Changing it
#: changes the last digits of every p and nothing else.
SEED = 0


@dataclass
class CladeRates:
    """Crown-to-tip distances for one clade."""

    clade: str
    n: int
    mean: float
    sd: float
    minimum: float
    maximum: float
    monophyletic: bool


@dataclass
class PairedTest:
    """One clade pair, compared over the taxa carrying both."""

    a: str
    b: str
    n_pairs: int
    median_difference: float
    ratio_of_means: float
    p: float
    at_floor: bool
    skipped: str = ""


@dataclass
class RateReport:
    per_clade: list[CladeRates] = field(default_factory=list)
    pairs: list[PairedTest] = field(default_factory=list)
    stems: dict[str, float] = field(default_factory=dict)
    n_taxa_paired: int = 0
    n_taxa_averaged: int = 0
    permutations: int = PERMUTATIONS


def _by_name(tree) -> dict[str, object]:
    return {t.name: t for t in tree.get_terminals() if t.name}


def crown_to_tip(tree, members, *, require_monophyly: bool = True
                 ) -> dict[str, float]:
    """Distance from the MRCA of ``members`` to each of them.

    ``require_monophyly`` matters more than it looks. When the group is not a
    clade its MRCA contains sequences that are not in it, so the node is not
    that group's crown and every distance measured from it is inflated by
    however deep the intruders sit. Returning an empty mapping makes the caller
    notice; silently measuring from the wrong node would not.
    """
    by_name = _by_name(tree)
    present = [m for m in dict.fromkeys(members) if m in by_name]
    if len(present) < 2:
        return {}
    crown = tree.common_ancestor([by_name[m] for m in present])
    inside = {t.name for t in crown.get_terminals()}
    if require_monophyly and inside != set(present):
        return {}
    return {m: float(tree.distance(crown, by_name[m])) for m in present}


def stem_lengths(tree, membership) -> dict[str, float]:
    """Length of each clade's stem: root to its crown.

    On a gene tree of paralogues this is the branch over which the paralogue
    diverged after the duplication, which is a quantity of interest in its own
    right and the reason crown-to-tip excludes it. Absent for a group that is
    not monophyletic, for the same reason as in :func:`crown_to_tip`.
    """
    by_name = _by_name(tree)
    out = {}
    for clade, members in membership.items():
        present = [m for m in dict.fromkeys(members) if m in by_name]
        if len(present) < 2:
            continue
        crown = tree.common_ancestor([by_name[m] for m in present])
        inside = {t.name for t in crown.get_terminals()}
        if inside != set(present):
            continue
        out[str(clade)] = float(tree.distance(tree.root, crown))
    return out


def per_clade(tree, membership) -> list[CladeRates]:
    """Descriptive crown-to-tip summary per clade.

    Descriptive is all it is. These means are NOT comparable to each other by a
    t-test or anything like one: the tips inside a clade share branches, so the
    n in each row overstates the independent information by an unknown factor.
    Use them to see the spread; use :func:`paired_tests` to compare.
    """
    rows = []
    for clade, members in membership.items():
        distances = crown_to_tip(tree, members)
        if not distances:
            rows.append(CladeRates(str(clade), 0, float("nan"), float("nan"),
                                   float("nan"), float("nan"), False))
            continue
        values = np.array(list(distances.values()), dtype=float)
        rows.append(CladeRates(
            clade=str(clade),
            n=int(values.size),
            mean=float(values.mean()),
            sd=float(values.std(ddof=1)) if values.size > 1 else 0.0,
            minimum=float(values.min()),
            maximum=float(values.max()),
            monophyletic=True,
        ))
    return rows


def paired_frame(tree, membership, taxon_of):
    """One row per taxon, one column per clade, crown-to-tip in the cells.

    A taxon with two sequences in the same clade contributes their mean, and
    the count of such taxa is reported: dropping them would discard exactly the
    lineages with a recent within-clade duplication, which is a biased subset,
    while letting one taxon contribute twice would weight it twice.
    """
    import pandas as pd

    distances = {str(clade): crown_to_tip(tree, members)
                 for clade, members in membership.items()}
    clades = [c for c, d in distances.items() if d]

    cells: dict[str, dict[str, list[float]]] = {}
    for clade in clades:
        for pid, value in distances[clade].items():
            taxon = taxon_of.get(pid)
            if not taxon:
                continue
            cells.setdefault(str(taxon), {}).setdefault(clade, []).append(value)

    averaged = sum(
        1 for row in cells.values() if any(len(v) > 1 for v in row.values())
    )
    frame = pd.DataFrame(
        [{clade: (float(np.mean(row[clade])) if clade in row else np.nan)
          for clade in clades} for row in cells.values()],
        index=list(cells),
        columns=clades,
    )
    frame.attrs["averaged"] = averaged
    return frame


def _paired_permutation(a: np.ndarray, b: np.ndarray, *,
                        permutations: int = PERMUTATIONS,
                        seed: int = SEED) -> tuple[float, bool]:
    """Sign-flip permutation p for paired differences.

    Under the null that the two clades evolve at the same rate, the sign of a
    paired difference is exchangeable; the test draws signs rather than
    reshuffling taxa, which is what keeps the pairing intact. Returns the
    two-sided p and whether it sits on the floor of ``1/(permutations + 1)``.
    """
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    diff = diff[np.isfinite(diff)]
    n = diff.size
    if n == 0:
        return float("nan"), False
    observed = abs(float(diff.mean()))
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(permutations, n))
    draws = np.abs((signs * diff).mean(axis=1))
    extreme = int((draws >= observed).sum())
    p = (extreme + 1) / (permutations + 1)
    return float(p), extreme == 0


def paired_tests(frame, *, permutations: int = PERMUTATIONS,
                 min_pairs: int = MIN_PAIRS, seed: int = SEED
                 ) -> list[PairedTest]:
    """Every clade pair, over the taxa carrying both."""
    clades = list(frame.columns)
    out = []
    for i, a in enumerate(clades):
        for b in clades[i + 1:]:
            both = frame[[a, b]].dropna()
            n = int(len(both))
            if n < min_pairs:
                out.append(PairedTest(
                    a, b, n, float("nan"), float("nan"), float("nan"), False,
                    skipped=(f"only {n} taxon/taxa carry both (need "
                             f"{min_pairs}); a sign-flip test over {n} pairs "
                             f"cannot reach a p below {1 / 2 ** n:.3g}"),
                ))
                continue
            left = both[a].to_numpy(dtype=float)
            right = both[b].to_numpy(dtype=float)
            p, at_floor = _paired_permutation(
                left, right, permutations=permutations, seed=seed)
            out.append(PairedTest(
                a=a, b=b, n_pairs=n,
                median_difference=float(np.median(left - right)),
                ratio_of_means=(float(left.mean() / right.mean())
                                if right.mean() else float("nan")),
                p=p, at_floor=at_floor,
            ))
    return out


def adjust(pairs: list[PairedTest], *, alpha: float = 0.05
           ) -> dict[tuple[str, str], float]:
    """Benjamini-Hochberg over the pairs that were actually tested.

    The family is the set of clade pairs compared on one tree, which is the
    scope a reader would take a significant pair from. Skipped pairs are not in
    it: a test that did not run is not a test that passed.
    """
    tested = [p for p in pairs if not p.skipped and np.isfinite(p.p)]
    if not tested:
        return {}
    order = sorted(tested, key=lambda p: p.p)
    m = len(order)
    out, running = {}, 1.0
    for rank in range(m, 0, -1):
        item = order[rank - 1]
        running = min(running, item.p * m / rank)
        out[(item.a, item.b)] = float(min(running, 1.0))
    return out


def report(tree, membership, taxon_of, *, permutations: int = PERMUTATIONS,
           min_pairs: int = MIN_PAIRS, seed: int = SEED) -> RateReport:
    """Everything above, assembled."""
    frame = paired_frame(tree, membership, taxon_of)
    pairs = paired_tests(frame, permutations=permutations,
                         min_pairs=min_pairs, seed=seed)
    return RateReport(
        per_clade=per_clade(tree, membership),
        pairs=pairs,
        stems=stem_lengths(tree, membership),
        n_taxa_paired=int(len(frame.dropna(thresh=2))) if len(frame) else 0,
        n_taxa_averaged=int(frame.attrs.get("averaged", 0)),
        permutations=permutations,
    )


@dataclass
class DatingInputs:
    """Whether an alignment and a tree describe the same sequences."""

    shared: list[str] = field(default_factory=list)
    tree_only: list[str] = field(default_factory=list)
    alignment_only: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return not self.tree_only and not self.alignment_only

    def message(self) -> str:
        if self.ready:
            return (f"alignment and tree agree on all {len(self.shared)} "
                    f"sequence(s)")
        lines = [
            f"alignment and tree do NOT describe the same sequences: "
            f"{len(self.shared)} shared, "
            f"{len(self.alignment_only)} in the alignment only, "
            f"{len(self.tree_only)} on the tree only."
        ]
        if self.alignment_only:
            lines.append(
                "  in the alignment, absent from the tree: "
                + ", ".join(self.alignment_only[:6])
                + (f" ... (+{len(self.alignment_only) - 6})"
                   if len(self.alignment_only) > 6 else "")
            )
            lines.append(
                "  The usual cause is the occupancy filter: the tree was "
                "inferred after dropping\n  sequences below "
                "MIN_TIP_OCCUPANCY, and the unfiltered alignment still "
                "carries them.\n  Build the matching alignment with "
                "matched_alignment() rather than re-running the tree."
            )
        if self.tree_only:
            lines.append(
                "  on the tree, absent from the alignment: "
                + ", ".join(self.tree_only[:6])
                + (f" ... (+{len(self.tree_only) - 6})"
                   if len(self.tree_only) > 6 else "")
            )
            lines.append(
                "  That direction is worse: the tree has tips the alignment "
                "cannot explain, so\n  the two came from different runs."
            )
        return "\n".join(lines)


def dating_inputs(tree, alignment) -> DatingInputs:
    """Compare a tree's tip labels with an alignment's identifiers.

    This exists because the mismatch is otherwise found by IQ-TREE, several
    hundred lines into a run, after it has read and analysed the whole
    alignment -- and because the two files that mismatch are exactly the two a
    pipeline naturally hands you: the alignment it aligned, and the tree it
    inferred from a filtered copy of that alignment it never wrote out.
    """
    tips = [t.name for t in tree.get_terminals() if t.name]
    tip_set, align_set = set(tips), {str(k) for k in alignment}
    return DatingInputs(
        shared=sorted(tip_set & align_set),
        tree_only=sorted(tip_set - align_set),
        alignment_only=sorted(align_set - tip_set),
    )


def matched_alignment(tree, alignment) -> dict[str, str]:
    """``alignment`` reduced to the sequences the tree actually carries.

    Columns are left alone. Dropping sequences can leave a column all-gap,
    which some tools object to, but removing those columns would change the
    coordinates every block and APR position in the project is reported
    against -- a worse trade than an empty column.
    """
    tips = [t.name for t in tree.get_terminals() if t.name]
    return {t: str(alignment[t]) for t in tips if t in alignment}


def dating_command(tree_path, alignment_path, inputs: DatingInputs, *,
                   dates_path=None, outgroup=None, threads: int = 4,
                   binary: str = "iqtree2") -> str:
    """The IQ-TREE least-squares dating call, with its own caveats attached.

    LSD2 rather than a Bayesian relaxed clock, and not as a compromise: the two
    differ in the width of an interval around a number that, without a
    calibration, neither of them measured. What LSD2 gives cheaply is relative
    node ordering and a per-branch rate, and ``--date-ci`` gives it an interval
    from resampling rather than from a prior.

    ``inputs`` is required, and the command is refused when it does not pass.
    An earlier version of this function took the two paths and trusted them,
    which produced a command pairing a 372-sequence alignment with a
    357-tip tree -- a combination IQ-TREE rejects only after reading both. A
    builder that cannot check its arguments should not be asked to.
    """
    if not isinstance(inputs, DatingInputs):
        raise TypeError(
            "dating_command needs the result of dating_inputs(tree, "
            "alignment); it will not emit a command for files it has not "
            "been told agree."
        )
    if not inputs.ready:
        raise ValueError(
            "refusing to build a dating command for an alignment and a tree "
            "that disagree.\n" + inputs.message()
        )
    parts = [
        binary,
        f"-s {alignment_path}",
        f"-te {tree_path}",
        "--date TAXNAME" if dates_path is None else f"--date {dates_path}",
        "--date-ci 100",
        "--clock-sd 0.2",
        f"-nt {threads}",
        "--prefix dated",
    ]
    if outgroup:
        parts.append(f"--date-root {outgroup}")
    return " \\\n    ".join(parts)


DATING_CAVEATS = """\
Three things decide whether a dated gene tree means anything here.

The duplication node is not a speciation, so no species divergence time
calibrates it -- but a whole-genome duplication does. Where the paralogues are
ohnologs, the duplication node IS that WGD and takes the WGD's own date, which
was estimated without reference to this family and is therefore a real external
calibration rather than an assumption wearing a date. Two consequences follow.
A WGD that happened in two rounds leaves the paralogues at two different node
depths, not one, so the tree should show two duplication nodes and a calibration
placed on the wrong one is wrong by the interval between the rounds. And the
WGD's date is as contested as the literature that produced it, so the interval
on every age downstream is at least that wide; quote it.

Inside a paralogue clade the nodes ARE speciations and can be calibrated
against a species tree independently, which is worth doing: two calibration
sources that agree are evidence, and one is an input.

Zero-length branches break every dating method, because a rate is a length
divided by a time and the numerator is zero. IQ-TREE reported 49 of 712
branches at or below zero on this family's tree. Collapse them, or set a
minimum, and say which.

A rate from substitutions per site is a rate of the sites that aligned. Where
the C-terminal third of the vertebrate proteins aligns to nothing in the
invertebrates, the rate is estimated over the conserved region only, and it
therefore understates divergence between those two sets rather than measuring
it."""


def format_rate_report(result: RateReport, adjusted=None) -> str:
    """The report as text, in the terms a methods section needs."""
    lines = ["=" * 70,
             "RELATIVE DIVERGENCE RATE (crown-to-tip, substitutions per site)",
             "=" * 70, ""]

    lines.append(f"  {'clade':24s} {'n':>4s} {'mean':>8s} {'sd':>8s} "
                 f"{'min':>8s} {'max':>8s} {'stem':>8s}")
    for row in result.per_clade:
        if not row.monophyletic:
            lines.append(f"  {row.clade:24s} {'--':>4s}   not monophyletic on "
                         f"this tree, so it has no crown to measure from")
            continue
        stem = result.stems.get(row.clade)
        lines.append(
            f"  {row.clade:24s} {row.n:4d} {row.mean:8.4f} {row.sd:8.4f} "
            f"{row.minimum:8.4f} {row.maximum:8.4f} "
            + (f"{stem:8.4f}" if stem is not None else f"{'--':>8s}")
        )
    lines += ["",
              "  Those means are descriptive. Tips inside a clade share "
              "branches, so the n",
              "  above overstates the independent information and no test "
              "compares the means",
              "  directly. The paired rows below are the comparison.", ""]

    lines += [f"-- paired within taxon ({result.n_taxa_paired} taxon/taxa "
              f"carry at least two clades) " + "-" * 10, ""]
    if result.n_taxa_averaged:
        lines.append(f"  {result.n_taxa_averaged} taxon/taxa carry more than "
                     f"one sequence in some clade; their distances were "
                     f"averaged\n  within that clade rather than entered "
                     f"twice.")
        lines.append("")
    lines.append(f"  {'pair':28s} {'n':>4s} {'median diff':>12s} "
                 f"{'ratio':>7s} {'p':>9s} {'q':>9s}")
    floor = 1.0 / (result.permutations + 1)
    any_floor = False
    for pair in result.pairs:
        label = f"{pair.a} vs {pair.b}"
        if pair.skipped:
            lines.append(f"  {label:28s} {pair.n_pairs:4d}   skipped: "
                         f"{pair.skipped}")
            continue
        q = (adjusted or {}).get((pair.a, pair.b))
        p_text = f"<{floor:.4g}" if pair.at_floor else f"{pair.p:9.4g}"
        any_floor = any_floor or pair.at_floor
        lines.append(
            f"  {label:28s} {pair.n_pairs:4d} {pair.median_difference:12.5f} "
            f"{pair.ratio_of_means:7.3f} {p_text:>9s} "
            + (f"{q:9.4g}" if q is not None else f"{'--':>9s}")
        )
    if any_floor:
        lines += ["", f"  !! a p shown as <{floor:.4g} is the permutation "
                      f"floor: no sign assignment was as",
                  "     extreme as the observed one. More permutations narrow "
                  "it; nothing else will."]

    lines += ["",
              "  'ratio' is the ratio of mean crown-to-tip distance over the "
              "paired taxa, so",
              "  1.20 means the first clade accumulated 20% more "
              "substitutions per site",
              "  within the same set of species. That is a RATE statement "
              "only because the",
              "  taxa are matched: unpaired, more substitutions and more time "
              "are the same",
              "  number.", "", "=" * 70, "", DATING_CAVEATS]
    return "\n".join(lines)


__all__ = [
    "CladeRates",
    "DATING_CAVEATS",
    "DatingInputs",
    "MIN_PAIRS",
    "PERMUTATIONS",
    "PairedTest",
    "RateReport",
    "SEED",
    "adjust",
    "crown_to_tip",
    "dating_command",
    "dating_inputs",
    "format_rate_report",
    "matched_alignment",
    "paired_frame",
    "paired_tests",
    "per_clade",
    "report",
    "stem_lengths",
]
