"""Per-column conservation: of sequence, of the APR call, and of the difference.

The project's question lives in the third of these. Sequence conservation at APR
positions is a well-trodden observation and mostly reports on the domain the APR
sits in; APR conservation on its own cannot be told apart from it. What
distinguishes amyloidogenicity as a selected property is a column whose residues
turn over while the amyloidogenic call persists -- and that is only visible as a
residual between the two measures.

Everything here is pure: no alignment parsing, no predictor calls, no I/O. The
inputs are per-column residue lists and per-column APR calls, produced by
``project`` and ``predict`` respectively.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Background amino-acid frequencies, UniProtKB/Swiss-Prot release statistics.
#: A uniform background would overstate conservation at columns made of common
#: residues: a column of pure leucine is less surprising than a column of pure
#: tryptophan, and only a real background distribution says so.
BACKGROUND_FREQUENCIES = {
    "A": 0.0825, "R": 0.0553, "N": 0.0406, "D": 0.0545, "C": 0.0137,
    "Q": 0.0393, "E": 0.0675, "G": 0.0707, "H": 0.0227, "I": 0.0596,
    "L": 0.0966, "K": 0.0584, "M": 0.0242, "F": 0.0386, "P": 0.0470,
    "S": 0.0656, "T": 0.0534, "W": 0.0108, "Y": 0.0292, "V": 0.0687,
}

#: Physicochemical classes, kept as a separate axis from the scores themselves.
AA_TYPES = {
    **{a: "hydrophobic" for a in "AVLIMC"},
    **{a: "aromatic" for a in "FWY"},
    **{a: "polar" for a in "STNQ"},
    **{a: "positive" for a in "KRH"},
    **{a: "negative" for a in "DE"},
    **{a: "special" for a in "GP"},
}


def aa_type(residue: str) -> str:
    return AA_TYPES.get(residue.upper(), "unknown")


@dataclass(frozen=True)
class ColumnConservation:
    """One alignment column, measured three ways."""

    column: int
    occupancy: int
    n_sequences: int
    shannon: float | None
    jensen_shannon: float | None
    apr_fraction: float | None
    defined: bool
    reason: str = ""

    @property
    def occupancy_fraction(self) -> float:
        return self.occupancy / self.n_sequences if self.n_sequences else 0.0


def shannon_entropy(residues: list[str]) -> float:
    """Shannon entropy in bits over the residues present at a column.

    Gaps are excluded by the caller rather than treated as a 21st character. A
    gap is missing evidence, not a residue that happens to be conserved, and
    counting it as one makes a sparsely occupied column look variable when it is
    merely unobserved.
    """
    if not residues:
        return 0.0
    total = len(residues)
    counts: dict[str, int] = {}
    for residue in residues:
        counts[residue.upper()] = counts.get(residue.upper(), 0) + 1
    return -sum(
        (count / total) * math.log2(count / total) for count in counts.values()
    )


def jensen_shannon_divergence(
    residues: list[str], background: dict[str, float] | None = None
) -> float:
    """JSD in bits between the column's distribution and the background.

    Bounded in [0, 1] for base-2 logarithms, symmetric, and finite even when a
    residue is absent from one distribution -- which is why it is preferred here
    over the KL divergence it is built from.
    """
    if not residues:
        return 0.0
    background = background or BACKGROUND_FREQUENCIES
    total = len(residues)
    observed: dict[str, float] = {}
    for residue in residues:
        key = residue.upper()
        observed[key] = observed.get(key, 0.0) + 1 / total

    divergence = 0.0
    for residue in set(observed) | set(background):
        p = observed.get(residue, 0.0)
        q = background.get(residue, 0.0)
        m = 0.5 * (p + q)
        if m == 0:
            continue
        if p > 0:
            divergence += 0.5 * p * math.log2(p / m)
        if q > 0:
            divergence += 0.5 * q * math.log2(q / m)
    return divergence


def apr_fraction(calls: list[bool]) -> float:
    """Fraction of orthologues called APR at this column."""
    return sum(1 for call in calls if call) / len(calls) if calls else 0.0


def measure_column(
    column: int,
    residues: list[str],
    calls: list[bool],
    *,
    n_sequences: int,
    min_occupancy: float = 0.5,
    background: dict[str, float] | None = None,
) -> ColumnConservation:
    """Measure one column, or record why it could not be measured.

    A column below ``min_occupancy`` is returned undefined rather than scored.
    With three sequences out of two hundred the entropy is near zero whatever
    the residues are, so a sparsely occupied column would enter the analysis as
    a maximally conserved one -- and insertions, which are exactly the sparse
    columns, would come out looking like the most conserved sites in the
    alignment.
    """
    occupancy = len(residues)
    if n_sequences == 0:
        raise ValueError("n_sequences must be positive")
    if occupancy / n_sequences < min_occupancy:
        return ColumnConservation(
            column=column,
            occupancy=occupancy,
            n_sequences=n_sequences,
            shannon=None,
            jensen_shannon=None,
            apr_fraction=None,
            defined=False,
            reason=(
                f"occupancy {occupancy}/{n_sequences} below {min_occupancy:.0%}"
            ),
        )
    return ColumnConservation(
        column=column,
        occupancy=occupancy,
        n_sequences=n_sequences,
        shannon=shannon_entropy(residues),
        jensen_shannon=jensen_shannon_divergence(residues, background),
        apr_fraction=apr_fraction(calls),
        defined=True,
    )


def apr_residual(
    columns: list[ColumnConservation],
) -> dict[int, float]:
    """APR conservation with sequence conservation regressed out.

    The headline metric. ``apr_fraction`` is regressed on ``jensen_shannon``
    across the defined columns, and each column's residual is what its APR
    conservation is NOT explained by its sequence conservation. A large positive
    residual marks a column that stays amyloidogenic while its residues change;
    a column where both are high is a conserved column and says nothing
    specifically about amyloid.

    Ordinary least squares on two variables, by hand, to keep this module free
    of a modelling dependency -- the fit is a line through at most a few thousand
    points and the closed form is exact.

    Returns an empty mapping when the fit is undefined (fewer than three defined
    columns, or no variation in sequence conservation). An undefined fit is not
    a residual of zero: zero would read as "fully explained", which is the
    opposite of "not measurable".
    """
    usable = [c for c in columns if c.defined]
    if len(usable) < 3:
        return {}

    xs = [c.jensen_shannon for c in usable]
    ys = [c.apr_fraction for c in usable]
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    variance = sum((x - mean_x) ** 2 for x in xs)
    if variance == 0:
        return {}

    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / variance
    intercept = mean_y - slope * mean_x
    return {
        c.column: c.apr_fraction - (intercept + slope * c.jensen_shannon)
        for c in usable
    }
