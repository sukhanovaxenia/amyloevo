"""Turning block occupancies into traits a phylogenetic test can accept.

``conserve.blocks`` produces a continuous ``(n_sequences, n_blocks)`` matrix:
the fraction of each block's columns on which a sequence carries an APR. Pagel's
lambda takes that as it stands. Fritz & Purvis's *D* is defined for a **binary**
trait, so a block has to become present or absent, and that step is where the
analysis can quietly decide its own answer.

**The threshold is a declaration.** There is no natural cut in a coverage
fraction, and the choice moves the prevalence of every trait at once -- which is
the quantity *D*'s null distributions are most sensitive to. It is therefore a
single named constant, written into the output, and not a default buried in a
call. :data:`PRESENCE_THRESHOLD` is 0.5: a block counts as present when the
sequence carries an APR over more than half of it. Report the sensitivity of
any conclusion to it; a result that holds at 0.4 and vanishes at 0.6 is a
result about the threshold.

**Prevalence is a precondition.** *D* compares the observed sum of sister-clade
differences against two simulated nulls, and both need the two states to be
present in quantity. A trait carried by 3 of 372 tips has an enormous sampling
variance: *D* will come back somewhere in a range that spans the whole
interpretable scale, and a reader sees a number rather than the absence of
information. The same holds for ``phyloglm``, whose Ives & Garland estimator
needs enough events to identify both the intercept and alpha. So a trait column
is screened before it is tested, and the screen's verdict travels with it
instead of being applied silently: :func:`screen` marks each column usable or
not, with the reason, and :func:`format_trait_report` prints the table.

**Multiplicity.** One test per block is a family of tests -- 15 or 16 on the
FXR1 set -- and they are not independent, since blocks neighbouring in the
alignment share sequences and sampling. Adjust across the tested columns with
``amyloevo.stats.multiple``; the floor of a permutation-based *p* also limits
what any adjustment can reject, which bit this project once already.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: A block counts as present when the sequence carries an APR over more than
#: this fraction of its columns. See the module docstring: this is a choice, not
#: a measurement, and it belongs in the provenance of every result built on it.
PRESENCE_THRESHOLD = 0.5

#: A binary trait needs at least this many tips in its minority state. Below it
#: *D* and ``phyloglm`` return numbers whose sampling variance covers the whole
#: interpretable range.
MIN_MINORITY = 10

#: And the minority state must be at least this fraction of the tips. The count
#: and the fraction catch different failures: 10 of 12 tips passes the count and
#: fails this, 10 of 5,000 the reverse.
MIN_PREVALENCE = 0.05


@dataclass
class TraitCheck:
    """One trait column, and whether it can carry a phylogenetic test."""

    label: str
    n: int
    n_present: int
    usable: bool = False
    reason: str = ""

    @property
    def prevalence(self) -> float:
        return self.n_present / self.n if self.n else float("nan")

    @property
    def minority(self) -> int:
        return min(self.n_present, self.n - self.n_present)


def binarise(
    profiles: np.ndarray,
    ids,
    labels,
    *,
    threshold: float = PRESENCE_THRESHOLD,
):
    """Block occupancies -> a 0/1 DataFrame indexed by sequence id.

    ``profiles`` is ``(n_sequences, n_blocks)`` with rows in the order of
    ``ids`` and columns in the order of ``labels``. Non-finite entries raise:
    an unscored block is not an absent block, and mapping it to 0 invents
    absences that *D* then reads as phylogenetic structure.
    """
    import pandas as pd

    profiles = np.asarray(profiles, dtype=float)
    ids = list(ids)
    labels = [str(x) for x in labels]
    if profiles.shape != (len(ids), len(labels)):
        raise ValueError(
            f"profiles are {profiles.shape} but there are {len(ids)} id(s) and "
            f"{len(labels)} label(s)"
        )
    if not np.isfinite(profiles).all():
        bad = int((~np.isfinite(profiles)).sum())
        raise ValueError(
            f"{bad} non-finite occupancy value(s). Drop or fill those rows "
            f"explicitly -- silently reading them as absence would create "
            f"absences the predictors never called."
        )
    if not 0.0 <= threshold < 1.0:
        raise ValueError(f"threshold must be in [0, 1), got {threshold}")
    binary = (profiles > threshold).astype(int)
    return pd.DataFrame(binary, index=pd.Index(ids, name="id"), columns=labels)


def continuous(profiles: np.ndarray, ids, labels):
    """The same matrix left continuous, for Pagel's lambda."""
    import pandas as pd

    profiles = np.asarray(profiles, dtype=float)
    ids = list(ids)
    labels = [str(x) for x in labels]
    if profiles.shape != (len(ids), len(labels)):
        raise ValueError(
            f"profiles are {profiles.shape} but there are {len(ids)} id(s) and "
            f"{len(labels)} label(s)"
        )
    return pd.DataFrame(profiles, index=pd.Index(ids, name="id"), columns=labels)


def screen(
    binary,
    *,
    min_minority: int = MIN_MINORITY,
    min_prevalence: float = MIN_PREVALENCE,
) -> list[TraitCheck]:
    """Decide, per column, whether it can carry *D* or a logistic regression."""
    checks = []
    n = len(binary)
    for label in binary.columns:
        col = binary[label].to_numpy()
        present = int(col.sum())
        check = TraitCheck(label=str(label), n=n, n_present=present)
        if present == 0 or present == n:
            check.reason = (
                f"constant: {'every' if present else 'no'} tip carries it, so "
                f"there is no variation to attribute to the tree"
            )
        elif check.minority < min_minority:
            check.reason = (
                f"minority state on {check.minority} tip(s) (need "
                f"{min_minority}); D's sampling variance here spans its whole "
                f"interpretable range"
            )
        elif check.prevalence < min_prevalence or check.prevalence > 1 - min_prevalence:
            check.reason = (
                f"prevalence {check.prevalence:.1%} outside "
                f"[{min_prevalence:.0%}, {1 - min_prevalence:.0%}]"
            )
        else:
            check.usable = True
        checks.append(check)
    return checks


def usable(checks) -> list[str]:
    return [c.label for c in checks if c.usable]


def covariates(
    ids,
    *,
    groups=None,
    lengths=None,
    root_distance=None,
):
    """Build the predictor table for ``phyloglm``.

    ``groups`` is the paralogue call per id, ``lengths`` the protein length,
    ``root_distance`` the patristic distance from the root to each tip.

    On ``root_distance``: the design asks whether APR presence depends on
    *divergence*, and root-to-tip distance is the obvious measure of it. It is
    also a function of the same tree that supplies the correlation structure
    ``phyloglm`` corrects with, so the predictor and the error model are not
    independent -- on an ultrametric tree it is constant by construction, and on
    a tree of substitutions per site it is largely rate, i.e. the thing the
    model is already absorbing. Fit it if the question demands it, report it
    beside the fit without it, and do not read a coefficient on it as a
    biological effect of time. ``lengths`` has no such problem and is the
    covariate actually worth carrying: a longer protein has more room for an
    APR, and that must be controlled before any clade effect is believed.
    """
    import pandas as pd

    ids = list(ids)
    frame = pd.DataFrame(index=pd.Index(ids, name="id"))
    if groups is not None:
        frame["group"] = [groups.get(i) for i in ids]
    if lengths is not None:
        frame["length"] = [lengths.get(i, np.nan) for i in ids]
    if root_distance is not None:
        frame["root_distance"] = [root_distance.get(i, np.nan) for i in ids]
    return frame


def format_trait_report(
    checks,
    *,
    threshold: float = PRESENCE_THRESHOLD,
    min_minority: int = MIN_MINORITY,
    min_prevalence: float = MIN_PREVALENCE,
) -> str:
    """Plain-text table of what will and will not be tested, and why."""
    keep = [c for c in checks if c.usable]
    lines = [
        "=" * 72,
        "TRAIT SCREEN  (binary block presence)",
        "=" * 72,
        f"present when occupancy > {threshold:.2f}; testable when the minority "
        f"state has >= {min_minority} tip(s) and >= {min_prevalence:.0%} of them",
        "",
        f"  {'block':16s} {'n':>5s} {'present':>8s} {'prev':>7s}  verdict",
    ]
    for c in checks:
        verdict = "testable" if c.usable else f"skipped: {c.reason}"
        lines.append(
            f"  {c.label:16s} {c.n:5d} {c.n_present:8d} {c.prevalence:6.1%}  {verdict}"
        )
    lines += [
        "",
        f"{len(keep)} of {len(checks)} block(s) will be tested. The threshold is "
        f"a choice, not a measurement:",
        "re-run at 0.4 and 0.6 and report whether the conclusion survives.",
        "=" * 72,
    ]
    return "\n".join(lines)


__all__ = [
    "MIN_MINORITY",
    "MIN_PREVALENCE",
    "PRESENCE_THRESHOLD",
    "TraitCheck",
    "binarise",
    "continuous",
    "covariates",
    "format_trait_report",
    "screen",
    "usable",
]
