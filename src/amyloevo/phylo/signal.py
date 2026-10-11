"""Does an APR profile track taxonomy, and at which rank?

The question in §4 of the design -- "does APR presence depend on divergence and
species group" -- cannot be answered by correlation, because species share
ancestry. This module does what can be done without a tree, and is explicit
about where that stops.

**Test the rank, inside the group.** Paralogue identity and taxonomy are
entangled in any pooled analysis: every mammal in a three-paralogue family
contributes one of each, so "mammal" and "paralogue" explain the same variance
and whichever is tested first wins. Rank effects are therefore tested WITHIN a
paralogue group, and the pooled test is reported only as the sanity check that
the groups separate at all.

**Replication is a precondition, not a diagnostic.** PERMANOVA's pseudo-F and
R^2 both rise mechanically as groups approach singletons: with 56 families among
77 sequences most families have one member, within-group scatter is zero by
construction and R^2 goes to 1 whatever the data say. A rank is therefore
dropped unless enough of its levels are replicated, and the result carries the
number of levels and members that survived so a reader can see what was tested.

**R^2 near 1 usually means the profiles are degenerate, not that the rank is
powerful.** Level replication is necessary and not sufficient: the same
zero-denominator pathology arrives through the *profiles* when many sequences
share an identical one. Observed on FXR2 at order level, which passes every
replication filter -- 7 orders, 48 sequences, none of them a singleton -- and
still returns R^2 = 0.989 with pseudo-F = 591. The reason is that those 48
sequences carry only **7 distinct block profiles** between them, and three
orders are internally identical, so within-group scatter is 1% of the total and
R^2 follows arithmetically. FXR1 over the same ranks has 13 distinct profiles
among 58 sequences and returns R^2 = 0.658.

The permutation p-value is honest in both cases -- the labels really do predict
the profile -- so the rank is reported rather than skipped. What cannot be read
off it is an effect size: "order explains 99% of the variance" is the wrong
sentence, and "FXR2's APR architecture takes only seven forms, and they are
clade-specific" is the right one. :attr:`RankResult.distinct_profiles` and
:attr:`RankResult.within_total` are therefore reported on every row and the
report flags a row whose profiles are near-degenerate.

**Several ranks is a family of tests.** Class, order, family and genus are four
tests on the same data, and the ranks are nested so they are strongly dependent.
p-values across ranks are Benjamini-Hochberg adjusted; see
``amyloevo.stats.multiple``.

**What this is not.** A significant rank effect here says profiles cluster by
clade, not that the clade CAUSED the profile. Nested sampling alone produces
that pattern, which is why a phylogenetic test on a real tree -- Fritz & Purvis
D, Pagel's lambda, phylogenetic logistic regression -- remains the stricter
instrument the design calls for. This is the distribution-free screen that says
whether it is worth building the tree.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..stats.multiple import benjamini_hochberg

#: A level must have at least this many members to be tested.
MIN_PER_LEVEL = 3
#: A rank must retain at least this many levels after that filter.
MIN_LEVELS = 4
#: And at least this many sequences in total.
MIN_MEMBERS = 12

DEFAULT_RANKS = ("class", "order", "family", "genus")

#: Below this fraction of within-group scatter, R^2 is driven by the absence of
#: within-group variation rather than by the rank, and the row is flagged.
#:
#: The flag deliberately keys on scatter and not on the count of distinct
#: profiles. Exact row-uniqueness is the intuitive measure and is useless on
#: continuous features: add a whisper of noise to a set of identical profiles
#: and every row becomes "distinct" while the scatter stays negligible, so the
#: count stops firing exactly where the pathology is unchanged. Scatter is
#: scale-free and behaves the same for block occupancies and for continuous
#: features.
#:
#: 0.05 separates the observed cases cleanly: FXR2 order sits at 0.011 and is
#: the case this exists for, while FXR1 order (0.342) and FMR1 family (0.187)
#: are an order of magnitude clear of it.
NEGLIGIBLE_WITHIN_SCATTER = 0.05


@dataclass
class RankResult:
    """One rank's test, with what it was actually run on."""

    rank: str
    levels: int
    n: int
    pseudo_f: float = float("nan")
    r2: float = float("nan")
    p_value: float = float("nan")
    q_value: float = float("nan")
    skipped: str = ""
    #: Number of exactly distinct profile rows among the tested sequences.
    #: Descriptive only -- it is the sentence a reader wants ("the profiles take
    #: seven forms") and is not what the flag keys on, since it is meaningless
    #: for continuous features. See :data:`NEGLIGIBLE_WITHIN_SCATTER`.
    distinct_profiles: int = 0
    #: Within-group sum of squares as a fraction of the total. This is
    #: ``1 - r2``, kept separately because it is the quantity that goes to zero
    #: and so the one worth looking at when R^2 looks too good.
    within_total: float = float("nan")

    @property
    def tested(self) -> bool:
        return not self.skipped and np.isfinite(self.pseudo_f)

    @property
    def scatter_negligible(self) -> bool:
        """Whether R^2 here reflects absent within-group variation, not the rank.

        When this is true the permutation p-value still stands but R^2 must not
        be quoted as an effect size; see the module docstring.
        """
        if not self.tested or not np.isfinite(self.within_total):
            return False
        return self.within_total < NEGLIGIBLE_WITHIN_SCATTER


@dataclass
class SignalResult:
    """Taxonomic signal for one set of sequences."""

    group: str
    n_sequences: int
    ranks: list[RankResult] = field(default_factory=list)
    draws: int = 999

    def significant(self, alpha: float = 0.05) -> list[RankResult]:
        return [r for r in self.ranks if r.tested and r.q_value <= alpha]


def permanova(
    distances: np.ndarray, labels: np.ndarray, *, draws: int = 999, seed: int = 0
) -> tuple[float, float, float]:
    """Pseudo-F, R^2 and a label-permutation p (Anderson 2001).

    Returns ``(nan, nan, nan)`` for a degenerate design -- fewer than two
    levels, as many levels as observations, or zero within-group scatter.
    The last is the important one: when every group is internally identical the
    F ratio has a zero denominator, and reporting it as infinite dresses a
    design with no replication as the strongest result in the table.
    """
    n = distances.shape[0]
    d2 = distances ** 2
    total = d2[np.triu_indices(n, 1)].sum() / n

    def ss_within(lab: np.ndarray) -> float:
        ss = 0.0
        for level in np.unique(lab):
            idx = np.flatnonzero(lab == level)
            if idx.size < 2:
                continue
            sub = d2[np.ix_(idx, idx)]
            ss += sub[np.triu_indices(idx.size, 1)].sum() / idx.size
        return ss

    def statistic(lab: np.ndarray) -> tuple[float, float]:
        a = len(np.unique(lab))
        if a < 2 or a >= n:
            return float("nan"), float("nan")
        within = ss_within(lab)
        between = total - within
        if within <= 1e-12 or total <= 0:
            return float("nan"), float("nan")
        return (between / (a - 1)) / (within / (n - a)), between / total

    f_obs, r2 = statistic(labels)
    if not np.isfinite(f_obs):
        return float("nan"), float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    permuted = labels.copy()
    extreme = 0
    for _ in range(draws):
        rng.shuffle(permuted)
        f, _ = statistic(permuted)
        if np.isfinite(f) and f >= f_obs - 1e-12:
            extreme += 1
    return f_obs, r2, (extreme + 1) / (draws + 1)


def _replicated(labels: np.ndarray, min_per_level: int) -> tuple[np.ndarray, np.ndarray]:
    values, counts = np.unique(labels, return_counts=True)
    keep = set(values[counts >= min_per_level])
    mask = np.array([v in keep for v in labels])
    return mask, labels[mask]


def taxonomic_signal(
    profiles: np.ndarray,
    metadata,
    *,
    group: str = "",
    ranks=DEFAULT_RANKS,
    metric: str = "euclidean",
    draws: int = 999,
    seed: int = 0,
    min_per_level: int = MIN_PER_LEVEL,
    min_levels: int = MIN_LEVELS,
    min_members: int = MIN_MEMBERS,
) -> SignalResult:
    """Test whether ``profiles`` separate by each taxonomic rank.

    ``profiles`` is ``(n_sequences, n_features)`` -- block occupancies, column
    fractions, whatever the caller is comparing -- with rows in the same order
    as ``metadata``, a table carrying the rank columns.

    Features matter more than the test. A raw per-column binary profile over a
    long alignment is sparse, and clustering or testing it directly measures
    mostly shared absence; reduce to blocks or occupancy first. See
    ``amyloevo.conserve``.
    """
    from scipy.spatial.distance import pdist, squareform

    profiles = np.asarray(profiles, dtype=float)
    n = profiles.shape[0]
    result = SignalResult(group=group, n_sequences=n, draws=draws)
    if n < min_members:
        for rank in ranks:
            result.ranks.append(
                RankResult(rank=rank, levels=0, n=n,
                           skipped=f"only {n} sequences (need {min_members})")
            )
        return result

    # A non-finite profile poisons every distance it takes part in, and the
    # pseudo-F then comes back nan for the whole rank -- which reads as a
    # degenerate design when it is actually a hole in the input. Say which.
    bad = np.flatnonzero(~np.isfinite(profiles).all(axis=1))
    if bad.size:
        reason = (
            f"{bad.size} profile row(s) carry non-finite values (first at index "
            f"{int(bad[0])}); every distance involving them is nan. Fill or drop "
            f"those rows -- an unscored column is not the same as a zero."
        )
        for rank in ranks:
            result.ranks.append(
                RankResult(rank=rank, levels=0, n=n, skipped=reason)
            )
        return result

    distances = squareform(pdist(profiles, metric=metric))

    for rank in ranks:
        if rank not in getattr(metadata, "columns", []):
            result.ranks.append(
                RankResult(rank=rank, levels=0, n=0, skipped="column absent")
            )
            continue
        labels = metadata[rank].fillna("").astype(str).to_numpy()
        mask, kept = _replicated(labels, min_per_level)
        n_levels = len(np.unique(kept))
        if n_levels < min_levels or mask.sum() < min_members:
            result.ranks.append(
                RankResult(
                    rank=rank, levels=n_levels, n=int(mask.sum()),
                    skipped=(
                        f"only {n_levels} level(s) with >={min_per_level} members "
                        f"across {int(mask.sum())} sequences -- too few to test "
                        f"without R^2 rising on singletons alone"
                    ),
                )
            )
            continue
        sub = distances[np.ix_(mask, mask)]
        f, r2, p = permanova(sub, kept, draws=draws, seed=seed)
        distinct = len(np.unique(profiles[mask], axis=0))
        result.ranks.append(
            RankResult(rank=rank, levels=n_levels, n=int(mask.sum()),
                       pseudo_f=f, r2=r2, p_value=p,
                       distinct_profiles=distinct,
                       within_total=float(1.0 - r2) if np.isfinite(r2) else float("nan"),
                       skipped="" if np.isfinite(f) else "degenerate design")
        )

    tested = [r for r in result.ranks if r.tested]
    if tested:
        for entry, q in zip(tested, benjamini_hochberg([r.p_value for r in tested])):
            entry.q_value = float(q)
    return result


def format_signal_report(result: SignalResult) -> str:
    """Plain-text report for one group's rank tests."""
    header = f"TAXONOMIC SIGNAL" + (f" -- {result.group}" if result.group else "")
    lines = [
        "=" * 70,
        header,
        "=" * 70,
        f"{result.n_sequences} sequence(s); PERMANOVA with "
        f"{result.draws:,} label permutations; p adjusted across ranks (BH)",
        "",
        f"  {'rank':10s} {'levels':>7s} {'n':>5s} {'distinct':>9s} "
        f"{'pseudo-F':>9s} {'R2':>7s} {'p':>8s} {'q':>8s}",
    ]
    flagged = False
    for r in result.ranks:
        if not r.tested:
            lines.append(f"  {r.rank:10s} {r.levels:7d} {r.n:5d}   -- skipped: {r.skipped}")
            continue
        mark = " (!)" if r.scatter_negligible else ""
        flagged |= r.scatter_negligible
        lines.append(
            f"  {r.rank:10s} {r.levels:7d} {r.n:5d} {r.distinct_profiles:9d} "
            f"{r.pseudo_f:9.2f} {r.r2:7.3f} {r.p_value:8.3g} {r.q_value:8.3g}{mark}"
        )
    lines.append("")
    if flagged:
        lines += [
            f"(!) Within-group scatter on this row is under "
            f"{NEGLIGIBLE_WITHIN_SCATTER:.0%} of the total, so R^2 is close to 1",
            "    for arithmetic reasons: the sequences within a level barely differ",
            "    from each other. The p-value still holds -- the labels do predict",
            "    the profile -- but R^2 is NOT an effect size here. Say 'the profiles",
            "    take only <distinct> forms and they are clade-specific', not 'the",
            "    rank explains <R^2> of the variance'.",
            "",
        ]
    lines += [
        "A rank effect here means profiles CLUSTER by clade, not that the clade",
        "caused the profile: nested sampling alone produces that pattern. Treat",
        "it as the screen that decides whether building the species tree and",
        "running a phylogenetic test is worth it, not as the test itself.",
        "=" * 70,
    ]
    return "\n".join(lines)
