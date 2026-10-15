"""The gene tree: reading it, rooting it, and reconciling it with the traits.

This module holds the steps between "IQ-TREE produced a treefile" and "a
phylogenetic test can be run", because every one of them is a place where the
analysis can proceed on the wrong set of tips and still produce a number.

**Rooting is a declaration, checked.** An unrooted tree has no ancestry, and
D and lambda both read ancestry off the branching order, so the tree must be
rooted before either is computed. The outgroup is a caller-supplied set of ids
-- it cannot be inferred from the identity margin, for the reason recorded in
``group.paralogues``: on the FXR1 set all 58 invertebrate sequences fall below
the 0.05 margin, but so do 20 vertebrate sequences that belong to a paralogue.
What this module adds is the check that the declared outgroup is *monophyletic
on this tree*. If it is not, its MRCA contains ingroup tips, and rooting there
moves part of the ingroup to the other side of the root. :func:`root_on_outgroup`
reports the intruders and, by default, refuses rather than rooting anyway.

**Identical tips are not interchangeable tips.** Two sequences identical over
the aligned columns sit at the ends of zero-length branches. A trait that
differs between them therefore changes with no branch length to explain the
change, which is the one configuration in which D and lambda are not merely
imprecise but ill-posed. Deduplicating the alignment before inference looks
like the fix and is worse: it picks one member as the representative and
silently discards the others' traits. On the FXR1 trimmed alignment 33 groups
of identical rows cover 90 of 372 sequences, only 7 of those groups are
identical before trimming, and 11 contain members whose APR masks differ -- by
up to 84 residues. So: keep every sequence, let the tree program collapse them
for the search and restore them in the output, and use
:func:`trait_conflicts` to find out how much of the trait variance has no
branch length under it.

**Reconciliation is reported, never silent.** Tips without traits and traits
without tips are both normal -- a sequence can fail the completeness filter and
still be in the alignment -- and intersecting them without a count is how an
analysis ends up running on half the data it claims.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Branch support thresholds for the summary. UFBoot and SH-aLRT are not on the
#: same scale and are conventionally read together: a clade is well supported
#: when UFBoot >= 95 AND SH-aLRT >= 80 (Minh et al. 2013; Guindon et al. 2010).
#: Either alone accepts clades the other rejects.
UFBOOT_OK = 95.0
ALRT_OK = 80.0

#: A tip whose row in the alignment is mostly gaps is placed by missing data.
#: Nothing in the likelihood objects to it -- absent characters simply do not
#: constrain the topology -- so such a tip lands wherever the remaining columns
#: happen to point and carries a branch length that means nothing. Default is
#: deliberately permissive; the FXR1 set has 15 sequences below 0.5.
MIN_TIP_OCCUPANCY = 0.5


def _require_biopython():
    try:
        from Bio import Phylo
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "amyloevo.phylo.tree needs Biopython for Newick parsing and "
            "rerooting (`pip install 'amyloevo[phylo]'`). The tree operations "
            "here -- MRCA, reroot, monophyly, prune -- are exactly the ones "
            "worth not reimplementing."
        ) from exc
    return Phylo


# ---------------------------------------------------------------------------
# reading


def read_tree(path, fmt: str = "newick"):
    """Read a tree. Any format Biopython reads; IQ-TREE writes Newick."""
    Phylo = _require_biopython()
    return Phylo.read(str(path), fmt)


def write_tree(tree, path, fmt: str = "newick") -> None:
    Phylo = _require_biopython()
    Phylo.write(tree, str(path), fmt)


def tip_labels(tree) -> list[str]:
    return [t.name for t in tree.get_terminals()]


def _clades_by_name(tree) -> dict[str, object]:
    return {t.name: t for t in tree.get_terminals()}


# ---------------------------------------------------------------------------
# support


@dataclass
class SupportSummary:
    """How much of the topology is actually resolved.

    Monophyly of a paralogue clade is a claim about the topology, and a
    topology whose internal nodes are unsupported makes no claims. This is the
    number to quote beside any statement that the groups came out monophyletic.
    """

    n_internal: int
    #: Internal nodes carrying a parsable support value.
    n_scored: int
    ufboot_ok: int = 0
    alrt_ok: int = 0
    both_ok: int = 0
    has_alrt: bool = False

    def __str__(self) -> str:
        if not self.n_scored:
            return (
                f"{self.n_internal} internal node(s), none carrying a support "
                f"value -- run IQ-TREE with -B (UFBoot) and --alrt, or the "
                f"topology is a point estimate with no error bars."
            )
        frac = self.both_ok / self.n_scored if self.has_alrt else self.ufboot_ok / self.n_scored
        criterion = (
            f"UFBoot >= {UFBOOT_OK:.0f} and SH-aLRT >= {ALRT_OK:.0f}"
            if self.has_alrt
            else f"UFBoot >= {UFBOOT_OK:.0f}"
        )
        return (
            f"{self.n_scored} of {self.n_internal} internal node(s) scored; "
            f"{frac:.0%} meet {criterion}"
            + ("" if self.has_alrt else "  (no SH-aLRT: UFBoot alone accepts "
                                        "clades SH-aLRT rejects)")
        )


def _parse_support(label) -> tuple[float | None, float | None]:
    """Pull (alrt, ufboot) out of an internal node label.

    IQ-TREE writes ``SH-aLRT/UFboot`` when both are requested and a single
    number when only one is. A single number is read as UFBoot, which is what
    ``-B`` alone produces; a tree from ``--alrt`` alone would be misread, and
    that is why :attr:`SupportSummary.has_alrt` is reported rather than assumed.
    """
    if label is None:
        return None, None
    text = str(label).strip()
    if not text:
        return None, None
    parts = text.split("/")
    try:
        if len(parts) >= 2:
            return float(parts[0]), float(parts[1])
        return None, float(parts[0])
    except ValueError:
        return None, None


def support_summary(tree) -> SupportSummary:
    """Count how many internal nodes pass the support thresholds."""
    internal = [c for c in tree.get_nonterminals() if c is not tree.root]
    out = SupportSummary(n_internal=len(internal), n_scored=0)
    for clade in internal:
        label = clade.name
        if label is None and clade.confidence is not None:
            label = clade.confidence
        alrt, boot = _parse_support(label)
        if alrt is None and boot is None:
            continue
        out.n_scored += 1
        if alrt is not None:
            out.has_alrt = True
            if alrt >= ALRT_OK:
                out.alrt_ok += 1
        if boot is not None and boot >= UFBOOT_OK:
            out.ufboot_ok += 1
        if alrt is not None and boot is not None and alrt >= ALRT_OK and boot >= UFBOOT_OK:
            out.both_ok += 1
    return out


# ---------------------------------------------------------------------------
# rooting


@dataclass
class Rooting:
    """The result of rooting on a declared outgroup, and whether it was sound."""

    monophyletic: bool
    n_outgroup: int
    missing: list[str] = field(default_factory=list)
    #: Tips that break the outgroup's monophyly: the ones inside its MRCA that
    #: were not declared. Non-empty means the outgroup is not one side of any
    #: split in this tree.
    intruders: list[str] = field(default_factory=list)
    tree: object | None = None
    refused: str = ""

    def __str__(self) -> str:
        if self.refused:
            return f"NOT ROOTED: {self.refused}"
        if self.monophyletic:
            return (
                f"rooted on {self.n_outgroup} outgroup tip(s), monophyletic"
                + (f"; {len(self.missing)} declared id(s) absent from the tree"
                   if self.missing else "")
            )
        return (
            f"rooted on the MRCA of {self.n_outgroup} outgroup tip(s), which is "
            f"NOT monophyletic: {len(self.intruders)} ingroup tip(s) fall inside "
            f"it, so the root separates a set nobody declared."
        )


def _split_intruders(tree, group: set[str], by_name) -> list[str]:
    """Tips that stop ``group`` being one side of a split in this tree.

    Empty means the group *is* one side of a split, which is what "monophyletic
    outgroup" can mean on a tree whose own root is arbitrary. The complement is
    checked as well as the group: IQ-TREE writes an unrooted tree as a
    trifurcation at the root, and a group made of two of those three children
    has the root itself as its MRCA while still being a perfectly good split.
    Checking only the group's own MRCA calls that case paraphyletic, which is
    the kind of wrong answer that looks like a biological finding.
    """
    all_tips = set(by_name)
    for side in (group, all_tips - group):
        if not side:
            continue
        if len(side) == 1:
            return []
        mrca = tree.common_ancestor([by_name[n] for n in side])
        if {t.name for t in mrca.get_terminals()} == side:
            return []
    if len(group) == 1:
        return []
    mrca = tree.common_ancestor([by_name[n] for n in group])
    return sorted({t.name for t in mrca.get_terminals()} - group)


def root_on_outgroup(
    tree,
    outgroup,
    *,
    require_monophyly: bool = True,
    copy: bool = True,
) -> Rooting:
    """Root ``tree`` on ``outgroup``, checking its monophyly first.

    ``outgroup`` is an iterable of tip labels. With ``require_monophyly`` the
    tree is left unrooted when the outgroup is paraphyletic on it, and the
    reason is returned -- a paraphyletic outgroup is a result about the data
    (a misassigned sequence, a misplaced long branch, or a genuine topology
    nobody expected) and rooting through it buries that result.

    **Where the root lands.** Biopython places it *at* the outgroup's MRCA
    rather than along the branch above it, so the two branches descending from
    the root are not what a reader of the original tree would expect: the stem
    that separated the outgroup is reassigned to the ingroup side. Topology and
    every tip-to-tip patristic distance are unaffected, which is what
    :func:`clade_checks` and the monophyly statements depend on, but
    root-to-tip distances are not comparable to those from a tree rooted
    elsewhere. For the estimators, root in R with
    ``ape::root(phy, outgroup, resolve.root = TRUE)``, which places the root on
    the stem and is the function the comparative-methods packages are written
    against; this one is for inspection on the Python side.
    """
    import copy as _copy

    if copy:
        tree = _copy.deepcopy(tree)
    by_name = _clades_by_name(tree)
    declared = list(dict.fromkeys(outgroup))
    present = [n for n in declared if n in by_name]
    missing = [n for n in declared if n not in by_name]

    if len(present) < 1:
        return Rooting(False, 0, missing, [], None,
                       refused="no declared outgroup id is present in the tree")
    if len(present) == len(by_name):
        return Rooting(False, len(present), missing, [], None,
                       refused="the declared outgroup is the whole tree")

    targets = [by_name[n] for n in present]
    intruders = _split_intruders(tree, set(present), by_name)
    monophyletic = not intruders

    if intruders and require_monophyly:
        return Rooting(
            False, len(present), missing, intruders, None,
            refused=(
                f"the declared outgroup is not monophyletic: its MRCA also "
                f"contains {len(intruders)} ingroup tip(s). Rooting there would "
                f"place them outside the ingroup. Inspect them before rooting, "
                f"or pass require_monophyly=False to root anyway."
            ),
        )

    tree.root_with_outgroup(*targets)
    _bifurcate_root(tree, set(present))
    return Rooting(monophyletic, len(present), missing, intruders, tree)


def _bifurcate_root(tree, outgroup: set[str]) -> None:
    """Make the root bifurcating, with the outgroup on one side of it.

    Biopython's ``root_with_outgroup`` roots *at* the MRCA of its targets, so an
    outgroup of more than one tip leaves the root with three or more children.
    That tree is not merely untidy: ``ape::is.rooted`` returns FALSE for a root
    of degree three, and ``phyloglm`` -- which builds its covariance from
    root-to-tip paths -- refuses an unrooted tree. The rooting would then have
    been performed, reported as successful, and rejected downstream.

    The inserted node carries a branch length of zero and the existing children
    keep theirs, so every root-to-tip distance is unchanged. A zero-length
    branch at the root asserts nothing about when the two sides split, which is
    the honest value: nothing in the data locates the root along that branch.
    """
    from Bio.Phylo.BaseTree import Clade

    children = list(tree.root.clades)
    if len(children) <= 2:
        return
    out_side = [c for c in children if {t.name for t in c.get_terminals()} <= outgroup]
    in_side = [c for c in children if c not in out_side]
    if not out_side or not in_side:
        return
    def collapse(group):
        if len(group) == 1:
            return group[0]
        return Clade(branch_length=0.0, clades=group)
    tree.root.clades = [collapse(out_side), collapse(in_side)]


# ---------------------------------------------------------------------------
# reconciliation


@dataclass
class Reconciliation:
    """Overlap between the tree's tips and the trait table's rows."""

    n_tips: int
    n_traits: int
    shared: list[str] = field(default_factory=list)
    tips_without_traits: list[str] = field(default_factory=list)
    traits_without_tips: list[str] = field(default_factory=list)

    @property
    def n_shared(self) -> int:
        return len(self.shared)

    def __str__(self) -> str:
        return (
            f"{self.n_shared} sequence(s) have both a tip and a trait "
            f"({self.n_tips} tips, {self.n_traits} trait rows); "
            f"{len(self.tips_without_traits)} tip(s) have no trait and "
            f"{len(self.traits_without_tips)} trait row(s) have no tip."
        )


def reconcile(tips, trait_ids) -> Reconciliation:
    """Intersect tips with trait rows, keeping tip order, and count the losses."""
    tips = list(tips)
    trait_set = set(trait_ids)
    tip_set = set(tips)
    return Reconciliation(
        n_tips=len(tips),
        n_traits=len(trait_set),
        shared=[t for t in tips if t in trait_set],
        tips_without_traits=[t for t in tips if t not in trait_set],
        traits_without_tips=sorted(i for i in trait_set if i not in tip_set),
    )


def root_to_tip(tree) -> dict[str, float]:
    """Patristic distance from the root to each tip.

    On a tree of substitutions per site this is divergence *and* rate mixed
    together, and it is not constant across tips the way it would be on an
    ultrametric tree. See the caveat in ``traits.covariates`` before using it as
    a predictor: it is derived from the same tree that supplies the correlation
    structure a phylogenetic regression corrects with.
    """
    return {t.name: float(tree.distance(tree.root, t)) for t in tree.get_terminals()}


def prune_to(tree, keep, *, copy: bool = True):
    """Return ``tree`` with only the tips in ``keep``.

    Raises when ``keep`` leaves fewer than three tips: a two-tip tree has one
    branch and no topology, and every phylogenetic statistic on it is a
    division by the thing that is missing.
    """
    import copy as _copy

    if copy:
        tree = _copy.deepcopy(tree)
    keep = set(keep)
    present = {t.name for t in tree.get_terminals()}
    unknown = keep - present
    if unknown:
        raise KeyError(
            f"{len(unknown)} id(s) to keep are not tips of this tree, first: "
            f"{sorted(unknown)[0]}"
        )
    if len(keep) < 3:
        raise ValueError(
            f"{len(keep)} tip(s) left after pruning; a tree needs at least 3 "
            f"for any topology to exist"
        )
    for name in sorted(present - keep):
        tree.prune(name)
    return tree


# ---------------------------------------------------------------------------
# identical tips and the traits that disagree across them


def identical_groups(alignment) -> list[list[str]]:
    """Groups of ids whose aligned rows are character-identical.

    Takes the alignment as a mapping id -> aligned string. Groups of one are
    not returned. Order within a group follows the mapping, so the first member
    is the one a deduplicating step would most likely have kept.
    """
    by_row: dict[str, list[str]] = {}
    for name, row in alignment.items():
        by_row.setdefault(str(row), []).append(name)
    return [g for g in by_row.values() if len(g) > 1]


@dataclass
class Conflict:
    """One group of identical tips whose trait values are not identical."""

    members: list[str]
    #: Number of trait columns on which the members differ.
    n_differing: int
    #: Which columns those are.
    columns: list[str] = field(default_factory=list)


@dataclass
class ConflictReport:
    """How much trait variation sits on zero-length branches."""

    n_groups: int
    n_members: int
    n_testable: int
    conflicts: list[Conflict] = field(default_factory=list)

    @property
    def n_conflicting(self) -> int:
        return len(self.conflicts)

    def __str__(self) -> str:
        if not self.n_groups:
            return "no identical tips: every trait change has a branch under it."
        head = (
            f"{self.n_groups} group(s) of identical tips covering "
            f"{self.n_members} sequence(s); {self.n_testable} group(s) had two "
            f"or more members with traits."
        )
        if not self.conflicts:
            return head + (
                " No group's traits differ internally, so collapsing them would "
                "have lost nothing -- but there was no way to know that without "
                "checking."
            )
        worst = max(c.n_differing for c in self.conflicts)
        return head + (
            f" {self.n_conflicting} of them carry DIFFERENT traits (up to "
            f"{worst} column(s)). That variation has no branch length under it: "
            f"D and lambda cannot attribute it to the tree, and a deduplicating "
            f"step would have deleted it instead."
        )


def trait_conflicts(groups, traits) -> ConflictReport:
    """Find identical-tip groups whose members disagree on the traits.

    ``traits`` is a DataFrame indexed by sequence id. Groups with fewer than two
    members present in it are counted but not tested -- absent is not the same
    as equal.
    """
    import pandas as pd

    groups = [list(g) for g in groups]
    report = ConflictReport(
        n_groups=len(groups),
        n_members=sum(len(g) for g in groups),
        n_testable=0,
    )
    if not groups or traits is None or len(traits) == 0:
        return report
    columns = [str(c) for c in traits.columns]
    for group in groups:
        have = [m for m in group if m in traits.index]
        if len(have) < 2:
            continue
        report.n_testable += 1
        block = traits.loc[have]
        varies = [
            columns[j]
            for j in range(block.shape[1])
            if not _all_equal(block.iloc[:, j].to_numpy())
        ]
        if varies:
            report.conflicts.append(
                Conflict(members=have, n_differing=len(varies), columns=varies)
            )
    return report


def _all_equal(values: np.ndarray) -> bool:
    values = np.asarray(values)
    if values.size < 2:
        return True
    first = values[0]
    if values.dtype.kind in "fc":
        finite = np.isfinite(values)
        if not finite.all():
            return bool(finite.sum() == 0)
        return bool(np.allclose(values, first))
    return bool((values == first).all())


# ---------------------------------------------------------------------------
# alignment-side filters that decide which tips exist at all


def tip_occupancy(alignment) -> dict[str, float]:
    """Fraction of non-gap characters in each aligned row."""
    out = {}
    for name, row in alignment.items():
        row = str(row)
        n = len(row)
        out[name] = 0.0 if n == 0 else 1.0 - (row.count("-") + row.count(".")) / n
    return out


def low_occupancy(alignment, minimum: float = MIN_TIP_OCCUPANCY) -> dict[str, float]:
    """Rows below ``minimum`` occupancy, worst first.

    These are the candidates for exclusion *before* tree inference, not after:
    a tip placed by missing data also drags the branch lengths of its
    neighbours, so removing it afterwards does not undo its effect.
    """
    occ = tip_occupancy(alignment)
    return dict(sorted(
        ((k, v) for k, v in occ.items() if v < minimum), key=lambda kv: kv[1]
    ))


# ---------------------------------------------------------------------------
# paralogue clades


@dataclass
class CladeCheck:
    """Whether one paralogue group is monophyletic on the gene tree."""

    group: str
    n: int
    monophyletic: bool
    intruders: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        if self.monophyletic:
            return f"{self.group}: {self.n} tip(s), monophyletic"
        return (
            f"{self.group}: {self.n} tip(s), NOT monophyletic -- its MRCA also "
            f"holds {len(self.intruders)} tip(s) from other groups"
        )


def clade_checks(tree, membership) -> list[CladeCheck]:
    """Monophyly of each group in ``membership`` (group -> ids) on this tree.

    A group that is not monophyletic is a finding, not an error: on a gene tree
    it means either a misassignment, an incomplete lineage, or a duplication
    history more complicated than one event. What it rules out is computing a
    per-clade D or lambda by taking that group's MRCA, because that subtree
    contains sequences from the other groups.
    """
    by_name = _clades_by_name(tree)
    out = []
    for group, ids in membership.items():
        ids = list(dict.fromkeys(ids))
        present = [i for i in ids if i in by_name]
        missing = [i for i in ids if i not in by_name]
        if len(present) < 2:
            out.append(CladeCheck(group, len(present), False, [], missing))
            continue
        mrca = tree.common_ancestor([by_name[i] for i in present])
        inside = {t.name for t in mrca.get_terminals()}
        intruders = sorted(inside - set(present))
        out.append(CladeCheck(group, len(present), not intruders, intruders, missing))
    return out


__all__ = [
    "ALRT_OK",
    "MIN_TIP_OCCUPANCY",
    "UFBOOT_OK",
    "CladeCheck",
    "Conflict",
    "ConflictReport",
    "Reconciliation",
    "Rooting",
    "SupportSummary",
    "clade_checks",
    "identical_groups",
    "low_occupancy",
    "prune_to",
    "read_tree",
    "reconcile",
    "root_on_outgroup",
    "root_to_tip",
    "support_summary",
    "tip_labels",
    "tip_occupancy",
    "trait_conflicts",
    "write_tree",
]
