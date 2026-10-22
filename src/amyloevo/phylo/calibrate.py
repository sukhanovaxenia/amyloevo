"""Node calibrations for dating a gene tree, taken from a dated species tree.

**The calibrations go on NODES, because the tips carry no time.** LSD2's date
file is built for serially sampled sequences -- viruses with a collection date
each -- where time enters from the tips. An orthologue set is the other case
entirely: every species is extant, so every tip is contemporaneous and a tree
with all tips at one date has no scale at all. So the tips go in at zero
(``--date-tip 0``, rather than listed) and every piece of temporal information
comes from internal nodes.

**Why a species tree supplies them and the duplication nodes do not.**
Calibrating the duplication with a whole-genome-duplication date and then
reporting the duplication's date is circular. Inside each paralogue clade the
nodes are speciations, which a dated species tree estimates independently of
this family; calibrate those, and the duplication nodes -- which have no
counterpart in a species tree at all, since a duplication is not a speciation
-- come out as a result. That asymmetry is the whole argument for doing it this
way round, and it is what makes a later comparison against the WGD literature
a check rather than an echo.

**Subspecies expansion is the first thing that goes wrong, and it is silent.**
TimeTree answers a query for a species by returning its subspecies and
dropping the name asked for. A list of 152 species came back as 223 leaves:
``Apis mellifera`` as fourteen subspecies, ``Ailuropoda melanoleuca`` as two,
and ``Homo sapiens`` and ``Mus musculus`` absent outright -- which removes the
best-dated node in vertebrate biology from the calibration set without any
error being raised. :func:`collapse_subspecies` folds them back, and
:func:`coverage` reports both directions of what is left over, because a
calibration set that silently lost its deepest nodes still dates the tree.

Collapsing loses nothing of interest. Keeping one subspecies tip as the
species' representative leaves it at the same distance from the root -- the
tree is ultrametric -- and any divergence involving that species is read at a
node above the subspecies clade, which is unaffected. What it does assume is
that the tree really is ultrametric, so :func:`node_ages` checks that rather
than trusting the file to be the dated download and not some other export.

**A species divergence corresponds to several gene-tree nodes, not one.** Each
species appears once per paralogue, so one speciation is recoverable inside
each paralogue clade, and all of them carry the same age. That multiplies the
constraints, which a least-squares method is glad of -- provided each one
actually lands on the node intended, which is what :func:`calibrations`
verifies through :func:`amyloevo.phylo.tree.mrca_pair` instead of assuming.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .tree import mrca_pair

#: Relative spread of root-to-tip distances still called ultrametric. A dated
#: tree is ultrametric by construction, so anything above this is a different
#: file than the one intended -- usually an undated topology.
ULTRAMETRIC_TOLERANCE = 0.01

#: Calibrations younger than this are dropped by default. On a tree whose
#: deepest node is ~750 Myr, a 2 Myr node contributes almost no leverage and
#: carries the largest relative error, so it buys noise.
MIN_AGE = 10.0

#: Branch length at or below which a dating method treats two nodes as one.
#: This is NOT a property of the data and must not be guessed: LSD2 derives it
#: from the tree and prints it ("Collapse N internal branches having branch
#: length <= X"), so take X from that line. The default here is only a
#: placeholder for a tree nobody has run yet.
COLLAPSE_THRESHOLD = 1e-6

#: Line shape for one node calibration. The documented format for a TIP is
#: ``NAME<tab>DATE``; this is the ancestral-node analogue, with the node named
#: by two of its descendants. IT IS NOT VERIFIED against any binary here --
#: see :func:`format_dates` for how to check it in one cheap run before
#: trusting a whole calibration set.
NODE_LINE = "{a},{b}\t{date}"


@dataclass
class Coverage:
    """How a species tree's leaves line up with the taxa of an analysis.

    Counts only. Every comparison here is made on binomials, so a species
    present as thirteen subspecies counts once -- which means these numbers do
    NOT change when :func:`collapse_subspecies` runs, and the only figure that
    does is ``leaves``. Reporting both is the point: the first version of this
    message gave the binomial counts alone and said the duplicates "were folded
    back", so it read identically before and after the collapse and claimed an
    action it had not taken.
    """

    matched: list[str] = field(default_factory=list)
    tree_only: list[str] = field(default_factory=list)
    taxa_only: list[str] = field(default_factory=list)
    collapsed: dict[str, int] = field(default_factory=dict)
    leaves: int = 0

    @property
    def ready(self) -> bool:
        return not self.taxa_only

    @property
    def species(self) -> int:
        return len(self.matched) + len(self.tree_only)

    def message(self) -> str:
        lines = [
            f"{self.leaves} leaf label(s) resolving to {self.species} "
            f"species; {len(self.matched)} of the analysis's taxa matched, "
            f"{len(self.taxa_only)} with no leaf, "
            f"{len(self.tree_only)} species the analysis does not use"
        ]
        if self.collapsed:
            worst = sorted(self.collapsed.items(), key=lambda kv: -kv[1])[:5]
            lines.append(
                f"  {len(self.collapsed)} species are present as several "
                f"leaves, counted once above: "
                + ", ".join(f"{name} ({n})" for name, n in worst)
            )
            lines.append(
                "  collapse_subspecies() reduces the tree to one leaf each; "
                "this function only counts,\n  so every number but the leaf "
                "count is the same before and after it runs."
            )
        if self.taxa_only:
            lines.append(
                "  no leaf for: " + ", ".join(self.taxa_only[:8])
                + (f" ... (+{len(self.taxa_only) - 8})"
                   if len(self.taxa_only) > 8 else "")
            )
            lines.append(
                "  Those species contribute no calibration. Check the deep "
                "ones first: a set that\n  lost its oldest nodes still dates "
                "the tree, and the loss shows up only as a\n  suspiciously "
                "tight interval."
            )
        return "\n".join(lines)


@dataclass
class Calibration:
    """One node of the gene tree, and the age to pin it at."""

    clade: str
    a: str
    b: str
    age: float
    n_tips: int
    species: int
    #: Tips inside this node whose species has no leaf in the species tree, so
    #: their placement could not be checked. They do not invalidate the node --
    #: see :func:`calibrations` -- but they are the part of it taken on trust.
    unplaceable: int = 0

    @property
    def date(self) -> float:
        """Tips sit at zero and the axis runs forward, so ages are negative."""
        return -float(self.age)


def normalise(label) -> str:
    """A leaf label as a comparable name: underscores out, quotes off."""
    return str(label).replace("_", " ").strip().strip("'\"")


def binomial(name: str) -> str:
    """``Apis mellifera carnica`` -> ``Apis mellifera``.

    Two tokens, which is what folds a subspecies back onto its species. Names
    already of two tokens are returned unchanged, and a single token is
    returned as it is rather than being discarded -- an unresolved label is
    still a leaf, and dropping it here would make it vanish without a count.
    """
    parts = normalise(name).split()
    return " ".join(parts[:2]) if len(parts) >= 2 else " ".join(parts)


def collapse_subspecies(tree, *, keep: str = "first"):
    """Reduce each species to one leaf, labelled with its binomial.

    Modifies ``tree`` in place and returns how many leaves each collapsed
    species arrived with. The kept leaf stays at its own distance from the
    root, which on an ultrametric tree is every leaf's distance, so no age
    moves; a divergence involving the species is read above the subspecies
    clade and is untouched by the pruning.
    """
    groups: dict[str, list] = {}
    for tip in tree.get_terminals():
        groups.setdefault(binomial(tip.name), []).append(tip)

    collapsed = {}
    for name, tips in groups.items():
        if len(tips) > 1:
            collapsed[name] = len(tips)
            order = sorted(tips, key=lambda t: normalise(t.name))
            kept = order[0] if keep == "first" else order[-1]
            for tip in order:
                if tip is not kept:
                    tree.prune(tip)
            kept.name = name
        else:
            tips[0].name = name
    return collapsed


def coverage(tree, taxa) -> Coverage:
    """Line a species tree's leaves up with the taxa of an analysis.

    Run this before anything else. A query for 152 species can come back as
    223 leaves, and the names that went missing in the process are the ones a
    calibration set needs most.
    """
    wanted = {binomial(t) for t in taxa if str(t).strip()}
    groups: dict[str, int] = {}
    tips = tree.get_terminals()
    for tip in tips:
        name = binomial(tip.name)
        groups[name] = groups.get(name, 0) + 1
    present = set(groups)
    return Coverage(
        matched=sorted(present & wanted),
        tree_only=sorted(present - wanted),
        taxa_only=sorted(wanted - present),
        collapsed={k: v for k, v in sorted(groups.items()) if v > 1},
        leaves=len(tips),
    )


def node_ages(tree) -> dict[int, float]:
    """Age of every internal node, by its distance to a descendant tip.

    Raises when the tree is not ultrametric. That is not pedantry: a dated
    download is ultrametric by construction, so a spread in root-to-tip
    distances means the file is a different export -- a topology, or a tree
    with substitution lengths -- and every age read off it would be a number
    with no meaning rather than an error.
    """
    tips = tree.get_terminals()
    depths = np.array([tree.distance(tree.root, t) for t in tips], dtype=float)
    if depths.size == 0:
        return {}
    mean = float(depths.mean())
    spread = float(depths.max() - depths.min())
    if mean <= 0 or spread / mean > ULTRAMETRIC_TOLERANCE:
        raise ValueError(
            f"this tree is not ultrametric: root-to-tip distances span "
            f"{spread:.4g} around a mean of {mean:.4g} "
            f"({spread / mean if mean else float('inf'):.1%}). A dated tree "
            f"is ultrametric by construction, so this is not the dated "
            f"export -- most likely a topology or a tree in substitutions "
            f"per site. No age can be read from it."
        )
    ages = {}
    for node in tree.get_nonterminals():
        descendant = node.get_terminals()[0]
        ages[id(node)] = float(tree.distance(node, descendant))
    return ages


def _clean_pair(gene_tree, by_name, ids, *, tolerated):
    """Name the MRCA of ``ids`` by two descendants, if it holds nothing else.

    ``tolerated`` are tips allowed inside without spoiling the node -- the ones
    whose species the species tree cannot place, so their presence is a gap in
    the metadata rather than evidence that the topologies disagree. Anything
    else inside means the node is not the one the species set names, and a
    calibration placed there would date a different node.

    Returns ``(a, b, n_tolerated)`` or ``None``. The pair is the first and last
    terminal of the MRCA, which under any rooting lie in different children of
    it, so their own MRCA is that node again -- verified rather than assumed.
    """
    present = [i for i in dict.fromkeys(ids) if i in by_name]
    if len(present) < 2:
        return None
    mrca = gene_tree.common_ancestor([by_name[i] for i in present])
    inside = [t.name for t in mrca.get_terminals()]
    extra = set(inside) - set(present)
    if extra - set(tolerated):
        return None
    first, last = inside[0], inside[-1]
    if gene_tree.common_ancestor([by_name[first], by_name[last]]) is not mrca:
        return None
    return first, last, len(extra)


def calibrations(species_tree, gene_tree, species_of, membership, *,
                 min_age: float = MIN_AGE) -> list[Calibration]:
    """Gene-tree nodes that correspond to dated speciations, and their ages.

    Both trees are loaded objects, not paths, as everywhere else in this
    package. The two mappings are keyed on GENE-tree tip labels and not on
    species, which is the part the parameter names do not tell you::

        species_of  = {"9606.ENSP00000350170": "Homo sapiens", ...}
        membership  = {"FXR1": ["9606.ENSP00000350170", ...],
                       "FMR1": [...], "FXR2": [...],
                       "pre-duplication": [...]}

    Both come out of one metadata table -- ``id`` to ``sci`` for the first,
    ``id`` grouped by ``paralogue`` for the second -- and the species names are
    matched to the species tree through :func:`binomial`, so subspecies labels
    on either side resolve.

    For every internal node of the species tree and every paralogue clade, the
    gene tips of that clade whose species descend from the node are collected;
    when they form a clean clade on the gene tree, that clade is a recoverable
    instance of the speciation and takes the node's age. One speciation
    therefore yields up to one calibration per paralogue.

    The cleanliness test is the point. A gene tree whose topology inside a
    clade disagrees with the species tree will put other sequences inside that
    MRCA, and a calibration placed there dates a node that is not the one
    named. :func:`amyloevo.phylo.tree.mrca_pair` returns ``None`` in exactly
    that case and the candidate is skipped rather than reported, which is why a
    constrained gene tree yields far more calibrations than a free one -- and
    why the count is worth reading as a measure of agreement between the two
    trees.
    """
    ages = node_ages(species_tree)
    by_species: dict[str, list[str]] = {}
    for gene_id, species in species_of.items():
        if species:
            by_species.setdefault(binomial(species), []).append(str(gene_id))

    leaves_of: dict[str, list] = {}
    for tip in species_tree.get_terminals():
        leaves_of.setdefault(binomial(tip.name), []).append(tip)

    gene_by_name = {t.name: t for t in gene_tree.get_terminals()}
    gene_tips = set(gene_by_name)

    # Gene tips whose species has no leaf in the species tree. They cannot be
    # placed, which is NOT the same as being in the wrong place, and the
    # distinction decides how many calibrations survive. A single such tip
    # inside a clade -- one aardvark sequence, in the set this was written for
    # -- otherwise makes every node enclosing it unclean, and the clade loses
    # its whole deep range: FXR1 and FXR2 fell from the 435 Myr jawed-
    # vertebrate root to the 305 Myr teleost crown on one missing species. So
    # they are tolerated inside a node rather than counted against it, and
    # recorded, because tolerating them is an assumption: their true position
    # is unknown by definition, and a tip that in truth sits outside the node
    # would make the node older than the age pinned to it.
    unplaceable = {
        gene_id for gene_id in gene_tips
        if binomial(species_of.get(gene_id, "")) not in leaves_of
    }

    out: list[Calibration] = []

    for node in species_tree.get_nonterminals():
        age = ages.get(id(node))
        if age is None or age < min_age:
            continue
        descendants = {binomial(t.name) for t in node.get_terminals()}
        if len(descendants) < 2:
            continue
        for clade, members in membership.items():
            members = {str(m) for m in members} & gene_tips
            species_here = {s for s in descendants
                            if any(g in members
                                   for g in by_species.get(s, []))}
            if len(species_here) < 2:
                continue

            # This node's age belongs to the divergence of ALL its descendants,
            # and a clade rarely carries all of them. The gene tips collected
            # above are only the species the clade does carry, so their MRCA is
            # the divergence of THOSE species -- which is younger, by however
            # much of the node's subtree the clade is missing. Taking the
            # node's own age there inflates it: the species-tree root would
            # hand its 892 Myr to a clade of vertebrates whose own crown is
            # less than half that. So the node is used only when it IS the MRCA
            # of the species the clade carries; otherwise that MRCA is a node
            # of its own and is reached on its own iteration, with its own age.
            mrca = species_tree.common_ancestor(
                [leaf for s in species_here for leaf in leaves_of[s]])
            if mrca is not node:
                continue

            ids = [g for species in species_here
                   for g in by_species.get(species, []) if g in members]
            if len(ids) < 2:
                continue
            resolved = _clean_pair(gene_tree, gene_by_name, ids,
                                   tolerated=unplaceable & members)
            if resolved is None:
                continue
            a, b, extra = resolved
            out.append(Calibration(
                clade=str(clade), a=a, b=b, age=float(age),
                n_tips=len(ids), species=len(species_here),
                unplaceable=extra,
            ))
    out.sort(key=lambda c: (-c.age, c.clade))
    return out


def drop_unresolvable(gene_tree, calibrations_: list[Calibration], *,
                      threshold: float = COLLAPSE_THRESHOLD,
                      keep: str = "none"
                      ) -> tuple[list[Calibration], list[Calibration]]:
    """Remove calibrations that a dating run would merge onto one node.

    Dating methods collapse internal branches below a threshold before they
    estimate anything, because a rate is a length over a time and a length of
    zero gives no rate. Collapsing merges nodes -- and two calibrations that
    land on one merged node are two dates for one node, which is not a weak
    constraint but a contradiction. LSD2 reports it as "conflict or not enough
    information in the input temporal constraints" and stops, having named
    neither the nodes nor the ages.

    On the set this was written for, 149 of 355 internal branches sat below the
    threshold LSD2 chose, five merged nodes carried two ages each, and one of
    those pairs was 99.9 against 318.8 Myr -- a 219 Myr contradiction on a
    branch the gene tree resolves as zero.

    ``keep="none"`` drops every calibration in a conflicting group, which is
    the default because choosing between them is guesswork: a near-zero branch
    where the species tree puts 219 Myr means the gene tree does not resolve
    that split, and pinning either age asserts a position the tree does not
    support. ``keep="deepest"`` keeps the oldest instead, on the argument that
    deep nodes carry the time signal and the shallow one is inside the noise;
    it is available because that argument is sometimes right, not because it is
    safe.

    Groups that agree on a single age are not conflicts. They are redundant --
    the same node constrained twice -- so one is kept and the rest dropped.

    Returns ``(kept, dropped)``. ``threshold`` must be the value the dating run
    will actually use; see :data:`COLLAPSE_THRESHOLD`.
    """
    parent = {}
    for clade in gene_tree.get_nonterminals():
        for child in clade.clades:
            parent[id(child)] = clade

    merged: dict[int, object] = {}

    def find(node):
        while merged.get(id(node), node) is not node:
            node = merged[id(node)]
        return node

    for clade in gene_tree.get_nonterminals():
        if clade is gene_tree.root or id(clade) not in parent:
            continue
        if (clade.branch_length or 0.0) <= threshold:
            merged[id(find(clade))] = find(parent[id(clade)])

    by_name = {t.name: t for t in gene_tree.get_terminals()}
    groups: dict[int, list[Calibration]] = {}
    orphans: list[Calibration] = []
    for item in calibrations_:
        if item.a not in by_name or item.b not in by_name:
            orphans.append(item)
            continue
        node = gene_tree.common_ancestor([by_name[item.a], by_name[item.b]])
        groups.setdefault(id(find(node)), []).append(item)

    kept, dropped = [], list(orphans)
    for group in groups.values():
        ages = {round(c.age, 6) for c in group}
        if len(ages) == 1:
            best = max(group, key=lambda c: (c.n_tips, c.a))
            kept.append(best)
            dropped.extend(c for c in group if c is not best)
        elif keep == "deepest":
            best = max(group, key=lambda c: c.age)
            kept.append(best)
            dropped.extend(c for c in group if c is not best)
        else:
            dropped.extend(group)
    kept.sort(key=lambda c: (-c.age, c.clade))
    dropped.sort(key=lambda c: (-c.age, c.clade))
    return kept, dropped


def format_dropped(dropped: list[Calibration], *, threshold: float) -> str:
    """What :func:`drop_unresolvable` removed, and why it had to be removed."""
    if not dropped:
        return (f"no calibration was removed: no two of them land on one node "
                f"once branches at or below {threshold:g} are collapsed")
    by_clade: dict[str, int] = {}
    for item in dropped:
        by_clade[item.clade] = by_clade.get(item.clade, 0) + 1
    lines = [
        f"{len(dropped)} calibration(s) removed as unresolvable at a collapse "
        f"threshold of {threshold:g}:",
        "  " + ", ".join(f"{clade} {n}" for clade, n in sorted(by_clade.items())),
        "",
        "  Each sat on a node that the dating run merges with another carrying "
        "a different age,",
        "  which is a contradiction rather than a weak constraint -- LSD2 "
        "answers it with",
        "  'conflict or not enough information' and names nothing. A node that "
        "no branch length",
        "  separates from its neighbour cannot hold a date of its own.",
        "",
        "  The threshold must be the one the run will use. LSD2 derives it and "
        "prints it as",
        "  'Collapse N internal branches having branch length <= X'; take X "
        "from there, not",
        "  from this module's placeholder.",
    ]
    return "\n".join(lines)


def format_dates(calibrations_: list[Calibration], *,
                 line_format: str = NODE_LINE, digits: int = 4) -> str:
    """The date file's text: one line per calibration, ages negative.

    ``line_format`` is a parameter and not a constant because the shape of an
    ancestral-node line is the one thing here that cannot be checked without
    running the binary. The documented tip line is ``NAME<tab>DATE``; this
    default names the node by two descendants in the same two columns. Verify
    it cheaply before trusting a whole set: write a file with two or three
    calibrations, run the dating, and read back how many constraints LSD2
    reports it accepted. A format it does not understand yields zero, not a
    wrong answer, which is the good case -- and the reason to check with three
    lines rather than three hundred.
    """
    lines = []
    for item in calibrations_:
        lines.append(line_format.format(
            a=item.a, b=item.b, date=f"{item.date:.{digits}f}",
            age=f"{item.age:.{digits}f}", clade=item.clade,
        ))
    return "\n".join(lines) + "\n" if lines else ""


def format_report(cover: Coverage, items: list[Calibration]) -> str:
    """Coverage and the calibration set, in the terms a methods section needs."""
    lines = ["=" * 70, "NODE CALIBRATIONS FROM A DATED SPECIES TREE", "=" * 70,
             "", cover.message(), ""]
    if not items:
        lines += [
            "no calibration survived. Either the species tree and the gene "
            "tree share too few",
            "taxa, or the gene tree's topology inside the clades disagrees "
            "with the species",
            "tree, so no MRCA is clean. The second is the common case on an "
            "unconstrained gene",
            "tree, and constraining it to the species topology is the fix.",
        ]
        return "\n".join(lines)

    by_clade: dict[str, list[Calibration]] = {}
    for item in items:
        by_clade.setdefault(item.clade, []).append(item)

    lines.append(f"{len(items)} calibration(s) over "
                 f"{len(by_clade)} clade(s):")
    for clade, group in sorted(by_clade.items()):
        ages = [c.age for c in group]
        tolerated = sum(1 for c in group if c.unplaceable)
        lines.append(f"  {clade:18s} {len(group):3d} node(s), "
                     f"ages {min(ages):8.2f} to {max(ages):8.2f}"
                     + (f"   ({tolerated} hold an unplaceable tip)"
                        if tolerated else ""))

    tolerated_total = sum(1 for c in items if c.unplaceable)
    if tolerated_total:
        lines += [
            "",
            f"  {tolerated_total} node(s) enclose a tip whose species has no "
            f"leaf in the species tree.",
            "  Those tips were tolerated rather than counted against the "
            "node: being unplaceable is",
            "  a gap in the metadata, not evidence that the topologies "
            "disagree, and excluding",
            "  them would cost whole clades their deepest calibrations -- one "
            "such tip can do it.",
            "  It is still an assumption. Their true position is unknown by "
            "definition, and one",
            "  that in truth sat outside its node would make that node older "
            "than the age pinned",
            "  to it. The number above is how much of the set rests on that.",
        ]
    lines += [
        "",
        "  Every age above is a SPECIATION inside that clade, so the maximum "
        "is the deepest",
        "  speciation the clade still retains a clean calibration for. It is "
        "not the age of",
        "  the clade, not the age of the duplication that produced it, and "
        "the four maxima do",
        "  not order the duplication events -- a clade spanning deeper taxa "
        "simply reaches a",
        "  deeper speciation. The duplication ages are what dating ESTIMATES; "
        "nothing in this",
        "  table contains them.",
        "",
        "  Tips carry no date here -- every species is extant -- so pass "
        "--date-tip 0 and let",
        "  these nodes be the only temporal information. Ages are written "
        "negative, because",
        "  the axis runs forward from the tips at zero.",
        "",
        "  These ages are INPUTS, from the species tree. The duplication "
        "nodes are not in a",
        "  species tree at all, so their ages come out as results -- which "
        "is what makes a",
        "  later comparison with the whole-genome-duplication literature a "
        "check and not an",
        "  echo. Say which is which.",
        "",
        "  Sanity check the output before reading it: with tips at zero and "
        "negative node",
        "  ages, an estimated root age that comes back positive means the "
        "sign convention is",
        "  inverted, and nothing downstream is worth looking at.",
    ]
    return "\n".join(lines)


__all__ = [
    "COLLAPSE_THRESHOLD",
    "Calibration",
    "Coverage",
    "MIN_AGE",
    "NODE_LINE",
    "ULTRAMETRIC_TOLERANCE",
    "binomial",
    "calibrations",
    "collapse_subspecies",
    "coverage",
    "drop_unresolvable",
    "format_dates",
    "format_dropped",
    "format_report",
    "node_ages",
    "normalise",
]
