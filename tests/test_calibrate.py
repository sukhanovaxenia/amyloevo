"""Calibrations from a dated species tree, and the ways the tree is not one.

The fixture is built from the failure that prompted the module: a query for a
list of species comes back with the species replaced by their subspecies, so
the names asked for are absent and the leaf count is larger than the list.
Nothing raises, and the calibration set quietly loses whichever species were
expanded -- on the real data, Homo sapiens and Mus musculus among them.
"""

from __future__ import annotations

import io

import pytest

from amyloevo.phylo import calibrate as C

pytest.importorskip("Bio")

#: Four species, two of them returned as subspecies pairs, ultrametric at a
#: total depth of 100. Apis arrives twice and Homo twice, so a query for four
#: names yields six leaves -- the shape TimeTree produced.
EXPANDED = (
    "(((Homo_sapiens_sapiens:10,Homo_sapiens_neanderthalensis:10):90,"
    "(Mus_musculus_musculus:10,Mus_musculus_domesticus:10):90):0,"
    "(Apis_mellifera:50,Drosophila_melanogaster:50):50);"
)

#: Five species, one leaf each, ultrametric at depth 150, with four distinct
#: node ages so that a calibration can be identified by its age: (Homo,Mus) 90,
#: +Gallus 100, (Apis,Drosophila) 75, root 150.
CLEAN = ("(((Homo_sapiens:90,Mus_musculus:90):10,Gallus_gallus:100):50,"
         "(Apis_mellifera:75,Drosophila_melanogaster:75):75);")

#: Not a dated tree: root-to-tip distances differ, which is what a topology or
#: a substitutions-per-site tree looks like.
NOT_DATED = "((Homo_sapiens:0.3,Mus_musculus:0.1):0.2,Apis_mellifera:0.9);"


def _read(newick: str):
    from Bio import Phylo

    return Phylo.read(io.StringIO(newick), "newick")


# --- names -----------------------------------------------------------------


def test_a_subspecies_folds_onto_its_species():
    assert C.binomial("Apis_mellifera_carnica") == "Apis mellifera"
    assert C.binomial("Homo sapiens") == "Homo sapiens"


def test_a_single_token_label_is_kept_rather_than_discarded():
    """An unresolved label is still a leaf; dropping it here would make it
    disappear without being counted anywhere."""
    assert C.binomial("Nematostella") == "Nematostella"


# --- the expansion ---------------------------------------------------------


def test_coverage_reports_the_species_that_went_missing():
    """The failure the module exists for: the names queried are absent, and
    the only visible symptom is that the leaf count is too large."""
    cover = C.coverage(_read(EXPANDED),
                       ["Homo sapiens", "Mus musculus", "Apis mellifera",
                        "Gallus gallus"])
    assert cover.matched == ["Apis mellifera", "Homo sapiens", "Mus musculus"]
    assert cover.taxa_only == ["Gallus gallus"]
    assert not cover.ready
    assert "no leaf for" in cover.message()


def test_coverage_counts_how_many_leaves_each_species_arrived_with():
    cover = C.coverage(_read(EXPANDED), ["Homo sapiens"])
    assert cover.collapsed["Homo sapiens"] == 2
    assert cover.collapsed["Mus musculus"] == 2
    assert "present as several" in cover.message()


def test_only_the_leaf_count_changes_when_the_tree_is_collapsed():
    """Every comparison in Coverage is made on binomials, so the species-level
    numbers are identical before and after collapsing -- which made the first
    version of this message read as though the collapse had done nothing, and
    worse, claim it had folded the duplicates itself. The leaf count is the one
    figure that moves, so it is reported.
    """
    taxa = ["Homo sapiens", "Mus musculus", "Apis mellifera",
            "Drosophila melanogaster"]
    tree = _read(EXPANDED)
    before = C.coverage(tree, taxa)
    assert before.leaves == 6
    assert len(before.matched) == 4

    C.collapse_subspecies(tree)
    after = C.coverage(tree, taxa)
    assert after.leaves == 4
    assert after.matched == before.matched
    assert after.collapsed == {}
    assert "6 leaf label(s) resolving to 4 species" in before.message()
    assert "4 leaf label(s) resolving to 4 species" in after.message()
    assert "only counts" in before.message()


def test_collapsing_leaves_one_tip_per_species_and_keeps_the_ages():
    """Pruning subspecies must not move any divergence. On an ultrametric tree
    the kept leaf stays at every leaf's distance from the root, and the
    species' divergence from anything else is read above the subspecies
    clade."""
    tree = _read(EXPANDED)
    before = tree.distance(
        tree.common_ancestor([t for t in tree.get_terminals()
                              if t.name.startswith("Homo")][0],
                             [t for t in tree.get_terminals()
                              if t.name.startswith("Mus")][0]),
        [t for t in tree.get_terminals() if t.name.startswith("Homo")][0],
    )
    collapsed = C.collapse_subspecies(tree)
    assert collapsed == {"Homo sapiens": 2, "Mus musculus": 2}
    names = sorted(t.name for t in tree.get_terminals())
    assert names == ["Apis mellifera", "Drosophila melanogaster",
                     "Homo sapiens", "Mus musculus"]
    by_name = {t.name: t for t in tree.get_terminals()}
    after = tree.distance(
        tree.common_ancestor(by_name["Homo sapiens"], by_name["Mus musculus"]),
        by_name["Homo sapiens"],
    )
    assert after == pytest.approx(before)


def test_collapsing_is_deterministic():
    first = C.collapse_subspecies(_read(EXPANDED))
    second = C.collapse_subspecies(_read(EXPANDED))
    assert first == second


# --- ages ------------------------------------------------------------------


def test_node_ages_come_from_tip_distance_on_an_ultrametric_tree():
    tree = _read(CLEAN)
    ages = sorted(C.node_ages(tree).values())
    assert ages == pytest.approx([75.0, 90.0, 100.0, 150.0])


def test_a_tree_that_is_not_ultrametric_is_refused():
    """A dated download is ultrametric by construction, so a spread means the
    file is a different export and every age read off it would be a number
    with no meaning rather than an error."""
    with pytest.raises(ValueError, match="not ultrametric"):
        C.node_ages(_read(NOT_DATED))


# --- the calibrations ------------------------------------------------------

SPECIES = ["hum", "mus", "gal", "api", "dro"]
SCIENTIFIC = {"hum": "Homo sapiens", "mus": "Mus musculus",
              "gal": "Gallus gallus", "api": "Apis mellifera",
              "dro": "Drosophila melanogaster"}


def _clade(prefix: str, *, scrambled: bool = False) -> str:
    """One paralogue clade over the five species.

    Unscrambled it matches the species topology, so every speciation inside it
    is recoverable. Scrambled swaps mus and api, which breaks every node except
    the clade's own crown.
    """
    order = (["hum", "api", "gal", "mus", "dro"] if scrambled
             else ["hum", "mus", "gal", "api", "dro"])
    a, b, c, d, e = (f"{prefix}_{s}" for s in order)
    return f"((({a}:1,{b}:1):1,{c}:2):1,({d}:1,{e}:1):2)"


GENE = f"({_clade('A')}:1,{_clade('B')}:1);"
SPECIES_OF = {f"{p}_{s}": SCIENTIFIC[s] for p in ("A", "B") for s in SPECIES}
MEMBERSHIP = {p: [f"{p}_{s}" for s in SPECIES] for p in ("A", "B")}


def test_one_speciation_yields_one_calibration_per_paralogue():
    """Each species appears once per paralogue, so a single species divergence
    is recoverable inside each clade and all of them carry the same age. That
    multiplies the constraints, which least squares is glad of."""
    items = C.calibrations(_read(CLEAN), _read(GENE), SPECIES_OF, MEMBERSHIP,
                           min_age=10)
    assert {c.clade for c in items} == {"A", "B"}
    ages = sorted({round(c.age, 6) for c in items})
    assert ages == pytest.approx([75.0, 90.0, 100.0, 150.0])
    for age in ages:
        matching = {c.clade for c in items if c.age == pytest.approx(age)}
        assert matching == {"A", "B"}


def test_the_date_is_negative_because_the_tips_sit_at_zero():
    items = C.calibrations(_read(CLEAN), _read(GENE), SPECIES_OF, MEMBERSHIP,
                           min_age=10)
    assert all(c.date < 0 for c in items)
    assert max(c.date for c in items) == pytest.approx(-75.0)


def test_young_nodes_are_dropped():
    items = C.calibrations(_read(CLEAN), _read(GENE), SPECIES_OF, MEMBERSHIP,
                           min_age=100)
    assert items
    assert sorted({round(c.age) for c in items}) == [100, 150]


def test_a_clade_whose_topology_disagrees_loses_its_internal_calibrations():
    """The cleanliness test, and the reason the surviving count is worth
    reading as a measure of agreement between the two trees.

    A clade's own crown survives whatever its internal topology, because the
    species at the species-tree root are all of them; everything below it goes.
    So a disagreeing clade keeps exactly one calibration and an agreeing one
    keeps every node -- which is the difference a constraint tree buys.
    """
    scrambled = f"({_clade('A', scrambled=True)}:1,{_clade('B')}:1);"
    items = C.calibrations(_read(CLEAN), _read(scrambled), SPECIES_OF,
                           MEMBERSHIP, min_age=10)
    per_clade = {c: [i for i in items if i.clade == c] for c in ("A", "B")}
    assert len(per_clade["A"]) == 1
    assert per_clade["A"][0].age == pytest.approx(150.0)   # the crown only
    assert len(per_clade["B"]) == 4
    assert 90.0 in [c.age for c in per_clade["B"]]
    assert 90.0 not in [c.age for c in per_clade["A"]]


def test_a_clade_does_not_inherit_the_age_of_species_it_does_not_carry():
    """The bug this pins inflated every deep calibration.

    A species-tree node's age is the divergence of ALL its descendants. A
    paralogue clade rarely carries all of them -- a vertebrate clade carries no
    sponges -- so the gene tips collected for that node are a subset, and their
    MRCA is the divergence of the subset, which is younger. Taking the node's
    own age handed the species-tree root's 892 Myr to clades whose own crown is
    less than half that, and the symptom was a maximum age per clade equal to
    the depth of the whole species tree.

    Clade B here carries only hum/mus/gal, whose divergence is 100. Before the
    fix it also received the root's 150.
    """
    membership = dict(MEMBERSHIP, B=["B_hum", "B_mus", "B_gal"])
    items = C.calibrations(_read(CLEAN), _read(GENE), SPECIES_OF, membership,
                           min_age=10)
    b_ages = sorted({round(c.age) for c in items if c.clade == "B"})
    assert b_ages == [90, 100]
    assert 150 not in b_ages
    # clade A carries every species, so it does keep the root
    assert 150 in {round(c.age) for c in items if c.clade == "A"}


def test_each_node_is_used_at_most_once_per_clade():
    """Several species-tree nodes can reduce to the same set of species a
    clade carries; only the one that is their MRCA is kept, so the same gene
    node is not calibrated twice at two different ages."""
    membership = dict(MEMBERSHIP, B=["B_hum", "B_mus"])
    items = [c for c in C.calibrations(_read(CLEAN), _read(GENE), SPECIES_OF,
                                       membership, min_age=10)
             if c.clade == "B"]
    assert len(items) == 1
    assert items[0].age == pytest.approx(90.0)


def test_a_tip_the_species_tree_cannot_place_does_not_spoil_its_node():
    """Being unplaceable is a gap in the metadata, not evidence that the
    topologies disagree, and the difference is expensive.

    On the real set one aardvark sequence sat inside FXR1 and FXR2 while its
    species was missing from the species tree. Counted as an intruder it made
    every node enclosing it unclean, and both clades fell from the 435 Myr
    jawed-vertebrate root to the 305 Myr teleost crown -- on one species.
    """
    species_of = dict(SPECIES_OF, A_odd="Orycteropus afer")   # not in CLEAN
    membership = {"A": MEMBERSHIP["A"] + ["A_odd"], "B": MEMBERSHIP["B"]}
    # A_odd sits INSIDE the clade, beside A_hum, which is where an
    # unannotated mammal actually lands. As a sibling of the whole clade it
    # would never fall inside any MRCA and there would be nothing to tolerate.
    inner = ("((((A_hum:1,A_odd:1):0.5,A_mus:1.5):1,A_gal:2):1,"
             "(A_api:1,A_dro:1):2)")
    gene = f"({inner}:1,{_clade('B')}:1);"
    items = C.calibrations(_read(CLEAN), _read(gene), species_of, membership,
                           min_age=10)
    a_ages = {round(c.age) for c in items if c.clade == "A"}
    assert 150 in a_ages, "the deep node must survive one unplaceable tip"
    assert any(c.unplaceable for c in items if c.clade == "A")


def test_a_placeable_tip_in_the_wrong_clade_still_spoils_it():
    """The tolerance is only for tips that cannot be checked. A tip whose
    species IS in the species tree and sits where the species tree says it
    should not is exactly the disagreement the test exists to catch."""
    membership = {"A": MEMBERSHIP["A"] + ["B_hum"], "B": MEMBERSHIP["B"]}
    items = C.calibrations(_read(CLEAN), _read(GENE), SPECIES_OF, membership,
                           min_age=10)
    assert not [c for c in items if c.clade == "A" and c.age > 100]


def test_a_gene_tip_absent_from_the_tree_is_simply_not_used():
    items = C.calibrations(_read(CLEAN), _read(GENE), SPECIES_OF,
                           dict(MEMBERSHIP, C=["C_ghost"]), min_age=10)
    assert "C" not in {c.clade for c in items}


# --- calibrations a dating run would merge ---------------------------------

#: A and B join their parent across a branch of 1e-7, so a dating run collapses
#: the two nodes into one. Two calibrations placed on them become two dates for
#: a single node.
COLLAPSING = "((((A:1,B:1):0.0000001,Cc:1):1,D:1):1,(E:1,F:1):1);"


def _cal(a, b, age):
    return C.Calibration("g", a, b, age, 2, 2)


def test_conflicting_calibrations_on_a_collapsed_node_are_both_dropped():
    """Two dates for one node is a contradiction, not a weak constraint. LSD2
    answers it with 'conflict or not enough information' and names nothing, so
    the pair has to go before the run rather than after it.

    Both are dropped by default because choosing between them is guesswork: a
    branch the gene tree resolves as zero, where the species tree puts 200 Myr,
    is a split the gene tree does not resolve at all.
    """
    tree = _read(COLLAPSING)
    kept, dropped = C.drop_unresolvable(
        tree, [_cal("A", "B", 100.0), _cal("A", "Cc", 300.0)], threshold=1e-6)
    assert kept == []
    assert len(dropped) == 2


def test_keeping_the_deepest_is_available_and_not_the_default():
    tree = _read(COLLAPSING)
    kept, dropped = C.drop_unresolvable(
        tree, [_cal("A", "B", 100.0), _cal("A", "Cc", 300.0)],
        threshold=1e-6, keep="deepest")
    assert [c.age for c in kept] == [300.0]
    assert [c.age for c in dropped] == [100.0]


def test_calibrations_that_agree_are_redundant_rather_than_conflicting():
    """One node constrained twice at the same age is not a conflict; it is the
    same constraint written twice, so one is kept."""
    tree = _read(COLLAPSING)
    kept, dropped = C.drop_unresolvable(
        tree, [_cal("A", "B", 100.0), _cal("A", "Cc", 100.0)], threshold=1e-6)
    assert len(kept) == 1
    assert len(dropped) == 1


def test_nodes_a_real_branch_separates_are_both_kept():
    tree = _read(COLLAPSING)
    kept, _ = C.drop_unresolvable(
        tree, [_cal("A", "Cc", 300.0), _cal("E", "F", 50.0)], threshold=1e-6)
    assert sorted(c.age for c in kept) == [50.0, 300.0]


def test_the_threshold_must_be_the_one_the_run_will_use():
    """Below the branch length nothing merges and both survive; at or above it
    they collide. The value is the dating method's, not this module's."""
    tree = _read(COLLAPSING)
    pair = [_cal("A", "B", 100.0), _cal("A", "Cc", 300.0)]
    assert len(C.drop_unresolvable(tree, pair, threshold=1e-9)[0]) == 2
    assert len(C.drop_unresolvable(tree, pair, threshold=1e-6)[0]) == 0


def test_the_dropped_report_names_the_threshold_and_the_reason():
    tree = _read(COLLAPSING)
    _, dropped = C.drop_unresolvable(
        tree, [_cal("A", "B", 100.0), _cal("A", "Cc", 300.0)], threshold=1e-6)
    text = C.format_dropped(dropped, threshold=1e-6)
    assert "conflict or not enough information" in text
    assert "Collapse N internal branches" in text
    assert C.format_dropped([], threshold=1e-6).startswith("no calibration")


# --- the file --------------------------------------------------------------


def test_the_date_file_is_parameterised_because_the_syntax_is_unverified():
    items = C.calibrations(_read(CLEAN), _read(GENE), SPECIES_OF, MEMBERSHIP,
                           min_age=10)
    default = C.format_dates(items)
    assert "\t-" in default
    custom = C.format_dates(items, line_format="mrca:{a},{b} {date}")
    assert custom.startswith("mrca:")
    assert "-100" in custom


def test_an_empty_calibration_set_produces_no_file_and_says_why():
    text = C.format_report(C.Coverage(), [])
    assert "no calibration survived" in text
    assert "constraining it to the species topology is the fix" in text
    assert C.format_dates([]) == ""


def test_the_report_separates_the_inputs_from_the_results():
    items = C.calibrations(_read(CLEAN), _read(GENE), SPECIES_OF, MEMBERSHIP,
                           min_age=10)
    text = C.format_report(C.coverage(_read(CLEAN), SPECIES_OF.values()), items)
    assert "--date-tip 0" in text
    assert "ages are written negative" in text.lower()
    assert "check and not an" in text
    assert "sign convention is" in text
