"""The gene tree: what rooting and reconciliation must refuse to do silently.

Each of these pins a way the phylogenetic step can run to completion on the
wrong tips and still print a number. The rooting tests matter most: D and lambda
read ancestry off the branching order, so a tree rooted through a paraphyletic
outgroup gives well-formed answers to a question nobody asked.
"""

from __future__ import annotations

import io

import pandas as pd
import pytest

from amyloevo.phylo import tree as T

pytest.importorskip("Bio")

#: Outgroup {O1,O2} is monophyletic here.
CLEAN = "(((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1):0.2,(O1:0.1,O2:0.1):0.2);"

#: Outgroup {O1,O2} is NOT: its MRCA also holds A and B.
MESSY = "(((A:0.1,O1:0.1):0.1,(B:0.1,O2:0.1):0.1):0.2,(C:0.1,D:0.1):0.2);"

#: With SH-aLRT/UFBoot labels on the internal nodes.
SUPPORTED = (
    "(((A:0.1,B:0.1)95/99:0.1,(C:0.1,D:0.1)60/80:0.1)99/100:0.2,"
    "(O1:0.1,O2:0.1)88/97:0.2);"
)


def _read(newick: str):
    from Bio import Phylo

    return Phylo.read(io.StringIO(newick), "newick")


# --- rooting ---------------------------------------------------------------


def test_roots_on_a_monophyletic_outgroup():
    result = T.root_on_outgroup(_read(CLEAN), ["O1", "O2"])
    assert result.monophyletic
    assert not result.refused
    assert result.tree is not None
    # the root now separates the outgroup from everything else
    sides = [{t.name for t in child.get_terminals()} for child in result.tree.root]
    assert {"O1", "O2"} in sides


def test_the_rooted_tree_is_bifurcating_at_the_root():
    """ape::is.rooted is FALSE for a root of degree three, and phyloglm then
    refuses the tree -- so a multi-tip outgroup must not leave a polytomy."""
    for newick in (CLEAN, "((A:.1,B:.1):.1,(C:.1,D:.1):.1,(O1:.1,O2:.1):.2);"):
        rooted = T.root_on_outgroup(_read(newick), ["O1", "O2"]).tree
        assert len(rooted.root.clades) == 2


def test_rooting_preserves_every_tip_to_tip_distance():
    """What rooting may not change. Root-to-tip distances DO move -- Biopython
    roots at the outgroup MRCA, not on its stem, and the module says so -- but a
    patristic distance between two tips is a property of the unrooted tree and
    must survive. clade_checks and the monophyly statements rest on this."""
    original = _read(CLEAN)
    rooted = T.root_on_outgroup(original, ["O1", "O2"]).tree
    pairs = [("A", "O1"), ("A", "D"), ("C", "O2"), ("B", "C")]
    for x, y in pairs:
        before = original.distance(x, y)
        after = rooted.distance(x, y)
        assert after == pytest.approx(before), f"{x}-{y} moved"


def test_an_outgroup_spanning_two_children_of_an_unrooted_root_is_still_a_split():
    """IQ-TREE writes an unrooted tree as a trifurcation. An outgroup made of
    two of those three children has the root as its MRCA, so an MRCA-only check
    calls it paraphyletic -- it is not, and the complement check catches that."""
    newick = "((O1:.1,O2:.1):.1,(O3:.1,O4:.1):.1,(A:.1,B:.1):.2);"
    result = T.root_on_outgroup(_read(newick), ["O1", "O2", "O3", "O4"])
    assert result.monophyletic
    assert result.intruders == []
    assert result.tree is not None


def test_a_single_tip_outgroup_needs_no_inserted_node():
    rooted = T.root_on_outgroup(_read(CLEAN), ["O1"]).tree
    assert len(rooted.root.clades) == 2
    assert sorted(T.tip_labels(rooted)) == ["A", "B", "C", "D", "O1", "O2"]


def test_refuses_a_paraphyletic_outgroup_and_names_the_intruders():
    result = T.root_on_outgroup(_read(MESSY), ["O1", "O2"])
    assert not result.monophyletic
    assert result.tree is None
    assert result.refused
    assert result.intruders == ["A", "B"]
    assert "not monophyletic" in str(result)


def test_roots_a_paraphyletic_outgroup_only_when_told_to():
    result = T.root_on_outgroup(_read(MESSY), ["O1", "O2"], require_monophyly=False)
    assert result.tree is not None
    assert not result.monophyletic          # still reported as unsound
    assert result.intruders == ["A", "B"]


def test_reports_declared_outgroup_ids_absent_from_the_tree():
    result = T.root_on_outgroup(_read(CLEAN), ["O1", "O2", "O3"])
    assert result.missing == ["O3"]
    assert result.monophyletic


def test_refuses_when_no_outgroup_id_is_present():
    result = T.root_on_outgroup(_read(CLEAN), ["Z1", "Z2"])
    assert result.tree is None
    assert "no declared outgroup id" in result.refused


def test_refuses_when_the_outgroup_is_everything():
    result = T.root_on_outgroup(_read(CLEAN), ["A", "B", "C", "D", "O1", "O2"])
    assert result.tree is None
    assert "whole tree" in result.refused


def test_rooting_does_not_mutate_the_input_tree():
    original = _read(CLEAN)
    before = [t.name for t in original.get_terminals()]
    T.root_on_outgroup(original, ["O1", "O2"])
    assert [t.name for t in original.get_terminals()] == before


# --- support ---------------------------------------------------------------


def test_support_summary_reads_alrt_and_ufboot():
    summary = T.support_summary(_read(SUPPORTED))
    assert summary.has_alrt
    # four labelled internal nodes; the root itself carries no label and is
    # excluded anyway, since a root split is not a hypothesis the data tested
    assert summary.n_internal == 4
    assert summary.n_scored == 4
    # 95/99, 99/100 and 88/97 pass both; 60/80 fails UFBoot and SH-aLRT
    assert summary.both_ok == 3
    assert summary.ufboot_ok == 3
    assert summary.alrt_ok == 3


def test_support_summary_says_so_when_there_is_no_support_at_all():
    summary = T.support_summary(_read(CLEAN))
    assert summary.n_scored == 0
    assert "none carrying a support value" in str(summary)


def test_single_support_value_is_read_as_ufboot_and_alrt_is_not_claimed():
    summary = T.support_summary(_read("((A:0.1,B:0.1)99:0.1,(C:0.1,D:0.1)70:0.1);"))
    assert not summary.has_alrt
    assert summary.ufboot_ok == 1
    assert "no SH-aLRT" in str(summary)


# --- identical tips --------------------------------------------------------


def test_identical_groups_finds_exact_duplicates_only():
    alignment = {
        "a": "MKV--LL",
        "b": "MKV--LL",          # identical to a
        "c": "MKV--LI",          # one residue different
        "d": "MKVQQLL",
        "e": "MKV--LL",          # identical to a and b
    }
    groups = T.identical_groups(alignment)
    assert len(groups) == 1
    assert sorted(groups[0]) == ["a", "b", "e"]


def test_trait_conflicts_flags_identical_tips_whose_traits_differ():
    groups = [["a", "b", "e"], ["x", "y"]]
    traits = pd.DataFrame(
        {"blk1": [1, 0, 1, 1, 1], "blk2": [0, 0, 1, 1, 1]},
        index=["a", "b", "e", "x", "y"],
    )
    report = T.trait_conflicts(groups, traits)
    assert report.n_groups == 2
    assert report.n_members == 5
    assert report.n_testable == 2
    assert report.n_conflicting == 1
    conflict = report.conflicts[0]
    assert sorted(conflict.members) == ["a", "b", "e"]
    assert sorted(conflict.columns) == ["blk1", "blk2"]
    assert "DIFFERENT traits" in str(report)


def test_trait_conflicts_is_silent_when_identical_tips_agree():
    traits = pd.DataFrame({"blk1": [1, 1], "blk2": [0, 0]}, index=["a", "b"])
    report = T.trait_conflicts([["a", "b"]], traits)
    assert report.n_conflicting == 0
    assert report.n_testable == 1
    assert "would have lost nothing" in str(report)


def test_a_group_with_one_trait_row_is_counted_but_not_tested():
    traits = pd.DataFrame({"blk1": [1]}, index=["a"])
    report = T.trait_conflicts([["a", "b"]], traits)
    assert report.n_groups == 1
    assert report.n_testable == 0           # absent is not the same as equal
    assert report.n_conflicting == 0


def test_no_identical_tips_is_reported_as_such():
    assert "no identical tips" in str(T.trait_conflicts([], pd.DataFrame()))


# --- occupancy -------------------------------------------------------------


def test_tip_occupancy_counts_gaps_and_dots():
    occ = T.tip_occupancy({"a": "AB--", "b": "A...", "c": "ABCD"})
    assert occ["a"] == pytest.approx(0.5)
    assert occ["b"] == pytest.approx(0.25)
    assert occ["c"] == pytest.approx(1.0)


def test_low_occupancy_returns_the_worst_first():
    low = T.low_occupancy({"a": "AB--", "b": "A---", "c": "ABCD"}, minimum=0.6)
    assert list(low) == ["b", "a"]
    assert "c" not in low


# --- reconciliation --------------------------------------------------------


def test_reconcile_counts_losses_in_both_directions():
    rec = T.reconcile(["a", "b", "c"], ["b", "c", "d"])
    assert rec.shared == ["b", "c"]
    assert rec.tips_without_traits == ["a"]
    assert rec.traits_without_tips == ["d"]
    assert rec.n_shared == 2


def test_prune_to_keeps_the_requested_tips():
    pruned = T.prune_to(_read(CLEAN), ["A", "B", "C"])
    assert sorted(T.tip_labels(pruned)) == ["A", "B", "C"]


def test_prune_to_refuses_an_id_that_is_not_a_tip():
    with pytest.raises(KeyError, match="not tips"):
        T.prune_to(_read(CLEAN), ["A", "B", "ZZ"])


def test_prune_to_refuses_a_tree_with_no_topology_left():
    with pytest.raises(ValueError, match="at least 3"):
        T.prune_to(_read(CLEAN), ["A", "B"])


# --- paralogue clades ------------------------------------------------------


def test_clade_checks_reports_monophyly_per_group():
    checks = {c.group: c for c in T.clade_checks(
        _read(CLEAN), {"left": ["A", "B"], "right": ["C", "D"], "out": ["O1", "O2"]}
    )}
    assert all(c.monophyletic for c in checks.values())
    assert checks["left"].n == 2


def test_clade_checks_names_the_intruders_of_a_scattered_group():
    checks = {c.group: c for c in T.clade_checks(
        _read(CLEAN), {"scattered": ["A", "C"], "rest": ["B", "D"]}
    )}
    assert not checks["scattered"].monophyletic
    assert sorted(checks["scattered"].intruders) == ["B", "D"]
    assert "NOT monophyletic" in str(checks["scattered"])


def test_clade_checks_reports_ids_missing_from_the_tree():
    checks = {c.group: c for c in T.clade_checks(
        _read(CLEAN), {"left": ["A", "B", "GHOST"]}
    )}
    assert checks["left"].missing == ["GHOST"]
    assert checks["left"].n == 2


# --- divergence ------------------------------------------------------------


def test_root_to_tip_measures_patristic_distance():
    distances = T.root_to_tip(_read(CLEAN))
    assert distances["A"] == pytest.approx(0.4)
    assert distances["O1"] == pytest.approx(0.3)
