"""Relative rates: what the comparison must measure, and what it must ignore.

The design decision these pin is crown-to-tip rather than root-to-tip. Measured
from the root, every path carries its clade's stem, which is one number shared
by every taxon in that clade; it would enter every paired difference as a
constant and read as a rate difference it is not. The regression test for that
is the last one here, and it is the reason the module exists in this shape.
"""

from __future__ import annotations

import io

import pytest

from amyloevo.phylo import rates as R

pytest.importorskip("Bio")

#: Clade A accumulates 0.1 from its crown to every tip, clade B accumulates
#: 0.2, and both carry the same four taxa. So B's rate is exactly twice A's,
#: every paired difference is identical, and the stems (0.5 and 0.3) differ
#: from each other so that including them would be visible.
CLADE_A = "((A_t1:0.1,A_t2:0.1):0.0,(A_t3:0.1,A_t4:0.1):0.0)"
CLADE_B = "((B_t1:0.2,B_t2:0.2):0.0,(B_t3:0.2,B_t4:0.2):0.0)"
TWO_RATES = f"({CLADE_A}:0.5,{CLADE_B}:0.3);"

#: The same tree with the stems swapped and lengthened. Crown-to-tip is
#: untouched, so every statistic about RATE must be identical.
STEMS_CHANGED = f"({CLADE_A}:2.0,{CLADE_B}:0.01);"

MEMBERSHIP = {
    "A": ["A_t1", "A_t2", "A_t3", "A_t4"],
    "B": ["B_t1", "B_t2", "B_t3", "B_t4"],
}
TAXON_OF = {f"{clade}_t{i}": f"t{i}"
            for clade in ("A", "B") for i in (1, 2, 3, 4)}


def _read(newick: str):
    from Bio import Phylo

    return Phylo.read(io.StringIO(newick), "newick")


# --- what crown-to-tip measures -------------------------------------------


def test_crown_to_tip_excludes_the_stem():
    distances = R.crown_to_tip(_read(TWO_RATES), MEMBERSHIP["A"])
    assert set(distances) == set(MEMBERSHIP["A"])
    for value in distances.values():
        assert value == pytest.approx(0.1)


def test_crown_to_tip_refuses_a_group_that_is_not_a_clade():
    """Its MRCA holds sequences that are not in it, so the node is not that
    group's crown and every distance from it is inflated by however deep the
    intruders sit. An empty mapping makes the caller notice."""
    tree = _read(TWO_RATES)
    assert R.crown_to_tip(tree, ["A_t1", "B_t1"]) == {}


def test_crown_to_tip_needs_at_least_two_members():
    """The MRCA of one tip is that tip, and its distance to itself is zero --
    a number that would pass straight through every summary."""
    assert R.crown_to_tip(_read(TWO_RATES), ["A_t1"]) == {}


def test_stem_lengths_are_reported_separately():
    stems = R.stem_lengths(_read(TWO_RATES), MEMBERSHIP)
    assert stems["A"] == pytest.approx(0.5)
    assert stems["B"] == pytest.approx(0.3)


def test_per_clade_summary_uses_crown_distances():
    rows = {row.clade: row for row in R.per_clade(_read(TWO_RATES), MEMBERSHIP)}
    assert rows["A"].mean == pytest.approx(0.1)
    assert rows["B"].mean == pytest.approx(0.2)
    assert rows["A"].sd == pytest.approx(0.0)
    assert rows["A"].n == 4


def test_a_group_that_is_not_monophyletic_is_flagged_not_dropped():
    membership = dict(MEMBERSHIP, mixed=["A_t1", "B_t1"])
    rows = {row.clade: row for row in R.per_clade(_read(TWO_RATES), membership)}
    assert rows["mixed"].monophyletic is False
    assert rows["mixed"].n == 0


# --- the paired comparison -------------------------------------------------


def test_paired_frame_is_one_row_per_taxon():
    frame = R.paired_frame(_read(TWO_RATES), MEMBERSHIP, TAXON_OF)
    assert sorted(frame.index) == ["t1", "t2", "t3", "t4"]
    assert sorted(frame.columns) == ["A", "B"]
    assert frame.loc["t1", "A"] == pytest.approx(0.1)
    assert frame.loc["t1", "B"] == pytest.approx(0.2)


def test_a_taxon_with_two_sequences_in_one_clade_is_averaged_and_counted():
    """Dropping it would discard exactly the lineages with a recent
    within-clade duplication, which is a biased subset; entering it twice
    would weight that taxon twice."""
    newick = ("(((A_t1:0.1,A_x:0.3):0.0,(A_t3:0.1,A_t4:0.1):0.0):0.5,"
              f"{CLADE_B}:0.3);")
    membership = {"A": ["A_t1", "A_x", "A_t3", "A_t4"], "B": MEMBERSHIP["B"]}
    taxon_of = dict(TAXON_OF, A_x="t1")
    frame = R.paired_frame(_read(newick), membership, taxon_of)
    assert frame.attrs["averaged"] == 1
    # 0.1 and 0.3 from the same crown, averaged
    assert frame.loc["t1", "A"] == pytest.approx(0.2)


def _two_clades(n: int, rate_a: float, rate_b: float,
                stem_a: float = 0.5, stem_b: float = 0.3) -> str:
    """Two star clades over the same n taxa, at the given crown-to-tip rates."""
    a = ",".join(f"A_t{i}:{rate_a}" for i in range(1, n + 1))
    b = ",".join(f"B_t{i}:{rate_b}" for i in range(1, n + 1))
    return f"(({a}):{stem_a},({b}):{stem_b});"


def _maps(n: int):
    membership = {"A": [f"A_t{i}" for i in range(1, n + 1)],
                  "B": [f"B_t{i}" for i in range(1, n + 1)]}
    taxon_of = {f"{c}_t{i}": f"t{i}" for c in ("A", "B")
                for i in range(1, n + 1)}
    return membership, taxon_of


def test_the_paired_test_reports_the_ratio_of_rates():
    frame = R.paired_frame(_read(TWO_RATES), MEMBERSHIP, TAXON_OF)
    pairs = R.paired_tests(frame, min_pairs=2, permutations=999)
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.n_pairs == 4
    ratio = pair.ratio_of_means if pair.a == "B" else 1 / pair.ratio_of_means
    assert ratio == pytest.approx(2.0)


def test_four_taxa_cannot_resolve_below_an_eighth_however_large_the_effect():
    """Clade B's rate is exactly twice A's and every paired difference is
    identical, so the effect could not be cleaner -- and the p still cannot go
    below 2/16, because a sign-flip test over 4 pairs has 16 assignments and
    only the two all-same-sign ones are as extreme as the observed. The limit
    is the design of the test, not the strength of the signal, and MIN_PAIRS
    exists to stop it being reported as a weak result.
    """
    frame = R.paired_frame(_read(TWO_RATES), MEMBERSHIP, TAXON_OF)
    pair = R.paired_tests(frame, min_pairs=2, permutations=9999)[0]
    assert pair.p == pytest.approx(2 / 16, abs=0.02)
    assert not pair.at_floor


def test_enough_taxa_put_the_same_effect_on_the_permutation_floor():
    membership, taxon_of = _maps(20)
    frame = R.paired_frame(_read(_two_clades(20, 0.1, 0.2)), membership,
                           taxon_of)
    pair = R.paired_tests(frame, min_pairs=10, permutations=999)[0]
    assert pair.n_pairs == 20
    assert pair.at_floor
    assert pair.p == pytest.approx(1 / 1000)


def test_equal_rates_give_a_p_that_is_not_near_the_floor():
    equal = (f"({CLADE_A}:0.5,"
             "((B_t1:0.1,B_t2:0.1):0.0,(B_t3:0.1,B_t4:0.1):0.0):0.3);")
    frame = R.paired_frame(_read(equal), MEMBERSHIP, TAXON_OF)
    pair = R.paired_tests(frame, min_pairs=2, permutations=999)[0]
    assert not pair.at_floor
    assert pair.p == pytest.approx(1.0)
    assert pair.median_difference == pytest.approx(0.0)


def test_too_few_pairs_is_skipped_with_the_attainable_floor_named():
    """A sign-flip test over n pairs has 2**n assignments, so below a handful
    of taxa no result can clear a corrected threshold however large the
    difference. Saying so beats reporting a p of 0.25."""
    frame = R.paired_frame(_read(TWO_RATES), MEMBERSHIP, TAXON_OF)
    pair = R.paired_tests(frame.head(3), min_pairs=10)[0]
    assert pair.skipped
    assert "need 10" in pair.skipped
    assert "0.125" in pair.skipped


def test_adjustment_excludes_the_pairs_that_were_not_tested():
    """A test that did not run is not a test that passed, so it is not in the
    Benjamini-Hochberg family either."""
    frame = R.paired_frame(_read(TWO_RATES), MEMBERSHIP, TAXON_OF)
    pairs = R.paired_tests(frame, min_pairs=10)
    assert all(p.skipped for p in pairs)
    assert R.adjust(pairs) == {}


def test_permutation_p_is_reproducible():
    frame = R.paired_frame(_read(TWO_RATES), MEMBERSHIP, TAXON_OF)
    first = R.paired_tests(frame, min_pairs=2, permutations=499)[0].p
    second = R.paired_tests(frame, min_pairs=2, permutations=499)[0].p
    assert first == second


# --- the regression that justifies the whole design -----------------------


def test_stem_length_cannot_change_a_rate_statement():
    """The reason crown-to-tip exists. A clade's stem is one branch shared by
    every taxon in it, so from the root it enters every paired difference as a
    constant; a rate comparison must not move when it changes. Here the stems
    go from 0.5/0.3 to 2.0/0.01 and nothing about the rates may differ.
    """
    kwargs = dict(min_pairs=2, permutations=499)
    before = R.paired_tests(
        R.paired_frame(_read(TWO_RATES), MEMBERSHIP, TAXON_OF), **kwargs)[0]
    after = R.paired_tests(
        R.paired_frame(_read(STEMS_CHANGED), MEMBERSHIP, TAXON_OF), **kwargs)[0]
    assert after.median_difference == pytest.approx(before.median_difference)
    assert after.ratio_of_means == pytest.approx(before.ratio_of_means)
    assert after.p == pytest.approx(before.p)
    # and the stems themselves did change, so the fixture is doing its job
    assert (R.stem_lengths(_read(STEMS_CHANGED), MEMBERSHIP)["A"]
            != pytest.approx(R.stem_lengths(_read(TWO_RATES),
                                            MEMBERSHIP)["A"]))


# --- the report ------------------------------------------------------------


def test_report_names_the_duplication_calibration_and_the_zero_branches():
    result = R.report(_read(TWO_RATES), MEMBERSHIP, TAXON_OF, min_pairs=2,
                      permutations=499)
    text = R.format_rate_report(result, R.adjust(result.pairs))
    assert "whole-genome duplication" in text
    assert "Zero-length branches" in text
    assert "ratio" in text


#: An alignment carrying two sequences the tree does not: exactly the shape the
#: occupancy filter produces, since the tree is inferred from a filtered copy
#: that the pipeline never writes out.
ALIGNMENT = {f"{c}_t{i}": "MKVL" for c in ("A", "B") for i in (1, 2, 3, 4)}
ALIGNMENT_PLUS = dict(ALIGNMENT, A_gappy="M---", B_gappy="M---")


def test_dating_inputs_names_both_directions_of_a_mismatch():
    tree = _read(TWO_RATES)
    inputs = R.dating_inputs(tree, ALIGNMENT_PLUS)
    assert not inputs.ready
    assert inputs.alignment_only == ["A_gappy", "B_gappy"]
    assert inputs.tree_only == []
    assert "occupancy filter" in inputs.message()


def test_dating_inputs_passes_when_the_two_agree():
    inputs = R.dating_inputs(_read(TWO_RATES), ALIGNMENT)
    assert inputs.ready
    assert len(inputs.shared) == 8
    assert "agree on all 8" in inputs.message()


def test_dating_command_refuses_a_mismatched_pair():
    """The failure this prevents: a command pairing a 372-sequence alignment
    with a 357-tip tree, which IQ-TREE rejects only after reading both files
    and several hundred lines of output."""
    inputs = R.dating_inputs(_read(TWO_RATES), ALIGNMENT_PLUS)
    with pytest.raises(ValueError, match="do NOT describe the same"):
        R.dating_command("tree.nwk", "aln.fa", inputs)


def test_dating_command_refuses_to_be_called_without_the_check():
    with pytest.raises(TypeError, match="dating_inputs"):
        R.dating_command("tree.nwk", "aln.fa", None)


def test_dating_command_carries_a_confidence_interval():
    inputs = R.dating_inputs(_read(TWO_RATES), ALIGNMENT)
    command = R.dating_command("tree.nwk", "aln.fa", inputs,
                               dates_path="dates.txt", binary="iqtree3")
    assert "--date dates.txt" in command
    assert "--date-ci" in command
    assert command.startswith("iqtree3")


def test_identical_sequences_are_kept_by_default():
    """Without this IQ-TREE reduces the alignment to unique sequences and then
    fails against a fixed tree that still carries every tip -- 24 of 357 on
    this family. And on this data it is a decision about the data: one of the
    33 groups of identical tips carries different APR traits between its
    members, so de-duplicating deletes an observation."""
    inputs = R.dating_inputs(_read(TWO_RATES), ALIGNMENT)
    assert "--keep-ident" in R.dating_command("t.nwk", "a.fa", inputs)
    assert "--keep-ident" not in R.dating_command(
        "t.nwk", "a.fa", inputs, keep_identical=False)


def test_the_tree_only_direction_names_de_duplication_as_a_cause():
    """The message this replaces said the files came from different runs,
    which is wrong for the case that actually happens: files that agree
    exactly, and a tool that de-duplicates one of them internally."""
    tree = _read(TWO_RATES)
    short = {k: v for k, v in ALIGNMENT.items() if k != "A_t4"}
    message = R.dating_inputs(tree, short).message()
    assert "de-duplicated" in message
    assert "--keep-ident" in message


def test_matched_alignment_keeps_only_the_tips_and_leaves_columns_alone():
    """Trimming all-gap columns would renumber every block coordinate the
    project reports, which is a worse trade than an empty column."""
    matched = R.matched_alignment(_read(TWO_RATES), ALIGNMENT_PLUS)
    assert set(matched) == set(ALIGNMENT)
    assert all(len(seq) == 4 for seq in matched.values())
    assert R.dating_inputs(_read(TWO_RATES), matched).ready
