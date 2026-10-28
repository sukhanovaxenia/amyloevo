"""Taxonomy resolution and the labelling policy."""

from __future__ import annotations

import pytest

from amyloevo.io.annotate import DEFAULT_LABEL_TEMPLATE, annotate, make_labels, render
from amyloevo.io.taxonomy import TaxonomyDB, load_taxonomy

# A miniature taxdump. The tab-pipe-tab delimiter is the format's own.
NAMES_DMP = """\
9606\t|\tHomo sapiens\t|\t\t|\tscientific name\t|
9606\t|\thuman\t|\t\t|\tgenbank common name\t|
9605\t|\tHomo\t|\t\t|\tscientific name\t|
9604\t|\tHominidae\t|\t\t|\tscientific name\t|
9443\t|\tPrimates\t|\t\t|\tscientific name\t|
40674\t|\tMammalia\t|\t\t|\tscientific name\t|
7227\t|\tDrosophila melanogaster\t|\t\t|\tscientific name\t|
7215\t|\tDrosophila\t|\t\t|\tscientific name\t|
1\t|\troot\t|\t\t|\tscientific name\t|
"""

NODES_DMP = """\
9606\t|\t9605\t|\tspecies\t|
9605\t|\t9604\t|\tgenus\t|
9604\t|\t9443\t|\tfamily\t|
9443\t|\t40674\t|\torder\t|
40674\t|\t1\t|\tclass\t|
7227\t|\t7215\t|\tspecies\t|
7215\t|\t1\t|\tgenus\t|
1\t|\t1\t|\tno rank\t|
"""

MERGED_DMP = "99999\t|\t9606\t|\n"


@pytest.fixture
def taxdump(tmp_path):
    (tmp_path / "names.dmp").write_text(NAMES_DMP)
    (tmp_path / "nodes.dmp").write_text(NODES_DMP)
    (tmp_path / "merged.dmp").write_text(MERGED_DMP)
    return tmp_path


# --------------------------------------------------------------------------- #
# Taxonomy
# --------------------------------------------------------------------------- #
def test_resolves_name_and_lineage(taxdump) -> None:
    db = TaxonomyDB.from_taxdump(taxdump)
    taxon = db.resolve(9606)
    assert taxon.name == "Homo sapiens"
    assert taxon.rank == "species"
    assert taxon.at("genus") == "Homo"
    assert taxon.at("family") == "Hominidae"
    assert taxon.at("order") == "Primates"
    assert taxon.at("class") == "Mammalia"


def test_only_the_scientific_name_is_taken(taxdump) -> None:
    """names.dmp carries common names and synonyms under the same taxid."""
    assert TaxonomyDB.from_taxdump(taxdump).resolve(9606).name == "Homo sapiens"


def test_a_retired_taxid_resolves_through_merged(taxdump) -> None:
    """An orthologue set assembled years ago carries taxids NCBI has merged.

    Without merged.dmp those lineages come back empty and their species leave
    the analysis without appearing in any denominator.
    """
    assert TaxonomyDB.from_taxdump(taxdump).resolve(99999).name == "Homo sapiens"


def test_a_missing_rank_is_empty_not_an_error(taxdump) -> None:
    assert TaxonomyDB.from_taxdump(taxdump).resolve(7227).at("family") == ""


def test_unknown_taxid_resolves_to_none(taxdump) -> None:
    db = TaxonomyDB.from_taxdump(taxdump)
    assert db.resolve(123456789) is None
    assert db.resolve(None) is None
    assert db.resolve("not-a-number") is None


def test_coverage_reports_what_was_not_resolved(taxdump) -> None:
    db = TaxonomyDB.from_taxdump(taxdump)
    assert db.coverage([9606, 7227, 123456789]) == {
        "taxa": 3, "resolved": 2, "unresolved": 1
    }


def test_missing_taxdump_names_the_download(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="taxdump.tar.gz"):
        TaxonomyDB.from_taxdump(tmp_path)


def test_table_fallback(tmp_path) -> None:
    path = tmp_path / "tax.tsv"
    path.write_text("id\ttaxid\tsci\tgenus\tclass\nx\t9606\tHomo sapiens\tHomo\tMammalia\n")
    db = TaxonomyDB.from_table(path)
    taxon = db.resolve(9606)
    assert taxon.name == "Homo sapiens"
    assert taxon.at("genus") == "Homo"


def test_table_without_taxid_is_refused(tmp_path) -> None:
    path = tmp_path / "bad.tsv"
    path.write_text("id\tname\nx\ty\n")
    with pytest.raises(ValueError, match="no 'taxid' column"):
        TaxonomyDB.from_table(path)


def test_empty_database_is_usable_not_fatal() -> None:
    db = load_taxonomy()
    assert db.is_empty
    assert db.resolve(9606) is None


# --------------------------------------------------------------------------- #
# Labelling
# --------------------------------------------------------------------------- #
def test_render_drops_missing_fields_without_leaving_gaps() -> None:
    assert render("{tax_name} {group}", {"tax_name": "Homo sapiens"}) == "Homo sapiens"
    assert render("{a} {b}", {}) == ""


def test_default_label_is_species_plus_group(taxdump) -> None:
    result = annotate(
        ["9606.ENSP1", "7227.FBpp1"],
        taxonomy=TaxonomyDB.from_taxdump(taxdump),
        groups={"9606.ENSP1": "FXR1", "7227.FBpp1": "FXR1"},
    )
    assert list(result.table["label"]) == [
        "Homo sapiens FXR1", "Drosophila melanogaster FXR1"
    ]


def test_collisions_are_disambiguated_on_every_member(taxdump) -> None:
    """Both rows get the accession, not just the second.

    A suffix on one row of a colliding pair reads as a correction applied to
    that row; on both it reads as a property of the pair, which is what it is.
    """
    result = annotate(
        ["9606.ENSP1", "9606.ENSP2"],
        taxonomy=TaxonomyDB.from_taxdump(taxdump),
        groups={"9606.ENSP1": "FXR1", "9606.ENSP2": "FXR1"},
    )
    labels = list(result.table["label"])
    assert labels == ["Homo sapiens FXR1 [ENSP1]", "Homo sapiens FXR1 [ENSP2]"]
    assert result.collisions_disambiguated == 2


def test_label_falls_back_down_the_chain(taxdump) -> None:
    """An unresolved taxon gets its id, never a blank or a guess."""
    result = annotate(
        ["9606.ENSP1", "123456789.XP1"],
        taxonomy=TaxonomyDB.from_taxdump(taxdump),
        groups={"9606.ENSP1": "FXR1"},
    )
    labels = list(result.table["label"])
    assert labels[0] == "Homo sapiens FXR1"
    assert labels[1] == "123456789.XP1"
    assert result.unresolved_taxa == ["123456789.XP1"]


def test_template_is_configurable(taxdump) -> None:
    """A figure of one species' paralogues wants the group alone."""
    result = annotate(
        ["9606.ENSP1", "9606.ENSP2"],
        taxonomy=TaxonomyDB.from_taxdump(taxdump),
        groups={"9606.ENSP1": "FXR1", "9606.ENSP2": "FMR1"},
        label_template="{group}",
    )
    assert list(result.table["label"]) == ["FXR1", "FMR1"]


def test_the_id_is_always_kept(taxdump) -> None:
    """label is for display; joins and provenance go through id."""
    result = annotate(["9606.ENSP1"], taxonomy=TaxonomyDB.from_taxdump(taxdump))
    assert result.table["id"].iloc[0] == "9606.ENSP1"
    assert result.table["protein"].iloc[0] == "ENSP1"


def test_rank_columns_are_present(taxdump) -> None:
    result = annotate(["9606.ENSP1"], taxonomy=TaxonomyDB.from_taxdump(taxdump))
    row = result.table.iloc[0]
    assert row["taxid"] == 9606
    assert row["tax_name"] == "Homo sapiens"
    assert row["genus"] == "Homo"
    assert row["class"] == "Mammalia"


def test_make_labels_on_a_plain_list_of_dicts() -> None:
    labels, n = make_labels(
        [{"id": "a", "tax_name": "Mus musculus", "group": "FXR1", "protein": "a"}],
        template=DEFAULT_LABEL_TEMPLATE,
    )
    assert labels == ["Mus musculus FXR1"]
    assert n == 0


def test_the_disambiguating_suffix_is_the_accession_not_the_full_id(taxdump) -> None:
    """Inside a colliding set the taxon is identical by construction.

    A full `9606.ENSP1` therefore spends axis width on a prefix that is the same
    on every member of the set, and the accession alone separates them.
    """
    result = annotate(
        ["9606.ENSP00000350170", "9606.ENSP00000359506"],
        taxonomy=TaxonomyDB.from_taxdump(taxdump),
        groups={"9606.ENSP00000350170": "FXR1", "9606.ENSP00000359506": "FXR1"},
    )
    assert list(result.table["label"]) == [
        "Homo sapiens FXR1 [ENSP00000350170]",
        "Homo sapiens FXR1 [ENSP00000359506]",
    ]


def test_this_matches_amyloscopes_implementation_of_the_same_policy() -> None:
    """amyloevo builds these tables and amyloscope labels figures from them.

    Two implementations of one policy that disagree on the output are worse than
    one implementation that is merely imperfect. This caught a real divergence:
    the suffix was the accession here and the full id there, differing on 52 of
    the 372 FXR1 rows. Skipped where amyloscope is not installed.
    """
    amyloscope = pytest.importorskip("amyloscope.io.metadata")
    rows = [
        {"id": "9606.ENSP1", "protein": "ENSP1", "taxid": 9606,
         "tax_name": "Homo sapiens", "group": "FXR1", "protein_name": "FXR1"},
        {"id": "9606.ENSP2", "protein": "ENSP2", "taxid": 9606,
         "tax_name": "Homo sapiens", "group": "FXR1", "protein_name": "FXR1"},
        {"id": "10090.ENSP3", "protein": "ENSP3", "taxid": 10090,
         "tax_name": "Mus musculus", "group": "FXR1", "protein_name": "Fxr1"},
        {"id": "7227.FBpp4", "protein": "FBpp4", "taxid": 7227,
         "tax_name": "Drosophila melanogaster", "group": "", "protein_name": "Fmr1"},
    ]
    ours, our_n = make_labels(rows, template=DEFAULT_LABEL_TEMPLATE)
    theirs, their_n = amyloscope.build_labels(rows)
    assert ours == [theirs[r["id"]] for r in rows]
    assert our_n == their_n


def test_a_missing_rank_does_not_become_a_group_named_nan():
    """pandas reads an absent cell as NaN, NaN is truthy, and str(NaN) is
    "nan" -- so an emptiness check that works for csv's "" lets NaN through
    and the legend gains a group called "nan". The row must fall through to
    the next rank instead."""
    import pandas as pd

    from amyloevo.io.annotate import major_groups

    table = pd.DataFrame({
        "id": ["a", "b", "c", "d"],
        "phylum": ["Arthropoda", "Arthropoda", "Arthropoda", None],
        "class": ["Insecta", "Insecta", "Insecta", "Choanoflagellata"],
    })
    groups = major_groups(table, ("phylum", "class"), min_size=1, suffix="")
    assert groups["d"] == "Choanoflagellata"
    assert "nan" not in set(groups.values())


def test_a_missing_rank_is_recognised_whatever_spells_it():
    import pandas as pd

    from amyloevo.io.annotate import major_groups

    table = pd.DataFrame({
        "id": ["a", "b", "c", "d", "e"],
        "phylum": ["Arthropoda", "", "  ", "NaN", "nan"],
        "class": ["Insecta", "Clitellata", "Cestoda", "Ascidiacea", "Trematoda"],
    })
    groups = major_groups(table, ("phylum", "class"), min_size=1, suffix="")
    assert groups["a"] == "Arthropoda"
    for key, expected in (("b", "Clitellata"), ("c", "Cestoda"),
                          ("d", "Ascidiacea"), ("e", "Trematoda")):
        assert groups[key] == expected, f"{key} fell through as {groups[key]!r}"
