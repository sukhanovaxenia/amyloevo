"""Header parsing decides who is on the species tree, so it fails loudly."""

from __future__ import annotations

import pytest

from amyloevo.io.headers import HeaderParseError, detect_format, parse_header, parse_headers


def test_eggnog_header():
    parsed = parse_header(">9606.ENSP00000373809")
    assert parsed.taxid == 9606
    assert parsed.protein_id == "ENSP00000373809"
    assert parsed.has_taxon


def test_uniprot_header():
    parsed = parse_header("sp|P51114|FXR1_HUMAN", fmt="uniprot")
    assert parsed.protein_id == "P51114"
    assert parsed.mnemonic == "HUMAN"
    assert parsed.has_taxon


def test_ncbi_header():
    parsed = parse_header("NP_005078.2 FXR1 isoform a [Homo sapiens]", fmt="ncbi")
    assert parsed.protein_id == "NP_005078.2"
    assert parsed.organism == "Homo sapiens"


def test_a_header_of_the_wrong_format_returns_none():
    assert parse_header("sp|P51114|FXR1_HUMAN", fmt="eggnog") is None


def test_unknown_format_is_rejected_by_name():
    with pytest.raises(ValueError, match="unknown header format"):
        parse_header(">9606.ENSP1", fmt="ensembl")


def test_strict_parsing_refuses_a_partial_match():
    """Silently analysing whichever orthologues matched leaves a result with no
    recoverable denominator -- the phylogenetic test would run on a subset
    nobody chose."""
    headers = [">9606.ENSP1", ">10090.ENSMUSP1", ">not-a-taxid|X"]
    with pytest.raises(HeaderParseError) as excinfo:
        parse_headers(headers)
    message = str(excinfo.value)
    assert "1/3" in message
    assert "species tree" in message


def test_non_strict_parsing_reports_what_it_could_not_read():
    headers = [">9606.ENSP1", ">10090.ENSMUSP1", ">not-a-taxid|X"]
    parsed, unparsed = parse_headers(headers, strict=False)
    assert len(parsed) == 2
    assert unparsed == ["not-a-taxid|X"]


def test_the_error_lists_only_the_first_few_offenders():
    headers = [f">bad-{i}" for i in range(10)]
    with pytest.raises(HeaderParseError, match="and 7 more"):
        parse_headers(headers)


@pytest.mark.parametrize(
    "headers,expected",
    [
        ([">9606.ENSP1", ">10090.ENSMUSP1"], "eggnog"),
        (["sp|P51114|FXR1_HUMAN", "tr|Q9Y6M1|IF2B1_HUMAN"], "uniprot"),
        (["NP_005078.2 x [Homo sapiens]"], "ncbi"),
        ([">???"], None),
    ],
)
def test_format_detection(headers, expected):
    assert detect_format(headers) == expected
