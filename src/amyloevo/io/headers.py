"""Parse orthologue FASTA headers into a taxon identity.

EggNOG writes ``>taxid.proteinid`` (``9606.ENSP00000373809``), which is the
default here. Other sources differ, and a header pattern that silently fails to
match is the most consequential failure in this package: an orthologue with no
taxid cannot be placed on the species tree, and a set with several such
orthologues yields a phylogenetic test run on a subset nobody chose.

So parsing is strict by default. ``parse_headers`` reports every header it could
not read rather than skipping them, and the caller decides.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: EggNOG: taxid.proteinid
EGGNOG_PATTERN = re.compile(r"^(?P<taxid>\d+)\.(?P<protein>\S+)")
#: UniProt: sp|ACC|NAME_SPECIES / tr|ACC|NAME_SPECIES
UNIPROT_PATTERN = re.compile(r"^(?:sp|tr)\|(?P<protein>[^|]+)\|\S*_(?P<mnemonic>\w+)")
#: NCBI-style: accession [Genus species]
NCBI_PATTERN = re.compile(r"^(?P<protein>\S+).*\[(?P<organism>[^\]]+)\]")

PATTERNS = {
    "eggnog": EGGNOG_PATTERN,
    "uniprot": UNIPROT_PATTERN,
    "ncbi": NCBI_PATTERN,
}


@dataclass(frozen=True)
class OrthologHeader:
    """One orthologue's identity, as far as its header states it."""

    raw: str
    protein_id: str
    taxid: int | None = None
    organism: str | None = None
    mnemonic: str | None = None

    @property
    def has_taxon(self) -> bool:
        """Whether this orthologue can be placed on a species tree at all."""
        return self.taxid is not None or bool(self.organism or self.mnemonic)


class HeaderParseError(ValueError):
    """Raised when headers cannot be read under the chosen format."""


def parse_header(header: str, *, fmt: str = "eggnog") -> OrthologHeader | None:
    """Parse one header, or None when it does not match ``fmt``."""
    header = header.lstrip(">").strip()
    if fmt not in PATTERNS:
        raise ValueError(f"unknown header format {fmt!r}; known: {sorted(PATTERNS)}")
    match = PATTERNS[fmt].match(header)
    if not match:
        return None
    groups = match.groupdict()
    taxid = groups.get("taxid")
    return OrthologHeader(
        raw=header,
        protein_id=groups["protein"],
        taxid=int(taxid) if taxid else None,
        organism=groups.get("organism"),
        mnemonic=groups.get("mnemonic"),
    )


def parse_headers(
    headers: list[str], *, fmt: str = "eggnog", strict: bool = True
) -> tuple[dict[str, OrthologHeader], list[str]]:
    """Parse many headers.

    Returns ``(parsed, unparsed)`` keyed by the raw header. With ``strict``, a
    single unreadable header raises, naming the format tried and the first few
    offenders -- because the alternative, quietly analysing whichever
    orthologues happened to match, produces a result with no recoverable
    denominator.
    """
    parsed: dict[str, OrthologHeader] = {}
    unparsed: list[str] = []
    for header in headers:
        result = parse_header(header, fmt=fmt)
        if result is None:
            unparsed.append(header.lstrip(">").strip())
        else:
            parsed[result.raw] = result

    if unparsed and strict:
        shown = ", ".join(repr(h) for h in unparsed[:3])
        more = f" (and {len(unparsed) - 3} more)" if len(unparsed) > 3 else ""
        raise HeaderParseError(
            f"{len(unparsed)}/{len(headers)} headers did not match format "
            f"{fmt!r}: {shown}{more}. Known formats: {sorted(PATTERNS)}. An "
            f"orthologue with no taxon cannot be placed on the species tree, so "
            f"these would drop out of the phylogenetic test without appearing "
            f"in its denominator."
        )
    return parsed, unparsed


def detect_format(headers: list[str]) -> str | None:
    """The format matching the most headers, or None if none matches any."""
    best, best_count = None, 0
    for name in PATTERNS:
        count = sum(1 for h in headers if parse_header(h, fmt=name) is not None)
        if count > best_count:
            best, best_count = name, count
    return best
