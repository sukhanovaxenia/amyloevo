"""Resolve a taxid to a scientific name and a lineage, offline.

``io/headers.py`` gets a taxid out of a header; this turns that number into
something a reader and a grouping can use. Both halves are needed before any
analysis can speak about species or clades, and a table keyed on
``9606.ENSP00000350170`` is not one anybody can read.

**Offline by construction.** NCBI's E-utilities would be the obvious route and
it is the wrong one here. An orthologue sweep is 10^4-10^5 sequences, E-utilities
is rate-limited to a few requests a second without a key, the answer for a given
taxid does not change between calls, and a pipeline that cannot run without a
network is a pipeline that cannot run on a cluster node. The authoritative
source is a file:

    curl -O https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz
    mkdir -p taxdump && tar xzf taxdump.tar.gz -C taxdump

``nodes.dmp`` gives parent and rank, ``names.dmp`` gives the scientific name,
``merged.dmp`` maps retired taxids onto their replacements. All three are read
here; ``merged.dmp`` matters more than it looks, because an orthologue set
assembled a few years ago will carry taxids NCBI has since merged, and without
it those lineages come back empty and their species silently leave the analysis.

**Two fallbacks, in order.** A user-supplied table (any TSV with a ``taxid``
column and whichever of ``tax_name``/``genus``/``family``/``order``/``class``/
``phylum`` it has) is accepted, because an existing annotation is often already
on disk and re-deriving it adds a dependency for nothing. Failing that,
resolution yields empty fields rather than raising -- but
:func:`TaxonomyDB.coverage` reports how many taxa were resolved, and the callers
in this package refuse to run a clade-level test on an unresolved set rather
than quietly testing whichever species happened to have names.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

#: Ranks carried on the metadata table, outermost last. Chosen because these are
#: the ranks a grouping is normally asked for; ``lineage_at`` reaches any rank
#: NCBI defines.
STANDARD_RANKS = ("genus", "family", "order", "class", "phylum", "superkingdom")

#: Scientific name class in names.dmp. The file also carries synonyms, common
#: names and misspellings under the same taxid, and taking the first row rather
#: than filtering on this gives a genus a fisherman's name for a species.
SCIENTIFIC_NAME = "scientific name"


@dataclass(frozen=True)
class Taxon:
    """One resolved taxon."""

    taxid: int
    name: str
    rank: str = ""
    #: rank -> name, for the ranks present in this taxon's lineage
    lineage: dict[str, str] = field(default_factory=dict)

    def at(self, rank: str) -> str:
        """The ancestor at ``rank``, or an empty string when absent.

        Not every lineage has every rank -- plenty of clades have no named
        subphylum, and some have no genus -- so a missing rank is normal and
        returns empty rather than raising.
        """
        return self.lineage.get(rank, "")


class TaxonomyDB:
    """Taxid -> name, rank and lineage.

    Build with :meth:`from_taxdump` or :meth:`from_table`, or
    :func:`load_taxonomy`, which tries both.
    """

    def __init__(
        self,
        names: dict[int, str] | None = None,
        nodes: dict[int, tuple[int, str]] | None = None,
        merged: dict[int, int] | None = None,
        rows: dict[int, dict[str, str]] | None = None,
    ) -> None:
        self._names = names or {}
        self._nodes = nodes or {}
        self._merged = merged or {}
        self._rows = rows or {}
        self._cache: dict[int, Taxon | None] = {}

    # -- construction ------------------------------------------------------ #

    @classmethod
    def from_taxdump(cls, directory: str | Path) -> "TaxonomyDB":
        """Read ``names.dmp``, ``nodes.dmp`` and (when present) ``merged.dmp``."""
        directory = Path(directory)
        names_path, nodes_path = directory / "names.dmp", directory / "nodes.dmp"
        missing = [p.name for p in (names_path, nodes_path) if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                f"{directory}: taxdump is missing {', '.join(missing)}. Get it with\n"
                f"  curl -O https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz\n"
                f"  mkdir -p {directory} && tar xzf taxdump.tar.gz -C {directory}"
            )

        names: dict[int, str] = {}
        for fields in _read_dmp(names_path):
            if len(fields) >= 4 and fields[3] == SCIENTIFIC_NAME:
                names[int(fields[0])] = fields[1]

        nodes: dict[int, tuple[int, str]] = {}
        for fields in _read_dmp(nodes_path):
            if len(fields) >= 3:
                nodes[int(fields[0])] = (int(fields[1]), fields[2])

        merged: dict[int, int] = {}
        merged_path = directory / "merged.dmp"
        if merged_path.is_file():
            for fields in _read_dmp(merged_path):
                if len(fields) >= 2:
                    merged[int(fields[0])] = int(fields[1])

        return cls(names=names, nodes=nodes, merged=merged)

    @classmethod
    def from_table(cls, path: str | Path) -> "TaxonomyDB":
        """Read a prepared table: a ``taxid`` column plus whatever ranks it has.

        Column names are matched case-insensitively, and both ``tax_name`` and
        ``sci``/``scientific_name``/``organism`` are accepted for the name,
        since every tool that writes one of these spells it differently.
        """
        path = Path(path)
        rows: dict[int, dict[str, str]] = {}
        with open(path, newline="", encoding="utf-8") as handle:
            sample = handle.read(4096)
            handle.seek(0)
            delimiter = "\t" if "\t" in sample.splitlines()[0] else ","
            reader = csv.DictReader(handle, delimiter=delimiter)
            lowered = {c.strip().lower(): c for c in (reader.fieldnames or [])}
            if "taxid" not in lowered:
                raise ValueError(
                    f"{path}: no 'taxid' column (have {list(reader.fieldnames or [])})"
                )
            name_col = next(
                (lowered[c] for c in
                 ("tax_name", "sci", "scientific_name", "organism", "species")
                 if c in lowered),
                None,
            )
            for row in reader:
                raw = (row.get(lowered["taxid"]) or "").strip()
                if not raw.isdigit():
                    continue
                entry = {"name": (row.get(name_col) or "").strip() if name_col else ""}
                for rank in STANDARD_RANKS:
                    if rank in lowered:
                        entry[rank] = (row.get(lowered[rank]) or "").strip()
                rows[int(raw)] = entry
        return cls(rows=rows)

    # -- lookup ------------------------------------------------------------ #

    def resolve(self, taxid: int | str | None) -> Taxon | None:
        """One taxon, or None when it cannot be resolved."""
        if taxid in (None, ""):
            return None
        try:
            key = int(taxid)
        except (TypeError, ValueError):
            return None
        if key in self._cache:
            return self._cache[key]
        taxon = self._resolve_uncached(key)
        self._cache[key] = taxon
        return taxon

    def _resolve_uncached(self, taxid: int) -> Taxon | None:
        if taxid in self._rows:
            row = self._rows[taxid]
            lineage = {r: row[r] for r in STANDARD_RANKS if row.get(r)}
            return Taxon(taxid=taxid, name=row.get("name", ""), lineage=lineage)

        # A retired taxid resolves through merged.dmp rather than to nothing.
        resolved = self._merged.get(taxid, taxid)
        if resolved not in self._nodes:
            return None

        lineage: dict[str, str] = {}
        current = resolved
        seen: set[int] = set()
        while current in self._nodes and current not in seen:
            seen.add(current)
            parent, rank = self._nodes[current]
            if rank and rank != "no rank":
                lineage.setdefault(rank, self._names.get(current, ""))
            if parent == current:  # root is its own parent
                break
            current = parent
        _, own_rank = self._nodes[resolved]
        return Taxon(
            taxid=taxid,
            name=self._names.get(resolved, ""),
            rank=own_rank,
            lineage={k: v for k, v in lineage.items() if v},
        )

    def coverage(self, taxids) -> dict[str, int]:
        """How many of ``taxids`` resolve. Report this before any clade test."""
        unique = {int(t) for t in taxids if str(t).isdigit()}
        resolved = sum(1 for t in unique if self.resolve(t) is not None)
        return {
            "taxa": len(unique),
            "resolved": resolved,
            "unresolved": len(unique) - resolved,
        }

    @property
    def is_empty(self) -> bool:
        return not (self._names or self._rows)


def _read_dmp(path: Path):
    """Yield the fields of an NCBI ``.dmp`` row.

    The format is ``field\\t|\\tfield\\t|\\t...\\t|`` -- tab-pipe-tab separated
    with a trailing delimiter -- so a plain split on tab leaves pipes stranded
    in every other field.
    """
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n").rstrip("\t|")
            if not line:
                continue
            yield [f.strip() for f in line.split("\t|\t")]


def load_taxonomy(
    taxdump: str | Path | None = None,
    table: str | Path | None = None,
) -> TaxonomyDB:
    """Load from a taxdump directory, else a table, else empty.

    An empty database is not an error: it lets the rest of the pipeline run and
    produce id-keyed output. It IS a reason to refuse a clade-level test, which
    is the caller's job -- see ``phylo.signal``.
    """
    if taxdump is not None and Path(taxdump).is_dir():
        return TaxonomyDB.from_taxdump(taxdump)
    if table is not None and Path(table).is_file():
        return TaxonomyDB.from_table(table)
    return TaxonomyDB()
