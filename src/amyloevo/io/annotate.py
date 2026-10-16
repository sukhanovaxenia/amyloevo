"""The metadata table every downstream table and figure is keyed on.

One row per sequence, carrying what the id alone cannot say: the taxon, its
lineage, which paralogue group the sequence belongs to, and a label a reader can
use. Built once and joined onto everything else, so a figure and a table never
disagree about what a sequence is called.

On labelling, which is a real choice and not a formatting detail
--------------------------------------------------------------

``9606.ENSP00000350170`` is unique, stable and unreadable. ``Homo sapiens FXR1``
is readable and not guaranteed unique -- a species with two isoforms assigned to
the same group produces two identical labels, and a reader cannot tell which
row is which. Neither is right on its own, so the policy here is:

1. **Label by meaning, not by accession.** The default template is
   ``{tax_name} {group}``, because that is what a reader of a comparative figure
   is actually comparing. An axis of accessions makes the reader do a lookup for
   every row.
2. **Fall back down a chain**, so a partially resolved set still labels: group,
   then species, then the bare id. A row with no taxonomy gets its id rather
   than a blank or a misleading guess.
3. **Disambiguate collisions with the accession, never silently.** Where a label
   would repeat, the id is appended to every member of the colliding set --
   all of them, not just the second onwards, so the suffix marks a group rather
   than looking like an afterthought on one row.
4. **Keep the id on the table.** ``label`` is for display; joins, provenance and
   any question of "which sequence exactly" go through ``id``. A pipeline that
   loses the accession at the first figure cannot answer the second question.

So: yes, plot by ``tax_name`` plus the group, and carry the id. The template is
configurable because a figure of one species' paralogues wants ``{group}`` alone
while a figure of one group's orthologues wants ``{tax_name}`` alone, and
hardcoding either produces an axis that repeats the same word on every row.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from .headers import parse_header
from .taxonomy import STANDARD_RANKS, TaxonomyDB

#: Default display template. Fields: id, protein, taxid, tax_name, group,
#: protein_name, plus every rank in STANDARD_RANKS.
DEFAULT_LABEL_TEMPLATE = "{tax_name} {group}"

#: Tried in order when the template renders empty for a row.
LABEL_FALLBACKS = ("{tax_name} {group}", "{tax_name}", "{group}", "{id}")

_FIELD = re.compile(r"\{(\w+)\}")


@dataclass
class AnnotationResult:
    """The metadata table plus what could not be resolved."""

    table: "object"  # pandas.DataFrame; typed loosely to keep the import local
    unresolved_taxa: list[str]
    collisions_disambiguated: int

    def __len__(self) -> int:  # pragma: no cover - convenience
        return len(self.table)


def annotate(
    sequence_ids,
    *,
    taxonomy: TaxonomyDB | None = None,
    header_format: str = "eggnog",
    groups: dict[str, str] | None = None,
    protein_names: dict[str, str] | None = None,
    label_template: str = DEFAULT_LABEL_TEMPLATE,
    ranks=STANDARD_RANKS,
):
    """Build the metadata table for ``sequence_ids``.

    ``groups`` maps id -> paralogue/group name (see :mod:`amyloevo.group`).
    ``protein_names`` maps id -> a gene or protein name, for sets that carry one;
    where it is absent the group name stands in, since for a paralogue family
    they are the same thing and inventing a second column of the same values
    helps nobody.

    Returns an :class:`AnnotationResult` whose ``table`` has one row per id with
    columns ``id, protein, taxid, tax_name, <ranks...>, group, protein_name,
    label``.
    """
    import pandas as pd

    taxonomy = taxonomy or TaxonomyDB()
    groups = groups or {}
    protein_names = protein_names or {}

    rows = []
    unresolved: list[str] = []
    for sid in sequence_ids:
        parsed = parse_header(sid, fmt=header_format)
        taxid = parsed.taxid if parsed else None
        accession = parsed.protein_id if parsed else sid
        taxon = taxonomy.resolve(taxid)
        if taxid is not None and taxon is None:
            unresolved.append(sid)

        group = groups.get(sid, "")
        row = {
            "id": sid,
            "protein": accession,
            "taxid": taxid if taxid is not None else "",
            "tax_name": taxon.name if taxon else (parsed.organism if parsed else "") or "",
            "group": group,
            "protein_name": protein_names.get(sid, "") or group,
        }
        for rank in ranks:
            row[rank] = taxon.at(rank) if taxon else ""
        rows.append(row)

    table = pd.DataFrame(rows)
    ordered = ["id", "protein", "taxid", "tax_name", *ranks, "group", "protein_name"]
    table = table[[c for c in ordered if c in table.columns]]

    labels, n_collisions = make_labels(table, template=label_template)
    table["label"] = labels
    return AnnotationResult(
        table=table,
        unresolved_taxa=unresolved,
        collisions_disambiguated=n_collisions,
    )


def render(template: str, row: dict) -> str:
    """Render one template against a row, dropping fields the row lacks.

    A missing field becomes empty rather than raising or printing ``{tax_name}``
    on an axis, and the result is whitespace-collapsed so a dropped field does
    not leave a double space behind.
    """
    def _sub(match: re.Match) -> str:
        value = row.get(match.group(1), "")
        return "" if value is None else str(value)

    return " ".join(_FIELD.sub(_sub, template).split())


def make_labels(
    table,
    *,
    template: str = DEFAULT_LABEL_TEMPLATE,
    fallbacks=LABEL_FALLBACKS,
    id_column: str = "id",
) -> tuple[list[str], int]:
    """Display labels for a metadata table, with collisions disambiguated.

    Returns ``(labels, n_disambiguated)``. Every member of a colliding set gets
    the accession appended -- not just the duplicates after the first -- so the
    suffix reads as a property of the set rather than as a correction applied to
    some rows and not others.
    """
    records = table.to_dict("records") if hasattr(table, "to_dict") else list(table)

    drafts = []
    for row in records:
        text = render(template, row)
        if not text:
            for alternative in fallbacks:
                text = render(alternative, row)
                if text:
                    break
        drafts.append(text or str(row.get(id_column, "")))

    counts = Counter(drafts)
    colliding = {text for text, n in counts.items() if n > 1}
    labels = [
        f"{text} [{row.get('protein') or row.get(id_column, '')}]"
        if text in colliding
        else text
        for text, row in zip(drafts, records)
    ]
    n_disambiguated = sum(1 for text in drafts if text in colliding)
    return labels, n_disambiguated


def label_map(table, **kwargs) -> dict[str, str]:
    """``{id: label}``, for joining onto a figure or another table."""
    labels, _ = make_labels(table, **kwargs)
    ids = table["id"].tolist() if hasattr(table, "__getitem__") else []
    return dict(zip(ids, labels))


#: A group smaller than this is not worth its own symbol or colour on a figure:
#: it is indistinguishable from the next one at a glance, and a legend of
#: twenty-five entries is not a legend. Groups below it fall back to a coarser
#: rank; see :func:`major_groups`.
MIN_GROUP_SIZE = 5


def major_groups(
    table,
    ranks=("class", "phylum"),
    *,
    min_size: int = MIN_GROUP_SIZE,
    other_label: str = "other",
    suffix: str = " (other)",
) -> dict[str, str]:
    """Collapse a taxonomy into a handful of groups a figure can show.

    NCBI's ranks are not uniform across a set that spans a kingdom: on the FXR1
    family ``class`` gives 25 values of which 11 have fewer than three members,
    while ``phylum`` gives 12 but puts every vertebrate in one bucket. Neither
    rank alone produces a legend.

    So the ranks are tried from finest to coarsest, and a group is kept the
    first time it reaches ``min_size``. **The remainder is then re-counted
    among itself**, which is the part that matters: falling back to ``phylum``
    for the three coelacanths must not produce "Chordata" as a group, because
    that name already covers the 217 mammals assigned at the previous rank.
    Counting only the sequences still unassigned turns it into "Chordata
    (other)" -- the chordates that were not large enough to name -- and the
    suffix says so. Whatever is still short at the last rank becomes
    ``other_label``.

    On the FXR1 set with the defaults this yields nine groups: Mammalia 217,
    Actinopteri 59, Insecta 33, other 17, Reptilia 14, Aves 12, Chordata
    (other) 8, Lepidosauria 7, Nematoda (other) 5.

    ``table`` is a DataFrame with an ``id`` column and one column per rank.
    """
    ranks = [r for r in ranks if r in getattr(table, "columns", [])]
    if not ranks:
        raise KeyError(
            f"none of the requested ranks are columns of this table; it has "
            f"{list(getattr(table, 'columns', []))}"
        )
    values = {r: dict(zip(table["id"], table[r])) for r in ranks}
    pending = [str(i) for i in table["id"]]
    assigned: dict[str, str] = {}

    for depth, rank in enumerate(ranks):
        counts = Counter(
            str(values[rank][i]) for i in pending
            if values[rank].get(i) and str(values[rank][i]).strip()
        )
        keep = {name for name, n in counts.items() if n >= min_size}
        remainder = []
        for i in pending:
            name = values[rank].get(i)
            name = str(name).strip() if name else ""
            if name in keep:
                assigned[i] = name if depth == 0 else name + suffix
            else:
                remainder.append(i)
        pending = remainder
        if not pending:
            break

    for i in pending:
        assigned[i] = other_label
    return assigned
