"""Assign every sequence to a paralogue group, before conservation is measured.

Thin: read the alignment, call `amyloevo.group.assign`, write the table and a
report. The method and every judgement in it live in the package.

With `group.references` unset this writes an empty table with the right columns
and says so. That is the correct behaviour for a set that really is one
orthologue group -- inventing a split there would be worse than having none --
and the empty table keeps the DAG and every downstream join intact.
"""

from __future__ import annotations

import sys
from pathlib import Path

snakemake = globals()["snakemake"]
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd  # noqa: E402

from amyloevo.group import assign, assignment_table, copy_number, summarise  # noqa: E402

COLUMNS = ["id", "group", "nearest", "best_identity", "margin",
           "core_coverage", "confident"]


def read_fasta(path: str) -> dict[str, str]:
    seqs: dict[str, str] = {}
    name, buf = None, []
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            if name is not None:
                seqs[name] = "".join(buf)
            name, buf = line[1:].split()[0], []
        else:
            buf.append(line.strip())
    if name is not None:
        seqs[name] = "".join(buf)
    return seqs


def main() -> None:
    params = snakemake.params
    references = dict(params.references or {})

    if len(references) < 2:
        note = (
            "group.references names fewer than two references, so no paralogue "
            "assignment was made. This is correct for a set that is one "
            "orthologue group. If the set is a paralogue family, name one "
            "well-annotated species' paralogues here -- the groups cannot be "
            "recovered from the set's NAME, and conservation measured over a "
            "mixture is a statement about the mixture.\n"
        )
        pd.DataFrame(columns=COLUMNS).to_csv(
            snakemake.output.table, sep="\t", index=False
        )
        Path(snakemake.output.report).write_text(note)
        print(note, file=sys.stderr)
        return

    alignment = read_fasta(snakemake.input.alignment)
    meta = pd.read_csv(snakemake.input.meta, sep="\t", dtype={"taxid": "string"})

    outgroup = params.outgroup
    if isinstance(outgroup, str):
        # A metadata column holding a truthy flag, e.g. a column of
        # "is_invertebrate". A column is the honest place for this: which
        # sequences predate the duplication is a statement about the tree, and
        # it belongs in the data rather than in a threshold.
        if outgroup not in meta.columns:
            raise KeyError(
                f"group.outgroup names the metadata column {outgroup!r}, which "
                f"is not in {snakemake.input.meta}. Columns: "
                f"{sorted(meta.columns)}"
            )
        flagged = meta.loc[meta[outgroup].astype(str).str.lower().isin(
            ("1", "true", "yes", "t", "y")
        ), "id"]
        outgroup = set(flagged)
    elif outgroup:
        outgroup = set(outgroup)
    else:
        outgroup = None

    assignments = assign(
        alignment,
        references,
        min_margin=params.min_margin,
        outgroup=outgroup,
        outgroup_label=params.outgroup_label,
    )
    table = assignment_table(assignments, min_margin=params.min_margin)
    table.to_csv(snakemake.output.table, sep="\t", index=False)

    report = [summarise(assignments, min_margin=params.min_margin)]
    taxid_of = dict(zip(meta["id"], meta["taxid"]))
    if any(taxid_of.values()):
        counts = copy_number(assignments, taxid_of)
        report += [
            "",
            f"{len(counts['per_taxon'])} taxon/taxa; "
            f"{len(counts['complete'])} carry all {counts['n_groups']} group(s); "
            f"{len(counts['deficient'])} carry fewer.",
            "",
            "A taxon carrying fewer sequences than there are groups may predate "
            "the duplication, may have lost a copy, or may simply be incomplete "
            "in this set. Those are different facts and only a tree separates "
            "them, so none of them is asserted here.",
        ]
    Path(snakemake.output.report).write_text("\n".join(report) + "\n")
    print("\n".join(report))


main()
