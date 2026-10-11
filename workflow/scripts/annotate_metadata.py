"""Build the metadata table: taxid, tax_name, protein, lineage, and the label.

Thin by design -- read the headers, resolve the taxonomy, write the table. Every
decision that could be wrong lives in `amyloevo.io`, where it is unit-tested.

The `unresolved` output is not a log line. A taxon that fails to resolve leaves
the analysis silently: it is dropped from every rank-wise denominator without
ever being counted as missing, so a sweep can lose twenty species and still
report clean numbers. Writing the ids to a file the DAG knows about makes that
loss something you have to look at.
"""

from __future__ import annotations

import sys
from pathlib import Path

snakemake = globals()["snakemake"]
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from amyloevo.io import annotate, load_taxonomy  # noqa: E402


def read_headers(path: str) -> list[str]:
    return [
        line[1:].split()[0]
        for line in Path(path).read_text().splitlines()
        if line.startswith(">")
    ]


def main() -> None:
    params = snakemake.params
    ids = read_headers(snakemake.input.fasta)

    taxonomy = load_taxonomy(
        taxdump=params.taxdump or None,
        table=params.table_fallback or None,
    )
    if taxonomy.is_empty:
        print(
            "WARNING: no taxonomy source configured (taxonomy.taxdump and "
            "taxonomy.table are both null). Rank columns will be empty and "
            "every rank-wise test will be skipped for want of levels. Point "
            "taxonomy.taxdump at an unpacked NCBI taxdump to fix this.",
            file=sys.stderr,
        )

    result = annotate(
        ids,
        taxonomy=taxonomy,
        header_format=params.header_format,
        label_template=params.label_template,
        ranks=tuple(params.ranks),
    )
    result.table.to_csv(snakemake.output.table, sep="\t", index=False)
    Path(snakemake.output.unresolved).write_text(
        "\n".join(result.unresolved_taxa) + ("\n" if result.unresolved_taxa else "")
    )

    print(
        f"{len(result.table)} sequence(s); "
        f"{len(result.unresolved_taxa)} unresolved taxon id(s); "
        f"{result.collisions_disambiguated} label(s) disambiguated"
    )


main()
