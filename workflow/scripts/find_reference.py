"""Locate the reference sequence in an orthologue group, or fail saying so.

Thin by design: header parsing lives in `amyloevo.io.headers`, where it is
unit-tested. This script resolves paths and writes one line.
"""

from __future__ import annotations

import sys
from pathlib import Path

from amyloevo.io.headers import parse_headers

snakemake = globals()["snakemake"]  # injected by Snakemake

fasta = Path(snakemake.input[0])
headers = [
    line[1:].strip()
    for line in fasta.read_text(encoding="utf-8").splitlines()
    if line.startswith(">")
]

parsed, unparsed = parse_headers(
    headers,
    fmt=snakemake.params.fmt,
    strict=bool(snakemake.params.strict),
)

match_field = snakemake.params.match
wanted = str(snakemake.params.value)

hits = []
for raw, record in parsed.items():
    value = {
        "taxid": str(record.taxid) if record.taxid is not None else None,
        "organism": record.organism,
        "mnemonic": record.mnemonic,
        "protein": record.protein_id,
    }.get(match_field)
    if value is not None and wanted.lower() in value.lower():
        hits.append(raw)

if not hits:
    message = (
        f"{fasta.name}: no reference sequence with {match_field}={wanted!r}. "
        f"APR coordinates are reported against the reference, so a group without "
        f"one has no anchor and its conservation table would carry positions that "
        f"mean nothing. Add the reference sequence to the group, or set "
        f"reference.required: false to skip this protein deliberately."
    )
    if snakemake.params.required:
        sys.exit(message)
    Path(snakemake.output[0]).write_text("", encoding="utf-8")
    sys.stderr.write(message + "\n")
else:
    if len(hits) > 1:
        sys.stderr.write(
            f"{fasta.name}: {len(hits)} sequences match the reference criterion "
            f"({', '.join(hits[:3])}...). Using the first; narrow "
            f"reference.value if that is not the intended one.\n"
        )
    Path(snakemake.output[0]).write_text(hits[0] + "\n", encoding="utf-8")
