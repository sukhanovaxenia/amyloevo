"""Per-column conservation for one protein: the rule's whole body is this call.

All the measurement logic lives in `amyloevo.project` and `amyloevo.conserve`,
where it is unit-tested. The residual is deliberately absent: it is defined
against a trend fitted across columns, and fitting per protein on a few hundred
columns gives a slope with no stability. `summarise.py` fits once, at run level.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from amyloevo.conserve.scores import BACKGROUND_FREQUENCIES, aa_type, measure_column
from amyloevo.project.columns import index_alignment

snakemake = globals()["snakemake"]


def read_fasta(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    name = None
    chunks: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            if name is not None:
                records[name] = "".join(chunks)
            name, chunks = line[1:].strip(), []
        elif name is not None:
            chunks.append(line.strip())
    if name is not None:
        records[name] = "".join(chunks)
    return records


alignment = read_fasta(Path(snakemake.input.alignment))
index = index_alignment(alignment)
n_sequences = len(alignment)

reference_line = Path(snakemake.input.reference).read_text(encoding="utf-8").strip()
reference_id = reference_line or next(iter(alignment))

# APR calls: one row per (sequence_id, residue_number) with a boolean. Absent
# sequences contribute no call rather than a False one -- "not scored" and "scored
# as negative" are different, and conflating them deflates APR conservation.
calls_frame = pd.read_csv(snakemake.input.calls, sep="\t")
calls: dict[tuple[str, int], bool] = {}
if not calls_frame.empty:
    for row in calls_frame.itertuples(index=False):
        calls[(str(row.sequence_id), int(row.residue_number))] = bool(row.apr)

background = (
    BACKGROUND_FREQUENCIES
    if snakemake.params.background == "swissprot"
    else {residue: 1 / 20 for residue in BACKGROUND_FREQUENCIES}
)

reference_map = index[reference_id]
rows = []
for column in range(1, index.alignment_length + 1):
    residues = index.column_residues(column)
    column_calls = []
    for sequence_id, mapping in index.mappings.items():
        residue_number = mapping.residue_at(column)
        if residue_number is None:
            continue                      # gapped: contributes no call
        call = calls.get((sequence_id, residue_number))
        if call is not None:
            column_calls.append(call)

    measured = measure_column(
        column,
        residues,
        column_calls,
        n_sequences=n_sequences,
        min_occupancy=float(snakemake.params.min_occupancy),
        background=background,
    )
    reference_residue_number = reference_map.residue_at(column)
    rows.append(
        {
            "protein": Path(snakemake.input.alignment).stem,
            "alignment_column": column,
            "reference_residue_number": reference_residue_number,
            "reference_residue": (
                reference_map.residue_character(column)
                if reference_residue_number is not None
                else "-"
            ),
            "reference_aa_type": (
                aa_type(reference_map.residue_character(column))
                if reference_residue_number is not None
                else "gap"
            ),
            "occupancy": measured.occupancy,
            "n_sequences": n_sequences,
            "occupancy_fraction": round(measured.occupancy_fraction, 4),
            "n_calls": len(column_calls),
            "shannon": measured.shannon,
            "jensen_shannon": measured.jensen_shannon,
            "apr_fraction": measured.apr_fraction,
            "defined": measured.defined,
            "undefined_reason": measured.reason,
        }
    )

pd.DataFrame(rows).to_csv(snakemake.output[0], sep="\t", index=False)
