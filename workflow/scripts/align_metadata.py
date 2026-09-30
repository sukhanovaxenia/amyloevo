"""Record the aligner, its version and its arguments beside the alignment.

Alignment choice moves column assignments, and the residual is a difference of two
column-dependent quantities, so this is part of the result rather than a log line.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import yaml

snakemake = globals()["snakemake"]

alignment = Path(snakemake.input[0])


def _version(binary: str) -> str:
    for flag in ("--version", "-version", "-v"):
        try:
            proc = subprocess.run(
                [binary, flag], capture_output=True, text=True, timeout=30, check=False
            )
        except (OSError, subprocess.SubprocessError):
            continue
        text = (proc.stdout or proc.stderr or "").strip()
        if text:
            return text.splitlines()[0]
    return "unknown"


text = alignment.read_text(encoding="utf-8")
records = [line for line in text.splitlines() if line.startswith(">")]
lengths = {
    len(seq)
    for seq in "".join(
        line if not line.startswith(">") else "\n" for line in text.splitlines()
    ).split("\n")
    if seq
}

Path(snakemake.output[0]).write_text(
    yaml.safe_dump(
        {
            "aligner": snakemake.params.tool,
            "binary": snakemake.params.binary,
            "version": _version(snakemake.params.binary),
            "arguments": snakemake.params.args,
            "n_sequences": len(records),
            "alignment_length": sorted(lengths)[-1] if lengths else 0,
            # A checksum makes "same alignment?" answerable without diffing.
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        },
        sort_keys=True,
    ),
    encoding="utf-8",
)
