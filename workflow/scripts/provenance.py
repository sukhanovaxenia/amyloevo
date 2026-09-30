"""Write everything a second run would need to reproduce this one.

Includes the two decisions that are easiest to lose and most consequential: which
predictors were in the panel, and how redundancy was corrected -- clustering and
tree weighting address the same bias, so a result that does not say which was used
has a denominator nobody can reconstruct.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import yaml

snakemake = globals()["snakemake"]
config = dict(snakemake.params.config)

try:
    import amyloevo

    package_version = amyloevo.__version__
except Exception:  # pragma: no cover - recorded, never fatal
    package_version = "unknown"

redundancy = config["redundancy"]
correction = redundancy["method"]
Path(snakemake.output[0]).write_text(
    yaml.safe_dump(
        {
            "amyloevo_version": package_version,
            "python": sys.version.split()[0],
            "platform": f"{platform.system()} {platform.machine()}",
            "panel": config["predict"]["panel"],
            "consensus": config["predict"]["consensus"],
            "predicted": not config["predict"].get("from_intervals"),
            "aligner": config["align"]["tool"],
            "aligner_arguments": config["align"][
                f"{config['align']['tool']}_args"
            ],
            "redundancy_correction": correction,
            "clustering_purpose": (
                "sampling correction"
                if correction == "cluster"
                else "compute cap only"
                if redundancy["cluster"].get("cap_only")
                else "not applied"
            ),
            "min_occupancy": config["conserve"]["min_occupancy"],
            "background": config["conserve"]["background"],
            "tree": config["phylo"]["tree"],
            "phylogenetic_signal": config["phylo"]["signal"],
            "regression": config["phylo"]["regression"],
        },
        sort_keys=True,
    ),
    encoding="utf-8",
)
