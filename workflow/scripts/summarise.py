"""Fit the residual trend once, across every protein, and report the fit with it.

Fitting per protein would give a slope with no stability; fitting globally gives
one trend against which each column's surprise is measured. Either way the SLOPE
belongs in the output: where sequence conservation carries no information about the
APR call it is ~0, the residual collapses to a centred APR fraction, and a reader
who cannot see that will read it as conservation-corrected when it is not.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from amyloevo.conserve.scores import ColumnConservation, apr_residual

snakemake = globals()["snakemake"]

frames = [pd.read_csv(path, sep="\t") for path in snakemake.input]
table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

defined = table[table["defined"]] if not table.empty else table

# apr_residual() keys on the column index, which repeats across proteins, so the
# fit is run over a re-indexed view and mapped back by row position.
columns = [
    ColumnConservation(
        column=position,
        occupancy=int(row.occupancy),
        n_sequences=int(row.n_sequences),
        shannon=float(row.shannon),
        jensen_shannon=float(row.jensen_shannon),
        apr_fraction=float(row.apr_fraction),
        defined=True,
    )
    for position, row in enumerate(defined.itertuples(index=False))
]
residuals = apr_residual(columns)

table["apr_residual"] = pd.NA
if residuals:
    table.loc[defined.index, "apr_residual"] = [
        residuals.get(position) for position in range(len(defined))
    ]

fit: dict[str, object] = {
    "n_columns_total": int(len(table)),
    "n_columns_defined": int(len(defined)),
    "residual_defined": bool(residuals),
}
if residuals and len(defined) >= 3:
    xs = defined["jensen_shannon"].astype(float)
    ys = defined["apr_fraction"].astype(float)
    variance = float(((xs - xs.mean()) ** 2).sum())
    slope = float(((xs - xs.mean()) * (ys - ys.mean())).sum() / variance)
    fit.update(
        {
            "slope": round(slope, 6),
            "intercept": round(float(ys.mean() - slope * xs.mean()), 6),
            "pearson_r": round(float(xs.corr(ys)), 6),
            "interpretation": (
                "sequence conservation predicts the APR call; the residual is "
                "conservation-corrected"
                if abs(slope) > 0.05
                else "NO trend: the slope is near zero, so apr_residual is only a "
                "centred apr_fraction and must NOT be read as "
                "conservation-corrected"
            ),
        }
    )
else:
    fit["interpretation"] = (
        "residual undefined -- fewer than three measurable columns, or no variation "
        "in sequence conservation. Undefined is not zero: zero would read as "
        "'fully explained by sequence'."
    )

table.to_csv(snakemake.output.table, sep="\t", index=False)
Path(snakemake.output.fit).write_text(yaml.safe_dump(fit, sort_keys=True), encoding="utf-8")
