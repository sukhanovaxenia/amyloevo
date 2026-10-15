"""Running the R estimators, and adjusting what comes back.

The estimators themselves are in ``workflow/scripts/phylo_signal.R``: Fritz &
Purvis *D* from ``caper``, Pagel's lambda from ``phytools``, phylogenetic
logistic regression from ``phylolm``. This module is the part worth unit
testing -- assembling the call, and deciding what the returned p-values mean
once there is one per block.

**One test per block is a family of tests.** Fifteen blocks on the FXR1 set,
and four p-value columns, so four families of fifteen. Each column is adjusted
separately across blocks, because they answer different questions: *D* against
a random null and *D* against a Brownian null can both be significant on the
same block, and pooling them into one adjustment would treat two readings of
one statistic as two findings.

**A permutation p has a floor, and the floor can make the adjustment unable to
reject anything.** ``phylo.d`` with *k* permutations cannot return a p below
roughly ``1/(k+1)``. Benjamini-Hochberg at level alpha over *m* blocks needs at
least one p at or below ``alpha/m`` before it can reject, so *k* must satisfy
``1/(k+1) <= alpha/m``. With 15 blocks at alpha = 0.05 that is 299
permutations; the default of 1,000 clears it. This project has already shipped
a table of all-null results produced by exactly this arithmetic going unchecked,
which is why :func:`floor_warning` exists rather than a comment.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

from ..stats.multiple import benjamini_hochberg

#: p-value columns the R script returns, each its own family across blocks,
#: mapped to the name its adjusted counterpart gets. Spelled out rather than
#: derived by string surgery: a q column silently named after the wrong p is
#: the sort of mistake that survives review.
Q_FOR_P = {
    "d_p_random": "d_q_random",
    "d_p_brownian": "d_q_brownian",
    "lambda_p": "lambda_q",
    "glm_p": "glm_q",
}
P_COLUMNS = tuple(Q_FOR_P)

DEFAULT_RSCRIPT = "Rscript"


def floor_warning(permutations: int, n_tests: int, alpha: float = 0.05) -> str:
    """Empty when the permutation count can support the adjustment, else why not."""
    if n_tests <= 0 or permutations <= 0:
        return ""
    floor = 1.0 / (permutations + 1)
    needed = alpha / n_tests
    if floor <= needed:
        return ""
    return (
        f"{permutations:,} permutations put the smallest attainable p at "
        f"{floor:.4g}, above the alpha/m = {needed:.4g} that Benjamini-Hochberg "
        f"at alpha = {alpha:g} over {n_tests} block(s) needs before it can "
        f"reject anything. Every q would come back non-significant for "
        f"arithmetic reasons, whatever the data say. Raise permutations to at "
        f"least {int(np.ceil(1 / needed)) - 1:,}."
    )


def run_estimators(
    tree_path,
    traits,
    covariates,
    outgroup,
    *,
    r_script,
    workdir,
    signal: str = "both",
    regression: str = "phyloglm",
    permutations: int = 1000,
    seed: int = 0,
    rscript: str = DEFAULT_RSCRIPT,
):
    """Write the R script's inputs, run it, and return its table.

    ``traits`` is the screened 0/1 DataFrame indexed by id, ``covariates`` the
    predictor table (and the continuous occupancies, as ``occupancy_<block>``
    columns), ``outgroup`` the tip labels to root on -- empty to leave the tree
    as it is, which the R side then refuses if it is unrooted.
    """
    import pandas as pd

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    if shutil.which(rscript) is None:
        raise FileNotFoundError(
            f"{rscript!r} is not on PATH. The phylogenetic estimators run in R "
            f"(caper, phytools, phylolm); there is no Python fallback here on "
            f"purpose -- see amyloevo.phylo.estimate."
        )

    traits_path = workdir / "traits.tsv"
    covar_path = workdir / "covariates.tsv"
    outgroup_path = workdir / "outgroup.txt"
    out_path = workdir / "phylo_signal.r.tsv"

    traits.reset_index().rename(columns={traits.index.name or "index": "id"}).to_csv(
        traits_path, sep="\t", index=False
    )
    covariates.reset_index().rename(
        columns={covariates.index.name or "index": "id"}
    ).to_csv(covar_path, sep="\t", index=False)
    outgroup_path.write_text("\n".join(str(x) for x in outgroup) + "\n")

    command = [
        rscript, str(r_script), str(tree_path), str(traits_path), str(covar_path),
        str(outgroup_path), str(out_path), signal, regression,
        str(permutations), str(seed),
    ]
    proc = subprocess.run(command, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "the R estimators failed.\n"
            f"command: {' '.join(command)}\n"
            f"stderr:\n{proc.stderr.strip()}"
        )
    if proc.stderr.strip():
        print(proc.stderr.strip())
    return pd.read_csv(out_path, sep="\t")


def adjust(table, *, alpha: float = 0.05, p_columns=P_COLUMNS):
    """Add a BH-adjusted ``q_`` column beside each p-value column.

    Each p-value column is one family across blocks. Rows whose p is missing --
    an estimator that declined or failed -- are excluded from the adjustment
    rather than filled, because a missing p is not a p of 1 and counting it as
    one inflates *m* and weakens every other block's q.
    """
    table = table.copy()
    for column in p_columns:
        if column not in table.columns:
            continue
        values = table[column].to_numpy(dtype=float)
        finite = np.flatnonzero(np.isfinite(values))
        q = np.full(values.shape, np.nan)
        if finite.size:
            q[finite] = benjamini_hochberg(values[finite])
        table[Q_FOR_P[column]] = q
    return table


def format_phylo_report(
    table,
    *,
    alpha: float = 0.05,
    permutations: int = 1000,
    support=None,
    rooting=None,
    conflicts=None,
    clades=None,
) -> str:
    """The report that travels with the table, including what not to read off it."""
    lines = [
        "=" * 72,
        "PHYLOGENETIC SIGNAL IN APR BLOCK PRESENCE",
        "=" * 72,
    ]
    if rooting is not None:
        lines.append(f"rooting:   {rooting}")
    if support is not None:
        lines.append(f"support:   {support}")
    if clades:
        lines.append("clades:")
        lines += [f"    {c}" for c in clades]
    if conflicts is not None:
        lines.append(f"identical tips: {conflicts}")
    lines.append("")

    warning = floor_warning(permutations, len(table), alpha)
    if warning:
        lines += ["  !! " + warning, ""]

    if len(table) == 0:
        lines += ["No block passed the trait screen, so nothing was tested.",
                  "=" * 72]
        return "\n".join(lines)

    lines.append(
        f"  {'block':16s} {'n':>5s} {'pres':>5s} {'D':>7s} {'p(rnd)':>8s} "
        f"{'p(BM)':>8s} {'lambda':>7s} {'p':>8s}"
    )
    for row in table.itertuples(index=False):
        def num(attr, fmt, _row=row):
            value = getattr(_row, attr, float("nan"))
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = float("nan")
            width = fmt.split(".")[0] or "7"
            return format(value, fmt) if np.isfinite(value) else f"{'--':>{width}}"
        lines.append(
            f"  {str(row.block):16s} {int(row.n):5d} {int(row.n_present):5d} "
            f"{num('d_estimate', '7.3f')} {num('d_p_random', '8.3g')} "
            f"{num('d_p_brownian', '8.3g')} {num('lambda', '7.3f')} "
            f"{num('lambda_p', '8.3g')}"
        )
    notes = []
    for row in table.itertuples(index=False):
        note = getattr(row, "note", "")
        if isinstance(note, str) and note.strip():
            notes.append((str(row.block), note.strip()))
    if notes:
        lines += ["", "estimator notes:"]
        lines += [f"    {b}: {n}" for b, n in notes]

    lines += [
        "",
        "Reading this table.",
        "  D = 0 is the Brownian expectation -- the trait follows the tree; D = 1",
        "  is phylogenetic randomness. Both p-values are reported because a D in",
        "  the middle can differ significantly from BOTH nulls, and that is the",
        "  informative case a single p hides. D above 1 means overdispersed:",
        "  relatives share the trait LESS than chance, which on a gene tree of",
        "  paralogues is worth looking at rather than rounding to 1.",
        "",
        "  Branch lengths here are substitutions per site, not time. Lambda is",
        "  calibrated against Brownian motion along those lengths, so it mixes",
        "  rate with history and is not a clock statement.",
        "",
        "  The clade term is deliberately absent from the regression. Each",
        "  paralogue group is one monophyletic clade from one duplication, so a",
        "  group contrast rests on two independent events no matter how many tips",
        "  it spans; a phyloglm coefficient on it would be computed as if every",
        "  tip were a datum. The paired within-taxon contrast over taxa carrying",
        "  all three copies is the controlled instrument for that question.",
        "=" * 72,
    ]
    return "\n".join(lines)


__all__ = [
    "DEFAULT_RSCRIPT",
    "P_COLUMNS",
    "adjust",
    "floor_warning",
    "format_phylo_report",
    "run_estimators",
]
