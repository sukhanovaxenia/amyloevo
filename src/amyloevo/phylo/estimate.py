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


#: Below this a p-value has left the range a double can represent usefully.
#: R prints 0.0 or a denormal like 7.16e-320; neither is a measurement, and
#: "p = 0" is not a sentence to put in a manuscript.
P_UNDERFLOW = 1e-300

#: At or above this, Pagel's lambda has reached the top of the scale and the
#: point estimate is not distinguishable from 1 (Brownian on the given tree).
LAMBDA_CEILING = 0.999

#: Two fitted values agreeing to this many significant digits are one value
#: found twice, not two measurements -- the optimiser stopped at the same bound.
TIE_DIGITS = 9


def boundary_warnings(table) -> list[str]:
    """Flag fitted values that are bounds of a search rather than estimates.

    An optimiser that stops at the edge of its interval still returns a number,
    and that number is then printed, adjusted and read like any other. Three
    ways it showed up on the first FXR1 run:

    * five blocks sharing a lambda of 0.999926686256414 -- twelve identical
      digits, which is one boundary found five times;
    * all fifteen sharing a phyloglm alpha of 1.51447 to six figures;
    * likelihood-ratio p-values at 0.0 and 7.16e-320, below what a double
      carries.

    None of these is an error, and none of them errored. They need saying.
    """
    import pandas as pd

    out: list[str] = []
    if len(table) == 0:
        return out

    def numeric(column):
        if column not in table.columns:
            return None
        values = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)
        return values[np.isfinite(values)]

    lam = numeric("pagel_lambda")
    if lam is None:
        lam = numeric("lambda")
    if lam is not None and lam.size:
        at_ceiling = int((lam >= LAMBDA_CEILING).sum())
        if at_ceiling:
            out.append(
                f"lambda is at the top of its scale on {at_ceiling} of "
                f"{lam.size} fitted row(s) (>= {LAMBDA_CEILING}). There the "
                f"estimate is 'at least as tree-following as Brownian motion' "
                f"and no more: the likelihood is flat against the bound, so the "
                f"usual standard error and the usual chi-square p do not apply "
                f"to it."
            )
        rounded = np.round(lam, TIE_DIGITS)
        values, counts = np.unique(rounded, return_counts=True)
        tied = [(v, c) for v, c in zip(values, counts) if c > 1]
        if tied:
            worst = max(tied, key=lambda vc: vc[1])
            out.append(
                f"{worst[1]} row(s) share a lambda of {worst[0]:.12g} to "
                f"{TIE_DIGITS} significant digits. Independent fits to "
                f"different traits do not agree to that precision; that is the "
                f"optimiser stopping in the same place, so treat those rows as "
                f"'lambda hit the bound', not as a repeated measurement."
            )

    bound = numeric("glm_alpha_bound")
    if bound is not None and bound.size and (bound != 0).any():
        out.append(
            f"phyloglm reported alpha on a search bound for "
            f"{int((bound != 0).sum())} of {bound.size} fit(s) (alphaWarn "
            f"!= 0). The coefficient and its p-value are conditioned on an "
            f"alpha that was not estimated."
        )
    else:
        alpha_values = numeric("glm_alpha")
        if alpha_values is not None and alpha_values.size > 1:
            rounded = np.round(alpha_values, 6)
            values, counts = np.unique(rounded, return_counts=True)
            tied = [(v, c) for v, c in zip(values, counts) if c > 1]
            if tied:
                worst = max(tied, key=lambda vc: vc[1])
                out.append(
                    f"{worst[1]} phyloglm fit(s) share alpha = {worst[0]:.6g} to "
                    f"six figures. alpha is the phylogenetic-correlation "
                    f"parameter and it is searched over a bounded interval, so "
                    f"one value arrived at this often is a bound rather than a "
                    f"finding. Check `fit$alphaWarn` before quoting any of these "
                    f"coefficients."
                )

    for column in ("lambda_p", "glm_p", "d_p_random", "d_p_brownian"):
        values = numeric(column)
        if values is not None and values.size:
            under = int((values < P_UNDERFLOW).sum())
            if under:
                out.append(
                    f"{under} value(s) in {column} are below {P_UNDERFLOW:g}, "
                    f"which is where double precision stops being meaningful. "
                    f"They are reported as <1e-300; do not quote them as "
                    f"numbers."
                )
    return out


def run_estimators(
    tree_path,
    traits,
    covariates,
    outgroup,
    *,
    tasks,
    r_script,
    workdir,
    signal: str = "both",
    regression: str = "phyloglm",
    permutations: int = 1000,
    seed: int = 0,
    rscript: str = DEFAULT_RSCRIPT,
):
    """Write the R script's inputs, run it, and return its table.

    ``traits`` is the 0/1 DataFrame indexed by id, ``covariates`` the predictor
    table (and the continuous occupancies, as ``occupancy_<block>`` columns),
    ``tasks`` the ``(scope, block)`` work list from
    ``traits.tasks`` -- the screening decisions, made once in Python, so that R
    never re-decides what is testable -- and ``outgroup`` the tip labels to root
    on, empty to leave the tree as it is, which the R side then refuses if it is
    unrooted.
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
    if len(tasks) == 0:
        raise ValueError("the work list is empty: no block passed the screen")

    traits_path = workdir / "traits.tsv"
    covar_path = workdir / "covariates.tsv"
    tasks_path = workdir / "tasks.tsv"
    outgroup_path = workdir / "outgroup.txt"
    out_path = workdir / "phylo_signal.r.tsv"

    traits.reset_index().rename(columns={traits.index.name or "index": "id"}).to_csv(
        traits_path, sep="\t", index=False
    )
    covariates.reset_index().rename(
        columns={covariates.index.name or "index": "id"}
    ).to_csv(covar_path, sep="\t", index=False)
    tasks.to_csv(tasks_path, sep="\t", index=False)
    outgroup_path.write_text("\n".join(str(x) for x in outgroup) + "\n")

    command = [
        rscript, str(r_script), str(tree_path), str(traits_path), str(covar_path),
        str(tasks_path), str(outgroup_path), str(out_path), signal, regression,
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


def adjust(table, *, alpha: float = 0.05, p_columns=P_COLUMNS, by="scope"):
    """Add a BH-adjusted ``q_`` column beside each p-value column.

    Each p-value column is one family across blocks, **within one scope**.
    Whole-tree and per-clade rows are separate families: they are different
    questions asked of different tips, and pooling them would inflate *m* by a
    factor of four here and weaken every row for no gain in error control. Pass
    ``by=None`` to adjust across the whole table instead.

    Rows whose p is missing -- an estimator that declined or failed -- are
    excluded from the adjustment rather than filled, because a missing p is not
    a p of 1 and counting it as one inflates *m* and weakens every other row.
    """
    table = table.copy()
    if by and by in table.columns:
        blocks = [np.flatnonzero((table[by] == s).to_numpy())
                  for s in dict.fromkeys(table[by])]
    else:
        blocks = [np.arange(len(table))]
    for column in p_columns:
        if column not in table.columns:
            continue
        values = table[column].to_numpy(dtype=float)
        q = np.full(values.shape, np.nan)
        for index in blocks:
            finite = index[np.isfinite(values[index])]
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

    # Rows as dicts, not itertuples. `itertuples` renames any column that is not
    # a valid Python identifier -- `lambda` is a keyword, so `row.lambda` cannot
    # exist and a getattr fallback silently printed a dash in every row of the
    # lambda column. A report that blanks a whole statistic without erroring is
    # the worst available failure, so the lookup no longer goes through
    # attributes at all.
    records = table.to_dict("records")

    def num(record, key, fmt, *, alt=None):
        value = record.get(key, record.get(alt) if alt else None)
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = float("nan")
        width = fmt.split(".")[0] or "7"
        if not np.isfinite(value):
            return f"{'--':>{width}}"
        # A p-value printed as 0, or as 7.16e-320 in the denormal range, is not
        # a measurement: the likelihood ratio overflowed what a double can
        # carry. Reporting it as a number invites "p = 0" in a manuscript.
        if key.endswith("_p") and 0.0 <= value < P_UNDERFLOW:
            return f"{'<1e-300':>{width}}"
        return format(value, fmt)

    scopes = list(dict.fromkeys(str(r.get("scope", "all")) for r in records))
    for scope in scopes:
        rows = [r for r in records if str(r.get("scope", "all")) == scope]
        if len(scopes) > 1:
            heading = "whole tree" if scope == "all" else f"within {scope}"
            lines += [f"-- {heading} " + "-" * 40]
        lines.append(
            f"  {'block':16s} {'n':>5s} {'pres':>5s} {'D':>7s} {'p(rnd)':>8s} "
            f"{'p(BM)':>8s} {'lambda':>7s} {'p':>8s}"
        )
        for record in rows:
            lines.append(
                f"  {str(record.get('block', '?')):16s} "
                f"{int(record.get('n') or 0):5d} "
                f"{int(record.get('n_present') or 0):5d} "
                f"{num(record, 'd_estimate', '7.3f')} "
                f"{num(record, 'd_p_random', '8.3g')} "
                f"{num(record, 'd_p_brownian', '8.3g')} "
                f"{num(record, 'pagel_lambda', '7.3f', alt='lambda')} "
                f"{num(record, 'lambda_p', '8.3g')}"
            )
        lines.append("")

    for warning in boundary_warnings(table):
        lines += ["  !! " + warning, ""]

    notes = []
    for record in records:
        note = record.get("note", "")
        if isinstance(note, str) and note.strip():
            label = str(record.get("block", "?"))
            scope = str(record.get("scope", "all"))
            notes.append((f"{label} [{scope}]" if len(scopes) > 1 else label,
                          note.strip()))
    if notes:
        lines += ["estimator notes:"]
        lines += [f"    {b}: {n}" for b, n in notes]

    lines += [
        "",
        "Reading this table.",
        "  D (Fritz & Purvis 2010) rescales ONE quantity: the sum, over the",
        "  nodes of the tree, of how much sister clades differ in the trait.",
        "  A trait concentrated in a few clades gives few differences, a trait",
        "  scattered across the tree gives many. That sum is then placed between",
        "  two simulated nulls -- the mean under random reshuffling of the tip",
        "  values, and the mean under a Brownian character evolved on this tree",
        "  and cut once at the observed prevalence. D = 1 sits at the random",
        "  null, D = 0 at the Brownian one. So D is a POSITION between two",
        "  models, not a correlation and not a proportion of variance.",
        "",
        "  Both p-values are reported because they test different nulls:",
        "  p(rnd) asks whether the clumping could be chance, p(BM) whether it",
        "  could be Brownian. A D in the middle can reject both, which is the",
        "  informative case a single p hides. D above 1 means overdispersed --",
        "  relatives share the trait LESS than chance -- which on a gene tree of",
        "  paralogues is worth looking at rather than rounding down.",
        "",
        "  Lambda (Pagel 1999) is a multiplier on the off-diagonal entries of",
        "  the phylogenetic covariance matrix: 1 leaves the tree as it is, so",
        "  the trait is Brownian on it, and 0 flattens it to a star, so the tips",
        "  are independent. It is fitted on the CONTINUOUS occupancy, not on the",
        "  binary column, and tested by a likelihood ratio against lambda = 0.",
        "",
        "  That test is not the interesting part on a whole gene tree. The",
        "  deepest split here IS the duplication and the paralogue groups carry",
        "  different APR profiles, so 'is this trait non-independent on this",
        "  tree' has to come back yes, and it comes back yes at p values below",
        "  what a double can hold. The per-clade rows are where the answer is",
        "  not foreordained: inside one paralogue group the duplication is gone,",
        "  and lambda there is a statement about that group's own evolution.",
        "",
        "  Branch lengths are substitutions per site, not time. Lambda is",
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
    "LAMBDA_CEILING",
    "P_COLUMNS",
    "P_UNDERFLOW",
    "Q_FOR_P",
    "TIE_DIGITS",
    "adjust",
    "boundary_warnings",
    "floor_warning",
    "format_phylo_report",
    "run_estimators",
]
