"""The three measures, per alignment column, and the run-level fit.

`conserve_protein` computes per-column sequence conservation, APR conservation and
occupancy. The residual is NOT computed here: it is defined against a trend fitted
across columns, and fitting per protein on a few hundred columns gives a slope with
no stability. The fit happens once, at run level, in `conservation_summary`.
"""


rule conserve_protein:
    input:
        alignment="results/align/{protein}.aln",
        metadata="results/align/{protein}.aln.yaml",
        reference="results/align/{protein}.reference.txt",
        calls=apr_calls,
    output:
        "results/conserve/{protein}.tsv",
    params:
        min_occupancy=config["conserve"]["min_occupancy"],
        background=config["conserve"]["background"],
        header_format=config["header_format"],
        weight_by_tree=DO_WEIGHT,
        tree=lambda w: (
            f"results/phylo/{w.protein}.treefile" if DO_WEIGHT else None
        ),
    log:
        "logs/conserve/{protein}.log",
    conda:
        "../envs/core.yaml"
    script:
        "../scripts/conserve.py"


rule conservation_summary:
    """Fit the trend once, across all proteins, and report the residual with it.

    The slope is part of the output, not a hidden intermediate. Where sequence
    conservation carries no information about the APR call the slope is ~0 and the
    residual collapses to a centred APR fraction -- which is not a
    conservation-corrected quantity and must not be read as one. A reader can only
    know that if the slope is on the page.
    """
    input:
        expand("results/conserve/{protein}.tsv", protein=PROTEINS),
    output:
        table="results/summary/conservation_summary.tsv",
        fit="results/summary/residual_fit.yaml",
    log:
        "logs/conserve/summary.log",
    conda:
        "../envs/core.yaml"
    script:
        "../scripts/summarise.py"
