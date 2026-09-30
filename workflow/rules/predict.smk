"""Run the predictor panel over every orthologue, including the reference.

The reference is scored here as one member of its own alignment, by the same tools
as everything else. There is no separate reference-panel stage, and a twelve-tool
call from another run must never be substituted for one of these rows: an APR
called by twelve tools set against orthologues called by six is a
panel-composition artefact in which every orthologue appears to have lost APRs
that were never tested for.
"""


rule predict_panel:
    """One aggressor-wrappers batch per protein's orthologue set.

    Batched per protein rather than per sequence because the runners amortise
    startup over a multifasta, and per protein rather than globally because the
    output has to stay attributable to a group.
    """
    input:
        fasta=sequences_for_alignment,
    output:
        merged=directory("results/predict/{protein}/merged"),
    params:
        panel=",".join(config["predict"]["panel"]),
        cfg=config["predict"]["aggressor_config"],
        aggressor=config["tools"]["aggressor"],
    threads: config["resources"]["predict_threads"]
    log:
        "logs/predict/{protein}.log",
    conda:
        "../envs/predict.yaml"
    shell:
        r"""
        AGGRESSOR_WRAPPERS_CONFIG={params.cfg} {params.aggressor} {input.fasta} \
            -o results/predict/{wildcards.protein} \
            --predictors {params.panel} > {log} 2>&1
        """


rule apr_calls:
    """Collapse each orthologue's merged table into one APR call per residue.

    The consensus rule is fractional: it counts the per-tool binary calls, each of
    which its own parser produced with that tool's threshold AND direction. So the
    evidence entering the count is scale-free and polarity-correct before it is
    counted -- which is why no per-tool threshold appears in this workflow's config.
    """
    input:
        merged="results/predict/{protein}/merged",
    output:
        "results/predict/{protein}/apr_calls.tsv",
    params:
        threshold=config["predict"]["apr_threshold"],
        consensus=config["predict"]["consensus"],
        panel=config["predict"]["panel"],
    log:
        "logs/predict/{protein}.calls.log",
    conda:
        "../envs/core.yaml"
    script:
        "../scripts/apr_calls.py"
