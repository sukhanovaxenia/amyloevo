"""Logos, MEME export and iTOL annotations.

Each is a thin call into the package's viz module. The iTOL files are annotations
for a tree this workflow already built, so they are generated rather than
hand-assembled -- 372 proteins is too many to configure by clicking.
"""


rule apr_logo:
    input:
        alignment="results/align/{protein}.aln",
        conserve="results/conserve/{protein}.tsv",
    output:
        "results/viz/logos/{protein}.png",
    log:
        "logs/viz/{protein}.logo.log",
    conda:
        "../envs/viz.yaml"
    script:
        "../scripts/apr_logo.py"


rule meme_export:
    """APR segments in MEME format, for motif discovery over the segments."""
    input:
        alignment="results/align/{protein}.aln",
        conserve="results/conserve/{protein}.tsv",
    output:
        "results/viz/meme/{protein}.meme",
    log:
        "logs/viz/{protein}.meme.log",
    conda:
        "../envs/core.yaml"
    script:
        "../scripts/meme_export.py"


rule itol_annotations:
    input:
        tree="results/phylo/{protein}.treefile",
        conserve="results/conserve/{protein}.tsv",
    output:
        apr="results/viz/itol/{protein}_apr.txt",
        clades="results/viz/itol/{protein}_clades.txt",
    params:
        header_format=config["header_format"],
    log:
        "logs/viz/{protein}.itol.log",
    conda:
        "../envs/core.yaml"
    script:
        "../scripts/itol_annotations.py"
