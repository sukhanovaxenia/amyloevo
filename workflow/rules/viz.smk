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
    """Annotation files for the gene tree, one upload per iTOL track.

    Reads the block profiles rather than the conservation table: the thing to
    put beside a tree of this family is which APR blocks each sequence carries,
    and that is the block matrix. A tip with no profile gets -1 (unknown), not
    0 (absent) -- 75 of the 357 FXR1 tips failed the completeness filter and
    drawing them as lacking every block would be an invented result.

    iTOL drops a row whose label is not a leaf without reporting it, so the
    script compares every dataset's ids against the tree's tips and logs the
    counts. An empty track otherwise looks like a finding.
    """
    input:
        tree="results/phylo/{protein}.treefile",
        profiles="results/blocks/{protein}.occupancy.npz",
        groups="results/group/{protein}.tsv",
        meta="results/meta/{protein}.tsv",
    output:
        apr="results/viz/itol/{protein}_apr.txt",
        clades="results/viz/itol/{protein}_clades.txt",
        labels="results/viz/itol/{protein}_labels.txt",
        load="results/viz/itol/{protein}_load.txt",
        load_bars="results/viz/itol/{protein}_load_bars.txt",
        taxa_symbols="results/viz/itol/{protein}_taxa_symbols.txt",
        taxa_strip="results/viz/itol/{protein}_taxa_strip.txt",
    params:
        presence_threshold=config["phylo"]["presence_threshold"],
    log:
        "logs/viz/{protein}.itol.log",
    conda:
        "../envs/core.yaml"
    script:
        "../scripts/itol_annotations.py"
