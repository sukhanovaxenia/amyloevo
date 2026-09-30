"""Trees, phylogenetic signal, and the dependency test.

"Does APR presence depend on divergence or clade" cannot be answered by
correlation: species share ancestry, the observations are not independent, and any
clade-level pattern will appear significant. Signal in a binary trait needs Fritz &
Purvis's D or Pagel's lambda; the dependency itself needs phylogenetic logistic
regression. Both live behind one rule each so the choice is visible in the DAG.
"""


rule infer_tree:
    input:
        "results/align/{protein}.aln",
    output:
        "results/phylo/{protein}.treefile",
    params:
        args=config["phylo"]["iqtree_args"],
        iqtree=config["tools"]["iqtree"],
        prefix=lambda w: f"results/phylo/{w.protein}",
    threads: config["resources"]["tree_threads"]
    log:
        "logs/phylo/{protein}.log",
    conda:
        "../envs/phylo.yaml"
    shell:
        "{params.iqtree} -s {input} --prefix {params.prefix} -T {threads} "
        "{params.args} > {log} 2>&1"


rule phylogenetic_signal:
    """D / lambda for APR presence, and the regression on divergence and clade."""
    input:
        trees=expand("results/phylo/{protein}.treefile", protein=PROTEINS),
        conserve=expand("results/conserve/{protein}.tsv", protein=PROTEINS),
    output:
        "results/summary/phylo_signal.tsv",
    params:
        signal=config["phylo"]["signal"],
        regression=config["phylo"]["regression"],
    log:
        "logs/phylo/signal.log",
    conda:
        "../envs/phylo.yaml"
    script:
        "../scripts/phylo_signal.py"
