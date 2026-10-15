"""Trees, phylogenetic signal, and the dependency test.

"Does APR presence depend on divergence or clade" cannot be answered by
correlation: species share ancestry, the observations are not independent, and
any clade-level pattern will appear significant. Signal in a binary trait needs
Fritz & Purvis's D or Pagel's lambda; the dependency itself needs phylogenetic
logistic regression. Both live behind one rule each so the choice is visible in
the DAG.

On a paralogue family `tree: infer` gives a GENE tree -- the duplication is
inside it. That is the stricter choice and now the default one; what it costs is
that "monophyletic" and "per-clade" become claims the tree has to be checked
for, which `rule phylogenetic_signal` does before it tests anything.
"""


rule tree_input:
    """The sequences the tree is inferred from, and why the rest were left out.

    Mostly-gap rows are dropped BEFORE inference: absent characters do not
    constrain the topology, so such a tip is placed by whatever columns remain
    and drags its neighbours' branch lengths with it. Pruning it from a finished
    tree does not undo that.

    Identical rows are KEPT. They produce zero-length branches, which look like
    something to clean up and are not -- IQ-TREE collapses them for the search
    and restores them in the output, whereas deduplicating here would pick one
    member as the representative and discard the others' traits.
    """
    input:
        "results/align/{protein}.aln",
    output:
        alignment="results/phylo/{protein}.treeinput.fasta",
        dropped="results/phylo/{protein}.dropped.tsv",
    params:
        min_occupancy=config["phylo"]["min_tip_occupancy"],
    log:
        "logs/phylo/{protein}.treeinput.log",
    script:
        "../scripts/tree_input.py"


rule infer_tree:
    input:
        "results/phylo/{protein}.treeinput.fasta",
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
    """D / lambda for APR presence, and the regression on divergence.

    Reads the gene tree, roots it on the declared outgroup -- refusing when that
    outgroup is not monophyletic, because the root is what D and lambda read
    ancestry off -- binarises block occupancy at one named threshold, screens
    each block for the prevalence the estimators need, and hands the survivors
    to R. The alignment is an input because the identical rows in it are where
    trait variation ends up with no branch length under it, and that has to be
    measured rather than assumed away.
    """
    input:
        tree="results/phylo/{protein}.treefile",
        profiles="results/blocks/{protein}.occupancy.npz",
        groups="results/group/{protein}.tsv",
        meta="results/meta/{protein}.tsv",
        alignment="results/align/{protein}.aln",
    output:
        table="results/summary/{protein}.phylo_signal.tsv",
        report="results/summary/{protein}.phylo_signal.txt",
    params:
        signal=config["phylo"]["signal"],
        regression=config["phylo"]["regression"],
        permutations=config["phylo"]["permutations"],
        seed=config["phylo"]["seed"],
        presence_threshold=config["phylo"]["presence_threshold"],
        min_minority=config["phylo"]["min_minority"],
        min_prevalence=config["phylo"]["min_prevalence"],
        alpha=config["phylo"]["alpha"],
        require_outgroup_monophyly=config["phylo"]["require_outgroup_monophyly"],
        outgroup_label=config["group"]["outgroup_label"],
    log:
        "logs/phylo/{protein}.signal.log",
    conda:
        "../envs/phylo.yaml"
    script:
        "../scripts/phylo_signal.py"
