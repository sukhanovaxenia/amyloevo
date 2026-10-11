"""Taxonomy, labels, and paralogue groups -- before anything is conserved.

These rules run early on purpose. An "orthologue set" from EggNOG is routinely a
paralogue family, and the paralogue differences are usually larger than the
orthologue ones, so a conservation measure computed before the groups are known
is a measure over a mixture. See section 4 of docs/DESIGN.md.

The same rules produce the metadata table every later rule joins against and
every figure labels from: `taxid`, `tax_name`, `protein`, the rank columns, and
the display `label`.
"""


rule taxonomy_table:
    """taxid -> name and lineage for every orthologue in a group.

    Resolves through merged.dmp, so a set assembled years ago whose taxids NCBI
    has since merged does not silently lose those species: an unresolved taxon
    leaves the analysis without appearing in any denominator, which is the
    failure this rule's `unresolved` output exists to make visible.
    """
    input:
        fasta=lambda w: ortholog_fasta(w.protein),
    output:
        table="results/meta/{protein}.tsv",
        unresolved="results/meta/{protein}.unresolved.txt",
    params:
        taxdump=config["taxonomy"]["taxdump"],
        table_fallback=config["taxonomy"]["table"],
        header_format=config["header_format"],
        label_template=config["label"]["template"],
        ranks=config["taxonomy"]["ranks"],
    log:
        "logs/meta/{protein}.log",
    script:
        "../scripts/annotate_metadata.py"


rule assign_groups:
    """Paralogue assignment by identity to the configured references.

    Skipped (an empty table with the right columns) when `group.references` is
    null, which is the correct configuration for a set that really is a single
    orthologue group. Assigning one anyway would invent a split.
    """
    input:
        alignment="results/align/{protein}.aln",
        meta="results/meta/{protein}.tsv",
    output:
        table="results/group/{protein}.tsv",
        report="results/group/{protein}.report.txt",
    params:
        references=config["group"]["references"],
        min_margin=config["group"]["min_margin"],
        outgroup=config["group"]["outgroup"],
        outgroup_label=config["group"]["outgroup_label"],
    log:
        "logs/group/{protein}.log",
    script:
        "../scripts/assign_groups.py"


rule block_profiles:
    """APR blocks, and each sequence's occupancy over them.

    The feature-construction step, and it matters more than any test applied
    afterwards: a per-column profile over a long alignment is sparse, and
    testing it directly measures mostly shared absence.

    Blocks are defined per group and then merged, because a block carried by a
    minority of one group is exactly the discriminative feature a group contrast
    needs and the pooled definition would miss it.
    """
    input:
        alignment="results/align/{protein}.aln",
        calls=apr_calls,
        groups="results/group/{protein}.tsv",
        reference="results/align/{protein}.reference.txt",
    output:
        profiles="results/blocks/{protein}.occupancy.npz",
        blocks="results/blocks/{protein}.tsv",
    params:
        min_occupancy=config["blocks"]["min_occupancy"],
        min_coverage=config["blocks"]["min_coverage"],
        min_width=config["blocks"]["min_width"],
        max_gap=config["blocks"]["max_gap"],
        min_group=config["blocks"]["min_group"],
    log:
        "logs/blocks/{protein}.log",
    script:
        "../scripts/block_profiles.py"


rule taxonomic_signal:
    """The distribution-free screen: do APR profiles cluster by clade?

    Run WITHIN each paralogue group, because every member of a three-paralogue
    family contributes one of each and so clade and paralogue explain the same
    variance in a pooled test.

    This is not the phylogenetic test of `rule phylogenetic_signal`, and does not
    discharge it. It says whether building the tree is worth it.
    """
    input:
        profiles="results/blocks/{protein}.occupancy.npz",
        meta="results/meta/{protein}.tsv",
        groups="results/group/{protein}.tsv",
    output:
        table="results/summary/{protein}.taxonomic_signal.tsv",
        report="results/summary/{protein}.taxonomic_signal.txt",
    params:
        ranks=config["taxonomy"]["ranks"],
        draws=config["phylo"]["permutations"],
        seed=config["phylo"]["seed"],
        min_per_level=config["phylo"]["min_per_level"],
        min_levels=config["phylo"]["min_levels"],
        min_members=config["phylo"]["min_members"],
    log:
        "logs/phylo/{protein}.taxonomic_signal.log",
    script:
        "../scripts/taxonomic_signal.py"


rule group_figures:
    """The three group-wise figures: occupancy, block heatmap, paired load.

    Figure size never scales with the number of sequences. A per-sequence figure
    over an orthologue sweep is how a plotting routine produces an image no
    viewer opens; these summarise by group, and the per-sequence view is the
    paginated table `results/blocks/{protein}.tsv` beside them.

    Axes label from the metadata `label` column, so a row reads
    `Homo sapiens FXR1` rather than `9606.ENSP00000350170`.
    """
    input:
        profiles="results/blocks/{protein}.occupancy.npz",
        blocks="results/blocks/{protein}.tsv",
        meta="results/meta/{protein}.tsv",
        groups="results/group/{protein}.tsv",
    output:
        occupancy="results/viz/groups/{protein}_occupancy.png",
        heatmap="results/viz/groups/{protein}_blocks.png",
        load="results/viz/groups/{protein}_load.png",
    params:
        reference_label=config["label"]["reference"],
    log:
        "logs/viz/{protein}.groups.log",
    script:
        "../scripts/group_figures.py"
