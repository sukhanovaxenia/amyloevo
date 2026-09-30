"""Cluster orthologues -- when clustering is the chosen correction, or a cap."""


rule cluster_mmseqs:
    """Reduce a group to representatives at a given identity.

    Whether this is a SAMPLING correction or only a compute cap is decided in
    config and recorded in the provenance, because the two make different claims:
    the first says the conservation denominator is now unbiased, the second says
    only that the job fits.
    """
    input:
        lambda w: str(ORTHOLOG_DIR / f"{w.protein}{SUFFIX}"),
    output:
        fasta="results/cluster/{protein}.fasta",
        tsv="results/cluster/{protein}_clusters.tsv",
    params:
        identity=config["redundancy"]["cluster"]["identity"],
        coverage=config["redundancy"]["cluster"]["coverage"],
        mmseqs=config["tools"]["mmseqs"],
    threads: config["resources"]["align_threads"]
    log:
        "logs/cluster/{protein}.log",
    conda:
        "../envs/cluster.yaml"
    shell:
        r"""
        tmp=$(mktemp -d)
        {params.mmseqs} easy-cluster {input} $tmp/clu $tmp \
            --min-seq-id {params.identity} -c {params.coverage} \
            --threads {threads} > {log} 2>&1
        cp $tmp/clu_rep_seq.fasta {output.fasta}
        cp $tmp/clu_cluster.tsv {output.tsv}
        rm -rf $tmp
        """


rule cluster_cdhit:
    input:
        lambda w: str(ORTHOLOG_DIR / f"{w.protein}{SUFFIX}"),
    output:
        fasta="results/cluster/{protein}.fasta",
        tsv="results/cluster/{protein}_clusters.tsv",
    params:
        identity=config["redundancy"]["cluster"]["identity"],
        cdhit=config["tools"]["cdhit"],
    threads: config["resources"]["align_threads"]
    log:
        "logs/cluster/{protein}.log",
    conda:
        "../envs/cluster.yaml"
    shell:
        r"""
        {params.cdhit} -i {input} -o {output.fasta} -c {params.identity} \
            -T {threads} -d 0 > {log} 2>&1
        mv {output.fasta}.clstr {output.tsv}
        """


# One rule per tool, one tool chosen: ruleorder resolves the ambiguity by config
# rather than by whichever rule Snakemake sees first.
if config["redundancy"]["cluster"]["tool"] == "mmseqs2":
    ruleorder: cluster_mmseqs > cluster_cdhit
else:
    ruleorder: cluster_cdhit > cluster_mmseqs
