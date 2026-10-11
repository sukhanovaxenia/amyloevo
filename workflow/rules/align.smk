"""Align each orthologue group, recording the aligner and its parameters.

Alignment choice moves column assignments, and the residual of DESIGN.md section 3
is a difference of two quantities that both depend on those columns. So the aligner
is not an implementation detail that can be swapped between runs: it is part of
the result, and `align_metadata` writes it beside the alignment.
"""


rule check_reference_present:
    """Fail loudly when a group lacks its reference sequence.

    APR coordinates are reported against the reference, so a group without one has
    no anchor. Measuring conservation anyway would produce a table whose positions
    mean nothing -- a failure worth having before the aligner runs, not after.
    """
    input:
        sequences_for_alignment,
    output:
        "results/align/{protein}.reference.txt",
    params:
        match=config["reference"]["match"],
        value=config["reference"]["value"],
        fmt=config["header_format"],
        strict=config["header_strict"],
        required=config["reference"]["required"],
    log:
        "logs/align/{protein}.reference.log",
    conda:
        "../envs/core.yaml"
    script:
        "../scripts/find_reference.py"


rule align_mafft:
    input:
        fasta=sequences_for_alignment,
        reference="results/align/{protein}.reference.txt",
    output:
        "results/align/{protein}.aln",
    params:
        args=config["align"]["mafft_args"],
        mafft=config["tools"]["mafft"],
    threads: config["resources"]["align_threads"]
    log:
        "logs/align/{protein}.log",
    conda:
        "../envs/align.yaml"
    shell:
        "{params.mafft} --thread {threads} {params.args} {input.fasta} "
        "> {output} 2> {log}"


rule align_muscle:
    input:
        fasta=sequences_for_alignment,
        reference="results/align/{protein}.reference.txt",
    output:
        "results/align/{protein}.aln",
    params:
        args=config["align"]["muscle_args"],
        muscle=config["tools"]["muscle"],
    threads: config["resources"]["align_threads"]
    log:
        "logs/align/{protein}.log",
    conda:
        "../envs/align.yaml"
    shell:
        "{params.muscle} -align {input.fasta} -output {output} "
        "-threads {threads} {params.args} > {log} 2>&1"


rule align_famsa:
    input:
        fasta=sequences_for_alignment,
        reference="results/align/{protein}.reference.txt",
    output:
        "results/align/{protein}.aln",
    params:
        args=config["align"]["famsa_args"],
        muscle=config["tools"]["famsa"],
    threads: config["resources"]["align_threads"]
    log:
        "logs/align/{protein}.log",
    conda:
        "../envs/align.yaml"
    shell:
        "{params.muscle} -t {threads} -v {params.args} "
        "{input.fasta} {output} > {log} 2>&1"


if config["align"]["tool"] == "mafft":
    ruleorder: align_mafft > align_muscle > align_famsa
elif config["align"]["tool"] == "muscle":
    ruleorder: align_muscle > align_mafft > align_famsa
else:
    ruleorder: align_famsa > align_muscle > align_mafft


rule align_metadata:
    """The aligner, its version and its arguments, beside the alignment itself."""
    input:
        "results/align/{protein}.aln",
    output:
        "results/align/{protein}.aln.yaml",
    params:
        tool=config["align"]["tool"],
        binary=lambda w: config["tools"][config["align"]["tool"]],
        args=lambda w: config["align"][f"{config['align']['tool']}_args"],
    conda:
        "../envs/align.yaml"
    script:
        "../scripts/align_metadata.py"
