import sys
sys.path.insert(0, "../../src")
from amyloevo.phylo.tree import low_occupancy, tip_occupancy, identical_groups

def read_fasta(path):
    seqs, name, buf = {}, None, []
    for line in open(path):
        if line.startswith(">"):
            if name: seqs[name] = "".join(buf)
            name, buf = line[1:].split()[0], []
        else:
            buf.append(line.strip())
    if name: seqs[name] = "".join(buf)
    return seqs

aln = read_fasta("/Users/kseniasuhanova/iCloud/Documents/Documents_icloud/Lab_amyloids/Projects/FXR1/alignment/FXR1_opistho.mafft.trim.fasta")
drop = low_occupancy(aln, 0.5)
print(f"{len(drop)} of {len(aln)} tips below 50% occupancy")
with open("/Users/kseniasuhanova/iCloud/Documents/Documents_icloud/Lab_amyloids/Projects/FXR1/alignment/tree/fxr_family.treeinput.fasta", "w") as out, \
     open("/Users/kseniasuhanova/iCloud/Documents/Documents_icloud/Lab_amyloids/Projects/FXR1/alignment/tree/fxr_family.dropped.tsv", "w") as log:
    log.write("id\toccupancy\n")
    for pid, row in aln.items():
        if pid in drop:
            log.write(f"{pid}\t{drop[pid]:.4f}\n")
            continue
        out.write(f">{pid}\n")
        for i in range(0, len(row), 60):
            out.write(row[i:i+60] + "\n")
groups = identical_groups({k: v for k, v in aln.items() if k not in drop})
print(f"{len(groups)} identical-row group(s) KEPT, "
      f"{sum(len(g)-1 for g in groups)} redundant records")
