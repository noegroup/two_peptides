

from two_peptides.meta import fast_folder_pairs
from bgmol.systems.minipeptides import AMINO_ACIDS
import os


TEST = True


submit_stub = lambda a,b: (
    f"sbatch "
    f"-J sim_{a}_{b} -o log/sim_{a}_{b}.log "
    f"--time 24:00:00 -p gpu --gres gpu:1 --mem 8GB "
    f"two_peptides -a {a} -b {b} "
)


for a in AMINO_ACIDS:
    for b in AMINO_ACIDS:
        command = submit_stub(a, b)
        if TEST:
            command += "--test"
        os.system(command)


for peptide1, peptide2 in fast_folder_pairs():
        command = submit_stub(peptide1, peptide2)
        if TEST:
            command += "--test"
        os.system(command)




