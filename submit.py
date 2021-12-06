

from two_peptides.meta import fast_folder_pairs
from bgmol.systems.minipeptides import AMINO_ACIDS
import os


TEST = False
DRYRUN = False

submit_stub = lambda a,b: (
    f"sbatch "
    f"-J sim_{a}_{b} -o log/sim_{a}_{b}.log "
    f"--time 24:00:00 -p gpu --gres gpu:1 --mem 6GB "
    f"--nodelist gpu[000-070] "
    f"two_peptides -a {a} -b {b} "
)

def submit(a,b):
    command = submit_stub(a, b)
    if TEST:
        command += "--test"
    if DRYRUN:
        print(command)
    else:
        os.system(command)


for a in AMINO_ACIDS:
    for b in AMINO_ACIDS:
        submit(a,b)

for a, b in fast_folder_pairs():
    submit(a,b)



