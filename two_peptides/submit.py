

from bgmol.systems.minipeptides import AMINO_ACIDS
from bgmol.systems.fastfolders import FAST_FOLDER_NAMES, FastFolder


for a in AMINO_ACIDS:
    for b in AMINO_ACIDS:
        submit_run(a, b)



