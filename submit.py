

import os
import time
from two_peptides.status import submitted, is_finished


TEST = False
DRYRUN = False

submit_stub = lambda a,b: (
    f"sbatch "
    f"-J sim_{a}_{b} -o log/sim_{a}_{b}.log "
    f"--time 24:00:00 -p gpu --gres gpu:1 --mem 8GB "
    f"--exclude gpu[100-130] "
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


for a, b in submitted():
    if not (is_finished(a, b) or is_finished(b, a)):
        print(a, b)
        submit(a, b)
        time.sleep(0.5)
