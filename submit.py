

import os
import click
import time
from two_peptides.status import submitted, is_finished


TEST = False
DRYRUN = False

submit_stub = lambda a,b: (
    f"sbatch "
    f"-J sim_{a}_{b} -o log/sim_{a}_{b}.log "
    f"--time 24:00:00 -p gpu --gres gpu:1 --mem 8GB "
    f"--exclude gpu[100-130],gpu064,gpu066,gpu077 "
    f"two_peptides run -a {a} -b {b} "
)


def submit(a,b):
    command = submit_stub(a, b)
    if TEST:
        command += "--test"
    if not (is_finished(a, b) or is_finished(b, a)):
        if DRYRUN:
            print(command)
        else:
            print(a, b)
            os.system(command)
            time.sleep(0.5)


@click.command()
@click.option("--many/--no-many", default=False)
@click.option("-a", "--aminoacids1", default=None)
@click.option("-b", "--aminoacids2", default=None)
def main(many, aminoacids1, aminoacids2):
    if many:
        to_submit = submitted()
    else:
        assert aminoacids1 is not None and aminoacids2 is not None
        to_submit = ((aminoacids1, aminoacids2), )
    for a, b in to_submit:
        submit(a, b)


if __name__ == "__main__":
    main()
