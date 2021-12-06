

__all__ = ["submitted", "status"]

import os
import click
from two_peptides.meta import fast_folder_pairs
from bgmol.systems.minipeptides import AMINO_ACIDS


def submitted():
    for a in AMINO_ACIDS:
        for b in AMINO_ACIDS:
            yield a, b

    for a, b in fast_folder_pairs():
        yield a, b


def is_finished(a, b, outdir="./data"):
    def filename(suffix, npz=False):
        return os.path.join(
            outdir,
            f"{a}_{b}_{suffix}."
            f"{'npz' if npz else 'npy'}"
        )
    if not os.path.exists(filename("coord")):
        return False
    if not os.path.exists(filename("force")):
        return False
    if not os.path.exists(filename("embed")):
        return False
    if not os.path.exists(filename("energies", npz=True)):
        return False
    return True


@click.command()
@click.option("-o", "--outdir", type=click.Path(exists=True, dir_okay=True), default="./data")
def main(outdir):
    for a, b in submitted():
        print("ok" if is_finished(a, b, outdir) else "--", a, b)


if __name__ == "__main__":
    main()
