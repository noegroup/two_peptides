

__all__ = ["submitted", "status"]

import os
import click
from subprocess import Popen, PIPE

from bgmol.systems.minipeptides import AMINO_ACIDS
from two_peptides.meta import fast_folder_pairs


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


def parse_squeue():
    p = Popen(["squeue", "-u", "kraemea88", '-o', '"%.15j %.8T"'], stdout=PIPE, stderr=PIPE)
    lines = [line.decode('utf-8').strip() for line in p.stdout.readlines()]
    status_dict = {"RUNNING": [], "PENDING": []}
    for line in lines:
        try:
            _, jobname, status = line.strip().split()
        except ValueError:
            continue
        if not jobname.startswith("sim_"):
            continue
        _, *peptides = jobname.strip().split("_")
        status = status[:-1]
        if status in status_dict:
            status_dict[status].append(tuple(peptides))
    return status_dict


def _determine_status(peptide1, peptide2, outdir, squeue_dict):
    if is_finished(peptide1, peptide2, outdir):
        return "C"
    elif (peptide1, peptide2) in squeue_dict["RUNNING"]:
        return "R"
    elif (peptide1, peptide2) in squeue_dict["PENDING"]:
        return "Q"
    else:
        return "E"


@click.command()
@click.option("-o", "--outdir", type=click.Path(exists=True, dir_okay=True), default="./data")
@click.option("--detailed/--no-detailed", default=False)
def main(outdir, detailed):
    n_jobs = {"C": 0, "R": 0, "E": 0, "Q": 0}
    squeue_dict = parse_squeue()
    for a, b in submitted():
        status = _determine_status(a, b, outdir, squeue_dict)
        if status == "E":
            status = _determine_status(b, a, outdir, squeue_dict)
        if detailed:
            print(status, a, b)
        n_jobs[status] += 1
    print(f"""----------------------------------------------------------------------------
{n_jobs['C']} finished, {n_jobs['E']} failed, {n_jobs['R']} running, {n_jobs['Q']} pending 
""")


if __name__ == "__main__":
    main()
