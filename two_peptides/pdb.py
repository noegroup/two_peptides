"""make pdb files"""

import os
import click
from bgmol.systems import TwoMiniPeptides
import mdtraj as md
from .status import submitted


def save_pdb(aminoacids1, aminoacids2, outdir):
    system = TwoMiniPeptides(aminoacids1, aminoacids2)
    traj = md.Trajectory(system.positions, topology=system.mdtraj_topology)
    traj.save_pdb(os.path.join(outdir, f"{aminoacids1}_{aminoacids2}.pdb")


@click.command()
@click.option("-o", "--outdir", type=click.Path(exists=True, writable=True, dir_okay=True), default="./data"))
def main(outdir):
    for peptide1, peptide2 in submitted():
        save_pdb(peptide1, peptide2, outdir)
        save_pdb(peptide2, peptide1, outdir)


if __name__ == "__main__":
    main()
