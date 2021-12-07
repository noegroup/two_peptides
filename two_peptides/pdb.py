"""make pdb files"""

import os
import click
import mdtraj as md
from two_peptides.status import submitted
from two_peptides.simulation import TwoPeptideSimulation


def save_pdb(aminoacids1, aminoacids2, outdir):
    sim = TwoPeptideSimulation(aminoacids1, aminoacids2)
    model = sim.model
    traj = md.Trajectory(model.positions[sim.beads], topology=model.mdtraj_topology.subset(sim.beads))
    traj.save_pdb(os.path.join(outdir, f"{aminoacids1}_{aminoacids2}.pdb"))


@click.command()
@click.option("-o", "--outdir", type=click.Path(exists=True, writable=True, dir_okay=True), default="./data")
def main(outdir):
    for peptide1, peptide2 in submitted():
        print(peptide1, peptide2)
        save_pdb(peptide1, peptide2, outdir)
        save_pdb(peptide2, peptide1, outdir)


if __name__ == "__main__":
    main()
