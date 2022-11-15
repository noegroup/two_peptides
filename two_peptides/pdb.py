"""make pdb files"""

import os
import click
import mdtraj as md
from two_peptides.status import submitted
from two_peptides.simulation import TwoPeptideSimulation


def save_pdb(aminoacids1, aminoacids2, filename, selection="saved_atoms"):
    sim = TwoPeptideSimulation(aminoacids1, aminoacids2)
    model = sim.model
    selected = {"saved_atoms": sim.saved_atoms, "protein": model.select("protein")}[selection]
    traj = md.Trajectory(model.positions[selected], topology=model.mdtraj_topology.subset(selected))
    traj.save_pdb(filename)


@click.command()
@click.option("-o", "--outdir", type=click.Path(exists=True, writable=True, dir_okay=True), default="./data")
@click.option("-s", "--selection", type=click.Choice(["saved_atoms", "protein"]), default="protein")
def main(outdir, selection):
    for peptide1, peptide2 in submitted():
        print(peptide1, peptide2)
        save_pdb(peptide1, peptide2, os.path.join(outdir, f"{peptide1}_{peptide2}_{selection}.pdb"), selection)
        save_pdb(peptide2, peptide1, os.path.join(outdir, f"{peptide2}_{peptide1}_{selection}.pdb"), selection)


if __name__ == "__main__":
    main()
