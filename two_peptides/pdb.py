"""make pdb files"""

import os
import click
import mdtraj as md
from two_peptides.status import submitted
from two_peptides.simulation import TwoPeptideSimulation


def save_pdb(aminoacids1, aminoacids2, outdir, selection="beads"):
    sim = TwoPeptideSimulation(aminoacids1, aminoacids2)
    model = sim.model
    selected = {"beads": sim.beads, "protein": model.select("protein")}[selection]
    traj = md.Trajectory(model.positions[selected], topology=model.mdtraj_topology.subset(selected))
    traj.save_pdb(os.path.join(outdir, f"{aminoacids1}_{aminoacids2}_{selection}.pdb"))


@click.command()
@click.option("-o", "--outdir", type=click.Path(exists=True, writable=True, dir_okay=True), default="./data")
@click.option("-s", "--selection", type=click.Choice(["beads", "protein"]), default="beads")
def main(outdir, selection):
    for peptide1, peptide2 in submitted():
        print(peptide1, peptide2)
        save_pdb(peptide1, peptide2, outdir, selection)
        save_pdb(peptide2, peptide1, outdir, selection)


if __name__ == "__main__":
    main()
