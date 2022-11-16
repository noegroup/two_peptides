"""make pdb files"""

import os
import click
from two_peptides.status import submitted
from two_peptides.simulation import TwoPeptideSimulation


@click.command()
@click.option("-o", "--outdir", type=click.Path(exists=True, writable=True, dir_okay=True), default="./data")
@click.option("-s", "--selection", type=click.Choice(["saved_atoms", "protein"]), default="protein")
def main(outdir, selection):
    for peptide1, peptide2 in submitted():
        sim = TwoPeptideSimulation(peptide1, peptide2)
        sim.save_pdb(os.path.join(outdir, f"{peptide1}_{peptide2}_{selection}.pdb"), selection)


if __name__ == "__main__":
    main()
