

__all__ = ["run"]

from two_peptides.cli import cli_main


import os
from itertools import product
from typing import Sequence
import numpy as np

import click

from bgmol.systems.minipeptides import AMINO_ACIDS
from .simulation import TwoPeptideSimulation, barostat
from .report import Report
from .meta import DEFAULT_DISTANCES


VALID_PEPTIDES = (
    list(AMINO_ACIDS.keys())
    +
    ["".join(x) for x in product(list(AMINO_ACIDS.keys()), list(AMINO_ACIDS.keys()))]
)


def run(
        aminoacids1: str,
        aminoacids2: str,
        outdir: str = "./data",
        distances: Sequence[float] = DEFAULT_DISTANCES,
        test: bool = False
):
    def filename(suffix, is_test=test, npz=False):
        return os.path.join(
            outdir,
            f"{'test_' if is_test else ''}"
            f"{aminoacids1}_{aminoacids2}_{suffix}."
            f"{'npz' if npz else 'npy'}"
        )
    assert not os.path.exists(filename("coord"))
    assert not os.path.exists(filename("force"))
    assert not os.path.exists(filename("pdb"))
    assert not os.path.exists(filename("energies", npz=True))

    simulation = TwoPeptideSimulation(aminoacids1, aminoacids2)
    simulation.d0 = distances[0]
    simulation.k = 1. if test else 500.

    with barostat(simulation.simulation):
        simulation.minimize()
        simulation.friction = 100.
        simulation.step(0 if test else 125000)
        simulation.friction = 0.1
        simulation.step(0 if test else 25000)

    reports = []

    for distance in distances:
        # equilibrate
        simulation.d0 = distance
        simulation.friction = 100.
        simulation.step(1 if test else 25000)
        simulation.friction = 0.1
        simulation.step(1 if test else 25000)
        for i in range(10 if test else 1000):
            simulation.step(1 if test else 500)
            reports.append(simulation.report())

    summary_report = Report.from_reports(*reports)
    np.save(filename("coord"), summary_report.positions)
    np.save(filename("force"), summary_report.unbiased_forces)
    summary_report.save_energies(filename("energies", npz=True))


@cli_main.command(name="run")
@click.option("-a", "--aminoacids1", type=click.Choice(VALID_PEPTIDES))
@click.option("-b", "--aminoacids2", type=click.Choice(VALID_PEPTIDES))
@click.option("-o", "--outdir", type=click.Path(exists=True, dir_okay=True, writable=True), default="./data")
@click.option("-d", "--distances", type=float, multiple=True, default=DEFAULT_DISTANCES.tolist())
@click.option("--test/--no-test", default=False)
def run_cmd(
        aminoacids1: str,
        aminoacids2: str,
        outdir: str = "./data",
        distances: Sequence[float] = DEFAULT_DISTANCES,
        test: bool = False
):
    """Run simulation of peptide dimer.
    Example: `two_peptides run -a AA -b YY` will run umbrella sampling between dialanine and dityrosine
    and save the results to a subdirectory data.
    """
    return run(aminoacids1, aminoacids2, outdir, distances, test)
