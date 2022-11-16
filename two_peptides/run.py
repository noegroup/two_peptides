

__all__ = ["run"]

import os
import socket
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
    def filename(name, is_test=test, suffix="npy"):
        return os.path.join(
            outdir,
            f"{'test_' if is_test else ''}"
            f"{aminoacids1}_{aminoacids2}_{name}."
            f"{suffix}"
        )
    assert not os.path.exists(filename("coord"))
    assert not os.path.exists(filename("force"))
    assert not os.path.exists(filename("energies", suffix="npz"))
    assert not os.path.exists(filename("equilibration", suffix="txt"))

    print("Running on node", socket.gethostname(), "using GPU id", os.getenv("SLURM_STEP_GPUS", default="NO GPU"))

    simulation = TwoPeptideSimulation(aminoacids1, aminoacids2)
    simulation.save_pdb(filename("solute", suffix="pdb"), selection="saved_atoms")
    simulation.d0 = distances[0]
    simulation.k = 1. if test else 500.

    print("Equilibrating", flush=True)
    with barostat(simulation.simulation):
        simulation.minimize()
        simulation.friction = 100.
        simulation.step(0 if test else 125000)
        simulation.friction = 0.1
        simulation.step(0 if test else 25000)

    reports = []

    for i, distance in enumerate(distances):
        print(f"... starting umbrella {i} / {len(distances)}", flush=True)
        # equilibrate
        simulation.d0 = distance
        simulation.friction = 100.
        simulation.step(1 if test else 25000)
        simulation.friction = 0.1
        for _ in range(5):
            simulation.step(1 if test else 5000)
        for i in range(10 if test else 1000):
            simulation.step(1 if test else 500, production=True)
            reports.append(simulation.report())

    summary_report = Report.from_reports(*reports)
    np.save(filename("coord"), summary_report.positions)
    np.save(filename("force"), summary_report.unbiased_forces)
    summary_report.save_energies(filename("energies", suffix="npz"))
    simulation.write_stats(filename("equilibration", suffix="txt"))


@click.command(name="run")
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
