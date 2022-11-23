"""Sanity checks"""
import glob
import os
import dataclasses
from typing import Union
import numpy as np
import click
from openmm import unit
from .report import centroid_distance
from .meta import DEFAULT_DISTANCES, VALID_PEPTIDES


@dataclasses.dataclass()
class EquilibrationStats:
    density: Union[np.ndarray, float]
    potential_energy: Union[np.ndarray, float]
    temperature: Union[np.ndarray, float]
    d0: Union[np.ndarray, float] = np.nan
    d: Union[np.ndarray, float] = np.nan
    production: Union[np.ndarray, float] = 0.0  # indicator whether this belongs to equilibration or production phase

    def __post_init__(self):
        for key in dataclasses.asdict(self).keys():
            value = getattr(self, key)
            if isinstance(value, float):
                setattr(self, key, np.array([value]))

    @staticmethod
    def empty():
        return EquilibrationStats(np.ndarray([]), np.ndarray([]), np.ndarray([]), np.ndarray([]), np.ndarray([]),)

    @staticmethod
    def from_stats(*stats: "EquilibrationStats") -> "EquilibrationStats":
        fields = list(dataclasses.asdict(stats[0]).keys())
        assert all(list(dataclasses.asdict(stat).keys()) == fields for stat in stats)
        result_dict = {
            field: np.stack([getattr(stat, field) for stat in stats])
            for field in fields
        }
        return EquilibrationStats(**result_dict)

    @staticmethod
    def from_array(array: np.ndarray):
        return EquilibrationStats(
            density=array[:, 0],
            potential_energy=array[:, 1],
            temperature=array[:, 2],
            d0=array[:, 3],
            d=array[:, 4],
            production=array[:, 5]
        )

    @staticmethod
    def from_simulation(simulation: "TwoPeptideSimulation", production=False):
        context = simulation.simulation.context
        state = context.getState(getEnergy=True, getParameterDerivatives=True, getParameters=True)
        potential_energy = state.getPotentialEnergy()
        box_volume = state.getPeriodicBoxVolume()
        density = simulation._total_mass / box_volume
        integrator = context.getIntegrator()
        if hasattr(integrator, 'computeSystemTemperature'):
            temperature = integrator.computeSystemTemperature()
        else:
            temperature = (2 * state.getKineticEnergy() / (simulation._n_dof * unit.MOLAR_GAS_CONSTANT_R))

        return EquilibrationStats(
            potential_energy=potential_energy.value_in_unit(unit.kilojoule_per_mole),
            temperature=temperature.value_in_unit(unit.kelvin),
            density=density.value_in_unit(unit.gram/unit.item/unit.milliliter),
            d0=simulation.d0,
            d=centroid_distance(state),
            production=float(production)
        )

    def as_array(self):
        return np.column_stack(
            [self.density, self.potential_energy, self.temperature, self.d0, self.d, self.production]
        )

    def str(self):
        return np.array2string(self.as_array(), precision=2, suppress_small=True)

    def append(self, other):
        return self.from_array(np.row_stack([self.as_array(), other.as_array()]))

    def save(self, filename):
        np.savetxt(filename, self.as_array(), fmt="%15.5f", header=self.header())

    def header(self):
        return " ".join(f"{key:>14}" for key in dataclasses.asdict(self).keys())


@click.command(name="sanity-check")
@click.option("--many/--no-many", default=False)
@click.option("-a", "--aminoacids1", type=click.Choice(VALID_PEPTIDES), default=None)
@click.option("-b", "--aminoacids2", type=click.Choice(VALID_PEPTIDES), default=None)
@click.option("-o", "--outdir", type=click.Path(exists=True, dir_okay=True, writable=True), default="./data")
def sanity_check_cmd(
        many: bool,
        aminoacids1: str,
        aminoacids2: str,
        outdir: str = "./data",
):
    """Run simulation of peptide dimer.
    Example: `two_peptides run -a AA -b YY` will run umbrella sampling between dialanine and dityrosine
    and save the results to a subdirectory data.
    """

    print("      rho    se   drift <3se   d0     T    se")

    peptide_pairs = ((aminoacids1, aminoacids2), )
    if many:
        files = glob.glob(os.path.join(outdir, f"*_*_equilibration.txt"))
        peptide_pairs = (filename.split("_")[:2] for filename in files)

    for peptide1, peptide2 in peptide_pairs:
        print(f"{peptide1:>2} {peptide2:>2} ", end="")

        checks = []

        equilibration_file = os.path.join(outdir, f"{aminoacids1}_{aminoacids2}_equilibration.txt")
        equilibration_stats = EquilibrationStats.from_array(np.loadtxt(equilibration_file))

        # density
        production = equilibration_stats.production.astype(bool)
        mean, se, drift = mean_se_drift(equilibration_stats.density[150:300])
        checks.append(np.allclose(mean, 1.0, atol=0.02))
        checks.append(np.abs(drift) < 0.005)
        checks.append(se < 0.005)
        checks.append(np.abs(drift) < 3 * se)

        # distance
        checks.append(
            np.percentile(np.abs(equilibration_stats.d[production] - equilibration_stats.d0[production]), 99) < 0.2)

        # temperature
        mean, se, drift = mean_se_drift(equilibration_stats.temperature[production])
        checks.append(np.allclose(mean, 300., atol=1.0))
        checks.append(se < 1.0)

        print(" ".join("  .  " if check else " XXX " for check in checks))


def mean_se_drift(ts, n=5):
    chunks = np.array_split(ts, n)
    means = [np.mean(chunk) for chunk in chunks]
    a, b, *_ = np.polyfit(np.arange(5), means, 1)
    return np.mean(means), np.std(means)/np.sqrt(5), np.mean(np.abs([a*0+b - means[0], a*4+b - means[4]]))
