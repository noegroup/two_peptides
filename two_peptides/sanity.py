"""Sanity checks"""

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
@click.option("-a", "--aminoacids1", type=click.Choice(VALID_PEPTIDES))
@click.option("-b", "--aminoacids2", type=click.Choice(VALID_PEPTIDES))
@click.option("-o", "--outdir", type=click.Path(exists=True, dir_okay=True, writable=True), default="./data")
@click.option("-d", "--distances", type=float, multiple=True, default=DEFAULT_DISTANCES.tolist())
@click.option("--test/--no-test", default=False)
def sanity_check_cmd(
        aminoacids1: str,
        aminoacids2: str,
        outdir: str = "./data",
):
    """Run simulation of peptide dimer.
    Example: `two_peptides run -a AA -b YY` will run umbrella sampling between dialanine and dityrosine
    and save the results to a subdirectory data.
    """
    equi_file = EquilibrationStats.from_array(np.load(os.path.join(outdir, f"{aminoacids1}_{aminoacids2}_equilibration.txt")))
    density_ok = check_up_down(equi_file.density[300:600])
    density_unity = equi_file.density[-1] > 0.95 and equi_file.density[-1] > 1.05
    print(density_ok, density_unity)


def check_up_down(time_series, tol=0.1):
    increase = time_series[1:] > time_series[:-1] + 1e-10
    decrease = time_series[1:] < time_series[:-1] - 1e-10
    increase_fraction = increase.sum() / decrease.sum()
    return increase_fraction > 1. - tol and increase < 1. + tol
