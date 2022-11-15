"""Sanity checks"""

import os
import glob
import dataclasses
from typing import Union
import numpy as np
import click
import mdtraj as md
from openmm import unit
from mdtraj.utils import box_vectors_to_lengths_and_angles
from bgmol.systems import TwoMiniPeptides
from .report import centroid_distance


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


@dataclasses.dataclass(frozen=True)
class SanityInfo:
    box_lengths: np.ndarray
    density : float
    max_distance : float
    max_d0 : float


def sanity_info(peptide1, peptide2, rootdir="./data"):
    z = np.load(os.path.join(rootdir, f"{peptide1}_{peptide2}_energies.npz"))
    box = z["box"][0]
    # assert NVT
    for box_t in z["box"]:
        assert np.allclose(box_t, box)
    model = TwoMiniPeptides(aminoacids1=peptide1, aminoacids2=peptide2)
    lengths_angles = box_vectors_to_lengths_and_angles(*box)
    lengths, angles = lengths_angles[:,3], lengths_angles[3:]
    traj = md.Trajectory(
        xyz=model.positions[None, ...],
        unitcell_lengths=lengths[None, :],
        unitcell_angles=angles[None, :],
        topology=model.mdtraj_topology
    )
    return SanityInfo(
        box_lengths=lengths,
        density=md.density(traj),
        max_distance=z["distance"].max(),
        max_d0=z["d0"].max()
    )


@click.command()
@click.option("-o", "--outdir", type=click.Path(exists=True, dir_okay=True), default="./data")
def main(outdir):
    datafiles = glob.glob(os.path.join(outdir, "*_*_energies.npz"))
    for f in datafiles:
        peptide1, peptide2, _ = os.path.basename(f).split("_")
        info = sanity_info(peptide1, peptide2, outdir)
        is_sane = info.max_distance > info.box_lengths
        print(peptide1)


if __name__ == "__main__":
    main()