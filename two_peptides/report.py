

__all__ = ["Report"]


import numpy as np
from dataclasses import dataclass, asdict
from typing import Union
import mdtraj as md
from bgmol.util.importing import import_openmm
mm, unit, app = import_openmm()


@dataclass(frozen=True)
class Report:
    positions: np.ndarray
    unbiased_forces: np.ndarray
    bias_forces: np.ndarray
    unbiased_energy: np.ndarray
    bias_energy: np.ndarray
    box: np.ndarray
    distance: Union[np.ndarray, float]
    d0: Union[np.ndarray, float]
    k: Union[np.ndarray, float]

    @staticmethod
    def from_context(context, atom_ids, topology, center_group) -> "Report":
        full_state = _get_state(context)
        unbiased_state = _get_state(context, groups={1})
        bias_state = _get_state(context, groups={11})
        # _assert_equal()
        report = Report(
            positions=_positions(full_state, topology, center_group)[atom_ids],
            unbiased_forces=_force(unbiased_state)[atom_ids],
            bias_forces=_force(bias_state)[atom_ids],
            unbiased_energy=_energy(unbiased_state),
            bias_energy=_energy(bias_state),
            box=full_state.getPeriodicBoxVectors(asNumpy=True),
            distance=_centroid_distance(full_state),
            d0=full_state.getParameters()["d0"],
            k=full_state.getParameters()["k"]
        )
        _assert_equal(_energy(full_state), _energy(unbiased_state) + _energy(bias_state))
        _assert_equal(_force(full_state), _force(unbiased_state) + _force(bias_state))
        _assert_equal(_energy(bias_state), report.k/2*(report.distance - report.d0)**2)
        return report

    @staticmethod
    def from_reports(*reports) -> "Report":
        fields = list(asdict(reports[0]).keys())
        assert all(list(asdict(report).keys()) == fields for report in reports)
        result_dict = {
            field: np.stack([getattr(report, field) for report in reports])
            for field in fields
        }
        return Report(**result_dict)

    def save_energies(self, outfile: str):
        np.savez(
            outfile,
            unbiased_energy=self.unbiased_energy,
            bias_energy=self.bias_energy,
            distance=self.distance,
            d0=self.d0,
            k=self.k,
            box=self.box
        )


def _get_state(context: mm.Context, **kwargs) -> mm.State:
    state = context.getState(
        getPositions=True,
        getParameters=True,
        getParameterDerivatives=True,
        enforcePeriodicBox=True,
        getForces=True,
        getEnergy=True,
        **kwargs
    )
    return state


def _force(state: mm.State):
    return state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)


def _positions(state: mm.State, topology: md.Topology, center_group: np.ndarray):
    xyz = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    box = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
    lengths_angles = md.utils.box_vectors_to_lengths_and_angles(*box)
    traj = md.Trajectory(
        xyz=xyz[None, ...],
        unitcell_lengths=lengths_angles[:3],
        unitcell_angles=lengths_angles[3:],
        topology=topology
    )
    traj.image_molecules(anchor_molecules=[{topology.atom(i) for i in center_group}])
    return traj.xyz[0, ...]


def _energy(state: mm.State):
    return state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)


def _centroid_distance(state: mm.State):
    du_dd = state.getEnergyParameterDerivatives()["d0"]
    k = state.getParameters()["k"]
    d0 = state.getParameters()["d0"]
    return d0 - du_dd/k


def _assert_equal(x, y):
    assert np.allclose(x, y, atol=1e-3)
