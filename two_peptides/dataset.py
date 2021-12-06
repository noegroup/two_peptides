

__all__ = ["TwoPeptidesDataset", "potential_of_mean_force"]


import numpy as np
from torchmdnet.datasets.in_mem_dataset import InMemoryDataset


class TwoPeptidesDataset(InMemoryDataset):
    def __init__(self, coordglob, forceglob, embedglob, stride=1):
        super().__init__(coordglob, forceglob, embedglob, stride=stride)
        # unit conversion
        self.coord_list = [_nanometer_to_angstrom(c) for c in self.coord_list]
        self.force_list = [_kj_per_mol_and_nm_to_kcal_per_mol_and_angstrom(f) for f in self.force_list]


def potential_of_mean_force(peptide1, peptide2, rootdir):
    from pymbar import MBAR
    z = np.load(f"K_F_energies.npz")


def score():
    pass


def force_error():
    pass


# unit conversion

def _nanometer_to_angstrom(coordinates):
    return 10. * coordinates


def _kj_per_mol_and_nm_to_kcal_per_mol_and_angstrom(forces):
    return 0.023900573613766716 * forces


def _kj_to_kcal(energies):
    return 0.23900573613766718 * energies


def _kcal_to_kt(energies):
    return 1.6773984449958859 * energies


def _kj_to_kt(energies):
    return 0.40090785014242025 * energies
