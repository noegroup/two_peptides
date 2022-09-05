

__all__ = ["TwoPeptidesDataset", "potential_of_mean_force"]


import os
from typing import Union, List, Tuple

import numpy as np
from dataclasses import dataclass
from .meta import DEFAULT_DISTANCES
from torchmdnet.datasets.in_mem_dataset import InMemoryDataset


class TwoPeptidesDataset(InMemoryDataset):
    """Pairs of small peptides in solution.
    """
    def __init__(self, coordglob, forceglob, embedglob, stride=1):
        super().__init__(coordglob, forceglob, embedglob, stride=stride)
        # unit conversion
        self.coord_list = [_nanometer_to_angstrom(c) for c in self.coord_list]
        self.force_list = [_kj_per_mol_and_nm_to_kcal_per_mol_and_angstrom(f) for f in self.force_list]

    def split(self, val_fraction: float, random_seed: int = 1) -> Tuple["TwoPeptidesDataset", "TwoPeptidesDataset"]:
        """deterministic split into two datasets"""
        n_files = len(self.coord_list)
        random_state = np.random.get_state()
        np.random.seed(random_seed)
        np.random.permutation(n_files)
        n_train = int((1.0 - val_fraction) * n_files)
        trainset = TwoPeptidesDataset("", "", "")
        valset = TwoPeptidesDataset("", "", "")
        trainset.coord_list = self.coord_list[:n_train]
        valset.coord_list = self.coord_list[n_train:]
        trainset.force_list = self.force_list[:n_train]
        valset.force_list = self.force_list[n_train:]
        trainset.embedding_list = self.embedding_list[:n_train]
        valset.embedding_list = self.embedding_list[n_train:]
        trainset.index = _make_index(trainset.coord_list)
        valset.index = _make_index(valset.coord_list)
        np.random.set_state(random_state)
        return trainset, valset

    def __add__(self, other: InMemoryDataset) -> InMemoryDataset:
        assert isinstance(other, InMemoryDataset)
        dataset = InMemoryDataset("", "", "")
        dataset.coord_list = self.coord_list + other.coord_list
        dataset.force_list = self.force_list + other.force_list
        dataset.embedding_list = self.embedding_list + other.embedding_list
        dataset.index = _make_index(dataset.coord_list)
        return dataset


def _make_index(coord_list):
    index = []
    for i, coords in enumerate(coord_list):
        size = len(coords)
        index.extend(list(zip([i] * size, range(size))))
    return index


@dataclass
class PMF:
    free_energy: np.ndarray  # pmf in kcal/mol
    uncertainty: np.ndarray  # uncertainty in kcal/mol
    bin_centers: np.ndarray  # bins in Angstrom


def potential_of_mean_force(peptide1: str, peptide2: str, rootdir: str) -> PMF:
    """Potential of mean force along the reaction coordinate
    (the distance between peptide1-beads and peptide2-beads center of mass).

    Example
    -------

    Plot the PMF between Ala and Tyr2 from the energies file in the current dir:
    >>> from two_peptides.dataset import potential_of_mean_force
    >>> pmf = potential_of_mean_force("A", "YY", rootdir="./")
    >>> plt.plot(pmf.bin_centers, pmf.free_energy)
    >>> plt.fill_between(
    >>>     pmf.bin_centers,
    >>>     pmf.free_energy - 2*pmf.uncertainty,
    >>>     pmf.free_energy + 2*pmf.uncertainty,
    >>>     alpha=0.1
    >>> )
    >>> plt.ylabel("Free Energy $F$ [kcal/mol]")
    >>> plt.xlabel("Distance $d0$ [A]")

    Notes
    -----
    Requires pymbar.
    """
    from pymbar import MBAR
    z = np.load(os.path.join(rootdir, f"{peptide1}_{peptide2}_energies.npz"))

    # compute free energies in kt and nm
    unbiased_energies = _kj_to_kt(z["unbiased_energy"])
    u_kn = unbiased_energies[None, :] + _umbrella_kt_nm(z["distance"][None, :], DEFAULT_DISTANCES[:, None])
    N_k = np.array([1000] * 27)
    mbar = MBAR(u_kn, N_k)

    # compute PMF
    n_bins = 50
    bins = np.linspace(0.4, 3.0, n_bins)
    digitized = np.digitize(z["distance"], bins)
    pmf, uncertainty = mbar.computePMF(unbiased_energies, digitized, n_bins)

    # return in kcal and angstrom
    return PMF(
        free_energy=_kt_to_kcal(pmf[:-1]),
        uncertainty=_kt_to_kcal(uncertainty[:-1]),
        bin_centers=_nanometer_to_angstrom(0.5*(bins[:-1]+bins[1:]))
    )


# standalone unit conversion

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


def _kt_to_kcal(energies):
    return 0.5961612775922492 * energies


def _umbrella_kt_nm(d, d0):
    # to this in nm and kt
    return _kj_to_kt(500.0) * (d - d0)**2
