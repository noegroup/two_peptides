"""Sanity checks"""

import os
import glob
import dataclasses
import numpy as np
import click
import mdtraj as md
from mdtraj.utils import box_vectors_to_lengths_and_angles
from bgmol.systems import TwoMiniPeptides


@dataclasses.dataclass()
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