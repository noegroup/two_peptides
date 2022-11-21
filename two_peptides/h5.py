
import os

import click
import h5py
import json
import warnings
import operator
import numpy as np
import mdtraj as md
from dataclasses import dataclass
from tqdm.auto import tqdm


@dataclass
class TwoPeptideSimulationOutput:
    peptide1: str
    peptide2: str
    _topology: md.Topology = None

    def filename(self, name, suffix):
        return f"data/{self.peptide1}_{self.peptide2}_{name}.{suffix}"

    @property
    def pdb(self):
        return self.filename("solute", "pdb")

    @property
    def coordinate_file(self):
        return self.filename("coord", "npy")

    @property
    def energy_file(self):
        return self.filename("energies", "npz")

    @property
    def force_file(self):
        return self.filename("force", "npy")

    @property
    def equilibration_file(self):
        return self.filename("equilibration", "txt")

    @property
    def topology(self):
        if self._topology is None:
            self._topology = md.load(self.pdb).top
        return self._topology

    def __str__(self):
        return f"{self.peptide1}_{self.peptide2}"

    def read_rewrapped_trajectory(self, topology, box_vectors) -> md.Trajectory:
        # read data
        lengths_angles = np.column_stack(
            md.utils.box_vectors_to_lengths_and_angles(box_vectors[:, 0], box_vectors[:, 1], box_vectors[:, 2]))
        traj = md.Trajectory(
            xyz=np.load(self.coordinate_file),
            unitcell_lengths=lengths_angles[:, :3],
            unitcell_angles=lengths_angles[:, 3:],
            topology=topology
        )
        rewrapped_trajectory = traj.image_molecules(
            anchor_molecules=[
                {topology.atom(i)
                 for i in range(traj.n_atoms)
                 if topology.atom(i) in traj.top.chain(0).atoms}
            ]
        )
        return rewrapped_trajectory

    def write_to_h5(self, destination):
        top = md.load(self.pdb).top
        energy_output = np.load(self.energy_file)
        box_vectors = energy_output["box"]
        trajectory = self.read_rewrapped_trajectory(top, box_vectors)
        forces = np.load(self.force_file)
        h5group = self.get_h5group(destination)
        nframes = trajectory.n_frames
        natoms = trajectory.n_atoms
        h5group.attrs["topology"] = topology_to_ascii(top)
        h5group.create_dataset("aa_coords", shape=(nframes, natoms, 3), dtype=np.float32, data=trajectory.xyz)
        h5group.create_dataset("aa_forces", shape=(nframes, natoms, 3), dtype=np.float32, data=forces)
        h5group.create_dataset("box_vectors", shape=(nframes, 3, 3), dtype=np.float32, data=box_vectors)
        h5group.create_dataset("unbiased_energy", shape=(nframes,), dtype=np.float32, data=energy_output["unbiased_energy"])
        h5group.create_dataset("bias_energy", shape=(nframes,), dtype=np.float32, data=energy_output["bias_energy"])
        h5group.create_dataset("distance", shape=(nframes,), dtype=np.float32, data=energy_output["distance"])
        h5group.create_dataset("d0", shape=(nframes,), dtype=np.float32, data=energy_output["d0"])
        h5group.create_dataset("k", shape=(nframes,), dtype=np.float32, data=energy_output["k"])

    def get_h5group(self, destination):
        h5 = h5py.File(destination, "a")
        main_group = h5.require_group("MINI")
        main_group.attrs["description"] = """Simulation output
        """
        h5group = main_group.require_group(str(self))
        return h5group


def topology_to_ascii(topology_object: md.Topology):
    """adapted from mdtraj"""

    try:
        topology_dict = {
            'chains': [],
            'bonds': []
        }

        for chain in topology_object.chains:
            chain_dict = {
                'residues': [],
                'index': int(chain.index)
            }
            for residue in chain.residues:
                residue_dict = {
                    'index': int(residue.index),
                    'name': str(residue.name),
                    'atoms': [],
                    "resSeq": int(residue.resSeq),
                    "segmentID": str(residue.segment_id)
                }

                for atom in residue.atoms:

                    try:
                        element_symbol_string = str(atom.element.symbol)
                    except AttributeError:
                        element_symbol_string = ""

                    residue_dict['atoms'].append({
                        'index': int(atom.index),
                        'name': str(atom.name),
                        'element': element_symbol_string
                    })
                chain_dict['residues'].append(residue_dict)
            topology_dict['chains'].append(chain_dict)

        for atom1, atom2 in topology_object.bonds:
            topology_dict['bonds'].append([
                int(atom1.index),
                int(atom2.index)
            ])

    except AttributeError as e:
        raise AttributeError('topology_object fails to implement the'
                             'chains() -> residue() -> atoms() and bond() protocol. '
                             'Specifically, we encountered the following %s' % e)

    data = json.dumps(topology_dict)
    if not isinstance(data, bytes):
        data = data.encode('ascii')

    return data


def string_to_topology(ascii_string):
    """adapted from mdtraj"""

    topology_dict = json.loads(ascii_string)
    topology = md.Topology()

    for chain_dict in sorted(topology_dict['chains'], key=operator.itemgetter('index')):
        chain = topology.add_chain()
        for residue_dict in sorted(chain_dict['residues'], key=operator.itemgetter('index')):
            try:
                resSeq = residue_dict["resSeq"]
            except KeyError:
                resSeq = None
                warnings.warn('No resSeq information found in HDF file, defaulting to zero-based indices')
            try:
                segment_id = residue_dict["segmentID"]
            except KeyError:
                segment_id = ""
            residue = topology.add_residue(residue_dict['name'], chain, resSeq=resSeq, segment_id=segment_id)
            for atom_dict in sorted(residue_dict['atoms'], key=operator.itemgetter('index')):
                try:
                    element = md.core.element.get_by_symbol(atom_dict['element'])
                except KeyError:
                    element = md.core.element.virtual
                topology.add_atom(atom_dict['name'], element, residue)

    atoms = list(topology.atoms)
    for index1, index2 in topology_dict['bonds']:
        topology.add_bond(atoms[index1], atoms[index2])

    return topology


@click.command(name="to-h5")
@click.option("-d", "--destination", type=click.Path(dir_okay=False), default="./data/allatom.h5")
@click.option("--many/--no-many", default=False)
@click.option("-a", "--aminoacids1", default=None)
@click.option("-b", "--aminoacids2", default=None)
def to_h5_cmd(destination, many, aminoacids1, aminoacids2):
    """convert simulation output to h5"""
    from two_peptides.status import submitted
    if many:
        dimers = submitted()
    else:
        assert aminoacids1 is not None and aminoacids2 is not None
        dimers = ((aminoacids1, aminoacids2), )

    dimers = list(dimers)
    pbar = tqdm(dimers)
    for twopep in pbar:
        sim_output = TwoPeptideSimulationOutput(*twopep)
        print(sim_output)
        if not os.path.exists(sim_output.pdb):
            print(sim_output.pdb, "does not exist. Skipping")
            continue
        if not os.path.exists(sim_output.coordinate_file):
            print(sim_output.pdb, "does not exist. Skipping")
            continue
        pbar.desc = str(sim_output)
        try:
            sim_output.write_to_h5(destination)
        except Exception as e:
            print(e)
            continue



