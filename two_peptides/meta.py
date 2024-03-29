

__all__ = ["embedding", "fast_folder_pairs", "DEFAULT_DISTANCES"]


import os
import pickle
from copy import deepcopy
import contextlib
import io
import random
from itertools import product

import numpy as np
from bgmol.systems import AMINO_ACIDS

DEFAULT_DISTANCES = np.arange(3.0, 0.3, -0.1)


embedding_map = {
    'ALA': 1,
    'CYS': 2,
    'ASP': 3,
    'GLU': 4,
    'PHE': 5,
    'GLY': 6,
    'HIS': 7,
    'ILE': 8,
    'LYS': 9,
    'LEU': 10,
    'MET': 11,
    'ASN': 12,
    'PRO': 13,
    'GLN': 14,
    'ARG': 15,
    'SER': 16,
    'THR': 17,
    'VAL': 18,
    'TRP': 19,
    'TYR': 20,
    'N': 21,
    'CA': 22,
    'C': 23,
    'O': 24
}


one_letter = {
    'ALA': 'A',
    'CYS': 'C',
    'ASP': 'D',
    'GLU': 'E',
    'PHE': 'F',
    'GLY': 'G',
    'HIS': 'H',
    'ILE': 'I',
    'LYS': 'K',
    'LEU': 'L',
    'MET': 'M',
    'ASN': 'N',
    'PRO': 'P',
    'GLN': 'Q',
    'ARG': 'R',
    'SER': 'S',
    'THR': 'T',
    'VAL': 'V',
    'TRP': 'W',
    'TYR': 'Y',
}


def embedding(atom: "mdtraj.core.topology.Atom") -> int:
    if atom.name == "CB" and atom.residue.name in embedding_map:
        return embedding_map[atom.residue.name]
    elif atom.name in ["N", "CA", "C", "O"]:
        return embedding_map[atom.name]
    else:
        return 0
        #raise ValueError(f"No embedding for atom {atom} in residue {atom.residue}")


def fast_folder_pairs():
    cachename = os.path.join(os.path.dirname(__file__), "_ffpairs.pic") 
    if os.path.isfile(cachename):
        with open(cachename, "rb") as f:
            return pickle.load(f)
    from bgmol.systems import FastFolder, FAST_FOLDER_NAMES
    lookup = deepcopy(one_letter)
    lookup["HSE"] = "H"
    lookup["HSD"] = "H"
    lookup["NLE"] = "L"
    pairs_of_pairs = set()
    for fast_folder in FAST_FOLDER_NAMES:
        with contextlib.redirect_stdout(io.StringIO()):
            with contextlib.redirect_stderr(io.StringIO()):
                model = FastFolder(fast_folder, solvated=False)
        top = model.mdtraj_topology
        pairs = set()
        for i in range(top.n_residues - 1):
            res1 = top.residue(i).name
            res2 = top.residue(i+1).name
            assert res1 in lookup
            assert res2 in lookup
            pairs.add((lookup[res1] + lookup[res2]))
        pairs = list(pairs)
        for i, pair1 in enumerate(pairs):
            for j, pair2 in enumerate(pairs):
                if j >= i:
                    continue
                pairs_of_pairs.add((pair1, pair2))
    pairs_of_pairs = random.choices(list(pairs_of_pairs), k=1000)
    with open(cachename, "wb") as f:
        pickle.dump(pairs_of_pairs, f)
    return pairs_of_pairs


VALID_PEPTIDES = (
    list(AMINO_ACIDS.keys())
    +
    ["".join(x) for x in product(list(AMINO_ACIDS.keys()), list(AMINO_ACIDS.keys()))]
)
