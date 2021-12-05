__all__ = ["embedding"]

import mdtraj.core.topology


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


def embedding(atom: mdtraj.core.topology.Atom):
    if atom.name == "CB" and atom.residue.name in embedding_map:
        return embedding_map[atom.residue.name]
    elif atom.name in ["N", "CA", "C", "O"]:
        return embedding_map[atom.name]
    else:
        raise ValueError(f"No embedding for atom {atom} in residue {atom.residue}")
