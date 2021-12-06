
# Simulation Data for Pairs of Small Peptides

Umbrella sampling between two peptides 
covering distances from 4 A to 30 A.
The distances are measured between the centers-of-mass
of the "beads" in peptide 1 and 2.



## Use the dataset
To load all available data:

```python
from two_peptides import TwoPeptidesDataset
dataset = TwoPeptidesDataset(
    coordglob="<DATADIR>/*_coord.npy",
    forceglob="<DATADIR>/*_force.npy",
    embedglob="<DATADIR>/*_embed.npy",
)
```

All energies are in kcal/mol.
All coorddinates are in Angstrom.
Embedding follows Nick's convention.

