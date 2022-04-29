
# Simulation Data for Pairs of Small Peptides

Umbrella sampling between two peptides 
covering distances from 4 A to 30 A.
The distances are measured between the centers-of-mass
of the "beads" in peptide 1 and 2.

Data is located in `/import/a12/users/kraemea88/two_peptides`.

## Units
The coordinates in data files are in units of nanometers.
The forces are unbiased forces in units of kJ/mol/nm.
Units are converted to Angstrom and kcal/mol/A when loading the
`TwoPeptidesDataset`.

## Use the dataset
To load all available data:

```python
from two_peptides.dataset import TwoPeptidesDataset
dataset = TwoPeptidesDataset(
    coordglob="<DATADIR>/*_coord.npy",
    forceglob="<DATADIR>/*_force.npy",
    embedglob="<DATADIR>/*_embed.npy",
)
```
All energies are in kcal/mol.
All coordinates are in Angstrom.
Embedding follows Nick's convention.

To split the dataset into training and validation in a determinstic manner,
you can do
```python
from two_peptides.dataset import TwoPeptidesDataset
dataset = TwoPeptidesDataset(...)
trainset, valset = dataset.split(val_fraction=0.2)
```

To combine this dataset with another `InMemoryDataset`, 
you can add the two datasets together.
```python
other_dataset = ...
dataset = TwoPeptidesDataset(...)
total_dataset = dataset + other_dataset
```

The dataset can be used within a `torch_geometric.DataLoader`:
```python
from torch_geometric.data import DataLoader
loader = DataLoader(dataset, shuffle=True, batch_size=256)
for batch in loader:
    ...
```
## Other Types of Analysis
To analyse the potential of mean force, check
```python
from two_peptides.dataset import potential_of_mean_force
help(potential_of_mean_force)
```



## Requirements
### Mandatory:
- numpy

### Optional:
- `torchmd_net` (for loading datasets)
- `pymbar` (for computing PMFs)
- `openmm` (for running simulations)
- `click` (for using the command-line interface)
- `bgmol` (branch `two_peptides`; for simulations and analysis)
- `mdtraj` (for simulations and analysis)


## Simulation Settings
The simulation settings are the same as for the octapeptides, i.e.
- 9 Angstrom LJ cutoff
- 7.5-9 Angstrom LJ switching
- PME electrostatics
- amber99sbildn + TIP3P force field
- HBOND constraints
- no hydrogen mass repartitioning

Umbrella sampling was run using
- 2 fs time step
- 0.1 / ps friction constant
- a force constant of 500. kJ/mol/nm^2
- umbrella windows over the centroid bond distance from 4 to 40 Angstrom spaced in 1A intervals
- each simuation is equilibrated in NPT for 300 ps
- for each umbrella, the simulation is run for 1 ns. Coordinate/force pairs are saved every 1 ps.
