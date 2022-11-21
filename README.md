
# Simulation Data for Pairs of Small Peptides

Umbrella sampling between two peptides 
covering distances from 4 A to 30 A.
The distances are measured between the centers-of-mass
of the "saved_atoms" in peptide 1 and 2.

Data is located in `/import/a12/users/kraemea88/two_peptides`.

## Units
The coordinates in data files are in units of nanometers.
The forces are unbiased forces in units of kJ/mol/nm.
Units are converted to Angstrom and kcal/mol/A when loading the
`TwoPeptidesDataset`.

## Loading and processing Data
Raw simulation output is located in the subdirectory `data`. 
Using this raw output directly is not recommended, as the peptide coordinates are not properly wrapped
in the primary unit cell. 

Processed all-atom data is located in the file ```/import/a12/users/kraemea88/two_peptides/data/allatom.h5```
The file structure is as follows



```python
from torch_geometric.data import DataLoader
loader = DataLoader(dataset, shuffle=True, batch_size=256)
for batch in loader:
    ...
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
- umbrella windows over the centroid bond distance from 3 to 30 Angstrom spaced in 1A intervals
- each simuation is equilibrated in NPT for 300 ps
- for each umbrella, the simulation is run for 1 ns. Coordinate/force pairs are saved every 1 ps.
