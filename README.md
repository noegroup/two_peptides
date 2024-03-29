
# Simulation Data for Pairs of Small Peptides

Umbrella sampling between two peptides 
covering distances from 3 A to 30 A.
The distances are measured between the centers-of-mass
of the "saved_atoms" in peptide 1 and 2.

Data is located in `/import/a12/users/kraemea88/two_peptides`.

## Units
The coordinates in data files are in units of nanometers.
The forces are unbiased forces in units of kJ/mol/nm.

## Loading and processing Data
Processed all-atom data is located in the file ```/import/a12/users/kraemea88/two_peptides/data/allatom.h5```
For an example on how to use it, see `example_usage.ipynb`.

TL;DR: 
```python
DATA_FILE = "/import/a12/users/kraemea88/two_peptides/data/allatom.h5"
import h5py
with h5py.File(DATA_FILE, "r") as data:
    print(data["MINI"])
    print("Available Data:", list(data["MINI"]["IL_LF"].keys()))
    print("Coordinate shape:", data["MINI"]["IL_LF"]["aa_coords"].shape)
    
    from two_peptides.h5 import string_to_topology
    print(string_to_topology(data["MINI"]["IL_LF"].attrs["topology"]))
```

gives
```
<HDF5 group "/MINI" (1213 members)>
Available Data: ['aa_coords', 'aa_forces', 'bias_energy', 'box_vectors', 'd0', 'distance', 'k', 'unbiased_energy']
Coordinate shape: (27000, 101, 3)
<mdtraj.Topology with 2 chains, 8 residues, 101 atoms, 124 bonds>
```

Note that access to the data directory can be very slow.
It makes sense to work on a local copy of the data file instead.


## Caveats
Raw simulation output is located in the subdirectory `data/raw`. 
Using this raw output directly is not recommended, as the peptide coordinates are not properly wrapped
in the primary unit cell. 

All minipeptides are capped.

## Requirements
### Mandatory:
- numpy

### Optional:
- `pymbar` (for computing PMFs)
- `openmm` (for running simulations)
- `click` (for using the command-line interface)
- `bgmol` (branch `two_peptides`; for simulations and analysis)
- `mdtraj` (for simulations and analysis)
- `h5py` (for processing simulation output)


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
