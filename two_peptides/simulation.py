
__all__ = ["TwoPeptideSimulation", "barostat"]


import warnings
import contextlib
import numpy as np

from bgmol.systems import TwoMiniPeptides
from bgmol.util.importing import import_openmm
import mdtraj as md
from .report import Report
from .sanity import EquilibrationStats
mm, unit, app = import_openmm()


class TwoPeptideSimulation:
    ATOM_SELECTION = "protein "
    
    def __init__(
            self,
            aminoacids1,
            aminoacids2,
            d0=3.0,
            friction=0.1,
            k=500.
    ):
        """Umbrella simulation of two peptides in OpenMM using the openmm.LangevinMiddleIntegrator.
        
        Attributes
        ----------
        friction : float
            Friction coefficient of the Langevin integrator
        d0 : float
            Center of the umbrella window; distance between centers of masses in nm
        k : float
            Force constant of the umbrella potential in kJ/mol/nm^2
        atomgroup1 : list of int
            Atom IDs of saved_atoms belonging to the first peptide
        atomgroup2 : list of int
            Atom IDs of saved_atoms belonging to the second peptide
        saved_atoms : list of int
            Atom IDs of all peptide saved_atoms
        """
        self.model = TwoMiniPeptides(aminoacids1=aminoacids1, aminoacids2=aminoacids2)
        system = add_umbrella(self.model.system, group1=self.atomgroup1, group2=self.atomgroup2, d0=d0, k=k)
        integrator = mm.LangevinMiddleIntegrator(300. * unit.kelvin, friction / unit.picoseconds, 2.0 * unit.femtoseconds)
        integrator.setConstraintTolerance(1e-7)
        try:
            platform = mm.Platform.getPlatformByName("CUDA")
            platform_properties = {"DeviceIndex": "0", "Precision": "mixed"}
        except mm.OpenMMException as e:
            warnings.warn(f"Using CPU platform. Caught exception: {str(e)}")
            platform = mm.Platform.getPlatformByName("CPU")
            platform_properties = dict()
        self.simulation = app.Simulation(self.model.topology, system, integrator, platform, platform_properties)
        self.simulation.context.setPositions(self.model.positions)
        self._n_dof, self._total_mass = _compute_n_dof_and_mass(self.simulation)
        self.stats = EquilibrationStats.empty()

    @property
    def friction(self):
        return self.simulation.context.getIntegrator().getFriction()

    @friction.setter
    def friction(self, gamma):
        self.simulation.context.getIntegrator().setFriction(gamma)

    @property
    def d0(self):
        return self.simulation.context.getParameters()["d0"]

    @d0.setter
    def d0(self, distance):
        self.simulation.context.setParameter("d0", distance)

    @property
    def k(self):
        return self.simulation.context.getParameters()["k"]

    @k.setter
    def k(self, force_constant):
        self.simulation.context.setParameter("k", force_constant)

    @property
    def atomgroup1(self):
        return self.model.select(TwoPeptideSimulation.ATOM_SELECTION + " and resid < 10")

    @property
    def atomgroup2(self):
        return self.model.select(TwoPeptideSimulation.ATOM_SELECTION + " and resid > 10")

    @property
    def saved_atoms(self):
        return np.sort(self.model.select(TwoPeptideSimulation.ATOM_SELECTION))

    def minimize(self):
        self.simulation.minimizeEnergy()

    def step(self, steps, log_stats_interval=500, production=False):
        self.stats = self.stats.append(EquilibrationStats.from_simulation(self, production=production))
        for step in range(steps // log_stats_interval):
            self.simulation.step(log_stats_interval)
            self.stats = self.stats.append(EquilibrationStats.from_simulation(self, production=production))
        self.simulation.step(steps % log_stats_interval)

    def report(self) -> Report:
        return Report.from_context(
            self.simulation.context,
            atom_ids=self.saved_atoms,
            center_group=self.atomgroup1,
            topology=self.model.mdtraj_topology
        )

    def write_stats(self, filename):
        self.stats.save(filename)

    def save_pdb(self, filename, selection="saved_atoms"):
        selected = {"saved_atoms": self.saved_atoms, "protein": self.model.select("protein")}[selection]
        traj = md.Trajectory(self.model.positions[selected], topology=self.model.mdtraj_topology.subset(selected))
        traj.save_pdb(filename)



@contextlib.contextmanager
def barostat(simulation: app.Simulation, pressure=1.0 * unit.atmosphere, temperature=300*unit.kelvin):
    barostat = mm.MonteCarloBarostat(pressure, temperature)
    force_id = simulation.context.getSystem().addForce(barostat)
    simulation.context.reinitialize(preserveState=True)
    yield
    simulation.context.getSystem().removeForce(force_id)
    simulation.context.reinitialize(preserveState=True)


def add_umbrella(system, group1, group2, d0=3.0, k=500.0):
    umbrella_system = copy_system(system)
    for force in umbrella_system.getForces():
        force.setForceGroup(1)
    force = mm.CustomCentroidBondForce(2, "0.5*k*(distance(g1,g2)-d0)^2")
    force.addGlobalParameter("d0", d0)
    force.addGlobalParameter("k", k)
    force.addEnergyParameterDerivative("d0")
    g1 = force.addGroup(group1.tolist())
    g2 = force.addGroup(group2.tolist())
    force.addBond([g1, g2], [])
    force.setForceGroup(11)
    force.setUsesPeriodicBoundaryConditions(True)
    umbrella_system.addForce(force)
    return umbrella_system


def copy_system(source: mm.System):
    return mm.XmlSerializer.deserializeSystem(mm.XmlSerializer.serializeSystem(source))


def _compute_n_dof_and_mass(simulation):
    """Compute number of degrees of freedom and total mass.
    from openmm.app.StateDataReporter
    """
    system = simulation.system

    # Compute the number of degrees of freedom.
    dof = 0
    for i in range(system.getNumParticles()):
        if system.getParticleMass(i) > 0*unit.dalton:
            dof += 3
    for i in range(system.getNumConstraints()):
        p1, p2, distance = system.getConstraintParameters(i)
        if system.getParticleMass(p1) > 0*unit.dalton or system.getParticleMass(p2) > 0*unit.dalton:
            dof -= 1
    if any(type(system.getForce(i)) == mm.CMMotionRemover for i in range(system.getNumForces())):
        dof -= 3

    # Compute the total system mass.
    mass = 0*unit.dalton
    for i in range(system.getNumParticles()):
        mass += system.getParticleMass(i)

    return dof, mass
