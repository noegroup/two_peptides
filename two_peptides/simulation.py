
__all__ = ["TwoPeptideSimulation", "barostat"]


import warnings
import contextlib
import numpy as np

from bgmol.systems import TwoMiniPeptides
from bgmol.util.importing import import_openmm
from .report import Report
from .meta import embedding
mm, unit, app = import_openmm()


class TwoPeptideSimulation:
    def __init__(
            self,
            aminoacids1,
            aminoacids2,
            d0=3.0,
            friction=0.1,
            k=500.
    ):
        self.model = TwoMiniPeptides(aminoacids1=aminoacids1, aminoacids2=aminoacids2)
        system = add_umbrella(self.model.system, group1=self.beadgroup1, group2=self.beadgroup2, d0=d0, k=k)
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
    def beadgroup1(self):
        return self.model.select(CGBEADS + "and resid < 10")

    @property
    def beadgroup2(self):
        return self.model.select(CGBEADS + "and resid > 10")

    @property
    def beads(self):
        return np.sort(self.model.select(CGBEADS))

    @property
    def embedding(self):
        return np.array([embedding(self.model.mdtraj_topology.atom(i)) for i in self.beads])

    def minimize(self):
        self.simulation.minimizeEnergy()

    def step(self, steps):
        self.simulation.step(steps)

    def report(self) -> Report:
        return Report.from_context(
            self.simulation.context,
            atom_ids=self.beads,
            center_group=self.beadgroup1,
            topology=self.model.mdtraj_topology
        )


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


CGBEADS = (
    "not water "
    "and resname != ACE "
    "and resname != NME "
    "and (backbone or name CB) "
)
