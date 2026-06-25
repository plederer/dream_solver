import ngsolve as ngs
import netgen.occ as occ
import dream.bla as bla

from dream.io import DomainL2Sensor
from dream.incompressible_flow import IncompressibleFlowSolver, flowfields, Force

# Channel flow
H = 2.0



face = occ.WorkPlane().RectangleC(4, H).Face()
for edge, name in zip(face.edges, ["bottom", "right", "top", "left"]):
    edge.name = name
face.edges[1].Identify(face.edges[3], "periodic", occ.IdentificationType.PERIODIC)
face.name = "channel"
geo = occ.OCCGeometry(face, dim=2)
mesh = ngs.Mesh(geo.GenerateMesh(maxh=0.2))
mesh.Refine()

# Set up the solver configuration
cfg = IncompressibleFlowSolver(mesh)
cfg.time = "stationary"

cfg.reynolds_number = 1
cfg.dynamic_viscosity = "constant"


cfg.fem = "taylor-hood"
cfg.fem.order = 2
cfg.fem.scheme = "direct"


# Set up the boundary conditions
PRESSURE_DROP = 2
cfg.bcs['left|right'] = "periodic"
cfg.bcs['top|bottom'] = "wall"
cfg.dcs['channel'] = Force((PRESSURE_DROP, 0))

# Initialize the finite element spaces, trial and test functions, gridfunctions, and boundary conditions
cfg.initialize()

# Get solution fields
uh = cfg.get_solution_fields()
cfg.io.draw(uh)

# Solve stationary solution
cfg.solve()

# Change the viscosity model to power-law fluid
r = 1.5
cfg.dynamic_viscosity = "powerlaw"
cfg.dynamic_viscosity.powerlaw_exponent = r

# Exact solution for the power-law fluid
ue = flowfields()
ue.u = ((r-1)/r * (PRESSURE_DROP)**(1/(r-1)) * (H/2)**(r/(r-1)) * (1 - (2*bla.abs(ngs.y)/H)**(r/(r-1))), 0)
ue.p = 0

ngs.Draw(ue.u, mesh, "exact_velocity")

l2 = flowfields()
l2.u = uh.u - ue.u
l2.p = uh.p - ue.p

l2_sensor = DomainL2Sensor(l2, mesh)
cfg.io.sensor.enable = True
cfg.io.sensor.add(l2_sensor)

cfg.fem.initialize_symbolic_forms()

cfg.fem.solver.method = "newton"
cfg.fem.solver.method.convergence_criterion = 1e-8
cfg.fem.solver.method.damping_factor = 1
cfg.fem.solver.method.max_iterations = 100

# Solve the system
cfg.solve()
