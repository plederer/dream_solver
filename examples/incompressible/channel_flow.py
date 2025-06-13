import ngsolve as ngs
from dream.incompressible import IncompressibleSolver, Inflow, flowfields, Force
import dream.bla as bla
from dream.io import DomainL2Sensor
import netgen.occ as occ

# Channel flow
H = 2.0

# Power-law fluid parameters
r = 1.5
dP = 2


face = occ.WorkPlane().RectangleC(4, H).Face()
for edge, name in zip(face.edges, ["bottom", "right", "top", "left"]):
    edge.name = name
face.edges[1].Identify(face.edges[3], "periodic", occ.IdentificationType.PERIODIC)
face.name = "channel"
geo = occ.OCCGeometry(face, dim=2)
mesh = ngs.Mesh(geo.GenerateMesh(maxh=0.2))
mesh.Refine()

# Set up the solver configuration
cfg = IncompressibleSolver(mesh)
cfg.time = "stationary"
cfg.reynolds_number = 1
cfg.dynamic_viscosity = "constant"


cfg.fem = "taylor-hood"
cfg.fem.order = 2
cfg.fem.scheme = "stationary"


# Set up the boundary conditions
cfg.bcs['left|right'] = "periodic"
cfg.bcs['top|bottom'] = "wall"
cfg.dcs['channel'] = Force((dP, 0))

# Initialize the finite element spaces, trial and test functions, gridfunctions, and boundary conditions
cfg.initialize()

# Get solution fields
uh = cfg.get_solution_fields()
cfg.io.draw(uh)

# Solve stationary solution
cfg.solve()

# Change the viscosity model to power-law fluid
cfg.dynamic_viscosity = "powerlaw"
cfg.dynamic_viscosity.powerlaw_exponent = r


# Exact solution for the power-law fluid
ue = flowfields()
ue.u = ((r-1)/r * (dP)**(1/(r-1)) * (H/2)**(r/(r-1)) * (1- (2*bla.abs(ngs.y)/H)**(r/(r-1))), 0)
ue.p = 0

ngs.Draw(ue.u, mesh, "exact_velocity")

l2 = flowfields()
l2.u = uh.u - ue.u
l2.p = uh.p - ue.p

l2_sensor = DomainL2Sensor(l2, mesh)
cfg.io.sensor.enable = True
cfg.io.sensor.add(l2_sensor)

cfg.fem.initialize_symbolic_forms()

cfg.nonlinear_solver.damping_factor = 1
cfg.nonlinear_solver.max_iterations = 100


# Solve the system
cfg.solve()



# cfg.initialize_finite_element_spaces()
# cfg.initialize_trial_and_test_functions()
# cfg.initialize_gridfunctions()
# cfg.initialize_boundary_conditions()
# cfg.initialize_symbolic_forms()

# # Draw quantities
# cfg.draw(cfg.get_fields(u=True, p=True))

# # Initialize the solver and solve the system
# cfg.solver.initialize()
# cfg.solver.solve()

# # Change the viscosity model
# cfg.dynamic_viscosity = "powerlaw"
# cfg.dynamic_viscosity.powerlaw_exponent = 1.2
# cfg.solver = "nonlinear"
# cfg.solver.method = "newton"
# cfg.solver.method.damping_factor = 0.5
# cfg.solver.max_iterations = 100

# # Initialize forms again without reinitializing the finite element spaces
# cfg.initialize_symbolic_forms()
# cfg.draw(cfg.get_fields(u=True, p=True))

# # Solve the system again
# cfg.solver.initialize()
# cfg.solver.solve()
# %%

# import matplotlib.pyplot as plt
# import numpy as np
# y = np.linspace(-H/2, H/2, 100)
# u = fields.u(mesh(0, y))

# fig, ax = plt.subplots()
# ax.plot(u, y)

# plt.show()
#