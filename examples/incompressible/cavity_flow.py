from ngsolve import *
from dream import SolverConfiguration
from dream.incompressible import *

mesh = Mesh(unit_square.GenerateMesh(maxh=0.1))

# Set up the solver configuration
cfg = SolverConfiguration(mesh)
cfg.pde = "incompressible"
cfg.pde.reynolds_number = 1
cfg.pde.fem.order = 4

cfg.solver.inverse.solver = "umfpack"

# Set up the boundary conditions
cfg.pde.bcs['left'] = Inflow(state=flowstate(u=(1, 0)))
cfg.pde.bcs['top|bottom'] = "wall"
cfg.pde.bcs['right'] = "outflow"

# Initialize the finite element spaces, trial and test functions, gridfunctions, and boundary conditions
cfg.pde.initialize_finite_element_spaces()
cfg.pde.initialize_trial_and_test_functions()
cfg.pde.initialize_gridfunctions()
cfg.pde.initialize_boundary_conditions()
cfg.pde.initialize_symbolic_forms()

# Draw quantities
cfg.pde.draw(cfg.pde.get_fields(u=True, p=True))

# Initialize the solver and solve the system
cfg.solver.initialize()
cfg.solver.solve()

# Change the viscosity model
cfg.pde.dynamic_viscosity = "powerlaw"
cfg.pde.dynamic_viscosity.powerlaw_exponent = 1.2
cfg.solver = "nonlinear"
cfg.solver.method = "newton"
cfg.solver.method.damping_factor = 0.5
cfg.solver.max_iterations = 100

# Initialize forms again without reinitializing the finite element spaces
cfg.pde.initialize_symbolic_forms()
cfg.pde.draw(cfg.pde.get_fields(u=True, p=True))

# Solve the system again
cfg.solver.initialize()
cfg.solver.solve()