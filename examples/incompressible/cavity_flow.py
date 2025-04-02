#%%
from ngsolve import *
from ngsolve.webgui import Draw
from dream.incompressible import IncompressibleSolver, Inflow, flowfields
from dream.compressible import CompressibleFlowSolver
mesh = Mesh(unit_square.GenerateMesh(maxh=0.1))

# Set up the solver configuration
cfg = IncompressibleSolver(mesh)
# cfg = CompressibleFlowSolver(mesh)

cfg.reynolds_number = 1
cfg.fem.order = 4
cfg.dynamic_viscosity = "powerlaw"

cfg.linear_solver = "umfpack"

# Set up the boundary conditions
a = flowfields()
cfg.bcs['top'] = Inflow(fields={"u": (1,0)})#flowfields(velocity=(1, 0)))
cfg.bcs['right|left|bottom'] = "wall"
# cfg.bcs['right'] = "outflow"


# Initialize the finite element spaces, trial and test functions, gridfunctions, and boundary conditions
cfg.initialize()

fields = cfg.get_fields(u=True)
# cfg.io.vtk = True
# cfg.io.vtk.fields = cfg.get_fields(u=True, p=True)

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
