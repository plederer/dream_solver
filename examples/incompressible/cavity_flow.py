# %%
from ngsolve import *
from dream.incompressible import IncompressibleSolver, Inflow, flowfields
mesh = Mesh(unit_square.GenerateMesh(maxh=0.05))

# Set up the solver configuration
cfg = IncompressibleSolver(mesh)
cfg.dynamic_viscosity = "constant"

cfg.time = "stationary"
cfg.fem = "taylor-hood"
cfg.fem = "hdivhdg"

cfg.fem.scheme = "direct"

# Set up the boundary conditions
a = flowfields()
cfg.bcs['top'] = Inflow(velocity=(x**2*(1-x)**2, 0))
cfg.bcs['right|left|bottom'] = "wall"

# Initialize the finite element spaces, trial and test functions, gridfunctions, and boundary conditions
cfg.initialize()

# Solve stokes system
fields = cfg.get_solution_fields()
cfg.io.draw(fields)

cfg.solve()


# Set up convection
cfg.convection = True
cfg.reynolds_number = 5000

cfg.fem.solver.method = "newton"
cfg.fem.solver.method.max_iterations = 100
cfg.fem.solver.method.damping_factor = 0.2
cfg.fem.solver.method.convergence_criterion = 1e-8

cfg.fem.initialize_symbolic_forms()
cfg.solve()
