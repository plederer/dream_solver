# %%
from ngsolve import *
# from ngsolve.webgui import Draw
from dream.incompressible import IncompressibleSolver, Inflow, flowfields
mesh = Mesh(unit_square.GenerateMesh(maxh=0.1))
# mesh.Refine()

# Set up the solver configuration
cfg = IncompressibleSolver(mesh)
cfg.time = "stationary"
# cfg.fem = "taylor-hood"
cfg.fem = "hdivhdg"
cfg.fem.scheme = "stationary"

cfg.reynolds_number = 1000
cfg.fem.order = 4
cfg.dynamic_viscosity = "constant"
cfg.convection = True



# cfg.nonlinear_solver.method.damping_factor = 0.5
# cfg.nonlinear_solver.max_iterations = 100
# Set up the boundary conditions
a = flowfields()
cfg.bcs['top'] = Inflow(velocity=(x**2*(1-x)**2,0)) #flowfields(velocity=(1, 0)))
# cfg.bcs['top'] = Inflow(velocity = CF((1, 0)))
cfg.bcs['right|left|bottom'] = "wall"

# Initialize the finite element spaces, trial and test functions, gridfunctions, and boundary conditions
cfg.initialize()


fields = cfg.get_solution_fields()
cfg.io.draw(fields)

cfg.solve()

# cfg.dynamic_viscosity = "powerlaw"
# cfg.dynamic_viscosity.powerlaw_exponent = 1.5
# cfg.nonlinear_solver.max_iterations = 100
# cfg.nonlinear_solver.damping_factor = 0.5
# cfg.fem.initialize_symbolic_forms()
# cfg.solve()


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

