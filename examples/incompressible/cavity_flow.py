from ngsolve import *
from dream import SolverConfiguration
from dream.incompressible import *

mesh = Mesh(unit_square.GenerateMesh(maxh=0.1))

cfg = SolverConfiguration(mesh)
cfg.pde = "incompressible"
cfg.pde.reynolds_number = 1e-2
cfg.pde.fe.order = 4

cfg.solver.inverse.solver = "umfpack"

# cfg.pde.bcs['top'] = Inflow(state=flowstate(u=(1, 0)))
# cfg.pde.bcs['left|bottom|right'] = "wall"

cfg.pde.bcs['left'] = Inflow(state=flowstate(u=(1, 0)))
cfg.pde.bcs['top|bottom'] = "wall"
cfg.pde.bcs['right'] = "outflow"

cfg.pde.set_finite_element_spaces()
cfg.pde.set_trial_and_test_functions()
cfg.pde.set_gridfunctions()
cfg.pde.set_boundary_conditions()
cfg.pde.set_discrete_system_tree()


drawing = cfg.pde.get_drawing_state(u=True, p=True, asdjas=True)
cfg.pde.draw()

cfg.solver.set_discrete_system()
cfg.solver.solve()


cfg.pde.dynamic_viscosity = "powerlaw"
cfg.pde.dynamic_viscosity.powerlaw_exponent = 1.2
cfg.solver = "nonlinear"
cfg.solver.method = "newton"
cfg.solver.method.damping_factor = 0.1
cfg.solver.max_iterations = 100

cfg.pde.set_discrete_system_tree()
drawing = cfg.pde.get_drawing_state(u=True, p=True)
cfg.pde.draw()

cfg.solver.set_discrete_system()
cfg.solver.solve()