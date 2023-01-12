from ngsolve import *

from dream import CompressibleHDGSolver, SolverConfiguration
from dream.utils.meshes import Get_Omesh

ngsglobals.msg_level = 0
SetNumThreads(8)


cfg = SolverConfiguration()
# Formulation settings
cfg.formulation = 'conservative'
cfg.mixed_method = None
cfg.dynamic_viscosity = None
cfg.riemann_solver = "hllem"

# Flow settings
cfg.Mach_number = 0.1
cfg.heat_capacity_ratio = 1.4

# Solver settings
cfg.order = 3
cfg.damping_factor = 1.0
cfg.time_scheme = 'BDF2'
cfg.time_step = 0.05
cfg.max_iterations = 100
cfg.convergence_criterion = 1e-12
cfg.linear_solver = 'pardiso'
cfg.simulation = 'transient'
cfg.compile_flag = True

# Dimensionless equations with diameter D
D = 1
R = D/2
R_farfield = D * 30

rho_inf = 1
u_inf = CF((1, 0))
p_inf = 1/(cfg.Mach_number**2 * cfg.heat_capacity_ratio)

mesh = Mesh(Get_Omesh(R, R_farfield, 32, 16, geom=1.9))
mesh.Curve(cfg.order)

solver = CompressibleHDGSolver(mesh, cfg)
solver.boundary_conditions.set_farfield('inflow', rho_inf, u_inf, pressure=p_inf)
solver.boundary_conditions.set_nonreflecting_outflow('outflow', p_inf)
solver.boundary_conditions.set_inviscid_wall('cyl')
solver.initial_condition.set(rho_inf, u_inf, pressure=p_inf)

saver = solver.get_saver("test2")

t = 0
tend = 40
with TaskManager():
    saver.save_configuration()
    saver.save_mesh()
    solver.setup()
    solver.solve_initial()
    solver.draw_solutions()

    formulation = solver.formulation

    Draw(formulation.pressure(solver.gfu.components[0]) - p_inf, mesh, "p'")
    Draw(formulation.velocity(solver.gfu.components[0]) - u_inf, mesh, "u'")

    while t < tend:
        t += solver.solver_configuration.time_step.Get()
        solver.solve_timestep(True)