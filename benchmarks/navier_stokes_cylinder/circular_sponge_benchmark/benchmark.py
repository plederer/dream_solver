from dream import CompressibleHDGSolver
from ngsolve import TaskManager, Draw
from dream import SolverConfiguration, ResultsDirectoryTree, Loader
from dream.utils.meshes import circular_cylinder_mesh, angular_cylinder_mesh
from ngsolve import *

ngsglobals.msg_level = 2
SetNumThreads(8)

# General Solver Configuration
cfg = SolverConfiguration()
cfg.formulation = 'conservative'
cfg.mixed_method = 'strain_heat'
cfg.dynamic_viscosity = 'constant'
cfg.riemann_solver = 'hllem'
cfg.Mach_number = 0.3
cfg.Reynolds_number = 150
cfg.Prandtl_number = 0.75
cfg.heat_capacity_ratio = 1.4
cfg.order = 3
cfg.compile_flag = True
cfg.damping_factor = 1
cfg.max_iterations = 100
cfg.time_scheme = 'BDF2'
cfg.linear_solver = 'pardiso'
cfg.bonus_int_order_bnd = cfg.order
cfg.bonus_int_order_vol = cfg.order

# Farfield Values
rho_inf = 1
u_inf = CF((1, 0))
p_inf = 1/(cfg.Mach_number**2 * cfg.heat_capacity_ratio)

# Geometry
R = 0.5
boundary_layer_levels = 1
boundary_layer_thickness = 0.08
transition_layer_levels = 4
transition_layer_growth = 1
transition_radial_factor = 10
farfield_radial_factor = 200
sponge_radial_factor = 3000
wake_factor = 1
wake_maxh = 3
farfield_maxh = 15
sponge_maxh = 800
weight_function_order = 3


tree = ResultsDirectoryTree()
mesh = Loader(tree).load_mesh('mesh')
mesh.Curve(cfg.order)

r = sqrt(x**2 + y**2)
sponge_start = R * farfield_radial_factor
sponge_length = (sponge_radial_factor - farfield_radial_factor) * R

weight_function = ((r - sponge_start)/sponge_length)**weight_function_order

solver = CompressibleHDGSolver(mesh, cfg, tree)
solver.boundary_conditions.set_farfield('inflow|outflow', rho_inf, u_inf, pressure=p_inf)
solver.boundary_conditions.set_adiabatic_wall('cylinder')
solver.domain_conditions.set_initial(rho_inf, u_inf, pressure=p_inf)
solver.domain_conditions.set_sponge_layer('sponge', weight_function, rho_inf,
                                          u_inf, pressure=p_inf, weight_function_order=weight_function_order)
solver.setup()

saver = solver.get_saver()
loader = solver.get_loader()

cfg.simulation = "transient"
cfg.time_step = 0.1
cfg.time_period = (600, 800)
cfg.convergence_criterion = 1e-12
cfg.save_state = True

loader.load_state(f'intermediate_{cfg.time_period.start}', load_time_scheme_components=True)

solver.drawer.draw()
solver.drawer.draw_particle_velocity(u_inf)
solver.drawer.draw_acoustic_pressure(p_inf, sd=3, autoscale=False, min=-1e-4, max=1e-4)

with TaskManager():
    solver.solve_transient("transient")
saver.save_configuration(name="transient_configuration_fine")
saver.save_state(f"intermediate_{cfg.time_period.end}", save_time_scheme_components=True)
