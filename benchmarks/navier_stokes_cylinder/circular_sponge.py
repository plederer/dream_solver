from dream import SolverConfiguration, CompressibleHDGSolver, ResultsDirectoryTree, Perturbation
from dream.utils.meshes import circular_cylinder_mesh
from ngsolve import *

load_stationary = False

ngsglobals.msg_level = 2
SetNumThreads(4)


cfg = SolverConfiguration()
cfg.formulation = 'conservative'
cfg.mixed_method = 'strain_heat'
cfg.dynamic_viscosity = 'constant'
cfg.riemann_solver = "hllem"
cfg.Mach_number = 0.3
cfg.Reynolds_number = 150
cfg.Prandtl_number = 0.75
cfg.heat_capacity_ratio = 1.4
cfg.order = 2
cfg.compile_flag = True
cfg.damping_factor = 1
cfg.time_scheme = 'BDF2'
cfg.linear_solver = 'pardiso'
cfg.simulation = 'stationary'
cfg.max_iterations = 100

rho_inf = 1
u_inf = (1, 0)
p_inf = 1/(cfg.Mach_number.Get()**2 * cfg.heat_capacity_ratio.Get())
weight_function_order = 3

R = 0.5
farfield_radial_factor = 100
sponge_radial_factor = 200
wake_maxh = 3
farfield_maxh = 15
sponge_maxh = 800

mesh = circular_cylinder_mesh(radius=R,
                              sponge_layer=True,
                              boundary_layer_levels=1,
                              boundary_layer_thickness=0.08,
                              transition_layer_levels=2,
                              transition_layer_growth=1.2,
                              transition_radial_factor=8,
                              farfield_radial_factor=farfield_radial_factor,
                              sponge_radial_factor=sponge_radial_factor,
                              wake_maxh=wake_maxh,
                              farfield_maxh=farfield_maxh,
                              sponge_maxh=sponge_maxh)
mesh = Mesh(mesh)
mesh.Curve(cfg.order)

directory_name = f"Re{int(cfg.Reynolds_number.Get())}_sponge_{cfg.order}"
tree = ResultsDirectoryTree(directory_name)

# Meta Data
cfg.info['Farfield Density'] = 1
cfg.info['Farfield Velocity'] = u_inf
cfg.info['Farfield Pressure'] = p_inf
cfg.info['Sponge Function'] = f'((r - r0)/𝚫r)^{weight_function_order}'

cfg.info['Radius'] = R
cfg.info['Farfield Radius'] = farfield_radial_factor * R
cfg.info['Sponge Radius'] = sponge_radial_factor * R
cfg.info['Wake Maxh'] = wake_maxh
cfg.info['Farfield Maxh'] = farfield_maxh
cfg.info['Sponge Maxh'] = sponge_maxh

r = sqrt(x**2 + y**2)
sponge_start = 0.5 * farfield_radial_factor
sponge_length = (sponge_radial_factor - farfield_radial_factor) * 0.5

weight_function = ((r - sponge_start)/sponge_length)**3

solver = CompressibleHDGSolver(mesh, cfg, tree)
solver.boundary_conditions.set_farfield('inflow|outflow', rho_inf, u_inf, pressure=p_inf)
solver.boundary_conditions.set_adiabatic_wall('cylinder')
solver.domain_conditions.set_sponge_layer('sponge', weight_function, rho_inf,
                                          u_inf, pressure=p_inf, weight_function_order=weight_function_order)

if not load_stationary:
    solver.domain_conditions.set_initial(rho_inf, u_inf, pressure=p_inf)

with TaskManager():
    solver.setup()
solver.drawer.draw()
solver.drawer.draw_acoustic_pressure(p_inf)

loader = solver.get_loader()
saver = solver.get_saver()
saver.save_mesh(name='mesh')

# Solve Stationary
if load_stationary:
    loader.load_configuration('stationary')
    loader.load_state_time_scheme('stationary')
else:
    cfg.time_step = 0.01
    cfg.time_step_max = 10
    cfg.convergence_criterion = 1e-10
    with TaskManager():
        solver.solve_stationary()
    saver.save_state(name='stationary')
    saver.save_state_time_scheme('stationary')
    saver.save_configuration(name="stationary")

# Solver Perturbation
Gamma = 1
Rv = 1
Mx = -3
My = 0.1
r = sqrt((x-Mx)**2 + (y-My)**2)
p_0 = Gamma * exp(-r**2/(Rv**2))
psi = Gamma * exp(-r**2/(Rv**2))
u_0 = CF((psi.Diff(y), -psi.Diff(x)))
rho_0 = (p_0/p_inf)**(1/cfg.heat_capacity_ratio) * rho_inf
perturbation = Perturbation(rho_0, u_0, pressure=p_0)
with TaskManager():
    solver.add_perturbation(perturbation)

# Transient
cfg.time_step = 1
cfg.convergence_criterion = 1e-8
cfg.time_period = (0, 10)
with TaskManager():
    solver.solve_transient()

saver.save_configuration(name="transient_configuration_coarse")
saver.save_state_time_scheme(name=f"intermediate_{cfg.time_period.end}")

cfg.time_period = (10, 30)
cfg.save_state = True

with TaskManager():
    solver.solve_transient(save_state_every_num_step=10)
saver.save_configuration(name="transient_configuration_fine")
saver.save_state_time_scheme(name=f"intermediate_{cfg.time_period.end}")
