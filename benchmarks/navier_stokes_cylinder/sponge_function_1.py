from dream import CompressibleHDGSolver, SolverConfiguration, ResultsDirectoryTree, Saver
from dream.utils.meshes import circular_cylinder_mesh
from ngsolve import *

ngsglobals.msg_level = 0
SetNumThreads(8)

# Enter Directory Name
directory_prefix = "sponge_function_1_ffr100"
parent_path = None

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
cfg.order = 4
cfg.compile_flag = True
cfg.damping_factor = 1
cfg.max_iterations = 100
cfg.time.scheme = 'BDF2'
cfg.linear_solver = 'pardiso'
cfg.bonus_int_order_bnd = cfg.order
cfg.bonus_int_order_vol = cfg.order

# Solver Options
load_stationary = False
draw = False

# Results Directory
directory_name = f"{directory_prefix}_Ma{cfg.Mach_number.Get()}_Re{cfg.Reynolds_number.Get()}_order{cfg.order}"
tree = ResultsDirectoryTree(directory_name, parent_path=parent_path)
saver = Saver(tree)

# Geometry
R = 0.5
boundary_layer_levels = 2
boundary_layer_thickness = 0.08
transition_layer_levels = 4
transition_layer_growth = 1
transition_radial_factor = 10
farfield_radial_factor = 200
sponge_radial_factor = 3000
wake_maxh = 2
farfield_maxh = 12
sponge_maxh = 600

mesh = circular_cylinder_mesh(R,
                              True,
                              boundary_layer_levels,
                              boundary_layer_thickness,
                              transition_layer_levels,
                              transition_layer_growth,
                              transition_radial_factor,
                              farfield_radial_factor,
                              sponge_radial_factor,
                              wake_maxh,
                              farfield_maxh,
                              sponge_maxh,
                              grading=0.2)

mesh = Mesh(mesh)
saver.save_mesh(mesh)


# Farfield Values
rho_inf = 1
u_inf = (1, 0)
p_inf = 1/(cfg.Mach_number.Get()**2 * cfg.heat_capacity_ratio.Get())

# Meta Data
cfg.info['Farfield Density'] = 1
cfg.info['Farfield Velocity'] = u_inf
cfg.info['Farfield Pressure'] = p_inf
cfg.info['Sponge Function'] = 'r⁵'

cfg.info['Radius'] = R
cfg.info['Farfield Radius'] = farfield_radial_factor * R
cfg.info['Sponge Radius'] = sponge_radial_factor * R
cfg.info['Wake Maxh'] = wake_maxh
cfg.info['Farfield Maxh'] = farfield_maxh
cfg.info['Sponge Maxh'] = sponge_maxh


### Solution process ###
mesh.Curve(cfg.order)

# Sponge
r = sqrt(x**2 + y**2)
sp_start = R * farfield_radial_factor
sp_length = (sponge_radial_factor - farfield_radial_factor) * R

r_ = (r - sp_start)/sp_length
weight_function = r_**5

solver = CompressibleHDGSolver(mesh, cfg, tree)
solver.boundary_conditions.set_farfield('inflow|outflow', u_inf, rho_inf, p_inf)
solver.boundary_conditions.set_adiabatic_wall('cylinder')
solver.domain_conditions.set_sponge_layer('sponge', weight_function, u_inf, rho_inf, p_inf, weight_function_order=5)
if not load_stationary:
    solver.domain_conditions.set_initial(u_inf, rho_inf, p_inf)

saver = solver.get_saver()
loader = solver.get_loader()

with TaskManager():
    solver.setup()

if draw:
    solver.drawer.draw()
    solver.drawer.draw_particle_velocity(u_inf)
    solver.drawer.draw_acoustic_pressure(p_inf, sd=3, autoscale=False, min=-1e-4, max=1e-4)

# Solve Stationary
if load_stationary:
    loader.load_configuration('stationary')
    loader.load_state_time_scheme('stationary')
else:
    cfg.time.step = 0.1
    cfg.time.max_step = 10
    with TaskManager():
        solver.solve_stationary()
    saver.save_state_time_scheme('stationary')
    saver.save_configuration(name="stationary")

# Add perturbation to the solution
r = sqrt((x+4)**2 + (y-0.1)**2)
p_0 = 1.2 * exp(-r**2)
psi = 1.2 * exp(-r**2)
u_0 = CF((psi.Diff(y), -psi.Diff(x)))
rho_0 = (p_0/p_inf)**(1/cfg.heat_capacity_ratio) * rho_inf

with TaskManager():
    solver.add_perturbation(u_0, rho_0, p_0)

# Solver Transient
cfg.time.step = 0.1
cfg.time.interval = (0, 100)
cfg.convergence_criterion = 1e-12

with TaskManager():
    solver.solve_transient()
saver.save_configuration(name=f"transient")
saver.save_state_time_scheme(f"transient_{cfg.time.interval.end}")

# Solver Transient
cfg.time.interval = (100, 400)
cfg.save_state = True

with TaskManager():
    solver.solve_transient()
saver.save_state_time_scheme(f"transient_{cfg.time.interval.end}")
