from dream import CompressibleHDGSolver
from ngsolve import TaskManager, Draw
from dream import SolverConfiguration, ResultsDirectoryTree
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
cfg.order = 2
cfg.compile_flag = True
cfg.damping_factor = 1
cfg.max_iterations = 100
cfg.time_scheme = 'BDF2'
cfg.linear_solver = 'pardiso'

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
farfield_radial_factor = 100
sponge_radial_factor = 600
wake_factor = 1
wake_maxh = 3
farfield_maxh = 15
sponge_maxh = 40
weight_function_order = 3

# Dimensionless numbers
Re = int(cfg.Reynolds_number.Get())
Ma = cfg.Mach_number.Get()

directory_prefix = f"Re{Re}_Ma{Ma}_k{cfg.order}_maxh{wake_maxh}"


def circular_sponge_setup():
    mesh = circular_cylinder_mesh(radius=R,
                                  sponge_layer=True,
                                  boundary_layer_levels=boundary_layer_levels,
                                  boundary_layer_thickness=boundary_layer_thickness,
                                  transition_layer_levels=transition_layer_levels,
                                  transition_layer_growth=transition_layer_growth,
                                  transition_radial_factor=transition_radial_factor,
                                  farfield_radial_factor=farfield_radial_factor,
                                  sponge_radial_factor=sponge_radial_factor,
                                  wake_maxh=wake_maxh,
                                  farfield_maxh=farfield_maxh,
                                  sponge_maxh=sponge_maxh)
    mesh = Mesh(mesh)
    mesh.Curve(cfg.order)

    directory = f"{directory_prefix}_circular_sponge"
    tree = ResultsDirectoryTree(directory)

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

    return solver


def circular_setup():
    mesh = circular_cylinder_mesh(radius=R,
                                  sponge_layer=False,
                                  boundary_layer_levels=boundary_layer_levels,
                                  boundary_layer_thickness=boundary_layer_thickness,
                                  transition_layer_levels=transition_layer_levels,
                                  transition_layer_growth=transition_layer_growth,
                                  transition_radial_factor=transition_radial_factor,
                                  farfield_radial_factor=farfield_radial_factor,
                                  sponge_radial_factor=sponge_radial_factor,
                                  wake_maxh=wake_maxh,
                                  farfield_maxh=farfield_maxh)
    mesh = Mesh(mesh)
    mesh.Curve(cfg.order)

    directory = f"{directory_prefix}_circular"
    tree = ResultsDirectoryTree(directory)

    solver = CompressibleHDGSolver(mesh, cfg, tree)
    solver.boundary_conditions.set_farfield('inflow|outflow', rho_inf, u_inf, pressure=p_inf)
    solver.boundary_conditions.set_adiabatic_wall('cylinder')
    solver.domain_conditions.set_initial(rho_inf, u_inf, pressure=p_inf)

    return solver


def angular_setup(set_pressure_outflow: bool = False):
    mesh = angular_cylinder_mesh(radius=R,
                                 sponge_layer=False,
                                 boundary_layer_levels=boundary_layer_levels,
                                 boundary_layer_thickness=boundary_layer_thickness,
                                 transition_layer_levels=transition_layer_levels,
                                 transition_layer_growth=transition_layer_growth,
                                 transition_radial_factor=transition_radial_factor,
                                 farfield_radial_factor=farfield_radial_factor,
                                 sponge_radial_factor=sponge_radial_factor,
                                 wake_factor=wake_factor,
                                 wake_maxh=wake_maxh,
                                 farfield_maxh=farfield_maxh,
                                 sponge_maxh=sponge_maxh)
    mesh = Mesh(mesh)
    mesh.Curve(cfg.order)

    directory = f"{directory_prefix}_angular"
    tree = ResultsDirectoryTree(directory)

    solver = CompressibleHDGSolver(mesh, cfg, tree)
    solver.boundary_conditions.set_farfield('inflow|outflow', rho_inf, u_inf, pressure=p_inf)
    if set_pressure_outflow:
        solver.boundary_conditions.set_outflow('outflow', p_inf)
    solver.boundary_conditions.set_adiabatic_wall('cylinder')
    solver.domain_conditions.set_initial(rho_inf, u_inf, pressure=p_inf)

    return solver


def angular_sponge_setup(set_pressure_outflow: bool = False):
    mesh = angular_cylinder_mesh(radius=R,
                                 sponge_layer=True,
                                 boundary_layer_levels=boundary_layer_levels,
                                 boundary_layer_thickness=boundary_layer_thickness,
                                 transition_layer_levels=transition_layer_levels,
                                 transition_layer_growth=transition_layer_growth,
                                 transition_radial_factor=transition_radial_factor,
                                 farfield_radial_factor=farfield_radial_factor,
                                 sponge_radial_factor=sponge_radial_factor,
                                 wake_factor=wake_factor,
                                 wake_maxh=wake_maxh,
                                 farfield_maxh=farfield_maxh,
                                 sponge_maxh=sponge_maxh)
    mesh = Mesh(mesh)
    mesh.Curve(cfg.order)

    directory = f"{directory_prefix}_angular"
    tree = ResultsDirectoryTree(directory)

    r = sqrt(x**2 + y**2)
    sponge_start = R * farfield_radial_factor
    sponge_length = (sponge_radial_factor - farfield_radial_factor) * R
    outflow_sponge_start = wake_factor * farfield_radial_factor * R

    weight_order = weight_function_order
    weight_inflow = ((r - sponge_start)/sponge_length)**weight_order
    weight_wake_top = ((y - sponge_start)/sponge_length)**weight_order
    weight_wake_bottom = ((-y - sponge_start)/sponge_length)**weight_order
    weight_outflow = ((x - outflow_sponge_start)/sponge_length)**weight_order
    weight_corner_top = weight_wake_top + weight_outflow
    weight_corner_bottom = weight_wake_bottom + weight_outflow

    solver = CompressibleHDGSolver(mesh, cfg, tree)
    solver.boundary_conditions.set_farfield('inflow|outflow', rho_inf, u_inf, pressure=p_inf)
    if set_pressure_outflow:
        solver.boundary_conditions.set_outflow('outflow', p_inf)
    solver.boundary_conditions.set_adiabatic_wall('cylinder')

    solver.domain_conditions.set_initial(rho_inf, u_inf, pressure=p_inf)
    solver.domain_conditions.set_sponge_layer('sponge_inflow', weight_inflow, rho_inf,
                                              u_inf, pressure=p_inf, weight_function_order=weight_order)
    solver.domain_conditions.set_sponge_layer('sponge_wake_top', weight_wake_top, rho_inf,
                                              u_inf, pressure=p_inf, weight_function_order=weight_order)
    solver.domain_conditions.set_sponge_layer('sponge_corner_top', weight_corner_top, rho_inf,
                                              u_inf, pressure=p_inf, weight_function_order=weight_order)
    solver.domain_conditions.set_sponge_layer('sponge_outflow', weight_outflow, rho_inf,
                                              u_inf, pressure=p_inf, weight_function_order=weight_order)
    solver.domain_conditions.set_sponge_layer('sponge_corner_bottom', weight_corner_bottom, rho_inf,
                                              u_inf, pressure=p_inf, weight_function_order=weight_order)
    solver.domain_conditions.set_sponge_layer('sponge_wake_bottom', weight_wake_bottom, rho_inf,
                                              u_inf, pressure=p_inf, weight_function_order=weight_order)

    return solver


def time_advancing_routine(solver: CompressibleHDGSolver,
                           stationary: bool = False,
                           transient: bool = False,
                           detailed_transient: bool = False,
                           time_periods: tuple = (0, 500, 800),
                           fine_time_step: float = 0.001,
                           draw: bool = False):

    cfg = solver.solver_configuration
    loader = solver.get_loader()
    saver = solver.get_saver()
    saver.save_mesh(name='mesh')
    solver.setup()

    if draw:
        solver.draw_solutions()
        Draw(solver.formulation.pressure(solver.gfu.components[0]) - p_inf, solver.mesh, "p'")
        Draw(solver.formulation.velocity(solver.gfu.components[0]) - u_inf, solver.mesh, "u'")

    # Solve Stationary
    cfg.time_step = 0.01
    cfg.time_step_max = 10
    cfg.convergence_criterion = 1e-12
    cfg.simulation = "stationary"
    if stationary:
        with TaskManager():
            solver.solve_initial()
            solver.solve_stationary(increment_at_iteration=10, increment_time_step_factor=10)

        saver.save_configuration(name='steady_configuration')
        saver.save_state(name='intermediate_0', save_time_scheme_components=True)
    else:
        loader.load_configuration('steady_configuration')
        loader.load_state(name='intermediate_0', load_time_scheme_components=True)

    # Solve coarse transient
    cfg.time_step = 1
    cfg.time_period = (time_periods[0], time_periods[1])
    cfg.convergence_criterion = 1e-8
    cfg.simulation = "transient"
    if transient:
        with TaskManager():
            solver.solve_transient()

        saver.save_configuration(name="transient_configuration_coarse")
        saver.save_state(f"intermediate_{cfg.time_period.end}", save_time_scheme_components=True)
    else:
        loader.load_configuration(name="transient_configuration_coarse")
        loader.load_state(f"intermediate_{cfg.time_period.end}", load_time_scheme_components=True)

    # Solve fine transient
    cfg.time_step = fine_time_step
    cfg.time_period = (time_periods[1], time_periods[2])
    cfg.convergence_criterion = 1e-12
    if detailed_transient:
        with TaskManager():
            solver.solve_transient()
        saver.save_configuration(name="transient_configuration_fine")
        saver.save_state(f"intermediate_{cfg.time_period.end}", save_time_scheme_components=True)


def saving_routine(solver: CompressibleHDGSolver, time_period=(800, 1000), save_state: bool = False, save_state_at_step=5,  draw=True):
    solver.setup()

    loader = solver.get_loader()
    loader.load_configuration(name="transient_configuration_fine")
    saver = solver.get_saver()

    cfg = solver.solver_configuration
    cfg.time_period = time_period
    cfg.save_state = save_state
    loader.load_state(f"intermediate_{cfg.time_period.start}", load_time_scheme_components=True)

    # Solve fine transient - save states

    if draw:
        solver.draw_solutions()
        Draw(solver.formulation.pressure(solver.gfu.components[0]) - p_inf, solver.mesh, "p'")
        Draw(solver.formulation.velocity(solver.gfu.components[0]) - u_inf, solver.mesh, "u'")

    with TaskManager():
        solver.solve_transient("transient", save_state_at_step=save_state_at_step)
    saver.save_state(f"intermediate_{cfg.time_period.end}", save_time_scheme_components=True)


if __name__ == '__main__':
    Draw(circular_sponge_setup().mesh)
