from dream import SolverConfiguration, CompressibleHDGSolver, ResultsDirectoryTree
from dream.utils.meshes import circular_cylinder_mesh
from ngsolve import *

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
u_inf = CF((1, 0))
p_inf = 1/(cfg.Mach_number**2 * cfg.heat_capacity_ratio)


farfield_radial_factor = 100
sponge_radial_factor = 400

mesh = circular_cylinder_mesh(radius=0.5,
                              sponge_layer=True,
                              boundary_layer_levels=1,
                              boundary_layer_thickness=0.08,
                              transition_layer_levels=2,
                              transition_layer_growth=1.2,
                              transition_radial_factor=8,
                              farfield_radial_factor=farfield_radial_factor,
                              sponge_radial_factor=sponge_radial_factor,
                              wake_maxh=1,
                              farfield_maxh=5, 
                              sponge_maxh=20)
mesh = Mesh(mesh)
mesh.Curve(cfg.order)
# input()
directory_name = f"Re{int(cfg.Reynolds_number.Get())}_sponge_{sponge_radial_factor}_{cfg.order}"
tree = ResultsDirectoryTree(directory_name)


r = sqrt(x**2 + y**2)
sponge_start = 0.5 * farfield_radial_factor
sponge_length = (sponge_radial_factor - farfield_radial_factor) * 0.5

weight_function = ((r - sponge_start)/sponge_length)**3

solver = CompressibleHDGSolver(mesh, cfg, tree)
solver.boundary_conditions.set_farfield('inflow|outflow', rho_inf, u_inf, pressure=p_inf)
solver.boundary_conditions.set_adiabatic_wall('cylinder')
solver.domain_conditions.set_initial(rho_inf, u_inf, pressure=p_inf)
solver.domain_conditions.set_sponge_layer('sponge', weight_function, rho_inf,
                                              u_inf, pressure=p_inf, weight_function_order=3)


# # Solve Stationary
cfg.time_step = 0.01
cfg.time_step_max = 10
cfg.convergence_criterion = 1e-10
with TaskManager():
    solver.setup()
    solver.solve_initial()
    solver.solve_stationary(increment_at_iteration=10, increment_time_step_factor=10)

loader = solver.get_loader()
saver = solver.get_saver()
saver.save_mesh(name='mesh')

saver.save_configuration(name='steady_configuration')
saver.save_state(name='intermediate_0')

solver.drawer.draw()

loader.load_state(name='intermediate_0')

Gamma = 1
Rv = 1
Mx = -3
My = 0.1
r = sqrt((x-Mx)**2 + (y-My)**2)
p_0 = Gamma * exp(-r**2/(Rv**2))
psi = Gamma * exp(-r**2/(Rv**2))
u_0 = CF((psi.Diff(y), -psi.Diff(x)))
rho_0 = (p_0/p_inf)**(1/cfg.heat_capacity_ratio) * rho_inf
rhoE_0 = p_0/(cfg.heat_capacity_ratio - 1) + rho_0 * 0.5 * (u_0[0]**2 + u_0[1]**2)
bump = CF((rho_0,rho_0 * u_0[0],rho_0 * u_0[1],rhoE_0))

gfu_pert = GridFunction(solver.formulation.fes)


fes = solver.formulation.fes
G, S = solver.formulation.TnT.MIXED
U, V = solver.formulation.TnT.PRIMAL
Uhat, Vhat = solver.formulation.TnT.PRIMAL_FACET

m = BilinearForm(fes)
m += G * S * dx()
m += U * V * dx()
m += Uhat * Vhat * dx(element_boundary = True)
m.Assemble()

f = LinearForm(fes)
f +=  bump * V * dx()
f += bump * Vhat * dx(element_boundary = True)
f.Assemble()

minv = m.mat.Inverse(fes.FreeDofs())
gfu_pert.vec.data = minv * f.vec

solver.formulation.gfu.vec.data += gfu_pert.vec
# solver.formulation.time_scheme.update_initial_solution(solver.formulation.gfu, *solver.formulation._gfu_old)
solver.formulation.update_gridfunctions(initial_value=True)
Redraw()


cfg.time_step = 0.01
cfg.convergence_criterion = 1e-8

cfg.time_period = (0, 100)
with TaskManager():
    solver.solve_transient()

saver.save_configuration(name="transient_configuration_initial")
saver.save_state_time_scheme(name=f"intermediate_{cfg.time_period.end}")

cfg.time_period = (100, 200)
cfg.save_state = True

with TaskManager():
    solver.solve_transient(save_state_every_num_step=10)
saver.save_configuration(name="transient_configuration_periodic")
saver.save_state_time_scheme(name=f"intermediate_{cfg.time_period.end}")
