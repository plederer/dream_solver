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


mesh = circular_cylinder_mesh(radius=0.5,
                              sponge_layer=False,
                              boundary_layer_levels=1,
                              boundary_layer_thickness=0.08,
                              transition_layer_levels=2,
                              transition_layer_growth=1.2,
                              transition_radial_factor=8,
                              farfield_radial_factor=100,
                              sponge_radial_factor=200,
                              wake_maxh=1,
                              farfield_maxh=5)
mesh = Mesh(mesh)
mesh.Curve(cfg.order)

directory_name = f"Re{int(cfg.Reynolds_number.Get())}_no_sponge_{cfg.order}"
tree = ResultsDirectoryTree(directory_name)

solver = CompressibleHDGSolver(mesh, cfg, tree)
solver.boundary_conditions.set_farfield('inflow|outflow', rho_inf, u_inf, pressure=p_inf)
solver.boundary_conditions.set_adiabatic_wall('cylinder')
solver.domain_conditions.set_initial(rho_inf, u_inf, pressure=p_inf)


loader = solver.get_loader()
saver = solver.get_saver()
saver.save_mesh(name='mesh')


# Solve Stationary
cfg.time_step = 0.01
cfg.time_step_max = 10
cfg.convergence_criterion = 1e-12
with TaskManager():
    solver.setup()
    solver.solve_initial()
    solver.solve_stationary(increment_at_iteration=10, increment_time_step_factor=10)

saver.save_configuration(name='steady_configuration')
saver.save_state(name='intermediate_0', save_time_scheme_components=True)

solver.setup()
solver.drawer.draw()

loader.load_state('intermediate_0', False)

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
solver.formulation.time_scheme.update_initial_solution(solver.formulation.gfu, *solver.formulation._gfu_old)

Redraw()


# Solve coarse transient
cfg.time_step = 0.1
cfg.time_period = (0, 100)
cfg.convergence_criterion = 1e-8

with TaskManager():
    solver.solve_transient()

saver.save_configuration(name="transient_configuration_coarse")
saver.save_state(f"intermediate_{cfg.time_period.end}", save_time_scheme_components=True)

# Solve fine transient
cfg.time_step = 0.01
cfg.time_period = (100, 300)
cfg.convergence_criterion = 1e-12
cfg.save_state = True

with TaskManager():
    solver.solve_transient()
saver.save_configuration(name="transient_configuration_fine")
saver.save_state(f"intermediate_{cfg.time_period.end}", save_time_scheme_components=True)
