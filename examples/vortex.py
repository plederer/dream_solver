from netgen.occ import OCCGeometry
from ngsolve import *
from dream import CompressibleHDGSolver, SolverConfiguration
from dream.utils.geometries import MakeOCCRectangle, MakeOCCCircle, MakeOCCCirclePlane

ngsglobals.msg_level = 0
SetNumThreads(8)

# geo = MakeOCCRectangle((-40, -40), (40, 40))
geo = MakeOCCCircle((0, 0), 40)
mesh = Mesh(geo.GenerateMesh(maxh=5))

cfg = SolverConfiguration()

cfg.formulation = "conservative"
cfg.simulation = "transient"
cfg.dynamic_viscosity = "constant"
cfg.mixed_method = "strain_heat"

cfg.Reynold_number = 1
cfg.Mach_number = 0.3
cfg.Prandtl_number = 0.72
cfg.heat_capacity_ratio = 1.4

cfg.order = 2
cfg.bonus_int_order_bnd = 10
cfg.bonus_int_order_vol = 10

cfg.time_scheme = "BDF2"
cfg.time_step = 0.05

cfg.linear_solver = "pardiso"
cfg.damping_factor = 1
cfg.max_iterations = 100
cfg.convergence_criterion = 1e-12

cfg.compile_flag = True
cfg.static_condensation = True

mesh.Curve(order=cfg.order)

gamma = cfg.heat_capacity_ratio
M = cfg.Mach_number

rho_inf = 1
T_inf = 1 / ((gamma - 1) * M**2)

u_inf = CF((1, 0))
p_inf = 1/(M**2 * gamma)

Gamma = 1
rv = 1
psi = Gamma * exp(-(x**2 + y**2)/(2*rv**2))

u_0 = u_inf
p_0 = p_inf + Gamma * exp(-(x**2 + y**2)/(2*rv**2))


solver = CompressibleHDGSolver(mesh, cfg)
solver.boundary_conditions.set_farfield("left", rho_inf, u_inf, pressure=p_inf)
solver.boundary_conditions.set_nonreflecting_outflow(
    "right|bottom|top", pressure=p_inf, tangential_convective_fluxes=True, tangential_viscous_fluxes=True,
    normal_viscous_fluxes=False)
# solver.boundary_conditions.set_dirichlet("left|bottom|right|top", rho_inf, (u_inf, v_inf), pressure=p_inf)
solver.initial_condition.set(rho_inf, u_inf, pressure=p_0)

t = 0
tend = 40
with TaskManager():
    solver.setup()
    solver.solve_initial()
    solver.draw_solutions()

    formulation = solver.formulation

    Draw(formulation.pressure(solver.gfu.components[0]) - p_inf, mesh, "p'")
    Draw(formulation.velocity(solver.gfu.components[0]) - u_inf, mesh, "u'")

    while t < tend:
        t += solver.solver_configuration.time_step.Get()
        solver.solve_timestep(True)
