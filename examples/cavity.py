from netgen.occ import OCCGeometry, WorkPlane, Glue
from ngsolve import *
from netgen.meshing import IdentificationType
from dream import CompressibleHDGSolver, SolverConfiguration
from dream.utils.geometries import MakeOCCRectangle, MakeOCCCircle, MakeOCCCirclePlane
ngsglobals.msg_level = 0
SetNumThreads(8)

air = WorkPlane().MoveTo(0, 0.2).Rectangle(12, 8.3).Face()
for edge, bc in zip(air.edges, ['default', 'right', 'top', 'left']):
    edge.name = bc
boundary_layer = WorkPlane().Rectangle(12, 0.2).Face()
boundary_layer.maxh = 0.1
for edge, bc in zip(boundary_layer.edges, ['wall', 'right', 'default', 'left']):
    edge.name = bc
cavity = WorkPlane().MoveTo(5, -1).Rectangle(2 , 1).Face()
cavity.maxh = 0.25
for edge, bc in zip(cavity.edges, ['wall', 'wall', 'default', 'wall']):
    edge.name = bc
geo = Glue([air, boundary_layer, cavity])
mesh = Mesh(OCCGeometry(geo, dim=2).GenerateMesh(maxh=1))

# face = WorkPlane().MoveTo(0, -0.5).Arc(0.5, 180).Arc(0.5, 180).Face()
# for bc, edge in zip(['right', 'left'], face.edges):
#     edge.name = bc
# mesh = Mesh(OCCGeometry(face, dim=2).GenerateMesh(maxh=0.05))

cfg = SolverConfiguration()
cfg.formulation = "conservative"
cfg.dynamic_viscosity = "constant"
# cfg.dynamic_viscosity = None
cfg.mixed_method = "strain_heat"
# cfg.mixed_method = None
cfg.riemann_solver = 'hllem'

cfg.Reynolds_number = 166
cfg.Mach_number = 0.42
cfg.Prandtl_number = 0.72
cfg.heat_capacity_ratio = 1.4

cfg.order = 3
cfg.bonus_int_order_bnd = 3
cfg.bonus_int_order_vol = 3

cfg.time.simulation = "transient"
cfg.time.scheme = "BDF2"
cfg.time.step = 0.02
cfg.time.interval = (0, 100)

cfg.linear_solver = "pardiso"
cfg.damping_factor = 1
cfg.max_iterations = 100
cfg.convergence_criterion = 1e-12

cfg.compile_flag = True
cfg.static_condensation = True

mesh.Curve(cfg.order)

gamma = cfg.heat_capacity_ratio
M = cfg.Mach_number

rho_inf = 1
alpha = pi/36
u_inf = CF((cos(alpha), sin(alpha)))
p_inf = 1/(M**2 * gamma)
T_inf = gamma/(gamma - 1) * p_inf/rho_inf
c = sqrt(gamma * p_inf/rho_inf)

Gamma = 0.005
Rv = 0.1

r = sqrt(x**2 + y**2)
psi = Gamma * exp(-r**2/(2*Rv**2))

u_0 = u_inf #+ CF((psi.Diff(y), -psi.Diff(x)))
p_0 = p_inf #* exp(-gamma/2*(Gamma/(c * Rv))**2 * exp(-r**2/Rv**2))
rho_0 = rho_inf

# u_0 = u_inf
# p_0 = p_inf * (1 + Gamma * exp(-r**2/(2*Rv**2)))
# rho_0 = gamma/(gamma - 1)/T_inf * p_0



solver = CompressibleHDGSolver(mesh, cfg)
# solver.boundary_conditions.set_nonreflecting_inflow("left", rho_inf, u_inf, pressure=p_inf, type="partially", reference_length=40, tangential_convective_fluxes=False)
solver.boundary_conditions.set_farfield("left|top|right", rho_inf, u_inf, pressure=p_inf)
solver.boundary_conditions.set_adiabatic_wall('wall')
solver.domain_conditions.set_initial(rho_0, u_0, pressure=p_0)


with TaskManager():
    solver.setup()

    formulation = solver.formulation
    TnT = formulation.TnT

    solver.drawer.draw(energy=True)
    solver.drawer.draw_acoustic_pressure(p_inf, autoscale=False, min=-1e-4, max=1e-4)
    solver.drawer.draw_particle_velocity(u_inf, autoscale=False, min=-1e-4, max=1e-4)

    solver.solve_transient()
