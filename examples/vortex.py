from netgen.occ import OCCGeometry, WorkPlane
from ngsolve import *
from netgen.meshing import IdentificationType
from dream import CompressibleHDGSolver, SolverConfiguration
from ngsolve.meshes import MakeStructured2DMesh

ngsglobals.msg_level = 0
SetNumThreads(8)

periodic = False
circle = False
structured = False
maxh = 0.1

cfg = SolverConfiguration()
cfg.formulation = "conservative"
# cfg.dynamic_viscosity = "constant"
# cfg.dynamic_viscosity = None
# cfg.mixed_method = "strain_heat"
# cfg.mixed_method = None
cfg.riemann_solver = 'roe'

cfg.Reynolds_number = 166
cfg.Mach_number = 0.42
cfg.Prandtl_number = 0.72
cfg.heat_capacity_ratio = 1.4

cfg.order = 4
cfg.bonus_int_order_bnd = 0
cfg.bonus_int_order_vol = 0

cfg.time.simulation = "transient"
cfg.time.scheme = "BDF2"
cfg.time.step = 0.005
cfg.time.interval = (0, 100)

cfg.linear_solver = "pardiso"
cfg.damping_factor = 1
cfg.max_iterations = 100
cfg.convergence_criterion = 1e-16

cfg.compile_flag = True
cfg.static_condensation = True
cfg.periodic = periodic

if circle:

    face = WorkPlane().MoveTo(0, -0.5).Arc(0.5, 180).Arc(0.5, 180).Face()
    for bc, edge in zip(['right', 'left'], face.edges):
        edge.name = bc
    mesh = Mesh(OCCGeometry(face, dim=2).GenerateMesh(maxh=maxh))
    mesh.Curve(cfg.order)

else:

    if structured:
        N = int(1 / maxh)
        mesh = MakeStructured2DMesh(False, N, N, periodic_y=periodic, mapping=lambda x, y: (x - 0.5, y - 0.5))
    else:
        face = WorkPlane().RectangleC(1, 1).Face()

        for bc, edge in zip(['bottom', 'right', 'top', 'left'], face.edges):
            edge.name = bc
        if periodic:
            periodic_edge = face.edges[0]
            periodic_edge.Identify(face.edges[2], "periodic", IdentificationType.PERIODIC)
        mesh = Mesh(OCCGeometry(face, dim=2).GenerateMesh(maxh=maxh))

gamma = cfg.heat_capacity_ratio
M = cfg.Mach_number

rho_inf = 1
# alpha = pi/9
alpha = 0
u_inf = CF((cos(alpha), sin(alpha)))
p_inf = 1/(M**2 * gamma)
T_inf = gamma/(gamma - 1) * p_inf/rho_inf
c = sqrt(gamma * p_inf/rho_inf)

Gamma = 0.02
Rv = 0.1

r = sqrt(x**2 + y**2)
psi = Gamma * exp(-r**2/(2*Rv**2))

# Vortex
u_0 = u_inf + CF((psi.Diff(y), -psi.Diff(x)))
p_0 = p_inf * exp(-gamma/2*(Gamma/(c * Rv))**2 * exp(-r**2/Rv**2))
rho_0 = gamma/(gamma - 1)/T_inf * p_0

# # Pressure Pulse
# u_0 = u_inf
# u_0 = CF((0,0))
# p_0 = p_inf * (1 + Gamma * exp(-r**2/Rv**2))
# rho_0 = gamma/(gamma - 1)/T_inf * p_0

p_00 = p_inf * exp(-gamma/2*(Gamma/(c * Rv))**2)


solver = CompressibleHDGSolver(mesh, cfg)
solver.boundary_conditions.set_farfield("left|right", u_inf, rho_inf, p_inf)
# solver.boundary_conditions.set_nonreflecting_outflow('right', p_inf, "perfect", 0.28, 1, True, False, False)

solver.boundary_conditions.set_inviscid_wall('top|bottom')
if periodic:
    solver.boundary_conditions.set_periodic('top|bottom')

solver.domain_conditions.set_initial(u_0, rho_0, p_0)

with TaskManager():
    solver.setup()

    solver.drawer.draw(energy=True)
    solver.drawer.draw_particle_velocity(u_inf, autoscale=False, min=-1e-4, max=1e-4)
    solver.drawer.draw_acoustic_pressure(p_inf, autoscale=False, min=-1e-4, max=1e-4)
    Draw((solver.formulation.pressure() - p_inf)/(p_00 - p_inf), mesh, "p*")
    solver.solve_transient()
