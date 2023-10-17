from netgen.occ import OCCGeometry, WorkPlane
from ngsolve import *
from dream import *
from netgen.meshing import IdentificationType
from ngsolve.meshes import MakeStructured2DMesh

ngsglobals.msg_level = 0
SetNumThreads(8)

periodic = True
circle = False
structured = False
maxh = 0.15

cfg = SolverConfiguration()
cfg.formulation = "conservative"
# cfg.dynamic_viscosity = "constant"
# cfg.dynamic_viscosity = None
# cfg.mixed_method = "strain_heat"
# cfg.mixed_method = None
cfg.scaling = "aeroacoustic"
cfg.riemann_solver = 'hllem'

cfg.Reynolds_number = 166
cfg.Mach_number = 0.05
cfg.Prandtl_number = 0.72
cfg.heat_capacity_ratio = 1.4

cfg.order = 6
cfg.bonus_int_order = {VOL: cfg.order, BND: cfg.order}

cfg.time.simulation = "transient"
cfg.time.scheme = "BDF2"
cfg.time.step = 0.2
cfg.time.interval = (0, 200)

cfg.linear_solver = "pardiso"
cfg.damping_factor = 1
cfg.max_iterations = 100
cfg.convergence_criterion = 1e-16

cfg.compile_flag = True
cfg.static_condensation = True

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
        face = WorkPlane().RectangleC(1, 2).Face()

        for bc, edge in zip(['bottom', 'right', 'top', 'left'], face.edges):
            edge.name = bc
        if periodic:
            periodic_edge = face.edges[0]
            periodic_edge.Identify(face.edges[2], "periodic", IdentificationType.PERIODIC)
        mesh = Mesh(OCCGeometry(face, dim=2).GenerateMesh(maxh=maxh))
# mesh.Refine()
gamma = cfg.heat_capacity_ratio
M = cfg.Mach_number

farfield = INF.farfield((1,0), cfg)
u_inf = CF(farfield.velocity)
rho_inf = farfield.density
p_inf = farfield.pressure
T_inf = farfield.temperature
c = INF.speed_of_sound(cfg)

M = cfg.Mach_number
Gamma = 0.01 * M/(M + 1)
Rv = 0.1
r = sqrt(x**2 + y**2)
psi = Gamma * exp(-r**2/(2*Rv**2))

# Vortex Isothermal
u_0 = u_inf + CF((psi.Diff(y), -psi.Diff(x)))
p_0 = p_inf * exp(-gamma/2*(Gamma/(c * Rv))**2 * exp(-r**2/Rv**2))
rho_0 = gamma/(gamma - 1)/T_inf * p_0
p_00 = p_inf * exp(-gamma/2*(Gamma/(c * Rv))**2)
initial = State(u_0, rho_0, p_0)

# # Vortex Isentropic
# constant = (gamma-1)/(2*gamma) * rho_inf/p_inf
# u_0 = u_inf + CF((psi.Diff(y), -psi.Diff(x)))
# p_0 = p_inf * (1 - constant * (Gamma/Rv)**2 * exp(-r**2/Rv**2))**(gamma/(gamma - 1))
# rho_0 = rho_inf * (1 - constant * (Gamma/Rv)**2 * exp(-r**2/Rv**2))**(1/(gamma - 1))
# p_00 = p_inf * (1 - constant * (Gamma/Rv)**2)**(gamma/(gamma - 1))
# initial = State(u_0, rho_0, p_0)

solver = CompressibleHDGSolver(mesh, cfg)
solver.boundary_conditions.set(bcs.FarField(farfield), "left")
solver.boundary_conditions.set(bcs.Outflow_NSCBC(p_inf, 0.28), "right")
solver.boundary_conditions.set(bcs.InviscidWall(), "top|bottom")
if periodic:
    solver.boundary_conditions.set(bcs.Periodic(), "top|bottom")
solver.domain_conditions.set(dcs.Initial(initial))


with TaskManager():
    solver.setup()

    solver.drawer.draw(energy=True)
    solver.drawer.draw_particle_velocity(u_inf, autoscale=True, min=-1e-4, max=1e-4)
    solver.drawer.draw_acoustic_pressure(p_inf, autoscale=True, min=-1e-2, max=1e-2)
    Draw((solver.formulation.pressure() - p_inf)/(p_00 - p_inf), mesh, "p*")
    solver.solve_transient()
