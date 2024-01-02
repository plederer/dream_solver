from netgen.occ import OCCGeometry, WorkPlane
from ngsolve import *
from dream import *
from netgen.meshing import IdentificationType
from ngsolve.meshes import MakeStructured2DMesh

ngsglobals.msg_level = 0
SetNumThreads(8)

periodic = False
circle = True
structured = False
maxh = 0.15

cfg = SolverConfiguration()
cfg.formulation = "conservative"
# cfg.dynamic_viscosity = "constant"
# cfg.mixed_method = "strain_heat"
cfg.scaling = "aerodynamic"
cfg.riemann_solver = 'hllem'

cfg.Reynolds_number = 150
cfg.Mach_number = 0.3
cfg.Prandtl_number = 0.72
cfg.equation_of_state.heat_capacity_ratio = 1.4

cfg.order = 6
cfg.bonus_int_order = {VOL: 6, BND: 6}

cfg.time.simulation = "transient"
cfg.time.scheme = "BDF2"
cfg.time.step = 0.1
cfg.time.interval = (0, 200)

cfg.linear_solver = "pardiso"
cfg.damping_factor = 1
cfg.max_iterations = 100
cfg.convergence_criterion = 1e-16

cfg.compile_flag = True
cfg.static_condensation = True

if circle:

    face = WorkPlane().MoveTo(0, -1).Arc(1, 180).Arc(1, 180).Face()
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


gamma = cfg.equation_of_state.heat_capacity_ratio
M = cfg.Mach_number

farfield = INF.state(M, (1, 0))
u_inf = CF(farfield.velocity)
rho_inf = farfield.density
p_inf = farfield.pressure
T_inf = farfield.temperature


# Vortex Pirozzoli
Mt = 0.01
R = 0.1
r = sqrt(x**2 + y**2)
vt = Mt/cfg.Mach_number * INF.velocity_magnitude(M) #cfg.scaling.farfield_velocity(cfg.Mach_number)

psi = vt * R * exp((R**2 - r**2)/(2*R**2))
u_0 = u_inf + CF((psi.Diff(y), -psi.Diff(x)))
p_0 = p_inf * (1 - (gamma - 1)/2 * Mt**2 * exp((R**2 - r**2)/(R**2)))**(gamma/(gamma - 1))
rho_0 = rho_inf * (1 - (gamma - 1)/2 * Mt**2 * exp((R**2 - r**2)/(R**2)))**(1/(gamma - 1))
p_00 = p_inf * (1 - (gamma - 1)/2 * Mt**2 * exp(1))**(gamma/(gamma - 1))
initial = State(rho_0, u_0, p_0)

solver = CompressibleHDGSolver(mesh, cfg)
solver.boundary_conditions.set(bcs.FarField(farfield), "left|right")
# solver.boundary_conditions.set(bcs.NSCBC(farfield, 0.28, tangential_convective_fluxes=True), "right")
# solver.boundary_conditions.set(bcs.NSCBC(farfield, 0.28, tangential_convective_fluxes=True), "left")
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
