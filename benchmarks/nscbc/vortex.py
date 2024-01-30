from netgen.occ import OCCGeometry, WorkPlane
from ngsolve import *
from dream import *
from netgen.meshing import IdentificationType

ngsglobals.msg_level = 0
SetNumThreads(8)

tree = ResultsDirectoryTree('vortex')
saver = Saver(tree)

maxh = 0.1
periodic = True

cfg = SolverConfiguration()
cfg.formulation = "conservative"
cfg.scaling = "aerodynamic"
cfg.riemann_solver = 'hllem'

cfg.Mach_number = 0.1
cfg.heat_capacity_ratio = 1.4

cfg.order = 6
cfg.bonus_int_order_bnd = cfg.order
cfg.bonus_int_order_vol = cfg.order

cfg.time.simulation = "transient"
cfg.time.scheme = "BDF2"
cfg.time.step = 1e-2
cfg.time.interval = (0, 20)
cfg.save_state = True

cfg.linear_solver = "pardiso"
cfg.damping_factor = 1
cfg.max_iterations = 100
cfg.convergence_criterion = 1e-16

cfg.compile_flag = True
cfg.static_condensation = True


face = WorkPlane().RectangleC(4, 2).Face()

for bc, edge in zip(['bottom', 'right', 'top', 'left'], face.edges):
    edge.name = bc
if periodic:
    periodic_edge = face.edges[0]
    periodic_edge.Identify(face.edges[2], "periodic", IdentificationType.PERIODIC)

mesh = Mesh(OCCGeometry(face, dim=2).GenerateMesh(maxh=maxh))
gamma = cfg.heat_capacity_ratio
M = cfg.Mach_number

farfield = INF.farfield((1, 0), cfg)
u_inf = CF(farfield.velocity)
rho_inf = farfield.density
p_inf = farfield.pressure
T_inf = farfield.temperature
c = INF.speed_of_sound(cfg)


# Vortex Pirozzoli
Mt = 0.01
R = 0.1
r = sqrt(x**2 + y**2)

if cfg.scaling is cfg.scaling.AERODYNAMIC:
    vt = Mt/cfg.Mach_number
elif cfg.scaling is cfg.scaling.ACOUSTIC:
    vt = Mt
elif cfg.scaling is cfg.scaling.AEROACOUSTIC:
    vt = Mt/(1 + cfg.Mach_number)

psi = vt * R * exp((R**2 - r**2)/(2*R**2))
u_0 = u_inf + CF((psi.Diff(y), -psi.Diff(x)))
p_0 = p_inf * (1 - (gamma - 1)/2 * Mt**2 * exp((R**2 - r**2)/(R**2)))**(gamma/(gamma - 1))
rho_0 = rho_inf * (1 - (gamma - 1)/2 * Mt**2 * exp((R**2 - r**2)/(R**2)))**(1/(gamma - 1))
p_00 = p_inf * (1 - (gamma - 1)/2 * Mt**2 * exp(1))**(gamma/(gamma - 1))
initial = State(u_0, rho_0, p_0)


def solve_farfield():

    tree.state_directory_name = "farfield"
    solver = CompressibleHDGSolver(mesh, cfg, tree)

    solver.boundary_conditions.set(bcs.FarField(farfield), "left|right|top|bottom")
    if periodic:
        solver.boundary_conditions.set(bcs.Periodic(), "top|bottom")

    solver.domain_conditions.set(dcs.Initial(initial))

    with TaskManager():
        solver.setup()
        solver.solve_transient()


def solve_yoo_tangential():

    tree.state_directory_name = "yoo2D"
    solver = CompressibleHDGSolver(mesh, cfg, tree)

    solver.boundary_conditions.set(bcs.NSCBC(farfield, "yoo", outflow_tangential_flux=True), "right|left|bottom|top")
    if periodic:
        solver.boundary_conditions.set(bcs.Periodic(), "top|bottom")

    solver.domain_conditions.set(dcs.Initial(initial))

    with TaskManager():
        solver.setup()
        solver.solve_transient()


def solve_yoo_lodi():

    tree.state_directory_name = "yoo1D"
    solver = CompressibleHDGSolver(mesh, cfg, tree)

    solver.boundary_conditions.set(bcs.NSCBC(farfield, "yoo", outflow_tangential_flux=False), "right|left|bottom|top")
    if periodic:
        solver.boundary_conditions.set(bcs.Periodic(), "top|bottom")

    solver.domain_conditions.set(dcs.Initial(initial))

    with TaskManager():
        solver.setup()
        solver.solve_transient()


def solve_poinsot2D():

    tree.state_directory_name = "poinsot"
    solver = CompressibleHDGSolver(mesh, cfg, tree)

    sigmas = State(velocity=4, pressure=0.275, temperature=4)
    solver.boundary_conditions.set(
        bcs.NSCBC(farfield, "poinsot", sigmas=sigmas, outflow_tangential_flux=True),
        "right|left|bottom|top")
    if periodic:
        solver.boundary_conditions.set(bcs.Periodic(), "top|bottom")

    solver.domain_conditions.set(dcs.Initial(initial))

    with TaskManager():
        solver.setup()
        solver.solve_transient()

def solve_mixed():

    tree.state_directory_name = "mixed"
    solver = CompressibleHDGSolver(mesh, cfg, tree)

    solver.boundary_conditions.set(
          solver.boundary_conditions.set(bcs.NSCBC(farfield, "yoo", outflow_tangential_flux=True)),
        "right|bottom|top")
    solver.boundary_conditions.set(bcs.FarField(farfield), 'left')
    if periodic:
        solver.boundary_conditions.set(bcs.Periodic(), "top|bottom")

    solver.domain_conditions.set(dcs.Initial(initial))

    with TaskManager():
        solver.setup()
        solver.solve_transient()


if __name__ == '__main__':
    saver.save_mesh(mesh)
    saver.save_configuration(cfg)
    solve_farfield()
    solve_yoo_tangential()
    solve_yoo_lodi()
    solve_poinsot2D()
    solve_mixed()
