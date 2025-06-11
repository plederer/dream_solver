import argparse
import ngsolve as ngs
from netgen import occ
from dream.compressible import CompressibleFlowSolver, flowfields, FarField, Outflow, GRCBC, NSCBC, Initial
from dream.io import DomainL2Sensor

ngs.ngsglobals.msg_level = 0
ngs.SetNumThreads(16)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Zero circulation vortex benchmark')
parser.add_argument('mach', metavar='M', type=float, help='Mach number')
parser.add_argument('alpha', metavar='a', type=float, help='Vortex strength')
OPTIONS = vars(parser.parse_args())

# Create mesh
MAXH = 0.15
L = 4
H = 2
R = 0.1

face = occ.WorkPlane().RectangleC(L, H).Face()
face.name = "air"

for bc, edge in zip(['bottom', 'right', 'top', 'left'], face.edges):
    edge.name = bc
face.edges[1].maxh = 2/3 * MAXH

mesh = ngs.Mesh(occ.OCCGeometry(face, dim=2).GenerateMesh(maxh=MAXH))


# Set configuration
cfg = CompressibleFlowSolver(mesh)
cfg.time = "transient"
cfg.time.timer.interval = (0, 5)
cfg.time.timer.step = 2e-3

cfg.fem = "conservative"
cfg.fem.order = 4
cfg.fem.method = "hdg"
cfg.fem.mixed_method = "inactive"
cfg.fem.scheme = "bdf2"

cfg.mach_number = OPTIONS['mach']
cfg.equation_of_state = "ideal"
cfg.equation_of_state.heat_capacity_ratio = 1.4
cfg.riemann_solver = 'upwind'
cfg.dynamic_viscosity = "inviscid"
cfg.scaling = "aerodynamic"

cfg.nonlinear_solver = "pardiso"
cfg.nonlinear_solver.method = "newton"
cfg.nonlinear_solver.method.damping_factor = 1
cfg.nonlinear_solver.max_iterations = 5
cfg.nonlinear_solver.convergence_criterion = 1e-8

cfg.optimizations.bonus_int_order.bnd = cfg.fem.order
cfg.optimizations.bonus_int_order.vol = cfg.fem.order
cfg.optimizations.static_condensation = True
cfg.optimizations.compile.realcompile = False


# Setup boundary and initial fields
Uinf = cfg.get_farfield_fields((1, 0))


def get_exact_solution_fields(x, y) -> flowfields:
    r = ngs.sqrt(x**2 + y**2)
    G = cfg.equation_of_state.heat_capacity_ratio
    PSI = OPTIONS['alpha']/cfg.mach_number * R * ngs.exp((1 - (r/R)**2)/2)

    U = flowfields()
    U.u = Uinf.u + ngs.CF((PSI.Diff(ngs.y), -PSI.Diff(ngs.x)))
    U.p = Uinf.p * (1 - (G - 1)/2 * OPTIONS['alpha']**2 * ngs.exp(1 - (r/R)**2))**(G/(G - 1))
    U.rho = Uinf.rho * (1 - (G - 1)/2 * OPTIONS['alpha']**2 * ngs.exp(1 - (r/R)**2))**(1/(G - 1))

    return U


t = cfg.time.timer.t
Ue = get_exact_solution_fields(ngs.x - Uinf.u[0] * t, ngs.y - Uinf.u[1] * t)

cfg.dcs['air'] = Initial(Ue)

cfg.info["Domain Length"] = L
cfg.info["Domain Height"] = H
cfg.info['Radius Vortex'] = R
cfg.info['Vortex Strength'] = OPTIONS['alpha']

# Set I/O options
cfg.io.settings.enable = True
cfg.io.settings.to_pickle = True
cfg.io.settings.to_txt = True

cfg.io.ngsmesh.enable = True
cfg.io.log.to_terminal = True

cfg.io.gfu.enable = True
cfg.io.gfu.rate = 5

cfg.io.sensor.enable = True


def zero_circulation_vortex_routine(func):

    def wrapper(*args, **kwargs):
        cfg.io.sensor.list.clear()

        # Set logging paths
        cfg.io.log.to_file = False
        cfg.io.path = func.__name__ + f"/M{cfg.mach_number.Get()}/alpha{OPTIONS['alpha']}"

        # Clear previous boundary conditions and set new ones
        cfg.bcs.clear()
        cfg.bcs['top|bottom|left'] = GRCBC(Uinf, relaxation_factor=0.01)
        func(*args, **kwargs)

        cfg.io.log.to_file = True
        # Initialize and solve the configuration
        cfg.initialize()

        Uh = cfg.get_solution_fields()

        l2_fields = flowfields()
        l2_fields.rho = Ue.rho - Uh.rho
        l2_fields.p = Ue.p - Uh.p
        l2_fields.u = Ue.u - Uh.u

        l2 = DomainL2Sensor(l2_fields, mesh, "air", "l2", rate=5)
        cfg.io.sensor.add(l2)

        with ngs.TaskManager():
            cfg.solve()

    return wrapper


@zero_circulation_vortex_routine
def farfield():
    cfg.bcs['right'] = FarField(Uinf)


@zero_circulation_vortex_routine
def outflow():
    cfg.bcs['right'] = Outflow(Uinf.p)


@zero_circulation_vortex_routine
def grcbc(target: str, CFL: float, beta: float):
    cfg.bcs['right'] = GRCBC(Uinf, target=target, relaxation_factor=CFL, tangential_relaxation=beta)
    cfg.io.path = cfg.io.path.joinpath(f'{target}/CFL{CFL}/beta{beta}')


@zero_circulation_vortex_routine
def nscbc(target: str, sigma: float, beta: float):
    cfg.bcs['right'] = NSCBC(Uinf, target=target, relaxation_factor=sigma, tangential_relaxation=beta)
    cfg.io.path = cfg.io.path.joinpath(f'{target}/Sigma{sigma}/beta{beta}')


if __name__ == '__main__':

    farfield()
    outflow()

    for target in ['farfield', 'outflow']:
        for CFL in [1e-1, 1e-2, 0]:
            for beta in [0, cfg.mach_number.Get()]:
                grcbc(target, CFL, beta)

    for target in ['farfield', 'outflow']:
        for sigma in [1, 0.28, 1e-2]:
            for beta in [0, cfg.mach_number.Get()]:
                nscbc(target, sigma, beta)
