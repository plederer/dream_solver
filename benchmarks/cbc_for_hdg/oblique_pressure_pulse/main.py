import argparse
import ngsolve as ngs
from netgen import occ
from dream.compressible import CompressibleFlowSolver, flowfields, FarField, Outflow, GRCBC, NSCBC, Initial

ngs.ngsglobals.msg_level = 0
ngs.SetNumThreads(16)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Oblique pressure pulse benchmark')
parser.add_argument('alpha', metavar='a', type=float, help='Pulse strength')
parser.add_argument('--reference', type=bool, default=False)
OPTIONS = vars(parser.parse_args())

# Create mesh
MAXH = 0.15
R = 0.2
H = 10*R

face = occ.WorkPlane().RectangleC(H, H).Face()
face.maxh = MAXH
face.name = "inner"

for bc, edge in zip(['bottom', 'right', 'top', 'left'], face.edges):
    edge.name = bc

if OPTIONS['reference']:
    outer = occ.WorkPlane().Circle(0, 0, 4*H).Face()
    outer.edges.maxh = 0.5
    outer.edges[0].name = "outer"
    outer.name = "outer"
    outer.maxh = 0.5
    face = occ.Glue([face, outer])

mesh = ngs.Mesh(occ.OCCGeometry(face, dim=2).GenerateMesh())

# Set configuration
cfg = CompressibleFlowSolver(mesh)
cfg.time = "transient"
cfg.time.timer.interval = (0, 4)
cfg.time.timer.step = 2e-3

cfg.fem = "conservative"
cfg.fem.order = 4
cfg.fem.method = "hdg"
cfg.fem.mixed_method = "inactive"
cfg.fem.scheme = "bdf2"

cfg.mach_number = 0.0
cfg.equation_of_state = "ideal"
cfg.equation_of_state.heat_capacity_ratio = 1.4
cfg.riemann_solver = 'lax_friedrich'
cfg.dynamic_viscosity = "inviscid"
cfg.scaling = "acoustic"

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

U0 = flowfields()
U0.p = Uinf.p * (1 + OPTIONS['alpha'] * ngs.exp(-(ngs.x**2 + ngs.y**2)/R**2))
U0.u = Uinf.u
U0.rho = Uinf.rho

cfg.dcs['inner'] = Initial(U0)
if OPTIONS['reference']:
    cfg.dcs['outer'] = Initial(U0)

cfg.info["Domain Length"] = H
cfg.info["Domain Height"] = H
cfg.info['Radius Pulse'] = R
cfg.info['Pulse Strength'] = OPTIONS['alpha']


# Set I/O options
cfg.io.settings.enable = True
cfg.io.settings.to_pickle = True
cfg.io.settings.to_txt = True

cfg.io.ngsmesh.enable = True
cfg.io.log.to_terminal = True

cfg.io.gfu.enable = True
cfg.io.gfu.rate = 5


def oblique_pressure_pulse_routine(func):

    def wrapper(*args, **kwargs):

        # Set logging paths
        cfg.io.log.to_file = False
        cfg.io.path = func.__name__ + f"/alpha{OPTIONS['alpha']}"

        # Clear previous boundary conditions and set new ones
        cfg.bcs.clear()
        func(*args, **kwargs)

        cfg.io.log.to_file = True
        # Initialize and solve the configuration
        cfg.initialize()
        with ngs.TaskManager():
            cfg.solve()

    return wrapper


@oblique_pressure_pulse_routine
def farfield():
    cfg.bcs['left|right|bottom|top'] = FarField(fields=Uinf, use_identity_jacobian=True)


@oblique_pressure_pulse_routine
def outflow():
    cfg.bcs['left|right|bottom|top'] = Outflow(pressure=Uinf.p)


@oblique_pressure_pulse_routine
def grcbc(target: str, CFL: float):
    cfg.bcs['left|right|top|bottom'] = GRCBC(Uinf, target=target, relaxation_factor=CFL)
    cfg.io.path = cfg.io.path.joinpath(f'{target}/CFL{CFL}')


@oblique_pressure_pulse_routine
def nscbc(target: str, sigma: float):
    cfg.bcs['left|right|top|bottom'] = NSCBC(Uinf, target=target, relaxation_factor=sigma)
    cfg.io.path = cfg.io.path.joinpath(f'{target}/Sigma{sigma}')


@oblique_pressure_pulse_routine
def reference_farfield():
    cfg.bcs['outer'] = FarField(Uinf, use_identity_jacobian=True)


if __name__ == '__main__':

    if OPTIONS['reference']:

        reference_farfield()

    else:

        farfield()
        outflow()

        for target in ['farfield', 'outflow', 'mass_inflow', 'temperature_inflow']:
            for CFL in [1e-1, 1e-2, 1e-3, 0]:
                grcbc(target, CFL)

        for target in ['farfield', 'outflow', 'mass_inflow', 'temperature_inflow']:
            for sigma in [1, 0.28, 1e-1, 1e-2]:
                nscbc(target, sigma)
