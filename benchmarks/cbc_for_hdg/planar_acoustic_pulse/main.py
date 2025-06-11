import argparse
import ngsolve as ngs
from netgen import occ
from dream.compressible import CompressibleFlowSolver, flowfields, FarField, Outflow, GRCBC, NSCBC, Initial

ngs.ngsglobals.msg_level = 0
ngs.SetNumThreads(16)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Planar acoustic pulse benchmark')
parser.add_argument('alpha', metavar='a', type=float, help='Pulse strength')
parser.add_argument('mach', metavar='M', type=float, help='Mach number')
parser.add_argument('--propagation', metavar='w', type=str, help='Wave direction', default="downstream")
parser.add_argument('--reference', type=bool, default=False)
OPTIONS = vars(parser.parse_args())

# Create mesh
MAXH = 0.07
X = 0.1
W = 8*X

face = occ.WorkPlane().RectangleC(W, 4*MAXH).Face()
face.name = "inner"
face.maxh = MAXH

if OPTIONS['reference']:

    buffer = occ.WorkPlane().RectangleC(14*W, 4*MAXH).Face()
    for edge, bc in zip(buffer.edges, ['bottom', 'right', 'top', 'left']):
        edge.name = bc

    for edge, bc in zip(face.edges, ['bottom', 'default', 'top', 'default']):
        edge.name = bc

    face = occ.Glue([buffer, face])

    for i, k in zip([1, 6, 9], [3, 4, 11]):
        face.edges[i].Identify(face.edges[k], f"periodic{i}_{k}", occ.IdentificationType.PERIODIC)
else:

    for bc, edge in zip(['bottom', 'right', 'top', 'left'], face.edges):
        edge.name = bc
    face.edges[0].Identify(face.edges[2], "periodic", occ.IdentificationType.PERIODIC)

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
DIR = {"downstream": 1, "upstream": -1}

Uinf = cfg.get_farfield_fields((1, 0))

U0 = flowfields()
U0.p = Uinf.p * (1 + OPTIONS['alpha'] * ngs.exp(- ngs.x**2/X**2))
U0.u = Uinf.u + DIR[OPTIONS['propagation']
                    ] * Uinf.p/(Uinf.rho * Uinf.c) * OPTIONS['alpha'] * ngs.exp(-ngs.x**2/X**2) * ngs.CF((1, 0))
U0.rho = Uinf.rho
cfg.dcs['inner'] = Initial(U0)
if OPTIONS['reference']:
    cfg.dcs['default'] = Initial(U0)


# Add information to the configuration
cfg.info["Domain Length"] = W
cfg.info["Domain Height"] = W
cfg.info['Width Pulse'] = X
cfg.info['Pulse Strength'] = OPTIONS['alpha']
cfg.info['Wave Direction'] = OPTIONS['propagation']

# Set I/O options
cfg.io.settings.enable = True
cfg.io.settings.to_pickle = True
cfg.io.settings.to_txt = True

cfg.io.ngsmesh.enable = True
cfg.io.log.to_terminal = True

cfg.io.gfu.enable = True
cfg.io.gfu.rate = 5


def planar_wave_routine(func):

    def wrapper(*args, **kwargs):

        # Set logging paths
        cfg.io.log.to_file = False
        cfg.io.path = func.__name__ + f"/M{cfg.mach_number.Get()}/alpha{OPTIONS['alpha']}/{OPTIONS['propagation']}"

        # Clear previous boundary conditions and set new ones
        cfg.bcs.clear()
        cfg.bcs['top|bottom'] = "periodic"
        func(*args, **kwargs)

        cfg.io.log.to_file = True
        # Initialize and solve the configuration
        cfg.initialize()
        with ngs.TaskManager():
            cfg.solve()

    return wrapper


@planar_wave_routine
def farfield_inflow_and_outflow():
    cfg.bcs['left|right'] = FarField(Uinf, use_identity_jacobian=True)


@planar_wave_routine
def farfield_inflow_and_pressure_outflow():
    cfg.bcs['left'] = FarField(Uinf, use_identity_jacobian=True)
    cfg.bcs['right'] = Outflow(Uinf.p)


@planar_wave_routine
def grcbc_farfield_inflow_and_outflow(CFL: float):
    cfg.bcs['left|right'] = GRCBC(Uinf, target="farfield", relaxation_factor=CFL)
    cfg.io.path = cfg.io.path.joinpath(f'CFL{CFL}')


@planar_wave_routine
def grcbc_farfield_inflow_and_pressure_outflow(CFL: float):
    cfg.bcs['left'] = GRCBC(Uinf, target="farfield", relaxation_factor=CFL)
    cfg.bcs['right'] = GRCBC(Uinf, target="outflow", relaxation_factor=CFL)
    cfg.io.path = cfg.io.path.joinpath(f'CFL{CFL}')


@planar_wave_routine
def grcbc_mass_inflow_and_pressure_outflow(CFL: float):
    cfg.bcs['left'] = GRCBC(Uinf, target="mass_inflow", relaxation_factor=CFL)
    cfg.bcs['right'] = GRCBC(Uinf, target="outflow", relaxation_factor=CFL)
    cfg.io.path = cfg.io.path.joinpath(f'CFL{CFL}')


@planar_wave_routine
def grcbc_temperature_inflow_and_pressure_outflow(CFL: float):
    cfg.bcs['left'] = GRCBC(Uinf, target="temperature_inflow", relaxation_factor=CFL)
    cfg.bcs['right'] = GRCBC(Uinf, target="outflow", relaxation_factor=CFL)
    cfg.io.path = cfg.io.path.joinpath(f'CFL{CFL}')


@planar_wave_routine
def nscbc_farfield_inflow_and_outflow(sigma: float):
    cfg.bcs['left|right'] = NSCBC(Uinf, target="farfield", relaxation_factor=sigma)
    cfg.io.path = cfg.io.path.joinpath(f'Sigma{sigma}')


@planar_wave_routine
def nscbc_farfield_inflow_and_pressure_outflow(sigma: float):
    cfg.bcs['left'] = NSCBC(Uinf, target="farfield", relaxation_factor=sigma)
    cfg.bcs['right'] = NSCBC(Uinf, target="outflow", relaxation_factor=sigma)
    cfg.io.path = cfg.io.path.joinpath(f'Sigma{sigma}')


@planar_wave_routine
def nscbc_mass_inflow_and_pressure_outflow(sigma: float):
    cfg.bcs['left'] = NSCBC(Uinf, target="mass_inflow", relaxation_factor=sigma)
    cfg.bcs['right'] = NSCBC(Uinf, target="outflow", relaxation_factor=sigma)
    cfg.io.path = cfg.io.path.joinpath(f'Sigma{sigma}')


@planar_wave_routine
def nscbc_temperature_inflow_and_pressure_outflow(sigma: float):
    cfg.bcs['left'] = NSCBC(Uinf, target="temperature_inflow", relaxation_factor=sigma)
    cfg.bcs['right'] = NSCBC(Uinf, target="outflow", relaxation_factor=sigma)
    cfg.io.path = cfg.io.path.joinpath(f'Sigma{sigma}')


@planar_wave_routine
def reference_farfield_inflow_and_outflow():
    cfg.bcs['left|right'] = FarField(Uinf, use_identity_jacobian=True)


if __name__ == '__main__':

    if OPTIONS['reference']:

        reference_farfield_inflow_and_outflow()

    else:

        farfield_inflow_and_outflow()
        farfield_inflow_and_pressure_outflow()

        for CFL in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 0]:
            grcbc_farfield_inflow_and_outflow(CFL)
            grcbc_farfield_inflow_and_pressure_outflow(CFL)
            grcbc_mass_inflow_and_pressure_outflow(CFL)
            grcbc_temperature_inflow_and_pressure_outflow(CFL)

        for sigma in [1, 0.28, 1e-1, 1e-2, 1e-3, 1e-4]:
            nscbc_farfield_inflow_and_outflow(sigma)
            nscbc_farfield_inflow_and_pressure_outflow(sigma)
            nscbc_mass_inflow_and_pressure_outflow(sigma)
            nscbc_temperature_inflow_and_pressure_outflow(sigma)
