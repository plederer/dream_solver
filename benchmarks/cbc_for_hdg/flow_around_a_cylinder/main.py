import argparse
import ngsolve as ngs
from netgen import occ
from dream.compressible import CompressibleFlowSolver, FarField, Outflow, GRCBC, NSCBC, AdiabaticWall, Initial
from dream.io import BoundarySensor

ngs.ngsglobals.msg_level = 0
ngs.SetNumThreads(16)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Flow around cylinder benchmark')
parser.add_argument('simulation', metavar='sim', type=str, help='Simulation')
parser.add_argument('--mach', metavar='M', type=float, help='Mach number', default=0.2)
OPTIONS = vars(parser.parse_args())

# Create mesh
OFFSET = 5
R = 0.5
WAKE_R = 4 * R

wp = occ.WorkPlane()

# Cylinder
cyl = wp.Circle(0, 0, R).Face()
cyl.edges.name = 'cylinder'
cyl.edges.maxh = 0.035

# Wake
wp.MoveTo(0, WAKE_R).Direction(-1, 0).Arc(WAKE_R, 180)
wp.LineTo((20+OFFSET)*R, -WAKE_R)
wp.LineTo((20+OFFSET)*R, WAKE_R)
wp.LineTo(0, WAKE_R)
wake = wp.Face()

wake.faces.maxh = 0.5
wake -= cyl

# domain
domain = wp.MoveTo(OFFSET*R, 0).RectangleC(40*R, 40*R).Face()
domain.faces.maxh = 1.5

if 'reference' not in OPTIONS['simulation']:
    for edge, name_ in zip(domain.edges, ['farfield', 'outflow', 'farfield', 'farfield']):
        edge.name = name_

domain -= cyl

dom = occ.Glue([wake, domain])
dom.faces.name = 'inner'

if 'reference' in OPTIONS['simulation']:
    wp.MoveTo(0, WAKE_R).Direction(-1, 0).Arc(WAKE_R, 180)
    wp.LineTo(200*R, -WAKE_R)
    wp.LineTo(200*R, WAKE_R)
    wp.LineTo(0, WAKE_R)
    wake = wp.Face()
    wake = wake - (wake - wp.MoveTo(0, 0).Circle(0, 0, R*200).Face())
    wake.faces.maxh = 1.5
    wake -= dom
    wake -= cyl

    sound = wp.MoveTo(0, 0).Circle(0, 0, R*200).Circle(0, 0, R).Reverse().Face()
    sound.faces.maxh = 15
    sound.faces.name = "sound"
    sound.edges[0].name = "farfield"

    dom = occ.Glue([dom, wake, sound])

mesh = ngs.Mesh(occ.OCCGeometry(dom, dim=2).GenerateMesh(maxh=15))

# Set configuration
cfg = CompressibleFlowSolver(mesh)
cfg.time = 'transient'
cfg.time.timer.step = 0.05

cfg.mach_number = OPTIONS['mach']
cfg.equation_of_state = 'ideal'
cfg.equation_of_state.heat_capacity_ratio = 1.4
cfg.reynolds_number = 150
cfg.prandtl_number = 0.72
cfg.riemann_solver = 'upwind'
cfg.dynamic_viscosity = 'constant'
cfg.scaling = 'aerodynamic'

cfg.fem = 'conservative'
cfg.fem.order = 4
cfg.fem.method = 'hdg'
cfg.fem.mixed_method = 'strain_heat'
cfg.fem.scheme = 'bdf2'

cfg.nonlinear_solver = 'pardiso'
cfg.nonlinear_solver.method = 'newton'
cfg.nonlinear_solver.method.damping_factor = 1
cfg.nonlinear_solver.max_iterations = 5
cfg.nonlinear_solver.convergence_criterion = 1e-8

cfg.optimizations.bonus_int_order.bnd = cfg.fem.order
cfg.optimizations.bonus_int_order.vol = cfg.fem.order
cfg.optimizations.static_condensation = True
cfg.optimizations.compile.realcompile = False

mesh.Curve(cfg.fem.order)

# Setup boundary and initial fields
Uinf = cfg.get_farfield_fields((1, 0))
cfg.dcs['inner'] = Initial(fields=Uinf)
if 'reference' in OPTIONS['simulation']:
    cfg.dcs['sound'] = Initial(fields=Uinf)


# Set I/O options
cfg.io.settings.enable = True
cfg.io.settings.to_pickle = True
cfg.io.settings.to_txt = True
cfg.io.ngsmesh.enable = True
cfg.io.log.to_terminal = True
cfg.io.gfu.enable = True


def flow_around_cylinder_routine(func):

    def wrapper(*args, **kwargs):

        cfg.io.sensor.list.clear()

        # Set logging paths
        cfg.io.log.to_file = False
        cfg.io.path = func.__name__ + f'/M{cfg.mach_number.Get()}'

        # Clear previous boundary conditions and set new ones
        cfg.bcs.clear()
        cfg.bcs['cylinder'] = AdiabaticWall()
        cfg.bcs['farfield'] = GRCBC(Uinf, relaxation_factor=0.01)
        func(*args, **kwargs)

        cfg.io.log.to_file = True

        # Initialize and solve the configuration
        cfg.initialize()

        Uh = cfg.get_solution_fields('strain_rate_tensor')
        cfg.time.timer.interval = (0, 100)

        cfg.io.gfu.rate = 0
        cfg.io.gfu.time_level_rate = 0
        cfg.io.settings.filename = 'init'
        cfg.io.gfu.filename = 'init'

        with ngs.TaskManager():
            cfg.solve()

        cl = BoundarySensor(
            {'c_l': cfg.lift_coefficient(Uh, Uh, Uinf, (0, 1), 1),
             'c_d': cfg.drag_coefficient(Uh, Uh, Uinf, (1, 0), 1)}, mesh, 'cylinder')

        cfg.io.sensor.add(cl)

        cfg.io.gfu.rate = 1
        cfg.io.gfu.time_level_rate = 200
        cfg.io.settings.filename = 'gfu'
        cfg.io.gfu.filename = 'gfu'

        cfg.time.timer.interval = (100, 750)
        with ngs.TaskManager():
            cfg.solve()

    return wrapper


@flow_around_cylinder_routine
def farfield():
    cfg.bcs['outflow'] = FarField(Uinf)


@flow_around_cylinder_routine
def outflow():
    cfg.bcs['outflow'] = Outflow(Uinf.p)


@flow_around_cylinder_routine
def grcbc(target: str, CFL: float):
    cfg.bcs['outflow'] = GRCBC(Uinf, target=target, relaxation_factor=CFL,
                               tangential_relaxation=cfg.mach_number,
                               is_viscous_fluxes=True)
    cfg.io.path = cfg.io.path.joinpath(f'{target}/CFL{CFL}')


@flow_around_cylinder_routine
def nscbc(target: str, sigma: float):
    cfg.bcs['outflow'] = NSCBC(Uinf, target=target, relaxation_factor=sigma,
                               tangential_relaxation=cfg.mach_number,
                               is_viscous_fluxes=True)
    cfg.io.path = cfg.io.path.joinpath(f'{target}/Sigma{sigma}')


@flow_around_cylinder_routine
def reference():
    ...


if __name__ == '__main__':

    match OPTIONS['simulation']:

        case 'farfield':
            farfield()

        case 'outflow':
            outflow()

        case 'grcbc':
            for target in ['farfield', 'outflow']:
                grcbc(target=target, CFL=1e-2)

        case 'nscbc':
            nscbc(target='outflow', sigma=0.28)

        case 'reference':
            reference()

        case _:
            raise ValueError('Simulation not available')
