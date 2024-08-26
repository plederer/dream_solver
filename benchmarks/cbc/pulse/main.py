from ngsolve import *
from dream import *
from netgen.occ import OCCGeometry, WorkPlane, IdentificationType
import argparse

parser = argparse.ArgumentParser(description='Vortex benchmark')
parser.add_argument('Alpha', metavar='a', type=float, help='Pulse strength')
args = vars(parser.parse_args())

ngsglobals.msg_level = 0
SetNumThreads(64)

draw = False

tree = ResultsDirectoryTree()
# tree.parent_path = ""

LOGGER.tree = tree
LOGGER.log_to_terminal = False

saver = Saver(tree)

name = "circle"
maxh = 0.07
R = 0.1
H = 8*R
alpha = args['Alpha']

cfg = SolverConfiguration()
cfg.formulation = "conservative"
cfg.scaling = "acoustic"
cfg.riemann_solver = 'lax_friedrich'

cfg.Mach_number = 0
cfg.heat_capacity_ratio = 1.4

cfg.order = 4
cfg.bonus_int_order_bnd = cfg.order
cfg.bonus_int_order_vol = cfg.order

cfg.time.simulation = "transient"
cfg.time.scheme = "BDF2"
cfg.time.step = 2e-3
save_step = 5
cfg.time.interval = (0, 15)
cfg.save_state = True

cfg.linear_solver = "pardiso"
cfg.damping_factor = 1
cfg.max_iterations = 5
cfg.convergence_criterion = 1e-8

cfg.compile_flag = True
cfg.static_condensation = True

face = WorkPlane().RectangleC(H, H).Face()
if name == "circle":
    face = WorkPlane().Circle(0, 0, H).Face()

for bc, edge in zip(['bottom', 'right', 'top', 'left'], face.edges):
    edge.name = bc

face.maxh = maxh
face.name = "inner"

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

r = sqrt(x**2 + y**2)

u_0 = u_inf
p_0 = p_inf * (1 - (gamma - 1)/2 * alpha**2 * exp(1 - (r/R)**2))**(gamma/(gamma - 1))
rho_0 = rho_inf * (1 - (gamma - 1)/2 * alpha**2 * exp(1 - (r/R)**2))**(1/(gamma - 1))
initial = State(u_0, rho_0, p_0)

cfg.info["Domain Length"] = H
cfg.info["Domain Height"] = H
cfg.info['Radius Pulse'] = R
cfg.info['Pulse Strength'] = alpha


def test(name: str = ""):

    def wraps(func):

        def wrapper(*args, **kwargs):

            info = cfg.info.copy()

            tree.directory_name = f"Ma{cfg.Mach_number.Get()}/alpha{alpha}/dt{cfg.time.step.Get()}"

            tree.state_directory_name = func.__name__
            if name:
                tree.state_directory_name += f"_{name}"

            solver = func(*args, **kwargs)
            solver.domain_conditions.set(dcs.Initial(initial))

            LOGGER.log_to_file = True

            with TaskManager():
                solver.setup()

                if draw:
                    solver.drawer.draw(energy=True)
                    solver.drawer.draw_particle_velocity(u_inf, autoscale=True, min=-1e-4, max=1e-4)
                    solver.drawer.draw_acoustic_pressure(p_inf, autoscale=True, min=-1e-2, max=1e-2)

                solver.solve_transient(save_state_every_num_step=save_step)

            saver.save_mesh(mesh)
            saver.save_configuration(cfg, name=f"{tree.state_directory_name}/cfg")
            LOGGER.log_to_file = False

            cfg._info = info

        return wrapper

    return wraps


@test(name)
def farfield_boundary():
    solver = CompressibleHDGSolver(mesh, cfg, tree)
    solver.boundary_conditions.set(bcs.FarField(farfield, Qform=True), "left|right|top|bottom")
    return solver


@test(name)
def gfarfield_boundary_farfield_outflow():
    solver = CompressibleHDGSolver(mesh, cfg, tree)
    solver.boundary_conditions.set(
        bcs.GFarField(farfield, sigma=State(velocity=0.01, pressure=0.01)),
        'left|top|bottom|right')
    return solver


@test(name)
def gfarfield_boundary_pressure_outflow():
    solver = CompressibleHDGSolver(mesh, cfg, tree)
    solver.boundary_conditions.set(
        bcs.GFarField(farfield, relaxation='outflow', sigma=State(velocity=0.01, pressure=0.01)),
        'left|top|bottom|right')
    return solver


@test(name)
def yoo():
    solver = CompressibleHDGSolver(mesh, cfg, tree)
    solver.boundary_conditions.set(bcs.GFarField(farfield, 'yoo', relaxation="outflow"), 'left|top|bottom|right')
    return solver


@test(name)
def outflow_boundary():
    solver = CompressibleHDGSolver(mesh, cfg, tree)
    solver.boundary_conditions.set(bcs.Outflow(farfield.pressure), 'left|top|bottom|right')
    return solver


if __name__ == '__main__':
    gfarfield_boundary_pressure_outflow()
    gfarfield_boundary_farfield_outflow()
    farfield_boundary()
    yoo()
    outflow_boundary()
