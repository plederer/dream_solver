from ngsolve import *
from dream import *
from netgen.occ import OCCGeometry, WorkPlane, IdentificationType
import argparse

parser = argparse.ArgumentParser(description='Vortex benchmark')
parser.add_argument('Alpha', metavar='a', type=float, help='Pulse strength')
parser.add_argument('Wave', metavar='w', type=str, help='Wave direction')
parser.add_argument('Mach', metavar='M', type=float, help='Mach number')
args = vars(parser.parse_args())

# args = {'Alpha': 0.01, 'Wave': 'right', 'Mach': 0.03}

ngsglobals.msg_level = 0
SetNumThreads(16)


draw = False

tree = ResultsDirectoryTree()
# tree.parent_path = ""

LOGGER.tree = tree
LOGGER.log_to_terminal = False

saver = Saver(tree)

name = ""
maxh = 0.07
X = 0.1
W = 8*X
alpha = args['Alpha'] #
wave = args['Wave']

if wave == "left":
    sign = -1
elif wave == "right":
    sign = 1
elif wave == "both":
    sign = 0
else:
    raise ValueError("Invalid wave type")

cfg = SolverConfiguration()
cfg.formulation = "conservative"
cfg.scaling = "acoustic"
cfg.riemann_solver = 'lax_friedrich'

cfg.Mach_number = args['Mach']
cfg.heat_capacity_ratio = 1.4

cfg.order = 4
cfg.bonus_int_order_bnd = cfg.order
cfg.bonus_int_order_vol = cfg.order

cfg.time.simulation = "transient"
cfg.time.scheme = "BDF2"
cfg.time.step = 2e-3
save_step = 5
cfg.time.interval = (0, 5)
cfg.save_state = True

cfg.linear_solver = "pardiso"
cfg.damping_factor = 1
cfg.max_iterations = 5
cfg.convergence_criterion = 1e-8

cfg.compile_flag = True
cfg.static_condensation = True

face = WorkPlane().RectangleC(W, 4*maxh).Face()
for bc, edge in zip(['bottom', 'right', 'top', 'left'], face.edges):
    edge.name = bc

periodic_edge = face.edges[0]
periodic_edge.Identify(face.edges[2], "periodic", IdentificationType.PERIODIC)

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
c_inf = INF.speed_of_sound(cfg)

p_0 = p_inf * (1 + alpha * exp(- x**2/X**2))
u_0 = u_inf + sign * p_inf/(rho_inf * c_inf) * alpha * exp(-x**2/X**2) * CF((1, 0))
rho_0 = rho_inf
initial = State(u_0, rho_0, p_0)

cfg.info["Domain Length"] = W
cfg.info["Domain Height"] = W
cfg.info['Width Pulse'] = X
cfg.info['Pulse Strength'] = alpha


def test(name: str = ""):

    def wraps(func):

        def wrapper(*args, **kwargs):

            info = cfg.info.copy()

            tree.directory_name = f"Ma_{cfg.Mach_number.Get()}/alpha_{alpha}/wave_{wave}"

            tree.state_directory_name = func.__name__
            if name:
                tree.state_directory_name += f"_{name}"

            solver = func(*args, **kwargs)
            solver.boundary_conditions.set(bcs.Periodic(), 'top|bottom')
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
def farfield_inflow_and_outflow():
    solver = CompressibleHDGSolver(mesh, cfg, tree)
    solver.boundary_conditions.set(bcs.FarField(farfield, Qform=True), "left|right")
    return solver


@test(name)
def farfield_inflow_and_pressure_outflow():
    solver = CompressibleHDGSolver(mesh, cfg, tree)
    solver.boundary_conditions.set(bcs.FarField(farfield, Qform=True), 'left')
    solver.boundary_conditions.set(bcs.Outflow(farfield.pressure), 'right')

    return solver


@test(name)
def grcbc_farfield_inflow_and_outflow(C: float = 0.01):
    solver = CompressibleHDGSolver(mesh, cfg, tree)
    solver.boundary_conditions.set(
        bcs.CBC(farfield, sigma=State(velocity=C, pressure=C)), 'left|right')

    tree.state_directory_name += f"_C{C}"
    cfg.info['CFL'] = C

    return solver


@test(name)
def grcbc_farfield_inflow_and_pressure_outflow(C: float = 0.01):
    solver = CompressibleHDGSolver(mesh, cfg, tree)
    solver.boundary_conditions.set(bcs.CBC(farfield, relaxation='outflow',
                                   sigma=State(velocity=C, pressure=C)), 'right')
    solver.boundary_conditions.set(bcs.CBC(farfield, relaxation='farfield',
                                   sigma=State(velocity=C, pressure=C)), 'left')

    tree.state_directory_name += f"_C{C}"
    cfg.info['CFL'] = C

    return solver


@test(name)
def grcbc_mass_inflow_and_pressure_outflow(C: float = 0.01):
    solver = CompressibleHDGSolver(mesh, cfg, tree)
    solver.boundary_conditions.set(bcs.CBC(farfield, relaxation='outflow',
                                   sigma=State(velocity=C, pressure=C)), 'right')
    solver.boundary_conditions.set(bcs.CBC(farfield, relaxation='mass_inflow',
                                   sigma=State(velocity=C, pressure=C)), 'left')

    tree.state_directory_name += f"_C{C}"
    cfg.info['CFL'] = C

    return solver


@test(name)
def grcbc_temperature_inflow_and_pressure_outflow(C: float = 0.01):
    solver = CompressibleHDGSolver(mesh, cfg, tree)
    solver.boundary_conditions.set(bcs.CBC(farfield, relaxation='outflow',
                                   sigma=State(velocity=C, pressure=C)), 'right')
    solver.boundary_conditions.set(bcs.CBC(farfield, relaxation='temperature_inflow',
                                   sigma=State(velocity=C, pressure=C)), 'left')

    tree.state_directory_name += f"_C{C}"
    cfg.info['CFL'] = C

    return solver


@test(name)
def nscbc_farfield_inflow_and_outflow(Sigma: float = 0.01):
    solver = CompressibleHDGSolver(mesh, cfg, tree)
    solver.boundary_conditions.set(
        bcs.CBC(farfield, type="nscbc", sigma=State(velocity=Sigma, pressure=Sigma)), 'left|right')

    tree.state_directory_name += f"_Sigma{Sigma}"
    cfg.info['Sigma'] = Sigma

    return solver


@test(name)
def nscbc_farfield_inflow_and_pressure_outflow(Sigma: float = 0.01):
    solver = CompressibleHDGSolver(mesh, cfg, tree)
    solver.boundary_conditions.set(bcs.CBC(farfield, type="nscbc", relaxation='outflow',
                                   sigma=State(velocity=Sigma, pressure=Sigma)), 'right')
    solver.boundary_conditions.set(bcs.CBC(farfield, type="nscbc", relaxation='farfield',
                                   sigma=State(velocity=Sigma, pressure=Sigma)), 'left')

    tree.state_directory_name += f"_Sigma{Sigma}"
    cfg.info['Sigma'] = Sigma

    return solver


@test(name)
def nscbc_mass_inflow_and_pressure_outflow(Sigma: float = 0.01):
    solver = CompressibleHDGSolver(mesh, cfg, tree)
    solver.boundary_conditions.set(bcs.CBC(farfield, type="nscbc", relaxation='outflow',
                                   sigma=State(velocity=Sigma, pressure=Sigma)), 'right')
    solver.boundary_conditions.set(bcs.CBC(farfield, type="nscbc", relaxation='mass_inflow',
                                   sigma=State(velocity=Sigma, pressure=Sigma)), 'left')

    tree.state_directory_name += f"_Sigma{Sigma}"
    cfg.info['Sigma'] = Sigma

    return solver


@test(name)
def nscbc_temperature_inflow_and_pressure_outflow(Sigma: float = 0.01):
    solver = CompressibleHDGSolver(mesh, cfg, tree)
    solver.boundary_conditions.set(bcs.CBC(farfield, type="nscbc", relaxation='outflow',
                                   sigma=State(velocity=Sigma, pressure=Sigma)), 'right')
    solver.boundary_conditions.set(bcs.CBC(farfield, type="nscbc", relaxation='temperature_inflow',
                                   sigma=State(velocity=Sigma, pressure=Sigma)), 'left')

    tree.state_directory_name += f"_Sigma{Sigma}"
    cfg.info['Sigma'] = Sigma

    return solver


if __name__ == '__main__':

    farfield_inflow_and_outflow()
    farfield_inflow_and_pressure_outflow()

    for C in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 0]:
        grcbc_farfield_inflow_and_outflow(C)
        grcbc_farfield_inflow_and_pressure_outflow(C)
        grcbc_mass_inflow_and_pressure_outflow(C)
        grcbc_temperature_inflow_and_pressure_outflow(C)

    for Sigma in [1, 0.28, 1e-1, 1e-2, 1e-3, 1e-4]:
        nscbc_farfield_inflow_and_outflow(Sigma)
        nscbc_farfield_inflow_and_pressure_outflow(Sigma)
        nscbc_mass_inflow_and_pressure_outflow(Sigma)
        nscbc_temperature_inflow_and_pressure_outflow(Sigma)
