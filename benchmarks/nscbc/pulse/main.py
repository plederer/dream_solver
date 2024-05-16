from ngsolve import *
from dream import *
from netgen.occ import OCCGeometry, WorkPlane, IdentificationType

ngsglobals.msg_level = 0
SetNumThreads(4)

draw = False

tree = ResultsDirectoryTree()
# tree.parent_path = ""

LOGGER.tree = tree
# LOGGER.log_to_terminal = False

saver = Saver(tree)

name = ""
maxh = 0.07
R = 0.1
H = 8*R
alpha = 0.4

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
cfg.time.step = 0.01
save_step = 1
cfg.time.interval = (0, 10)
cfg.time.interval = (0, 0.01)
cfg.save_state = False

cfg.linear_solver = "pardiso"
cfg.damping_factor = 1
cfg.max_iterations = 5
cfg.convergence_criterion = 1e-8

cfg.compile_flag = False
cfg.static_condensation = True

face = WorkPlane().RectangleC(H, H).Face()

for bc, edge in zip(['bottom', 'right', 'top', 'left'], face.edges):
    edge.name = bc

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

            i = 0
            while tree.state_path.exists():
                labels = tree.state_path.name.split("_")

                if f"{i}" in labels:
                    labels = labels[:-1]

                i += 1
                labels.append(f"{i}")

                tree.state_directory_name = '_'.join(labels)

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
            saver.state_path
            saver.save_configuration(cfg, name=f"{tree.state_directory_name}/cfg", comment=func.__doc__)

            cfg._info = info
            LOGGER.log_to_file = False


        return wrapper

    return wraps


@test(name)
def farfield_boundary(Qform=False):
    """ In this test we set the classical farfield condition known from literature as inflow and outflow. """

    if Qform:
        tree.state_directory_name += f"_Qform"

    solver = CompressibleHDGSolver(mesh, cfg, tree)
    solver.boundary_conditions.set(bcs.FarField(farfield, Qform=Qform), "left|right|bottom|top")

    return solver


@test(name)
def outflow_boundary():
    """ In this test we set the classical farfield condition known from literature as inflow and outflow. """

    solver = CompressibleHDGSolver(mesh, cfg, tree)
    solver.boundary_conditions.set(bcs.Outflow(farfield.pressure), "left|right|bottom|top")

    return solver


@test(name)
def yoo_boundary(glue: bool = False):
    """ In this test we set the classical farfield condition known from literature as inflow and as outflow the nonreflecting
    boundary condition of yoo with sigma_p = 0.278, sigma_T = 4 and sigma_u = 4. """

    if glue:
        tree.state_directory_name += f"_glue"

    solver = CompressibleHDGSolver(mesh, cfg, tree)
    solver.boundary_conditions.set(
        bcs.NSCBC(farfield, 'yoo', tangential_flux=False, glue=glue),
        "right|left|top|bottom")

    cfg.info["Glue"] = glue

    return solver


@test(name)
def poinsot_boundary(sigmas: State, glue: bool = False):
    """ In this test we set the classical farfield condition known from literature as inflow and as outflow the nonreflecting
    boundary condition of poinsot with different sigmas. """

    tree.state_directory_name += f"_sigma{sigmas.pressure}"
    if glue:
        tree.state_directory_name += f"_glue"

    solver = CompressibleHDGSolver(mesh, cfg, tree)
    solver.boundary_conditions.set(
        bcs.NSCBC(farfield, 'poinsot', sigmas, tangential_flux=False, glue=glue),
        "right|left|top|bottom")

    cfg.info["Glue"] = glue
    cfg.info['Sigma'] = sigmas.pressure

    return solver


@test(name)
def gfarfield_boundary(sigma=1, Ut: bool = False, p_relaxation: bool = False, glue: bool = False, Qform: bool = False):
    """ In this test we set the classical farfield condition known from literature as inflow and as outflow the generalized farfield condition in development with 
    additional time derivatives and tangential terms. """

    Utime = {False: 'Uhat', True: 'U'}
    rel = {False: 'Uinf', True: 'pinf'}

    def add_gfarfield_bilinearform(self, blf,  boundary: Region, bc: bcs.GFarField):
        compile_flag = self.cfg.compile_flag
        bonus_order_bnd = self.cfg.bonus_int_order_bnd

        U, _ = self.TnT.PRIMAL
        Uhat, Vhat = self.TnT.PRIMAL_FACET
        dt = self.cfg.time.step

        state = self.calc.determine_missing(bc.state)
        farfield = CF((state.density, state.momentum, state.energy))
        if bc.pressure_relaxation:
            farfield = CF(
                (self.density(U),
                 self.momentum(U),
                 state.pressure / (self.cfg.heat_capacity_ratio - 1) + self.kinetic_energy(U)))

        time = self._gfus.get_component(1)
        time['n+1'] = Uhat
        if Ut:
            time = self._gfus.get_component(0)
            time['n+1'] = U

        if Qform:

            Qin = self.DME_from_CHAR_matrix(self.identity_matrix(Uhat, self.normal, 'in', True), Uhat, self.normal)
            Qout = self.DME_from_CHAR_matrix(self.identity_matrix(Uhat, self.normal, 'out', True), Uhat, self.normal)

            cf = Uhat - Qout*U - Qin*farfield - (1 - bc.sigma) * Qin * (Uhat - farfield)
            cf += Qin * self.time_scheme.apply(time) * dt

        else:

            Ain = self.DME_convective_jacobian_incoming(Uhat, self.normal)
            Aout = self.DME_convective_jacobian_outgoing(Uhat, self.normal)

            cf = Aout * (Uhat - U) - bc.sigma * Ain * (Uhat - farfield)
            cf -= Ain * self.time_scheme.apply(time) * dt

        cf = cf * Vhat * ds(skeleton=True, definedon=boundary, bonus_intorder=bonus_order_bnd)
        blf += cf.Compile(compile_flag)

        if bc.glue:
            self.glue(blf, boundary)

    tree.state_directory_name += f"_{Utime[Ut]}_sigma{sigma}_{rel[p_relaxation]}"
    if glue:
        tree.state_directory_name += f"_glue"
    if Qform:
        tree.state_directory_name += f"_Qform"
    

    solver = CompressibleHDGSolver(mesh, cfg, tree)
    type(solver.formulation)._add_gfarfield_bilinearform = add_gfarfield_bilinearform

    solver.boundary_conditions.set(bcs.GFarField(farfield, sigma, p_relaxation, glue=glue), "left|right|top|bottom")

    cfg.info['Sigma'] = sigma
    cfg.info["Glue"] = glue
    cfg.info["Relaxation"] = rel[p_relaxation]

    return solver


if __name__ == '__main__':

    # for Ut in [True, False]:
    #     for tg_flux in [True, False]:
    #         for sigma in [1, cfg.Mach_number.Get(), 1e-3]:
    #             for glue in [True, False]:
    #                 for pressure_relaxation in [True, False]:
    #                     farfield_inflow_and_generalized_farfield_outflow(sigma, tg_flux, Ut, pressure_relaxation, glue)

    # farfield_inflow_and_generalized_farfield_outflow(1e-3, True, False)

    gfarfield_boundary(Qform=True)

    # for tg_flux in [True, False]:
    #     for glue in [True, False]:
    #         farfield_inflow_and_yoo_outflow(tg_flux, glue)

    # for tg_flux in [True, False]:
    #     for glue in [True, False]:
    #         for sigma in [State(4, pressure=0.28, temperature=4),
    #                       State(4, pressure=1e-3, temperature=4),
    #                       State(4, pressure=5, temperature=4)]:
    #             farfield_inflow_and_poinsot_outflow(sigma, tg_flux, glue)
