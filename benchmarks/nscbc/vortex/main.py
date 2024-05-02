from ngsolve import *
from dream import *
from netgen.occ import OCCGeometry, WorkPlane, IdentificationType

ngsglobals.msg_level = 0
SetNumThreads(4)

draw = True

tree = ResultsDirectoryTree()
saver = Saver(tree)

name = ""
maxh = 0.1
L = 4
H = 2
Mt = 0.01
R = 0.1

cfg = SolverConfiguration()
cfg.formulation = "conservative"
cfg.scaling = "aerodynamic"
cfg.riemann_solver = 'hllem'

cfg.Mach_number = 0.1
cfg.heat_capacity_ratio = 1.4

cfg.order = 4
cfg.bonus_int_order_bnd = cfg.order
cfg.bonus_int_order_vol = cfg.order

cfg.time.simulation = "transient"
cfg.time.scheme = "BDF2"
cfg.time.step = 1e-2
save_step = 1
cfg.time.interval = (0, 6)
cfg.save_state = True

cfg.linear_solver = "pardiso"
cfg.damping_factor = 1
cfg.max_iterations = 100
cfg.convergence_criterion = 1e-16

cfg.compile_flag = False
cfg.static_condensation = True

face = WorkPlane().RectangleC(L, H).Face()

for bc, edge in zip(['bottom', 'right', 'top', 'left'], face.edges):
    edge.name = bc
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

r = sqrt(x**2 + y**2)

if cfg.scaling is cfg.scaling.AERODYNAMIC:
    vt = Mt/cfg.Mach_number
elif cfg.scaling is cfg.scaling.ACOUSTIC:
    vt = Mt
elif cfg.scaling is cfg.scaling.AEROACOUSTIC:
    vt = Mt/(1 + cfg.Mach_number)

psi = vt * R * exp((1 - (r/R)**2)/2)
u_0 = u_inf + CF((psi.Diff(y), -psi.Diff(x)))
p_0 = p_inf * (1 - (gamma - 1)/2 * Mt**2 * exp(1 - (r/R)**2))**(gamma/(gamma - 1))
rho_0 = rho_inf * (1 - (gamma - 1)/2 * Mt**2 *  exp(1 - (r/R)**2))**(1/(gamma - 1))
initial = State(u_0, rho_0, p_0)
p_00 = p_inf * (1 - (gamma - 1)/2 * Mt**2 * exp(1))**(gamma/(gamma - 1))

cfg.info["Domain Length"] = L
cfg.info["Domain Height"] = H
cfg.info['Radius Vortex'] = R
cfg.info['Mach Vortex'] = Mt


def test(name: str = ""):

    def wraps(func):

        def wrapper(*args, **kwargs):

            info = cfg.info.copy()

            tree.directory_name = f"Ma{cfg.Mach_number.Get()}/Mat{Mt}/dt{cfg.time.step.Get()}"

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


            solver.boundary_conditions.set(bcs.Periodic(), "top|bottom")
            solver.domain_conditions.set(dcs.Initial(initial))

            with TaskManager():
                solver.setup()

                if draw:
                    solver.drawer.draw(energy=True)
                    solver.drawer.draw_particle_velocity(u_inf, autoscale=True, min=-1e-4, max=1e-4)
                    solver.drawer.draw_acoustic_pressure(p_inf, autoscale=True, min=-1e-2, max=1e-2)
                    Draw((solver.formulation.pressure() - p_inf)/(p_00 - p_inf),
                         mesh, "p*",  autoscale=False, min=-1e-8, max=1e-8)

                solver.solve_transient(save_state_every_num_step=save_step)

            saver.save_mesh(mesh)
            saver.save_configuration(cfg, name=f"{tree.state_directory_name}/cfg", comment=func.__doc__)

            cfg._info = info

        return wrapper

    return wraps


@test(name)
def farfield_inflow_and_outflow():
    """ In this test we set the classical farfield condition known from literature as inflow and outflow. """

    solver = CompressibleHDGSolver(mesh, cfg, tree)
    solver.boundary_conditions.set(bcs.FarField(farfield), "left|right")

    cfg.info["Inflow"] = r"$\bm{FF}_{\bm{U}_{\infty}}$"
    cfg.info["Outflow"] = r"$\bm{FF}_{\bm{U}_{\infty}}$"

    return solver


@test(name)
def farfield_inflow_and_yoo_outflow(tg_flux: bool = False, glue: bool = False):
    """ In this test we set the classical farfield condition known from literature as inflow and as outflow the nonreflecting
    boundary condition of yoo with sigma_p = 0.278, sigma_T = 4 and sigma_u = 4. """

    flux = {False: "1d", True: "2d"}

    tree.state_directory_name += f"_{flux[tg_flux]}"
    if glue:
        tree.state_directory_name += f"_glue"

    solver = CompressibleHDGSolver(mesh, cfg, tree)
    solver.boundary_conditions.set(bcs.FarField(farfield), 'left')
    solver.boundary_conditions.set(bcs.NSCBC(farfield, tangential_flux=tg_flux, glue=glue), "right")

    cfg.info["Inflow"] = r"$\bm{FF}_{\bm{U}_{\infty}}$"
    cfg.info["Outflow"] = rf"$\bm{{NSCBC}}_{{yoo, sigma_p={0.278}}}$"
    cfg.info["Glue"] = glue

    return solver


@test(name)
def farfield_inflow_and_poinsot_outflow(sigmas: State, tg_flux: bool = False, glue: bool = False):
    """ In this test we set the classical farfield condition known from literature as inflow and as outflow the nonreflecting
    boundary condition of poinsot with different sigmas. """

    flux = {False: "1d", True: "2d"}
    tree.state_directory_name += f"_{flux[tg_flux]}_sigma{sigmas.pressure}"
    if glue:
        tree.state_directory_name += f"_glue"

    solver = CompressibleHDGSolver(mesh, cfg, tree)

    solver.boundary_conditions.set(bcs.FarField(farfield), 'left')
    solver.boundary_conditions.set(bcs.NSCBC(farfield, 'poinsot', sigmas=sigmas,
                                   tangential_flux=tg_flux, glue=glue), "right")

    cfg.info["Inflow"] = r"$\bm{FF}_{\bm{U}_{\infty}}$"
    cfg.info["Outflow"] = rf"$\bm{{NSCBC}}_{{pst, sigma_p={sigmas.pressure}}}$"
    cfg.info["Glue"] = glue
    cfg.info['Sigma'] = sigmas.pressure

    return solver

@test(name)
def farfield_inflow_and_generalized_Qfarfield_outflow(
        sigma=1, tg_flux: bool = False, Ut: bool = False, p_relaxation: bool = False, glue: bool = False):
    """ In this test we set the classical farfield condition known from literature as inflow and as outflow the generalized farfield condition in development with 
    additional time derivatives and tangential terms. """

    flux = {False: "1d", True: "2d"}
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

        Mn = self.cfg.Mach_number * IfPos(self.normal[0], self.normal[0], -self.normal[0])

        time = self._gfus.get_component(1)
        time['n+1'] = Uhat
        B = Mn * self.DME_convective_jacobian(Uhat, self.tangential) * (grad(Uhat) * self.tangential)
        if Ut:
            time = self._gfus.get_component(0)
            time['n+1'] = U
            B = Mn * self.DME_convective_jacobian(Uhat, self.tangential) * (grad(U) * self.tangential)

        Qin = self.DME_from_CHAR_matrix(self.identity_matrix(Uhat, self.normal, 'in', True), Uhat, self.normal)
        Qout = self.DME_from_CHAR_matrix(self.identity_matrix(Uhat, self.normal, 'out', True), Uhat, self.normal)

        cf = Uhat - Qout*U - Qin*farfield - (1 - bc.sigma) * Qin * (Uhat - farfield)
        cf += Qin * self.time_scheme.apply(time) * dt
        if bc.tangential_flux:
            cf += Qin * B * dt

        cf = cf * Vhat * ds(skeleton=True, definedon=boundary, bonus_intorder=bonus_order_bnd)
        blf += cf.Compile(compile_flag)

        if bc.glue:
            self.glue(blf, boundary)

    tree.state_directory_name += f"_{Utime[Ut]}_{flux[tg_flux]}_sigma{sigma}_{rel[p_relaxation]}"
    if glue:
        tree.state_directory_name += f"_glue"
    
    solver = CompressibleHDGSolver(mesh, cfg, tree)
    type(solver.formulation)._add_gfarfield_bilinearform = add_gfarfield_bilinearform

    solver.boundary_conditions.set(bcs.FarField(farfield), 'left')
    solver.boundary_conditions.set(bcs.GFarField(farfield, sigma, p_relaxation, tg_flux, glue), "right")

    label = {False: r"{\bm{u}_{\infty}}", True: r"{p_{\infty}}" }

    cfg.info["Inflow"] = r"$\bm{FF}_{\bm{U}_{\infty}}$"
    cfg.info["Outflow"] = rf"$\bm{{GFF}}_{label[p_relaxation]}$"
    cfg.info['Sigma'] = sigma
    cfg.info["Glue"] = glue
    cfg.info["Relaxation"] = rel[p_relaxation]

    return solver

@test(name)
def farfield_inflow_and_generalized_farfield_outflow(
        sigma=1, tg_flux: bool = False, Ut: bool = False, p_relaxation: bool = False, glue: bool = False):
    """ In this test we set the classical farfield condition known from literature as inflow and as outflow the generalized farfield condition in development with 
    additional time derivatives and tangential terms. """

    flux = {False: "1d", True: "2d"}
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

        Mn = self.cfg.Mach_number * IfPos(self.normal[0], self.normal[0], -self.normal[0])

        time = self._gfus.get_component(1)
        time['n+1'] = Uhat
        B = Mn * self.DME_convective_jacobian(Uhat, self.tangential) * (grad(Uhat) * self.tangential)
        if Ut:
            time = self._gfus.get_component(0)
            time['n+1'] = U
            B = Mn * self.DME_convective_jacobian(Uhat, self.tangential) * (grad(U) * self.tangential)

        Ain = self.DME_convective_jacobian_incoming(Uhat, self.normal)
        Aout = self.DME_convective_jacobian_outgoing(Uhat, self.normal)

        cf = Aout * (Uhat - U) - bc.sigma * Ain * (Uhat - farfield)
        cf -= Ain * self.time_scheme.apply(time) * dt
        if bc.tangential_flux:
            cf -= Ain * B * dt

        cf = cf * Vhat * ds(skeleton=True, definedon=boundary, bonus_intorder=bonus_order_bnd)
        blf += cf.Compile(compile_flag)

        if bc.glue:
            self.glue(blf, boundary)

    tree.state_directory_name += f"_{Utime[Ut]}_{flux[tg_flux]}_sigma{sigma}_{rel[p_relaxation]}"
    if glue:
        tree.state_directory_name += f"_glue"
    
    solver = CompressibleHDGSolver(mesh, cfg, tree)
    type(solver.formulation)._add_gfarfield_bilinearform = add_gfarfield_bilinearform

    solver.boundary_conditions.set(bcs.FarField(farfield), 'left')
    solver.boundary_conditions.set(bcs.GFarField(farfield, sigma, p_relaxation, tg_flux, glue), "right")

    label = {False: r"{\bm{u}_{\infty}}", True: r"{p_{\infty}}" }

    cfg.info["Inflow"] = r"$\bm{FF}_{\bm{U}_{\infty}}$"
    cfg.info["Outflow"] = rf"$\bm{{GFF}}_{label[p_relaxation]}$"
    cfg.info['Sigma'] = sigma
    cfg.info["Glue"] = glue
    cfg.info["Relaxation"] = rel[p_relaxation]

    return solver


if __name__ == '__main__':

    for Ut in [True, False]:
        for tg_flux in [True, False]:
            for sigma in [1, cfg.Mach_number.Get(), 1e-3]:
                for glue in [True, False]:
                    for pressure_relaxation in [True, False]:
                        farfield_inflow_and_generalized_farfield_outflow(sigma, tg_flux, Ut, pressure_relaxation, glue)

    farfield_inflow_and_generalized_farfield_outflow(1e-3, True, False)

    farfield_inflow_and_outflow()

    for tg_flux in [True, False]:
        for glue in [True, False]:
            farfield_inflow_and_yoo_outflow(tg_flux, glue)

    for tg_flux in [True, False]:
        for glue in [True, False]:
            for sigma in [State(4, pressure=0.28, temperature=4),
                          State(4, pressure=1e-3, temperature=4),
                          State(4, pressure=5, temperature=4)]:
                farfield_inflow_and_poinsot_outflow(sigma, tg_flux, glue)
