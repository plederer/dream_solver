# Import modules
import numpy as np
import ngsolve as ngs
import netgen.occ as occ
import pandas as pd
import matplotlib.pyplot as plt

from dream.compressible import CompressibleFlowSolver, Force, FarField, Initial, flowfields,  Dirichlet
from dream.time import TransientRoutine


def get_geometry(periodic: bool = False):

    face = occ.WorkPlane().Rectangle(1, 1).Face()
    for edge, name in zip(face.edges, ('bottom', 'right', 'top', 'left')):
        edge.name = name

    if periodic:
        face.edges[0].Identify(face.edges[2], "periodic_x", occ.IdentificationType.PERIODIC)
        face.edges[1].Identify(face.edges[3], "periodic_y", occ.IdentificationType.PERIODIC)

    return occ.OCCGeometry(face, dim=2)


def div(F):
    return ngs.CF(tuple(F[i, 0].Diff(ngs.x) + F[i, 1].Diff(ngs.y) for i in range(F.dims[0])))


SCHEME_ORDER = {
    'implicit_euler': 1,
    'explicit_euler': 1,
    'bdf2': 2,
    'bdf3': 3,
    'rk_ars22': 2,
    'rk_ars33': 3,
    'crk4': 4,
    'sdirk22': 2,
    'sdirk33': 3,
    'sdirk43': 3,
    'sdirk54': 4
}

TRANSIENT_CFG = {
    'mach_number': 0.2,
    'reynolds_number': 1.0,
    'prandtl_number': 0.72,
    'equation_of_state': 'ideal',
    'equation_of_state.heat_capacity_ratio': 1.4,
    'scaling': 'aerodynamic',
    'riemann_solver': 'lax_friedrich',
    'time': 'transient',
    'time.timer.interval': (0.0, 10.0),
    'time.timer.step': 0.0001,
    'fem': 'conservative_hdg',
    'fem.order': 0,
    'fem.solver': 'direct',
    'fem.solver.method': 'newton',
    'fem.solver.method.max_iterations': 10,
    'fem.solver.method.convergence_criterion': 1e-10,
    'fem.solver.method.damping_factor': 1.0,
    'fem.bonus_int_order': 4,
    'fem.scheme': 'implicit_euler',
    'fem.scheme.compile': {'realcompile': False, 'wait': False, 'keep_files': False},
    'fem.viscous_treatment': None,
    'io.path': 'test',
    'io.vtk.enable': False,
    'io.vtk.rate': 1,
    'io.vtk.subdivision': 2,
}

PSEUDO_STATIONARY_CFG = {
    'mach_number': 0.2,
    'reynolds_number': 1.0,
    'prandtl_number': 0.72,
    'equation_of_state.heat_capacity_ratio': 1.4,
    'riemann_solver': 'lax_friedrich',
    'scaling': 'aerodynamic',
    'time': 'pseudo_time_stepping',
    'time.timer.step': 1,
    'time.max_time_step': 1,
    'time.increment_at': 1,
    'time.increment_factor': 5,
    'fem': 'conservative_hdg',
    'fem.order': 1,
    'fem.solver': 'direct',
    'fem.solver.method': 'newton',
    'fem.solver.method.max_iterations': 300,
    'fem.solver.method.convergence_criterion': 1e-20,
    'fem.solver.method.damping_factor': 1.0,
    'fem.scheme': "implicit_euler",
    'fem.scheme.compile': {'realcompile': False, 'wait': False, 'keep_files': False},
    'io.path': 'test',
    'io.vtk.enable': False,
    'io.vtk.rate': 1,
    'io.vtk.subdivision': 2,
}


class MMS:

    def __init__(self,
                 cfg: dict,
                 mms: callable,
                 filename: str | None = None,
                 flow_direction: tuple = (1.0, 1.0),
                 length: float = 1.0,
                 periods: int = 5,
                 init_exact: bool = False,
                 is_periodic: bool = False,
                 **export):

        if filename is None:
            filename = f"{str(self)}"

        if is_periodic:
            filename += "_periodic"

        self._cfg = cfg
        self._mms = mms
        self.flow_direction = flow_direction
        self.length = float(length)
        self.periods = int(periods)
        self.filename = filename
        self.is_periodic = is_periodic
        self.init_exact = init_exact
        self.export = {'to_fig': True, 'to_csv': True, 'to_dat': True}
        self.export.update(export)

    @property
    def is_transient(self) -> bool:
        return isinstance(self.cfg.time, TransientRoutine)

    def get_exact_fields(self, cfg: CompressibleFlowSolver) -> flowfields:
        # Construct dimensionless fields

        t = None
        if self.is_transient:
            t = cfg.time.timer.t

        U = self(t)
        U.u = cfg.velocity(U)
        U.rho_Ek = cfg.kinetic_energy(U)
        U.rho_Ei = cfg.inner_energy(U)
        U.p = cfg.pressure(U)
        U.T = cfg.temperature(U)

        U.grad_rho = ngs.CF((U.rho.Diff(ngs.x), U.rho.Diff(ngs.y)))
        U.grad_p = ngs.CF((U.p.Diff(ngs.x), U.p.Diff(ngs.y)))
        U.grad_u = ngs.CF((U.u[0].Diff(ngs.x), U.u[0].Diff(ngs.y),
                           U.u[1].Diff(ngs.x), U.u[1].Diff(ngs.y)), dims=(2, 2))
        U.grad_T = cfg.temperature_gradient(U, U)
        U['Ma'] = cfg.get_local_mach_number(U)

        return U

    def get_initial_fields(self, cfg: CompressibleFlowSolver) -> flowfields:
        # Construct dimensionless initial fields from constant terms only

        if self.is_transient:
            U = self(t=0)
        elif self.init_exact:
            U = self()
        else:
            U = cfg.get_farfield_fields(self.flow_direction)

        U.rho_Ek = cfg.kinetic_energy(U)
        U.rho_Ei = cfg.inner_energy(U)
        U.p = cfg.pressure(U)
        U.u = cfg.velocity(U)

        U.grad_rho = ngs.CF((U.rho.Diff(ngs.x), U.rho.Diff(ngs.y)))
        U.grad_p = ngs.CF((U.p.Diff(ngs.x), U.p.Diff(ngs.y)))
        U.grad_u = ngs.CF((U.u[0].Diff(ngs.x), U.u[0].Diff(ngs.y),
                           U.u[1].Diff(ngs.x), U.u[1].Diff(ngs.y)), dims=(2, 2))
        U.grad_T = cfg.temperature_gradient(U, U)

        return U

    def set_conditions(self, cfg: CompressibleFlowSolver):
        self.cfg = cfg
        cfg.update(self._cfg)

        cfg.bcs.clear()
        cfg.dcs.clear()

        self.Ue = self.get_exact_fields(cfg)
        self.U0 = self.get_initial_fields(cfg)

    def set_solution_fields(self, Uh: flowfields):
        Uh['Ma'] = self.cfg.get_local_mach_number(Uh)
        Uh.rho_u = self.cfg.momentum(Uh)
        Uh.rho_E = self.cfg.energy(Uh)
        Uh.T = self.cfg.temperature(Uh)
        self.Uh = Uh

    def write_to_streams(self, t: float = None, **log):
        self.log = log

        Ue = self.Ue
        Uh = self.Uh

        order = self.cfg.fem.order

        key = tuple(log.values())
        if t is not None:
            key = key + (t,)
            self.log['t'] = t

        Ue_ = ngs.CF((Ue.rho, Ue.rho_u, Ue.rho_E))
        Uh_ = ngs.CF((Uh.rho, Uh.rho_u, Uh.rho_E))

        self.errors[key] = {
            'rho': ngs.sqrt(ngs.Integrate((Uh.rho - Ue.rho) ** 2, self.cfg.mesh, order=order + 10)),
            'u': ngs.sqrt(ngs.Integrate(ngs.InnerProduct(Uh.u - Ue.u, Uh.u - Ue.u), self.cfg.mesh, order=order + 10)),
            'p': ngs.sqrt(ngs.Integrate((Uh.p - Ue.p) ** 2, self.cfg.mesh, order=order + 10)),
            'rho_u': ngs.sqrt(ngs.Integrate(ngs.InnerProduct(Uh.rho_u - Ue.rho_u, Uh.rho_u - Ue.rho_u), self.cfg.mesh, order=order + 10)),
            'rho_E': ngs.sqrt(ngs.Integrate((Uh.rho_E - Ue.rho_E) ** 2, self.cfg.mesh, order=order + 10)),
            'U': ngs.sqrt(ngs.Integrate(ngs.InnerProduct(Uh_ - Ue_, Uh_ - Ue_), self.cfg.mesh, order=order + 10)),
        }

    def open_output_streams(self):
        if any(self.export.values()):
            self.cfg.io.path.mkdir(parents=True, exist_ok=True)

        self.errors = {}

    def open_vtk_stream(self, **log):

        filename = self.filename
        for key, value in log.items():
            if isinstance(value, str):
                filename += f"_{value}"
            else:
                filename += f"_{key}{value}"

        self.cfg.io.vtk.filename = filename
        export = ['density', 'velocity', 'pressure', 'temperature', 'Ma', 'energy']
        fields = {f"{key}_h": value for key, value in self.Uh.items() if key in export}
        fields.update({f"{key}_e": value for key, value in self.Ue.items() if key in export})
        self.cfg.io.vtk.fields = fields

    def close_output_streams(self):

        if self.export['to_csv']:
            self.export_errors_to_csv()

        if self.export['to_fig']:
            self.export_errors_to_fig()

        if self.export['to_dat']:
            self.export_errors_to_dat()

    def export_errors_to_csv(self):

        filepath = self.cfg.io.path.joinpath(f"{self.filename}.csv")

        with filepath.open('w', newline='') as file:

            if not self.is_transient:

                LEVELS = max([level for level, _, _ in self.errors.keys()])

                file.write(f"levels (refinement): {LEVELS}\n")
                file.write("\n")
                file.write("------------------------------------------------------------------------\n")
                file.write("h, dt, eq, field, order, error\n")
                file.write("------------------------------------------------------------------------\n")

                errors = sorted(self.errors.items(), key=lambda x: x[0][2])
                for field in ['rho', 'u', 'p', 'rho_u', 'rho_E', 'U']:

                    for key, error in errors:
                        _, h, order = key
                        row = [f"{h:.15e}", f"{1.0:.15e}", str(self), f"{field:5}", str(order), f"{error[field]:.15e}"]
                        file.write(",\t".join(row) + "\n")

            else:

                SCHEMES = list(set(scheme for scheme, _, _ in self.errors))
                TIME_STEPS = list(set(dt for _, dt, _ in self.errors))
                TIME_STEPS.sort(reverse=True)
                file.write(f"Schemes: {SCHEMES}, Time steps: {TIME_STEPS}\n")
                file.write("\n")
                file.write("------------------------------------------------------------------------\n")
                file.write("scheme, dt, eq, field, t, error\n")
                file.write("------------------------------------------------------------------------\n")

                for field in ['rho', 'u', 'p', 'rho_u', 'rho_E', 'U']:

                    for key, error in self.errors.items():
                        scheme, dt, t = key
                        row = [f"{scheme:15}", f"{dt:5}", str(self), f"{field:5}", f"{t:.8e}", f"{error[field]:.15e}"]
                        file.write(",\t".join(row) + "\n")

    def export_errors_to_fig(self):

        # L2 errors
        labels = {'rho': r"$\| \rho_h - \rho_e \|$", 'u': r"$\| \mathbf{u}_h - \mathbf{u}_e \|$",
                  'p': r"$\| p_h - p_e \|$", 'rho_u': r"$\| \rho \mathbf{u}_h - \rho \mathbf{u}_e \|$",
                  'rho_E': r"$\| \rho E_h - \rho E_e \|$", 'U': r"$\| \bm{U}_h - \bm{U}_e \|$"}

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        df = pd.DataFrame.from_dict(self.errors, orient='index')
        df.index.names = list(self.log)

        if 'h' in df.index.names:
            h = df.index.get_level_values('h').unique()

            for ax, field in zip(axes, labels):

                data = df.xs(field, axis=1)

                for order in df.index.levels[2]:
                    error = data.loc[:, :, order]
                    ax.loglog(h, error, marker='o',  label=fr"$p={order}$")
                    ax.loglog(h, h**(order+1)/h[0]**(order+1) * error.iloc[0], ls='--', color='k')

                ax.set_xlabel(r"$h$")
                ax.set_title(labels[field])
                ax.legend()

                ax.set_ylim(1e-10, 1)

            fig.savefig(self.cfg.io.path.joinpath(f"{self.filename}.png"))

        elif 'dt' in df.index.names:
            schemes = df.index.get_level_values('scheme').unique()

            for scheme in schemes:
                scheme_ = df.xs(scheme, level='scheme')
                dt = scheme_.index.get_level_values('dt').unique()

                mean = pd.DataFrame([scheme_.loc[j].mean() for j in dt])
                mean.index = dt

                for ax, field in zip(axes, labels):

                    error = mean.xs(field, axis=1)
                    ax.loglog(dt, error, marker='o',  label=fr"{scheme}")

                    if scheme in SCHEME_ORDER:
                        order = SCHEME_ORDER[scheme]
                        ax.loglog(dt, dt**order/dt[0]**order * error.iloc[0], ls='--', color='k')

            for ax, field in zip(axes, labels):
                ax.set_xlabel(r"$\Delta t_{ny}$")
                ax.set_title(labels[field])
                ax.legend()

                ax.set_ylim(1e-6, 1)

            fig.savefig(self.cfg.io.path.joinpath(f"{self.filename}.png"))

    def export_errors_to_dat(self):
        df = pd.DataFrame.from_dict(self.errors, orient='index')
        df.index.names = list(self.log)
        if 'h' in df.index.names:
            df.sort_index(axis=0, level=2, inplace=True)
        df.to_csv(self.cfg.io.path.joinpath(f"{self.filename}.dat"))

    def __call__(self, t: float = None) -> flowfields:
        return self._mms(self, t)


class EulerMMS(MMS):

    def set_conditions(self, cfg: CompressibleFlowSolver):

        cfg.dynamic_viscosity = "inviscid"

        super().set_conditions(cfg)

        F = div(cfg.get_convective_flux(self.Ue))
        if self.is_transient:
            t = cfg.time.timer.t
            F += ngs.CF((self.Ue.rho.Diff(t), self.Ue.rho_u.Diff(t), self.Ue.rho_E.Diff(t)))

        cfg.dcs['default'] = Initial(fields=self.U0)
        cfg.dcs['default'] = Force(F[0], F[1:3], F[3], order=10, is_constant=not self.is_transient)

        if self.is_periodic:
            cfg.bcs['left|right|top|bottom'] = "periodic"
        else:
            cfg.bcs['left|right|top|bottom'] = FarField(fields=self.Ue)

    def __str__(self):
        return "EE"


class NavierStokesMMS(MMS):

    def set_conditions(self, cfg: CompressibleFlowSolver):

        cfg.dynamic_viscosity = "constant"

        super().set_conditions(cfg)

        F = div(cfg.get_convective_flux(self.Ue) - cfg.get_diffusive_flux(self.Ue, self.Ue))
        if self.is_transient:
            t = cfg.time.timer.t
            F += ngs.CF((self.Ue.rho.Diff(t), self.Ue.rho_u.Diff(t), self.Ue.rho_E.Diff(t)))

        cfg.dcs['default'] = Initial(fields=self.U0)
        cfg.dcs['default'] = Force(F[0], F[1:3], F[3], order=10, is_constant=not self.is_transient)

        if self.is_periodic:
            cfg.bcs['left|right|top|bottom'] = "periodic"
        else:
            cfg.bcs['left|right|top|bottom'] = Dirichlet(fields=self.Ue)

    def __str__(self):
        return "NS"


def get_constant_mach_mms(mms: MMS, t: float = None) -> flowfields:

    k = 2*ngs.pi/mms.length
    wave = ngs.sin(k*(ngs.x + ngs.y))

    if t is not None:
        T = mms.cfg.time.timer.interval[1] - mms.cfg.time.timer.interval[0]
        omega = 2*np.pi*mms.periods/T
        wave = ngs.sin(k*(ngs.x + ngs.y) - omega * t)

    U_inf = mms.cfg.get_farfield_fields(mms.flow_direction)
    U_inf.E = mms.cfg.specific_energy(U_inf)

    U = flowfields()

    U.rho = U_inf.rho * (1 + 0.1 * wave)
    U.rho_u = U.rho * U_inf.u
    U.rho_E = U.rho * U_inf.E

    U.u = mms.cfg.velocity(U)
    U.rho_Ek = mms.cfg.kinetic_energy(U)
    U.rho_Ei = mms.cfg.inner_energy(U)
    U.p = mms.cfg.pressure(U)
    U.T = mms.cfg.temperature(U)
    return U


def get_roy_mms(mms: MMS, t: float = None) -> flowfields:

    def sin(a):
        return ngs.sin(a * ngs.pi)

    def cos(a):
        return ngs.cos(a * ngs.pi)

    rho = {'c':   1.0, 'x': (0.1, 0.75, sin), 'y': (0.15,  1.0, cos), 'xy': (0.08, 1.25, cos), 't': (0.05, sin)}
    u = {'c':  70.0, 'x': (4.0,  5/3, sin), 'y': (-12.0,  1.5, cos), 'xy': (7.0,  0.6, cos), 't': (2.0, cos)}
    v = {'c':  90.0, 'x': (-20.0,  1.5, cos), 'y': (4.0,  1.0, sin), 'xy': (-11.0,  0.9, cos), 't': (4.0, sin)}
    p = {'c': 1.0e5, 'x': (-0.3e5, 1.0, cos), 'y': (0.2e5, 1.25, sin),
         'xy': (-0.25e5, 0.75, sin), 't': (0.1e5, cos)}
    U_inf = mms.cfg.get_farfield_fields((u['c'], v['c']))

    def to_dimensionless(*U):

        ref = ngs.sqrt(sum(u['c']**2 for u in U))
        cf = []
        for u in U:

            val = u['c']/ref

            if 't' in u and t is not None:
                amp, func = u['t']
                period = mms.cfg.time.timer.interval[1] - mms.cfg.time.timer.interval[0]
                val += amp/ref * func(2*mms.periods/period * t)

            if 'x' in u:
                amp, k, func = u['x']
                val += amp/ref * func(k*ngs.x/mms.length)

            if 'y' in u:
                amp, k, func = u['y']
                val += amp/ref * func(k*ngs.y/mms.length)

            if 'xy' in u:
                amp, k, func = u['xy']
                val += amp/ref * func(k*ngs.y * ngs.x / mms.length**2)

            cf.append(val)

        return ngs.CF(tuple(cf))

    U = flowfields()
    U.rho = U_inf.rho * to_dimensionless(rho)
    U.u = mms.cfg.scaling.velocity * to_dimensionless(u, v)
    U.p = U_inf.p * to_dimensionless(p)

    U.rho_u = mms.cfg.momentum(U)
    U.T = mms.cfg.temperature(U)
    U.rho_Ei = mms.cfg.inner_energy(U)
    U.rho_Ek = mms.cfg.kinetic_energy(U)
    U.rho_E = mms.cfg.energy(U)

    return U


def get_gassner_mms(mms: MMS, t: float = None) -> flowfields:

    if t is None:
        t = 0.0

    c = 4.0
    gamma = mms.cfg.equation_of_state.heat_capacity_ratio

    k = 2*ngs.pi/mms.length
    T = mms.cfg.time.timer.interval[1] - mms.cfg.time.timer.interval[0]
    omega = 2*np.pi*mms.periods/T
    wave = ngs.sin(k*(ngs.x + ngs.y) - omega * t)

    U = flowfields()
    U.rho = 1.0 + wave/c
    U.u = 1/ngs.sqrt(2) * ngs.CF((1, 1))
    U.T = gamma/2 * (c - 1 + wave)

    U.rho_u = mms.cfg.momentum(U)
    U.rho_Ek = mms.cfg.kinetic_energy(U)
    U.p = mms.cfg.pressure(U)
    U.rho_Ei = mms.cfg.inner_energy(U)
    U.rho_E = mms.cfg.energy(U)

    return U


def time_refinement_routine(mesh: ngs.Mesh, schemes: list, *simulations: MMS, levels: int = 1):

    for simulation in simulations:

        # Define common solver configuration
        cfg = CompressibleFlowSolver(mesh)
        simulation.set_conditions(cfg)

        if not simulation.is_transient:
            raise ValueError("Time refinement routine only works for transient simulations!")

        simulation.open_output_streams()
        for scheme in schemes:
            simulation.cfg.fem.scheme = scheme

            time_steps = [schemes[scheme]/(2**i) for i in range(levels)]

            # Solve for different time integration schemes
            time_step_routine(simulation, time_steps, scheme=scheme)

        simulation.close_output_streams()


def time_step_routine(simulation: MMS, time_steps: list, **log):

    for dt in time_steps:
        simulation.cfg.time.timer.step = dt
        solution_routine(simulation, **log, dt=dt)


def mesh_refinement_routine(*simulations: MMS,
                            levels=4,
                            orders=[1, 2, 3, 4, 5],
                            maxh=0.5,
                            quad_dominated=True):

    for simulation in simulations:

        # Define initial mesh size
        mesh = ngs.Mesh(get_geometry(simulation.is_periodic).GenerateMesh(maxh=maxh, quad_dominated=quad_dominated))

        # Define common solver configuration
        cfg = CompressibleFlowSolver(mesh)
        simulation.set_conditions(cfg)

        simulation.open_output_streams()
        for level in range(levels):

            # Refine Mesh
            if level > 0:
                mesh.Refine()
                maxh *= 0.5

            polynomial_order_routine(simulation, orders=orders, level=level+1, h=maxh)

        simulation.close_output_streams()


def polynomial_order_routine(simulation: MMS, orders=[1, 2, 3, 4, 5], **log):

    for order in orders:
        simulation.cfg.fem.order = order

        # Solve for different polynomial orders
        solution_routine(simulation, **log, order=order)


def solution_routine(simulation: MMS, **log):

    cfg = simulation.cfg

    # Initialize the solver
    cfg.initialize()

    # Get solution fields
    Uh = cfg.get_solution_fields()
    simulation.set_solution_fields(Uh)

    if cfg.io.vtk.enable:
        simulation.open_vtk_stream(**log)

    if simulation.is_transient:
        simulation.write_to_streams(0.0, **log)

        old = {'bdf2': ['n-1'], 'bdf3': ['n-1', 'n-2']}
        if cfg.fem.scheme.name in old:
            for i, n in enumerate(old[cfg.fem.scheme.name], 1):
                Uold = simulation(t=-i*cfg.time.timer.step.Get())
                cfg.fem.scheme.gfus['U'][n].Set(ngs.CF((Uold.rho, Uold.rho_u, Uold.rho_E)))


    # Solve the system
    with ngs.TaskManager():

        for rate, t in enumerate(cfg.time.start_solution_routine()):
            simulation.write_to_streams(t, **log)

            if cfg.timestep_controller is not None:
                cfg.timestep_controller.process_iteration(rate)


def test_configuration(expected: dict, result: CompressibleFlowSolver):
    logger = result.io.log.logger
    result_ = result.to_dict()
    for key, value in expected.items():
        if key not in result_:
            raise ValueError(f"Setting {key} not found in configuration!")
        if value != result_[key]:
            logger.warning(f"Configuration {key} value differs! Expected: {value}, got: {result_[key]}")


def test_exact_solution(cfg):
    sim = EulerMMS(PSEUDO_STATIONARY_CFG, filename='test')

    # Define initial mesh size
    maxh = 0.5
    mesh = ngs.Mesh(get_geometry().GenerateMesh(maxh=maxh, quad_dominated=True))

    # Define common solver configuration
    cfg = CompressibleFlowSolver(mesh)
    cfg.update(PSEUDO_STATIONARY_CFG)
    sim.cfg = cfg
    cfg.io.path.mkdir(parents=True, exist_ok=True)

    sim.errors = {}
    sim.Ue = sim.get_exact_fields(cfg)
    for level in range(5):

        # Refine Mesh
        if level > 0:
            mesh.Refine()
            maxh *= 0.5

        for order in [1, 2, 3, 4, 5]:
            Uh = ngs.GridFunction(ngs.L2(mesh, order=order)**4)
            Uh.Set(ngs.CF((sim.Ue.rho, sim.Ue.rho_u, sim.Ue.rho_E)))
            Uh = flowfields(rho=Uh.components[0],
                            rho_u=ngs.CF((Uh.components[1], Uh.components[2])),
                            rho_E=Uh.components[3])
            Uh.u = cfg.velocity(Uh)
            Uh.rho_Ek = cfg.kinetic_energy(Uh)
            Uh.rho_Ei = cfg.inner_energy(Uh)
            Uh.p = cfg.pressure(Uh)
            Uh.rho_u = cfg.momentum(Uh)

            sim.set_solution_fields(Uh)
            sim.open_vtk_stream(level=level, order=order)
            sim.write_to_streams(level=level, h=maxh, order=order)

    sim.export_errors_to_fig()


if __name__ == "__main__":

    cfg = ngs.Mesh(get_geometry().GenerateMesh(maxh=1, quad_dominated=True))
    cfg = CompressibleFlowSolver(cfg)

    # Test configurations
    cfg.update(TRANSIENT_CFG)
    test_configuration(TRANSIENT_CFG, cfg)

    cfg.update(PSEUDO_STATIONARY_CFG)
    test_configuration(PSEUDO_STATIONARY_CFG, cfg)

    # Test convergence of exact solution
    test_exact_solution(cfg)
