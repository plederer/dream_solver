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


TRANSIENT_CFG = {
    'mach_number': 0.2,
    'reynolds_number': 1.0,
    'prandtl_number': 0.72,
    'equation_of_state': 'ideal',
    'equation_of_state.heat_capacity_ratio': 1.4,
    'dynamic_viscosity': 'inviscid',
    'scaling': 'acoustic',
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
    'fem.mixed_method': 'inactive',
    'io.path': 'test',
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
}


class MMS:

    def __init__(self,
                 *configurations: dict,
                 filename: str | None = None,
                 flow_direction: tuple = (1.0, 1.0),
                 amplitude=0.1,
                 frequency=5.0,
                 length=1.0,
                 period=1.0,
                 init_exact: bool = False,
                 is_periodic: bool = False,
                 **export):

        if filename is None:
            filename = f"{str(self)}"

        if is_periodic:
            filename += "_periodic"

        self.configurations = configurations
        self.flow_direction = flow_direction
        self.amplitude = amplitude
        self.frequency = frequency
        self.length = length
        self.period = period
        self.filename = filename
        self.is_periodic = is_periodic
        self.init_exact = init_exact
        self.export = {'to_fig': True, 'to_csv': True, 'to_dat': True, 'to_vtk': True}
        self.export.update(export)

    @property
    def is_transient(self) -> bool:
        return isinstance(self.cfg.time, TransientRoutine)

    def get_exact_fields(self, cfg: CompressibleFlowSolver) -> flowfields:
        # Construct dimensionless fields

        U_inf = cfg.get_farfield_fields(self.flow_direction)
        U_inf.E = cfg.specific_energy(U_inf)

        t = 0
        if self.is_transient:
            t = cfg.time.timer.t

        U = self(t)
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

    def get_initial_fields(self, cfg: CompressibleFlowSolver) -> flowfields:
        # Construct dimensionless initial fields from constant terms only

        if isinstance(cfg, TransientRoutine) or self.init_exact:
            U = self(t=0)
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

        cfg.bcs.clear()
        cfg.dcs.clear()

        self.Ue = self.get_exact_fields(cfg)
        self.U0 = self.get_initial_fields(cfg)

    def set_solution_fields(self, Uh: flowfields):
        Uh['Ma'] = self.cfg.get_local_mach_number(Uh)
        self.Uh = Uh

    def write_to_output_streams(self):
        Ue = self.Ue
        Uh = self.Uh

        order = self.log['order']
        key = (self.log['level'], self.log['h'], self.log['order'])

        Ue_ = ngs.CF((Ue.rho, Ue.rho_u, Ue.rho_E))
        Uh_ = ngs.CF((Uh.rho, Uh.rho_u, Uh.rho_E))

        # weight = ngs.sin(ngs.pi * ngs.x/self.length) * ngs.sin(ngs.pi * ngs.y/self.length)
        # J_e = 4/ngs.pi**2

        self.errors[key] = {
            'rho': ngs.sqrt(ngs.Integrate((Uh.rho - Ue.rho) ** 2, self.cfg.mesh, order=order + 10)),
            'u': ngs.sqrt(ngs.Integrate(ngs.InnerProduct(Uh.u - Ue.u, Uh.u - Ue.u), self.cfg.mesh, order=order + 10)),
            'p': ngs.sqrt(ngs.Integrate((Uh.p - Ue.p) ** 2, self.cfg.mesh, order=order + 10)),
            'rho_u': ngs.sqrt(ngs.Integrate(ngs.InnerProduct(Uh.rho_u - Ue.rho_u, Uh.rho_u - Ue.rho_u), self.cfg.mesh, order=order + 10)),
            'rho_E': ngs.sqrt(ngs.Integrate((Uh.rho_E - Ue.rho_E) ** 2, self.cfg.mesh, order=order + 10)),
            'U': ngs.sqrt(ngs.Integrate(ngs.InnerProduct(Uh_ - Ue_, Uh_ - Ue_), self.cfg.mesh, order=order + 10)),
            # 'J(rho)': abs(ngs.Integrate(ngs.InnerProduct(Uh.rho, weight), self.cfg.mesh, order=order + 10) - J_e)
        }

        if not self.is_transient and self.export['to_vtk']:
            self.export_to_vtk()

    def open_output_streams(self):
        if any(self.export.values()):
            self.cfg.io.path.mkdir(parents=True, exist_ok=True)

        self.log = {'level': 0, 'h': 0.0, 'order': 0}

        self.errors = {}

    def close_output_streams(self):

        if self.export['to_csv']:
            self.export_errors_to_csv()

        if self.export['to_fig']:
            self.export_errors_to_fig()

        if self.export['to_dat']:
            self.export_errors_to_dat()

    def update_configurations(self, cfg: CompressibleFlowSolver, config: dict):
        cfg.update(config)

    def export_errors_to_csv(self):

        filepath = self.cfg.io.path.joinpath(f"{self.filename}.csv")

        with filepath.open('w', newline='') as file:

            LEVELS = max([level for level, _, _ in self.errors.keys()])

            file.write(f"levels (refinement): {LEVELS}\n")
            file.write("\n")
            file.write("------------------------------------------------------------------------\n")
            file.write("h, dt, eq, field, order, error\n")
            file.write("------------------------------------------------------------------------\n")

            errors = sorted(self.errors.items(), key=lambda x: x[0][2])
            for field in ['rho', 'u', 'p', 'rho_u', 'rho_E', 'U']:

                for key, error in errors:
                    level, h, order = key
                    dt = 1.0  # just a dummy, for the output.
                    row = [f"{h:.15e}", f"{dt:.15e}", str(self), f"{field:5}", str(order), f"{error[field]:.15e}"]
                    file.write(",\t".join(row) + "\n")

    def export_errors_to_fig(self):

        # L2 errors
        labels = {'rho': r"$\| \rho_h - \rho_e \|$", 'u': r"$\| \mathbf{u}_h - \mathbf{u}_e \|$",
                  'p': r"$\| p_h - p_e \|$", 'rho_u': r"$\| \rho \mathbf{u}_h - \rho \mathbf{u}_e \|$",
                  'rho_E': r"$\| \rho E_h - \rho E_e \|$", 'U': r"$\| \bm{U}_h - \bm{U}_e \|$"}

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        df = pd.DataFrame.from_dict(self.errors, orient='index')
        df.index.names = ['level', 'h', 'order']
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

        # Mean density functional
        # labels = {'J(rho)': r"$\| J(\rho_h) - J(\rho_e) \|$"}
        # fig, ax = plt.subplots(1, 1, figsize=(5, 5))

        # data = df.xs('J(rho)', axis=1)
        # for order in df.index.levels[2]:
        #     error = data.loc[:, :, order]
        #     ax.loglog(h, error, marker='o',  label=fr"$p={order}$")
        #     ax.loglog(h, h**(2*order)/h[0]**(2*order) * error.iloc[0], ls='--', color='k')

        #     ax.set_xlabel(r"$h$")
        #     ax.set_title(labels['J(rho)'])
        #     ax.legend()

        #     ax.set_ylim(1e-16, 2)

        # fig.savefig(self.cfg.io.path.joinpath(f"{self.filename}_J(rho).png"))

    def export_errors_to_dat(self):
        df = pd.DataFrame.from_dict(self.errors, orient='index')
        df.index.names = ['level', 'h', 'order']
        df.sort_index(axis=0, level=2, inplace=True)
        df.to_csv(self.cfg.io.path.joinpath(f"{self.filename}.dat"))

    def export_to_vtk(self, vtk: ngs.VTKOutput = None, t: float = -1):

        if vtk is None:
            filepath = self.cfg.io.path.joinpath(f"{self.filename}_{self.log['order']}_level{self.log["level"]}")
            vtk = ngs.VTKOutput(
                self.cfg.mesh, list(self.Uh.values()),
                list(self.Uh.keys()),
                filename=str(filepath),
                subdivision=3)

        vtk.Do(t)

    def __call__(self, t: float = 0) -> flowfields:

        k = 2*ngs.pi/self.length
        omega = 2*np.pi*self.frequency/self.period
        wave = ngs.sin(k*(ngs.x + ngs.y) - omega*t)

        U_inf = self.cfg.get_farfield_fields(self.flow_direction)
        U_inf.E = self.cfg.specific_energy(U_inf)

        U = flowfields()

        U.rho = U_inf.rho * (1 + self.amplitude * wave)
        U.rho_u = U.rho * U_inf.u
        U.rho_E = U.rho * U_inf.E

        U.u = self.cfg.velocity(U)
        U.rho_Ek = self.cfg.kinetic_energy(U)
        U.rho_Ei = self.cfg.inner_energy(U)
        U.p = self.cfg.pressure(U)
        U.T = self.cfg.temperature(U)
        return U


class EulerMMS(MMS):

    def update_configurations(self, cfg, config):
        cfg.dynamic_viscosity = "inviscid"
        super().update_configurations(cfg, config)

    def set_conditions(self, cfg: CompressibleFlowSolver):

        super().set_conditions(cfg)

        F = cfg.get_convective_flux(self.Ue)
        cfg.dcs['default'] = Initial(fields=self.U0)
        cfg.dcs['default'] = Force(flux=F, order=10)

        if self.is_periodic:
            cfg.bcs['left|right|top|bottom'] = "periodic"
        else:
            cfg.bcs['left|right|top|bottom'] = FarField(fields=self.Ue)

    def __str__(self):
        return "EE"


class NavierStokesMMS(MMS):

    def update_configurations(self, cfg, config):
        cfg.dynamic_viscosity = "constant"
        super().update_configurations(cfg, config)

    def set_conditions(self, cfg: CompressibleFlowSolver) -> flowfields:

        super().set_conditions(cfg)

        F = cfg.get_convective_flux(self.Ue) - cfg.get_diffusive_flux(self.Ue, self.Ue)
        cfg.dcs['default'] = Initial(fields=self.U0)
        cfg.dcs['default'] = Force(flux=F, order=10)

        if self.is_periodic:
            cfg.bcs['left|right|top|bottom'] = "periodic"
        else:
            cfg.bcs['left|right|top|bottom'] = Dirichlet(fields=self.Ue)

    def __str__(self):
        return "NS"


class RoyMMS(NavierStokesMMS):

    def __call__(self, t: float = 0) -> flowfields:

        U_inf = self.cfg.get_farfield_fields(self.flow_direction)

        def sin(a):
            return ngs.sin(a * ngs.pi/self.length)

        def cos(a):
            return ngs.cos(a * ngs.pi/self.length)

        rho = {'c':   1.0, 'x': (0.1, 0.75, sin), 'y': (0.15,  1.0, cos), 'xy': (0.08, 1.25, cos), 't': (0.1, 1.0, sin)}
        u = {'c':  70.0, 'x': (4.0,  5/3, sin), 'y': (-12.0,  1.5, cos), 'xy': (7.0,  0.6, cos), 't': (5, 1.0, sin)}
        v = {'c':  90.0, 'x': (-20.0,  1.5, cos), 'y': (4.0,  1.0, sin), 'xy': (-11.0,  0.9, cos), 't': (10, 1.0, sin)}
        p = {'c': 1.0e5, 'x': (-0.3e5, 1.0, cos),'y': (0.2e5, 1.25, sin),'xy': (-0.25e5, 0.75, sin), 't': (0.02e5, 1.0, sin)}

        def to_dimensionless(*U):

            ref = ngs.sqrt(sum(u['c']**2 for u in U))
            cf = []
            for u in U:

                val = u['c']/ref

                if 't' in u and t != 0:
                    amp, factor, func = u['t']
                    val += amp/ref * func(factor*ngs.pi/self.period * t)

                if 'x' in u:
                    amp, factor, func = u['x']
                    val += amp/ref * func(factor*ngs.x)

                if 'y' in u:
                    amp, factor, func = u['y']
                    val += amp/ref * func(factor*ngs.y)

                if 'xy' in u:
                    amp, factor, func = u['xy']
                    val += amp/ref * func(factor*ngs.y * ngs.x / self.length)

                cf.append(val)

            return ngs.CF(tuple(cf))

        U = flowfields()
        U.rho = U_inf.rho * to_dimensionless(rho)
        U.u = self.cfg.scaling.velocity * to_dimensionless(u, v)
        U.p = U_inf.p * to_dimensionless(p)

        U.rho_u = self.cfg.momentum(U)
        U.T = self.cfg.temperature(U)
        U.rho_Ei = self.cfg.inner_energy(U)
        U.rho_Ek = self.cfg.kinetic_energy(U)
        U.rho_E = self.cfg.energy(U)

        return U


def mesh_routine(*simulations: MMS,
                 levels=4,
                 orders=[1, 2, 3, 4, 5],
                 maxh=0.5,
                 quad_dominated=True):

    for simulation in simulations:

        # Define initial mesh size
        mesh = ngs.Mesh(get_geometry(simulation.is_periodic).GenerateMesh(maxh=maxh, quad_dominated=quad_dominated))

        # Define common solver configuration
        cfg = CompressibleFlowSolver(mesh)

        for sim_cfg in simulation.configurations:

            simulation.update_configurations(cfg, sim_cfg)
            simulation.set_conditions(cfg)

            simulation.open_output_streams()
            for level in range(levels):

                # Refine Mesh
                if level > 0:
                    mesh.Refine()
                    maxh *= 0.5

                simulation.log['level'] = level+1
                simulation.log['h'] = maxh
                simulation_routine(simulation, orders=orders)

            simulation.close_output_streams()


def simulation_routine(simulation: MMS, orders=[1, 2, 3, 4, 5]):

    for order in orders:
        simulation.log['order'] = order

        # Solve for different polynomial orders
        polyomial_order_routine(simulation, order)


def polyomial_order_routine(simulation: EulerMMS, order: int):

    cfg = simulation.cfg

    # Set polynomial order
    cfg.fem.order = order

    # Initialize the solver
    cfg.initialize()

    # Get solution fields
    Uh = cfg.get_solution_fields('rho_u', 'rho_E')
    simulation.set_solution_fields(Uh)

    # Solve the system
    with ngs.TaskManager():

        if simulation.is_transient:
            for t in cfg.time.start_transient_routine():
                simulation.write_to_output_streams_transient(t)

        else:
            cfg.solve()
            # Set solution fields
            simulation.write_to_output_streams()


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

            sim.log = {'level': level, 'h': maxh, 'order': order}
            sim.write_to_output_streams(Uh)

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
