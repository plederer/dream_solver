# %%
import numpy as np
import ngsolve as ngs
from netgen import occ
from dream.compressible import CompressibleFlowSolver, flowfields, Initial, InterfaceBC
from dream.time import MultizoneIMEXTimeRoutine, LocalTimeIMEXRoutine
from dream.mesh import get_rectangular_mesh
import matplotlib.pyplot as plt
import pandas as pd

def get_meshes(Nx = 5, Ny = 10, Nl = 4, quads: bool = True) -> ngs.Mesh:

    def union(*args):
        x_ = args[0]
        for arg in args[1:]:
            x_ = np.union1d(x_, arg)
        return x_

    def iend(*args):
        return sum([arg.size for arg in args]) + 1 - len(args)

    def ibegin(*args):
        return sum([arg.size for arg in args]) - len(args)
    
    xl = np.linspace(-0.25, 0.25, Nx+1)
    xm = union(np.linspace(0.25, 0.75, Nx+1))#, np.linspace(0.45, 0.55, Nl+1))
    xr = np.linspace(0.75, 1.25, Nx+1)

    x = union(xl, xm, xr)
    y = union(np.linspace(-0.5, 0.5, Ny+1), np.linspace(-0.05, 0.05, Nl+1))

    ys = slice(0, iend(y))

    # Full mesh
    domains = {'explicit_left': [(slice(0, iend(xl)), ys)],
               'implicit': [(slice(ibegin(xl), iend(xl, xm)), ys)],
               'explicit_right': [(slice(ibegin(xl, xm), iend(x)), ys)]
               }

    boundaries = {'bottom': [(slice(0, iend(x)), 0)],
                  'right': [(ibegin(x), ys)],
                  'top': [(slice(0, iend(x)), ibegin(y))],#, (slice(ibegin(xl, xm), iend(x)), ibegin(y)) ],
                  'left': [(0, slice(0, iend(y)))],
                  'interface': [(i-1, slice(y.size)) for i in [iend(xl), iend(xl, xm)]]
                  }
    
    mesh = get_rectangular_mesh(x, y, domains, boundaries, quads, True, True)

    # Explicit mesh
    domains = {'explicit_left': [(slice(0, iend(xl)), ys)],
               'explicit_right': [(slice(ibegin(xl, xm), iend(x)), ys)]
               }

    boundaries = {'bottom': [(slice(*i), 0) for i in [(0, iend(xl)), (ibegin(xl, xm), iend(x))]],
                  'right': [(ibegin(x), ys)],
                  'top': [(slice(*i), ibegin(y)) for i in [(0, iend(xl)), (ibegin(xl, xm), iend(x))]],
                  'left': [(0, slice(0, iend(y)))],
                  'interface': [(i, slice(y.size)) for i in [ibegin(xl), ibegin(xl, xm)]]
                  }

    explicit_mesh = get_rectangular_mesh(x, y, domains, boundaries, quads, True, True)

    # Implicit mesh
    domains = {'implicit': [(slice(ibegin(xl), iend(xl, xm)), ys)]}

    boundaries = {'bottom': [(slice(*i), 0) for i in [(ibegin(xl), iend(xl, xm))]],
                  'top': [(slice(*i), ibegin(y)) for i in [(ibegin(xl), iend(xl, xm))]],
                  'interface': [(i, slice(y.size)) for i in [ibegin(xl), ibegin(xl, xm)]]
                  }

    implicit_mesh = get_rectangular_mesh(x, y, domains, boundaries, quads, False, True)
    return mesh, implicit_mesh, explicit_mesh


def get_geometry():

    wp = occ.WorkPlane()
    centers = [0.0, 1.0, 2.0]
    faces = [wp.MoveTo(x_, 0).RectangleC(1, 1).Face() for x_ in centers]

    def assign_meta(face, name: str, left: str, right: str, maxh: float = 1.0):
        face.name = name
        for edge, name in zip(face.edges, ('bottom', right, 'top', left)):
            edge.name = name
        face.maxh = maxh

        face.edges[0].Identify(face.edges[2], f"periodic_y_{name}", occ.IdentificationType.PERIODIC)

    assign_meta(faces[0], 'explicit_left', 'left', 'interface')
    assign_meta(faces[1], 'implicit', 'interface', 'interface')
    assign_meta(faces[2], 'explicit_right', 'interface', 'right')

    faces[0].edges[3].Identify(faces[2].edges[1], f"periodic_x", occ.IdentificationType.PERIODIC)

    face = occ.Glue(faces)
    return occ.OCCGeometry(face, dim=2)


TRANSIENT_CFG = {
    'reynolds_number': 1.0,
    'prandtl_number': 0.72,
    'equation_of_state': 'ideal',
    'equation_of_state.heat_capacity_ratio': 1.4,
    'riemann_solver': 'upwind',
    'time': 'transient',
    'time.timer.interval': (0.0, 1.0),
    'time.timer.step': 0.1,
    'fem': 'conservative_hdg',
    'fem.scheme': 'implicit_euler',
    'fem.order': 3,
    'fem.solver': 'direct',
    'fem.solver.method': 'newton',
    'fem.solver.method.max_iterations': 5,
    'fem.solver.method.convergence_criterion': 1e-10,
    'fem.solver.method.damping_factor': 1.0,
    'fem.bonus_int_order': 4,
    'fem.scheme.compile': {'realcompile': False, 'wait': False, 'keep_files': False},
    'fem.viscous_treatment': None,
    'io.path': 'test2',
    'io.vtk.enable': True,
    'io.vtk.rate': 10,
    'io.vtk.subdivision': 2,
    # 'io.log.enable': False,
}

SCHEME_ORDER = {
    'implicit_euler': 1,
    'explicit_euler': 1,
    'bdf2': 2,
    'bdf3': 3,
    'rk_ars22': 2,
    'rk_ars33': 3,
    'rk_ars43': 3,
    'crk4': 4,
    'sdirk22': 2,
    'sdirk33': 3,
    'sdirk43': 3,
    'sdirk54': 4
}


class Vortex:

    def __init__(self,
                 cfg: dict,
                 filename: str | None = None,
                 domain: str = "explicit_left|implicit|explicit_right",
                 periodic: str = "bottom|top|left|right",
                 **export):

        if filename is None:
            filename = f"{str(self)}"

        self._cfg = cfg
        self.filename = filename
        self.domain = domain
        self.periodic = periodic
        self.export = {'to_fig': True, 'to_csv': True, 'to_dat': True}
        self.export.update(export)

    def _set_fields(self, U):
        U.rho = self.cfg.density(U)
        U.u = self.cfg.velocity(U)
        U.rho_u = self.cfg.momentum(U)
        U.rho_Ek = self.cfg.kinetic_energy(U)
        U.rho_Ei = self.cfg.inner_energy(U)
        U.p = self.cfg.pressure(U)
        U.T = self.cfg.temperature(U)
        U.rho_E = self.cfg.energy(U)

    def get_exact_fields(self, cfg: CompressibleFlowSolver) -> flowfields:
        # Construct dimensionless fields

        t = cfg.time.timer.t
        U = self(t)
        self._set_fields(U)

        U.grad_rho = ngs.CF((U.rho.Diff(ngs.x), U.rho.Diff(ngs.y)))
        U.grad_p = ngs.CF((U.p.Diff(ngs.x), U.p.Diff(ngs.y)))
        U.grad_u = ngs.CF((U.u[0].Diff(ngs.x), U.u[0].Diff(ngs.y),
                           U.u[1].Diff(ngs.x), U.u[1].Diff(ngs.y)), dims=(2, 2))
        U.grad_T = cfg.temperature_gradient(U, U)
        U['Ma'] = cfg.get_local_mach_number(U)

        return U

    def get_initial_fields(self, cfg: CompressibleFlowSolver) -> flowfields:
        # Construct dimensionless initial fields from constant terms only

        U = self(t=0.0)
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

    def set_conditions(self, cfg: CompressibleFlowSolver, **cfgs):
        self.cfg = cfg

        cfgs['dynamic_viscosity'] = "inviscid"
        cfgs['scaling'] = "acoustic"

        self._cfg.update(**cfgs)
        cfg.update(self._cfg)

        cfg.bcs.clear()
        cfg.dcs.clear()

        self.Ue = self.get_exact_fields(cfg)
        self.U0 = self.get_initial_fields(cfg)

        cfg.dcs[self.domain] = Initial(fields=self.U0, bonus_int_order=0)
        cfg.bcs[self.periodic] = "periodic"

    def set_solution_fields(self, Uh: flowfields):
        self._set_fields(Uh)
        Uh['Ma'] = self.cfg.get_local_mach_number(Uh)
        self.Uh = Uh

    def set_filenames(self, **log):
        filename = self.filename
        for key, value in log.items():
            if isinstance(value, str):
                filename += f"_{value}"
            else:
                filename += f"_{key}{value}"

        self.cfg.io.vtk.filename = filename
        self.cfg.io.gfu.filename = filename

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

    def open_vtk_stream(self):
        export = ['density', 'velocity', 'pressure', 'temperature', 'Ma', 'energy']
        fields = {f"{key}_h": value for key, value in self.Uh.items() if key in export}
        fields.update({f"{key}_e": value for key, value in self.Ue.items() if key in export})
        self.cfg.io.vtk.fields = fields

    def close_output_streams(self):

        if self.export['to_fig']:
            self.export_errors_to_fig()

        if self.export['to_dat']:
            self.export_errors_to_dat()

    def export_errors_to_fig(self):

        # L2 errors
        labels = {'rho': r"$\| \rho_h - \rho_e \|$", 'u': r"$\| \mathbf{u}_h - \mathbf{u}_e \|$",
                  'p': r"$\| p_h - p_e \|$", 'rho_u': r"$\| \rho \mathbf{u}_h - \rho \mathbf{u}_e \|$",
                  'rho_E': r"$\| \rho E_h - \rho E_e \|$", 'U': r"$\| \mathbf{U}_h - \mathbf{U}_e \|$"}

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        df = pd.DataFrame.from_dict(self.errors, orient='index')
        df.index.names = list(self.log)

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
        df.to_csv(self.cfg.io.path.joinpath(f"{self.filename}.dat"))

    def __call__(self, beta: float, R: float, t: float) -> flowfields:

        M = self.cfg.mach_number
        gamma = self.cfg.equation_of_state.heat_capacity_ratio

        x = ngs.x - M * t
        y = ngs.y
        alpha = 0 * ngs.pi/360

        f = -1/2 * ((x**2 + y**2)/R**2)
        Omega = beta * ngs.exp(f)

        dT = -(gamma - 1)/2 * Omega**2
        du = -y/R * Omega
        dv = x/R * Omega

        U = flowfields()
        U.rho = (1 + dT)**(1/(gamma - 1))
        U.u = M*ngs.CF((ngs.cos(alpha), ngs.sin(alpha))) + ngs.CF((du, dv))
        U.p = 1/gamma * (1 + dT)**(gamma/(gamma - 1))

        return U

    def __str__(self):
        return self.__class__.__name__


class FastVortex(Vortex):

    def set_conditions(self, cfg: CompressibleFlowSolver, **cfgs):

        cfgs['mach_number'] = 0.5
        super().set_conditions(cfg, **cfgs)

    def __call__(self, t: float) -> flowfields:
        return super().__call__(1/5, 0.1, t)


class SlowVortex(Vortex):

    def set_conditions(self, cfg: CompressibleFlowSolver, **cfgs):

        cfgs['mach_number'] = 0.05
        super().set_conditions(cfg, **cfgs)

    def __call__(self, t: float) -> flowfields:
        return super().__call__(1/50, 0.1, t)


def multizone_imex_time_refinement_routine(explicit_mesh: ngs.Mesh, implicit_mesh: ngs.Mesh,
                                           explicit_sim: Vortex, implicit_sim: Vortex,
                                           pair_schemes: list, levels: int = 1):

    # Define common solver configuration
    EXP = CompressibleFlowSolver(explicit_mesh)
    IMP = CompressibleFlowSolver(implicit_mesh)

    time = MultizoneIMEXTimeRoutine(IMP, EXP)
    explicit_sim.set_conditions(EXP, time=time)
    implicit_sim.set_conditions(IMP, time=time)

    imp_bc = InterfaceBC(fields=None)
    exp_bc = InterfaceBC(fields=None)

    IMP.bcs['interface'] = imp_bc
    EXP.bcs['interface'] = exp_bc

    explicit_sim.open_output_streams()
    implicit_sim.open_output_streams()
    for scheme in pair_schemes:
        implicit_scheme, explicit_scheme = scheme
        IMP.fem.scheme = implicit_scheme
        EXP.fem.scheme = explicit_scheme

        time_steps = [pair_schemes[scheme]/(2**i) for i in range(levels)]

        EXP.fem.initialize_finite_element_spaces()
        EXP.fem.initialize_trial_and_test_functions()
        EXP.fem.initialize_gridfunctions()
        EXP.fem.initialize_time_scheme_gridfunctions()

        Uh_exp = EXP.get_all_solution_fields()
        explicit_sim.set_solution_fields(Uh_exp)
        imp_bc.fields = Uh_exp

        IMP.fem.initialize_finite_element_spaces()
        IMP.fem.initialize_trial_and_test_functions()
        IMP.fem.initialize_gridfunctions()
        IMP.fem.initialize_time_scheme_gridfunctions()

        Uh_imp = IMP.get_all_solution_fields()
        implicit_sim.set_solution_fields(Uh_imp)
        exp_bc.fields = Uh_imp

        rate_imp = IMP.io.gfu.rate
        rate_exp = EXP.io.gfu.rate
        for dt in time_steps:
            time.timer.t = 0.0
            time.timer.step = dt

            # Set filenames for output
            explicit_sim.set_filenames(scheme=f"{explicit_scheme}", dt=dt)
            implicit_sim.set_filenames(scheme=f"{implicit_scheme}", dt=dt)

            EXP.fem.set_boundary_conditions()
            EXP.fem.set_initial_conditions()
            EXP.fem.initialize_symbolic_forms()

            IMP.fem.set_boundary_conditions()
            IMP.fem.set_initial_conditions()
            IMP.fem.initialize_symbolic_forms()

            if IMP.io.vtk.enable:
                explicit_sim.open_vtk_stream()
                implicit_sim.open_vtk_stream()

            explicit_sim.write_to_streams(0.0, scheme=f"{explicit_scheme}", dt=dt)
            implicit_sim.write_to_streams(0.0, scheme=f"{implicit_scheme}", dt=dt)

            # Solve the system
            with ngs.TaskManager():

                for t in time.start_solution_routine():
                    explicit_sim.write_to_streams(t, scheme=f"{explicit_scheme}", dt=dt)
                    implicit_sim.write_to_streams(t, scheme=f"{implicit_scheme}", dt=dt)

            IMP.io.gfu.rate *= 2
            EXP.io.gfu.rate *= 2

        IMP.io.gfu.rate = rate_imp
        EXP.io.gfu.rate = rate_exp

    explicit_sim.close_output_streams()
    implicit_sim.close_output_streams()


def local_imex_time_refinement_routine(explicit_mesh: ngs.Mesh, implicit_mesh: ngs.Mesh,
                                       explicit_sim: Vortex, implicit_sim: Vortex,
                                       pair_schemes: list, levels: int = 1):

    # Define common solver configuration
    EXP = CompressibleFlowSolver(explicit_mesh)
    IMP = CompressibleFlowSolver(implicit_mesh)

    time = LocalTimeIMEXRoutine(IMP, EXP)
    explicit_sim.set_conditions(EXP, time="transient")
    implicit_sim.set_conditions(IMP, time="transient")

    imp_bc = InterfaceBC(fields=None)
    exp_bc = InterfaceBC(fields=None)

    IMP.bcs['interface'] = imp_bc
    EXP.bcs['interface'] = exp_bc

    explicit_sim.open_output_streams()
    implicit_sim.open_output_streams()
    for scheme in pair_schemes:
        implicit_scheme, explicit_scheme = scheme
        IMP.fem.scheme = implicit_scheme
        EXP.fem.scheme = explicit_scheme

        ratio, dt_max = pair_schemes[scheme]

        time_steps = [dt_max/(2**i) for i in range(levels)]

        EXP.fem.initialize_finite_element_spaces()
        EXP.fem.initialize_trial_and_test_functions()
        EXP.fem.initialize_gridfunctions()
        EXP.fem.initialize_time_scheme_gridfunctions()

        Uh_exp = EXP.get_all_solution_fields()
        explicit_sim.set_solution_fields(Uh_exp)
        imp_bc.fields = Uh_exp

        IMP.fem.initialize_finite_element_spaces()
        IMP.fem.initialize_trial_and_test_functions()
        IMP.fem.initialize_gridfunctions()
        IMP.fem.initialize_time_scheme_gridfunctions()

        Uh_imp = IMP.get_all_solution_fields()
        implicit_sim.set_solution_fields(Uh_imp)
        exp_bc.fields = Uh_imp

        rate_imp = IMP.io.gfu.rate
        rate_exp = EXP.io.gfu.rate
        rate_imp_vtk = IMP.io.vtk.rate
        rate_exp_vtk = EXP.io.vtk.rate
        for dt in time_steps:
            EXP.time.timer.t = 0.0
            IMP.time.timer.t = 0.0
            EXP.time.timer.step = dt
            IMP.time.timer.step = ratio * dt

            # Set filenames for output
            explicit_sim.set_filenames(scheme=f"{explicit_scheme}", dt=dt)
            implicit_sim.set_filenames(scheme=f"{implicit_scheme}", dt=dt)

            EXP.fem.set_boundary_conditions()
            EXP.fem.set_initial_conditions()
            EXP.fem.initialize_symbolic_forms()

            IMP.fem.set_boundary_conditions()
            IMP.fem.set_initial_conditions()
            IMP.fem.initialize_symbolic_forms()

            if IMP.io.vtk.enable:
                explicit_sim.open_vtk_stream()
                implicit_sim.open_vtk_stream()

            explicit_sim.write_to_streams(0.0, scheme=f"{explicit_scheme}", dt=dt)
            implicit_sim.write_to_streams(0.0, scheme=f"{implicit_scheme}", dt=dt)

            # Solve the system
            with ngs.TaskManager():

                for t in time.start_solution_routine():
                    explicit_sim.write_to_streams(t, scheme=f"{explicit_scheme}", dt=dt)
                    implicit_sim.write_to_streams(t, scheme=f"{implicit_scheme}", dt=dt)

            IMP.io.gfu.rate *= 2
            EXP.io.gfu.rate *= 2
            IMP.io.vtk.rate *= 2
            EXP.io.vtk.rate *= 2

        IMP.io.gfu.rate = rate_imp
        EXP.io.gfu.rate = rate_exp
        IMP.io.vtk.rate = rate_imp_vtk
        EXP.io.vtk.rate = rate_exp_vtk

    explicit_sim.close_output_streams()
    implicit_sim.close_output_streams()


def time_refinement_routine(mesh: ngs.Mesh, schemes: list, simulation: Vortex, levels: int = 1):

    # Define common solver configuration
    cfg = CompressibleFlowSolver(mesh)
    simulation.set_conditions(cfg)

    simulation.open_output_streams()
    for scheme in schemes:
        cfg.fem.scheme = scheme

        time_steps = [schemes[scheme]/(2**i) for i in range(levels)]

        # Solve for different time integration schemes
        rate = cfg.io.gfu.rate
        for dt in time_steps:

            cfg.time.timer.t = 0.0
            cfg.time.timer.step = dt

            # Set filenames for output
            simulation.set_filenames(scheme=scheme, dt=dt)

            # Initialize the solver
            cfg.initialize()

            # Get solution fields
            Uh = cfg.get_solution_fields()
            simulation.set_solution_fields(Uh)

            if cfg.io.vtk.enable:
                simulation.open_vtk_stream()

            simulation.write_to_streams(0.0, scheme=scheme, dt=dt)

            # Solve the system
            with ngs.TaskManager():

                for rate, t in enumerate(cfg.time.start_solution_routine()):
                    simulation.write_to_streams(t, scheme=scheme, dt=dt)

            simulation.cfg.io.gfu.rate *= 2
        simulation.cfg.io.gfu.rate = rate

    simulation.close_output_streams()


if __name__ == "__main__":
    from ngsolve.webgui import Draw
    mesh, implicit_mesh, explicit_mesh = get_meshes()
    Draw(mesh)
    Draw(implicit_mesh)
    Draw(explicit_mesh)
