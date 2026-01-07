# %%
import numpy as np
import ngsolve as ngs
from dream.compressible import CompressibleFlowSolver, flowfields, Initial, InterfaceBC
from dream.time import IMEXTimeRoutine, Timer
from dream.mesh import get_rectangular_mesh
from dream.io import DomainL2Sensor
from time import time as clock


def get_uniform_meshes(Nx, Ny, quads: bool = True) -> ngs.Mesh:

    def union(*args):
        x_ = args[0]
        for arg in args[1:]:
            x_ = np.union1d(x_, arg)
        return x_

    def iend(*args):
        return sum([arg.size for arg in args]) + 1 - len(args)

    def ibegin(*args):
        return sum([arg.size for arg in args]) - len(args)

    xl = np.linspace(-0.5, -0.25, Nx//4+1)
    xm = np.linspace(-0.25, 0.25, Nx//2+1)
    xr = np.linspace(0.25, 0.5, Nx//4+1)

    x = union(xl, xm, xr)
    y = np.linspace(-0.5, 0.5, Ny+1)

    ys = slice(0, iend(y))

    # Full mesh
    domains = {'explicit_left': [(slice(0, iend(xl)), ys)],
               'implicit': [(slice(ibegin(xl), iend(xl, xm)), ys)],
               'explicit_right': [(slice(ibegin(xl, xm), iend(x)), ys)]
               }

    boundaries = {'bottom': [(slice(0, iend(x)), 0)],
                  'right': [(ibegin(x), ys)],
                  'top': [(slice(0, iend(x)), ibegin(y))],  # , (slice(ibegin(xl, xm), iend(x)), ibegin(y)) ],
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


def get_squashed_meshes(Nx, Ny, quads: bool = True) -> ngs.Mesh:

    def union(*args):
        x_ = args[0]
        for arg in args[1:]:
            x_ = np.union1d(x_, arg)
        return x_

    def iend(*args):
        return sum([arg.size for arg in args]) + 1 - len(args)

    def ibegin(*args):
        return sum([arg.size for arg in args]) - len(args)

    xl = np.linspace(-0.5, -0.0625, Nx//2 - Nx//16 + 1)

    xm = [0.0]
    for i in [12, 6, 4, 2]:
        xm.append(xm[-1] + 0.0625/i)
    xm = np.array(xm)
    xm = union(-xm[::-1], xm)
    xr = np.linspace(0.0625, 0.5, Nx//2 - Nx//16 + 1)

    x = union(xl, xm, xr)
    y = np.linspace(-0.5, 0.5, Ny+1)

    ys = slice(0, iend(y))

    # Full mesh
    domains = {'explicit_left': [(slice(0, iend(xl)), ys)],
               'implicit': [(slice(ibegin(xl), iend(xl, xm)), ys)],
               'explicit_right': [(slice(ibegin(xl, xm), iend(x)), ys)]
               }

    boundaries = {'bottom': [(slice(0, iend(x)), 0)],
                  'right': [(ibegin(x), ys)],
                  'top': [(slice(0, iend(x)), ibegin(y))],  # , (slice(ibegin(xl, xm), iend(x)), ibegin(y)) ],
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

STAGE_TO_SCHEME = {
    1: ('implicit_euler', 'explicit_euler'),
    2: ('sdirk22', 'rk_ars22'),
    3: ('sdirk33', 'rk_ars33'),
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
        U['entropy'] = self.cfg.specific_entropy(U)

    def get_exact_fields(self, t: float) -> flowfields:
        # Construct dimensionless fields

        U = self(t)
        self._set_fields(U)

        U.grad_rho = ngs.CF((U.rho.Diff(ngs.x), U.rho.Diff(ngs.y)))
        U.grad_p = ngs.CF((U.p.Diff(ngs.x), U.p.Diff(ngs.y)))
        U.grad_u = ngs.CF((U.u[0].Diff(ngs.x), U.u[0].Diff(ngs.y),
                           U.u[1].Diff(ngs.x), U.u[1].Diff(ngs.y)), dims=(2, 2))
        U.grad_T = self.cfg.temperature_gradient(U, U)
        U['Ma'] = self.cfg.get_local_mach_number(U)

        return U

    def set_conditions(self, cfg: CompressibleFlowSolver, **cfgs):
        self.cfg = cfg

        cfgs['dynamic_viscosity'] = "inviscid"
        cfgs['scaling'] = "acoustic"

        self._cfg.update(**cfgs)
        cfg.update(self._cfg)

        cfg.bcs.clear()
        cfg.dcs.clear()

        self.ptimer = Timer()
        self.ptimer.t = cfg.time.timer.t.Get()
        self.Ue = self.get_exact_fields(self.ptimer.t)
        self.U0 = self.get_exact_fields(0.0)

        cfg.dcs[self.domain] = Initial(fields=self.U0, bonus_int_order=10)
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

    def set_sensor_stream(self):

        order = self.cfg.fem.order

        Ue = self.Ue
        Uh = self.Uh

        Ue_ = ngs.CF((Ue.rho, Ue.rho_u, Ue.rho_E))
        Uh_ = ngs.CF((Uh.rho, Uh.rho_u, Uh.rho_E))

        fields = {'rho': Uh.rho - Ue.rho, 'u': Uh.u - Ue.u, 'p': Uh.p - Ue.p,
                  'rho_u': Uh.rho_u - Ue.rho_u, 'rho_E': Uh.rho_E - Ue.rho_E, 'U': Uh_ - Ue_}
        sensor = DomainL2Sensor(fields, self.cfg.mesh, self.domain, name=f"L2_{self.filename}",
                                integration_order=order + 10)

        self.cfg.io.sensor.add(sensor)

    def set_vtk_stream(self):
        export = ['density', 'velocity', 'pressure', 'temperature', 'Ma', 'energy']
        fields = {f"{key}_h": value for key, value in self.Uh.items() if key in export}
        fields.update({f"{key}_e": value for key, value in self.Ue.items() if key in export})
        self.cfg.io.vtk.fields = fields

    def update_timer(self, t):
        raise NotImplementedError()

    def __call__(self, beta: float, R: float, t: float) -> flowfields:

        M = self.cfg.mach_number
        gamma = self.cfg.equation_of_state.heat_capacity_ratio

        def x(xc):
            return (ngs.x - xc) - M * t

        def y(yc):
            return (ngs.y - yc)

        rho = 1
        u = ngs.CF((M, 0))
        p = 1

        for xc in [-2, -1, 0, 1]:
            for yc in [-1, 0, 1]:
                f = -1/2 * ((x(xc)**2 + y(yc)**2)/R**2)
                Omega = beta * ngs.exp(f)

                dT = -(gamma - 1)/2 * Omega**2
                du = -y(yc)/R * Omega
                dv = x(xc)/R * Omega

                rho += dT
                u += ngs.CF((du, dv))
                p += dT

        U = flowfields()
        U.rho = rho**(1/(gamma - 1))
        U.u = u
        U.p = 1/gamma * p**(gamma/(gamma - 1))

        return U

    def __str__(self):
        return self.__class__.__name__


class FastVortex(Vortex):

    def set_conditions(self, cfg: CompressibleFlowSolver, **cfgs):

        cfgs['mach_number'] = 0.5
        super().set_conditions(cfg, **cfgs)

    def update_timer(self, t: float):
        self.ptimer.t = t % 2.0

    def __call__(self, t: float) -> flowfields:
        return super().__call__(1/5, 0.1, t)


class SlowVortex(Vortex):

    def set_conditions(self, cfg: CompressibleFlowSolver, **cfgs):

        cfgs['mach_number'] = 0.05
        super().set_conditions(cfg, **cfgs)

    def __call__(self, t: float) -> flowfields:
        return super().__call__(1/50, 0.1, t)


def single_transient_routine(simulation: Vortex):

    # Define common solver configuration
    cfg = simulation.cfg

    # Set filenames for output
    simulation.set_filenames()

    # Initialize the solver
    cfg.initialize()

    # Get solution fields
    Uh = cfg.get_solution_fields()
    simulation.set_solution_fields(Uh)

    if cfg.io.vtk.enable:
        simulation.set_vtk_stream()

    if cfg.io.sensor.enable:
        simulation.set_sensor_stream()

    # Solve the system
    start = clock()
    with ngs.TaskManager():
        for _, t in enumerate(cfg.time.start_solution_routine()):
            simulation.update_timer(t)
    end = clock()

    with cfg.io.path.joinpath("runtime.txt").open("w") as file:
        file.write(f"{cfg.fem.scheme.name}_{cfg.time.timer.step.Get()}: {end - start}\n")


def imex_transient_routine(routine: IMEXTimeRoutine, *simulations: Vortex):

    SIMP, SEXP = simulations
    IMP, EXP = tuple(sim.cfg for sim in simulations)

    imp_bc = InterfaceBC(fields=None)
    exp_bc = InterfaceBC(fields=None)

    IMP.bcs['interface'] = imp_bc
    EXP.bcs['interface'] = exp_bc

    EXP.fem.initialize_finite_element_spaces()
    EXP.fem.initialize_trial_and_test_functions()
    EXP.fem.initialize_gridfunctions()
    EXP.fem.initialize_time_scheme_gridfunctions()

    Uh_exp = EXP.get_all_solution_fields()
    SEXP.set_solution_fields(Uh_exp)
    imp_bc.fields = Uh_exp

    IMP.fem.initialize_finite_element_spaces()
    IMP.fem.initialize_trial_and_test_functions()
    IMP.fem.initialize_gridfunctions()
    IMP.fem.initialize_time_scheme_gridfunctions()

    Uh_imp = IMP.get_all_solution_fields()
    SIMP.set_solution_fields(Uh_imp)
    exp_bc.fields = Uh_imp

    # Set filenames for output
    SEXP.set_filenames()
    SIMP.set_filenames()

    EXP.fem.set_boundary_conditions()
    EXP.fem.set_initial_conditions()
    EXP.fem.initialize_symbolic_forms()

    IMP.fem.set_boundary_conditions()
    IMP.fem.set_initial_conditions()
    IMP.fem.initialize_symbolic_forms()

    if IMP.io.vtk.enable:
        SEXP.set_vtk_stream()
        SIMP.set_vtk_stream()

    if IMP.io.sensor.enable:
        SEXP.set_sensor_stream()
        SIMP.set_sensor_stream()

    # Solve the system
    start = clock()
    with ngs.TaskManager():
        for t in routine.start_solution_routine():
            for sim in simulations:
                sim.update_timer(t)
    end = clock()

    with IMP.io.path.joinpath("runtime.txt").open("w") as file:
        file.write(f"{IMP.fem.scheme.name}_{IMP.time.timer.step.Get()}: {end - start}\n")


if __name__ == "__main__":
    from ngsolve.webgui import Draw
    mesh, implicit_mesh, explicit_mesh = get_squashed_meshes(32, 32)
    Draw(mesh)
    Draw(implicit_mesh)
    Draw(explicit_mesh)

# %%
