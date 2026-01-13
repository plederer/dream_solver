# Import modules
import numpy as np
import ngsolve as ngs

from dream.compressible import CompressibleFlowSolver, Force, Initial, flowfields,  Dirichlet, InterfaceBC
from dream.time import TransientRoutine, SynchronizedIMEXTimeRoutine
from dream.mesh import get_rectangular_mesh
from dream.io import DomainL2Sensor
from time import time as clock


def get_meshes(x, yl, yt, quads: bool = True) -> ngs.Mesh:

    domains = (('explicit', (x, yt)),
               ('implicit', (x, yl)),
               )

    boundaries = (('bottom', (x, yl.min())),
                  ('right', (x.max(), (yl.min(), yt.max()))),
                  ('top', (x, yt.max())),
                  ('left', (x.min(), (yl.min(), yt.max()))),
                  ('interface', (x, yl.max())),
                  )

    mesh = get_rectangular_mesh(domains, boundaries, quads, False, False)

    domains = (('explicit', (x, yt)),
               )

    boundaries = (('right', (x.max(), yt)),
                  ('top', (x, yt.max())),
                  ('left', (x.min(), yt)),
                  ('interface', (x, yt.min())),
                  )

    explicit_mesh = get_rectangular_mesh(domains, boundaries, quads, False, False)

    domains = (('implicit', (x, yl)),
               )

    boundaries = (('bottom', (x, yl.min())),
                  ('right', (x.max(), yl)),
                  ('left', (x.min(), yl)),
                  ('interface', (x, yl.max())),
                  )

    implicit_mesh = get_rectangular_mesh(domains, boundaries, quads, False, False)

    return mesh, implicit_mesh, explicit_mesh


def get_uniform_meshes(Nx, Ny, quads: bool = True) -> ngs.Mesh:

    yt = np.linspace(0.0, 0.5, Ny//2 + 1)
    yl = np.linspace(-0.5, 0.0, Ny//2 + 1)
    x = np.linspace(-0.5, 0.5, Nx + 1)

    return get_meshes(x, yl, yt, quads)


def get_refined_meshes(Nx, Ny, quads: bool = True, refinements: int = 0) -> ngs.Mesh:

    dy = 2.0 / Ny
    series = np.concatenate([[0.0], np.power(0.5, tuple(i for i in range(1, refinements+2)))[::-1], [1.0]])

    yt = np.linspace(-0.5 + dy, 0.5, Ny - 2 + 1)
    yl = -0.5 + dy * series
    x = np.linspace(-0.5, 0.5, Nx + 1)

    return get_meshes(x, yl, yt, quads)


def div(F):
    return ngs.CF(tuple(F[i, 0].Diff(ngs.x) + F[i, 1].Diff(ngs.y) for i in range(F.dims[0])))



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
    'io.path': 'test',
    'io.vtk.enable': True,
    'io.vtk.rate': 10,
    'io.vtk.subdivision': 2,
    # 'io.log.enable': False,
}

PSEUDO_STATIONARY_CFG = {
    'reynolds_number': 1.0,
    'prandtl_number': 0.72,
    'equation_of_state.heat_capacity_ratio': 1.4,
    'riemann_solver': 'upwind',
    'scaling': 'aerodynamic',
    'time': 'pseudo_time_stepping',
    'time.timer.step': 1,
    'time.max_time_step': 1,
    'time.increment_at': 1,
    'time.increment_factor': 5,
    'fem': 'conservative_hdg',
    'fem.scheme': "implicit_euler",
    'fem.order': 3,
    'fem.solver': 'direct',
    'fem.solver.method': 'newton',
    'fem.solver.method.max_iterations': 5,
    'fem.solver.method.convergence_criterion': 1e-10,
    'fem.solver.method.damping_factor': 1.0,
    'fem.bonus_int_order': 4,
    'fem.scheme.compile': {'realcompile': False, 'wait': False, 'keep_files': False},
}


class MMS:

    def __init__(self,
                 cfg: dict,
                 mms: callable,
                 filename: str | None = None,
                 domain: str = "explicit|implicit",
                 boundaries: str = "bottom|top|left|right",
                 length: float = 1.0,
                 periods: int = 5):

        if filename is None:
            filename = f"{str(self)}"

        self._cfg = cfg
        self._mms = mms
        self.length = float(length)
        self.periods = int(periods)
        self.filename = filename
        self.domain = domain
        self.boundaries = boundaries

    @property
    def is_transient(self) -> bool:
        return isinstance(self.cfg.time, TransientRoutine)

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
        self._cfg.update(**cfgs)

        cfg.update(self._cfg)

        cfg.bcs.clear()
        cfg.dcs.clear()

        t = 0.0
        if self.is_transient:
            t = cfg.time.timer.t

        self.Ue = self.get_exact_fields(t)
        self.U0 = self.get_exact_fields(0.0)

        cfg.dcs[self.domain] = Initial(fields=self.U0)
        cfg.bcs[self.boundaries] = Dirichlet(fields=self.Ue)

    def set_solution_fields(self, Uh: flowfields):
        Uh['Ma'] = self.cfg.get_local_mach_number(Uh)
        Uh.rho_u = self.cfg.momentum(Uh)
        Uh.rho_E = self.cfg.energy(Uh)
        Uh.T = self.cfg.temperature(Uh)
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

        self.L2_sensor = f"L2_{filename}"

    def set_sensor_stream(self):

        self.cfg.io.sensor.list.clear()

        order = self.cfg.fem.order

        Ue = self.Ue
        Uh = self.Uh

        Ue_ = ngs.CF((Ue.rho, Ue.rho_u, Ue.rho_E))
        Uh_ = ngs.CF((Uh.rho, Uh.rho_u, Uh.rho_E))

        fields = {'rho': Uh.rho - Ue.rho, 'u': Uh.u - Ue.u, 'p': Uh.p - Ue.p,
                  'rho_u': Uh.rho_u - Ue.rho_u, 'rho_E': Uh.rho_E - Ue.rho_E, 'U': Uh_ - Ue_}
        sensor = DomainL2Sensor(fields, self.cfg.mesh, self.domain, name=self.L2_sensor,
                                integration_order=order + 10)

        self.cfg.io.sensor.add(sensor)

    def set_vtk_stream(self):
        export = ['density', 'velocity', 'pressure', 'temperature', 'Ma', 'energy']
        fields = {f"{key}_h": value for key, value in self.Uh.items() if key in export}
        fields.update({f"{key}_e": value for key, value in self.Ue.items() if key in export})
        self.cfg.io.vtk.fields = fields

    def __call__(self, t: float = None) -> flowfields:
        return self._mms(self, t)


class EulerMMS(MMS):

    def set_conditions(self, cfg: CompressibleFlowSolver, **cfgs):

        cfg.dynamic_viscosity = "inviscid"

        super().set_conditions(cfg, **cfgs)

        F = div(cfg.get_convective_flux(self.Ue))
        if self.is_transient:
            t = cfg.time.timer.t
            F += ngs.CF((self.Ue.rho.Diff(t), self.Ue.rho_u.Diff(t), self.Ue.rho_E.Diff(t)))

        cfg.dcs[self.domain] = Force(F[0], F[1:3], F[3], order=10, is_constant=not self.is_transient)

    def __str__(self):
        return "EE"


class NavierStokesMMS(MMS):

    def set_conditions(self, cfg: CompressibleFlowSolver, **cfgs):

        cfg.dynamic_viscosity = "constant"

        super().set_conditions(cfg, **cfgs)

        F = div(cfg.get_convective_flux(self.Ue) - cfg.get_diffusive_flux(self.Ue, self.Ue))
        if self.is_transient:
            t = cfg.time.timer.t
            F += ngs.CF((self.Ue.rho.Diff(t), self.Ue.rho_u.Diff(t), self.Ue.rho_E.Diff(t)))

        cfg.dcs[self.domain] = Force(F[0], F[1:3], F[3], order=10, is_constant=not self.is_transient)

    def __str__(self):
        return "NS"


def get_constant_mach_mms(mms: MMS, t: float = None) -> flowfields:

    k = 2*ngs.pi/mms.length
    wave = ngs.sin(k*(ngs.x + ngs.y))

    if t is not None:
        T = mms.cfg.time.timer.interval[1] - mms.cfg.time.timer.interval[0]
        omega = 2*np.pi*mms.periods/T
        wave = ngs.sin(k*(ngs.x + ngs.y) - omega * t)

    U_inf = mms.cfg.get_farfield_fields((1, 0))
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
    r""" The mach number is fixed by the choice of the conservative variables! 
    
        The average Reynolds number is approximately

        ..math::
            Re = \frac{4 \sqrt{2} 1}{0.01} = 565.685424949238
    """

    c = 4.0
    k = 2*ngs.pi/mms.length
    T = mms.cfg.time.timer.interval[1] - mms.cfg.time.timer.interval[0]
    omega = 2*np.pi*mms.periods/T
    wave = ngs.sin(k*(ngs.x + ngs.y) - omega * t)

    U = flowfields()

    U.rho = c + wave
    U.rho_u = (c + wave) * ngs.CF((1, 1))
    U.rho_E = (c + wave)**2

    U.rho_u = mms.cfg.momentum(U)
    U.rho_Ek = mms.cfg.kinetic_energy(U)
    U.rho_Ei = mms.cfg.inner_energy(U)
    U.p = mms.cfg.pressure(U)

    return U


def steady_convergence_routine(mesh: ngs.Mesh, simulation: MMS, levels: tuple = (0, 5)):

    # Define common solver configuration
    cfg = CompressibleFlowSolver(mesh)
    simulation.set_conditions(cfg)

    start, end = levels
    for _ in range(start - 1):
        mesh.Refine()

    for level in range(start, end):

        # Refine Mesh
        if level > 0:
            mesh.Refine()

        # Set filenames for output
        simulation.set_filenames(level=level)

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
            cfg.solve()
        end = clock()

        with cfg.io.path.joinpath(f"runtime_level{level}.txt").open("w") as file:
            file.write(f"{cfg.fem.scheme.name}_{cfg.time.timer.step.Get()}: {end - start}\n")


def transient_convergence_routine(mesh: ngs.Mesh, simulation: MMS, time_steps: tuple = (0.01,)):

    # Define common solver configuration
    cfg = CompressibleFlowSolver(mesh)
    simulation.set_conditions(cfg)

    if not simulation.is_transient:
        raise ValueError("Time refinement routine only works for transient simulations!")

    for dt in time_steps:
        cfg.time.timer.reset()
        cfg.time.timer.step = dt

        # Set filenames for output
        simulation.set_filenames(dt=dt)

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
            cfg.solve()
        end = clock()

        with cfg.io.path.joinpath(f"runtime_dt{dt}.txt").open("w") as file:
            file.write(f"{cfg.fem.scheme.name}_{cfg.time.timer.step.Get()}: {end - start}\n")


def transient_imex_convergence_routine(imp_mesh: ngs.Mesh, 
                                       exp_mesh: ngs.Mesh, 
                                       imp_simulation: MMS, 
                                       exp_simulation: MMS, 
                                       time_steps: tuple = (0.01,)):
    
    IMP = CompressibleFlowSolver(imp_mesh)
    EXP = CompressibleFlowSolver(exp_mesh)
    SIMP, SEXP = imp_simulation, exp_simulation

    routine = SynchronizedIMEXTimeRoutine(IMP, EXP)

    SIMP.set_conditions(IMP)
    SEXP.set_conditions(EXP)

    imp_bc = InterfaceBC(fields=None)
    exp_bc = InterfaceBC(fields=None)

    IMP.bcs['interface'] = imp_bc
    EXP.bcs['interface'] = exp_bc

    for dt in time_steps:
        IMP.time.timer.reset()
        EXP.time.timer.reset()
        IMP.time.timer.step = dt
        EXP.time.timer.step = dt

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
        SEXP.set_filenames(dt=dt)
        SIMP.set_filenames(dt=dt)

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
            for _ in routine.start_solution_routine():
                ...
        end = clock()

        with IMP.io.path.joinpath(f"runtime_dt{dt}.txt").open("w") as file:
            file.write(f"{IMP.fem.scheme.name}_{IMP.time.timer.step.Get()}: {end - start}\n")

if __name__ == "__main__":
    mesh, imp, exp = get_refined_meshes(32, 32, refinements=5)

