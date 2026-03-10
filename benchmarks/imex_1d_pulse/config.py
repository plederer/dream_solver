# %%
import numpy as np
import ngsolve as ngs
from dream.bla import fixpoint_iteration
from dream.mesh import get_rectangular_mesh
from dream.compressible import CompressibleFlowSolver, Initial, flowfields,  InterfaceBC
from dream.io import BoundarySensor
from dream.time import SynchronizedIMEXTimeRoutine
from pathlib import Path


def get_coordinates(N, Ni, dxi0) -> ngs.Mesh:

    N = N//2
    Ni = Ni//2
    Ne = N - Ni

    y = np.linspace(-1, 1, 3)
    if np.isclose(dxi0, 1.0):
        x = np.linspace(-N, N, 2*N+1)
        return x, y
    
    x = np.zeros(N + 1)
    geom_i = np.power(1/dxi0, 1/Ni)
    xi = dxi0 * np.power(geom_i, np.arange(Ni))
    x[1:Ni+1] = np.cumsum(xi)

    Lxi = N - x[Ni]

    def fp_geometrical(x0):
        return np.power(1+Lxi*x0, 1/Ne) - 1

    geom_e = fixpoint_iteration(0.3, fp_geometrical, it=1000, tol=1e-16)
    xe = np.power(1+geom_e, np.arange(Ne))

    x[Ni+1:] = x[Ni]
    x[Ni+1:] += np.cumsum(xe)

    return np.union1d(-x, x), y


def get_single_mesh(N, Ni, dxi0) -> ngs.Mesh:

    x, y = get_coordinates(N, Ni, dxi0)

    left = slice(0, N//2-Ni//2+1)
    middle = slice(N//2-Ni//2, N//2+Ni//2+1)
    right = slice(N//2+Ni//2, N+1)

    domains = (
        ('explicit', (x[left], y)),
        ('implicit', (x[middle], y)),
        ('explicit', (x[right], y)),
    )

    boundaries = (('left', (x[0], y)),
                  ('right', (x[-1], y)),
                  ('top', (x, y[-1])),
                  ('bottom', (x, y[0])))

    mesh = get_rectangular_mesh(domains, boundaries, True, True, True)

    domains = (
        ('explicit', (x[left], y)),
        ('explicit', (x[right], y)),
    )

    boundaries = (('left', (x[0], y)),
                  ('right', (x[-1], y)),
                  ('top', (x[left], y[-1])),
                  ('top', (x[right], y[-1])),
                  ('bottom', (x[left], y[0])),
                  ('bottom', (x[right], y[0])),
                  ('interface', (x[N//2-Ni//2], y)),
                  ('interface', (x[N//2+Ni//2], y)))

    exp = get_rectangular_mesh(domains, boundaries, True, True, True)

    domains = (
        ('implicit', (x[middle], y)),
    )

    boundaries = (
        ('top', (x[middle], y[-1])),
        ('bottom', (x[middle], y[0])),
        ('interface', (x[N//2-Ni//2], y)),
        ('interface', (x[N//2+Ni//2], y)))

    imp = get_rectangular_mesh(domains, boundaries, True, False, True)

    return mesh, imp, exp


TRANSIENT_CFG = {
    'reynolds_number': 150.0,
    'prandtl_number': 0.72,
    'mach_number': 0.2,
    'equation_of_state': 'ideal',
    'equation_of_state.heat_capacity_ratio': 1.4,
    'riemann_solver': 'upwind',
    'dynamic_viscosity': 'constant',
    'scaling': 'aerodynamic',
    'time': 'transient',
    'time.timer.interval': (0.0, 50.0),
    'time.timer.step': 0.1,
    'fem': 'conservative_hdg',
    'fem.scheme': 'implicit_euler',
    'fem.scheme.compile': {'realcompile': False, 'wait': False, 'keep_files': False},
    'fem.order': 3,
    'fem.solver': 'direct',
    'fem.solver.method': 'newton',
    'fem.solver.method.max_iterations': 5,
    'fem.solver.method.convergence_criterion': 1e-14,
    'fem.solver.method.damping_factor': 1.0,
    'fem.bonus_int_order': 6,
    'fem.viscous_treatment': 'mixed_strain_temperature_gradient',
    'io.path': 'cylinder',
    'io.vtk.enable': False,
    'io.vtk.rate': 1,
    'io.vtk.subdivision': 2,
    # 'io.log.enable': False,
}


class Pulse:

    def __init__(self,
                 cfg: dict,
                 filename: str | None = None,
                 alpha: float = 1e-3,
                 X: float = 0.1,
                 domain: str | None = "implicit|explicit",
                 boundaries: str = "left|right|top|bottom"):

        if filename is None:
            filename = f"{str(self)}"

        self._cfg = cfg
        self.filename = filename
        self.domain = domain
        self.boundaries = boundaries
        self.alpha = alpha
        self.X = X

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
        U['thermodynamic_entropy'] = self.cfg.thermodynamic_entropy(U)

    def _get_initial_fields(self, Uinf: flowfields):

        U0 = flowfields()
        U0.rho = Uinf.rho
        U0.u = Uinf.u * (1 + Uinf.p/(Uinf.rho * Uinf.c) * self.alpha * ngs.exp(-(ngs.x/self.X)**2))
        U0.p = Uinf.p * (1 + self.alpha * ngs.exp(-(ngs.x/self.X)**2))
        U0.rho = Uinf.rho * (1 + self.alpha * ngs.exp(-(ngs.x/self.X)**2))**(1/self.cfg.equation_of_state.heat_capacity_ratio)
        self._set_fields(U0)
        return U0

    def set_conditions(self, cfg: CompressibleFlowSolver, **cfgs):
        self.cfg = cfg
        self._cfg.update(**cfgs)
        cfg.update(self._cfg)

        cfg.bcs.clear()
        cfg.dcs.clear()

        self.Uinf = self.cfg.get_farfield_fields((1, 0))
        self.U0 = self._get_initial_fields(self.Uinf)

        cfg.dcs[self.domain] = Initial(fields=self.U0, bonus_int_order=10)
        cfg.bcs[self.boundaries] = "periodic"

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
        self.cfg.io.ngsmesh.filename = filename
        self.sensor_name = filename

    def set_sensor_stream(self):

        self.cfg.io.sensor.list.clear()

        order = self.cfg.fem.order
        Uh = self.Uh

    def set_vtk_stream(self):
        export = ['density', 'velocity', 'pressure', 'temperature', 'Ma', 'energy', 'thermodynamic_entropy']
        fields = {f"{key}_h": value for key, value in self.Uh.items() if key in export}
        self.cfg.io.vtk.fields = fields

    def __str__(self):
        return self.__class__.__name__


def single_transient_routine(simulation: Pulse,  test: bool = False, **log):

    # Define common solver configuration
    cfg = simulation.cfg

    # Set filenames for output
    simulation.set_filenames(**log)

    # Initialize the solver
    cfg.fem.initialize_finite_element_spaces()
    cfg.fem.initialize_trial_and_test_functions()
    cfg.fem.initialize_gridfunctions()
    cfg.fem.initialize_time_scheme_gridfunctions()
    cfg.fem.set_boundary_conditions()
    cfg.fem.set_initial_conditions()
    cfg.fem.initialize_symbolic_forms()

    # Get solution fields
    Uh = cfg.get_all_solution_fields()
    simulation.set_solution_fields(Uh)

    if cfg.io.vtk.enable:
        simulation.set_vtk_stream()

    if cfg.io.sensor.enable:
        simulation.set_sensor_stream()

    if test:
        with ngs.TaskManager(pajetrace=int(10e9)):
            cfg.solve()
        return

    # Solve the system
    with ngs.TaskManager():
        timings = {}
        for _ in cfg.time.start_timing_solution_routine(timings):
            pass

    csv = {}
    info = {}
    for key, value in timings.items():
        if isinstance(value, np.ndarray):
            csv[key] = value
        else:
            info[key] = value

    cfg.io.path.mkdir(parents=True, exist_ok=True)
    filename = f"runtime_{simulation.filename}"
    label = f"{cfg.fem.scheme.name}"

    with cfg.io.path.joinpath(f"{filename}_info.txt").open("a") as file:
        file.write(f"{label} {cfg.time.timer.interval} {cfg.time.timer.step.Get()}:\n")
        for key, value in info.items():
            file.write(f"{key}: {value}\n")

    import pandas as pd
    df = pd.DataFrame(csv)
    df.to_csv(cfg.io.path.joinpath(f"{filename}_data.csv"), index=False)


def imex_transient_routine(implicit_simulation: Pulse,
                           explicit_simulation: Pulse,
                           test: bool = False,
                           **log):

    # Define common solver configuration
    IMP = implicit_simulation.cfg
    EXP = explicit_simulation.cfg

    time_routine = SynchronizedIMEXTimeRoutine(IMP, EXP)

    # Set filenames for output
    implicit_simulation.set_filenames(**log)
    explicit_simulation.set_filenames(**log)

    imp_bc = InterfaceBC(fields=None)
    exp_bc = InterfaceBC(fields=None)

    IMP.bcs['interface'] = imp_bc
    EXP.bcs['interface'] = exp_bc

    # Initialize the solver
    IMP.fem.initialize_finite_element_spaces()
    IMP.fem.initialize_trial_and_test_functions()
    IMP.fem.initialize_gridfunctions()
    IMP.fem.initialize_time_scheme_gridfunctions()
    IMP.fem.set_initial_conditions()

    EXP.fem.initialize_finite_element_spaces()
    EXP.fem.initialize_trial_and_test_functions()
    EXP.fem.initialize_gridfunctions()
    EXP.fem.initialize_time_scheme_gridfunctions()
    EXP.fem.set_initial_conditions()

    # Get solution fields
    Uhi = IMP.get_all_solution_fields()
    implicit_simulation.set_solution_fields(Uhi)
    exp_bc.fields = Uhi

    Uhe = EXP.get_all_solution_fields()
    explicit_simulation.set_solution_fields(Uhe)
    imp_bc.fields = Uhe

    IMP.fem.set_boundary_conditions()
    IMP.fem.initialize_symbolic_forms()

    EXP.fem.set_boundary_conditions()
    EXP.fem.initialize_symbolic_forms()

    if IMP.io.vtk.enable:
        implicit_simulation.set_vtk_stream()
        explicit_simulation.set_vtk_stream()

    if IMP.io.sensor.enable:
        implicit_simulation.set_sensor_stream()

    if test:
        with ngs.TaskManager(pajetrace=int(10e9)):
            for _ in time_routine.start_solution_routine():
                ...
        return

    # Solve the system
    with ngs.TaskManager():
        timings = {}

        for _ in time_routine.start_timing_solution_routine(timings):
            pass

    csv = {}
    info = {}
    for key, value in timings.items():
        if isinstance(value, np.ndarray):
            csv[key] = value
        else:
            info[key] = value

    IMP.io.path.mkdir(parents=True, exist_ok=True)
    filename = f"runtime_{implicit_simulation.filename}_{explicit_simulation.filename}"
    label = f"{IMP.fem.scheme.name}|{EXP.fem.scheme.name}"

    with IMP.io.path.joinpath(f"{filename}_info.txt").open("a") as file:
        file.write(f"{label} {IMP.time.timer.interval} {IMP.time.timer.step.Get()}:\n")
        for key, value in info.items():
            file.write(f"{key}: {value}\n")

    import pandas as pd
    df = pd.DataFrame(csv)
    df.to_csv(IMP.io.path.joinpath(f"{filename}_data.csv"), index=False)


def single_stable_time_step_routine(simulation: Pulse, tol: float = 1e-5, outputfile: Path = None, **log):

    # Define common solver configuration
    cfg = simulation.cfg

    # Set filenames for output
    simulation.set_filenames(**log)

    # Initialize the solver
    cfg.fem.initialize_finite_element_spaces()
    cfg.fem.initialize_trial_and_test_functions()
    cfg.fem.initialize_gridfunctions()
    cfg.fem.initialize_time_scheme_gridfunctions()
    cfg.fem.set_boundary_conditions()
    cfg.fem.initialize_symbolic_forms()

    # Get solution fields
    Uh = cfg.get_all_solution_fields()
    simulation.set_solution_fields(Uh)

    if cfg.io.vtk.enable:
        simulation.set_vtk_stream()

    if cfg.io.sensor.enable:
        simulation.set_sensor_stream()

    process_routine = None
    if outputfile is not None:
        import csv
        outputfile.parent.mkdir(parents=True, exist_ok=True)
        file = outputfile.open(mode='w')
        writer = csv.writer(file)
        writer.writerow(["time_step", "is_stable"])
        file.flush()

        def process_routine(dt, is_stable):
            writer.writerow([dt, is_stable])
            file.flush()

    # Solve the system
    with ngs.TaskManager():
        for _ in cfg.time.find_stable_time_step(tol=tol, process_routine=process_routine):
            cfg.fem.set_initial_conditions()

    if outputfile is not None:
        file.close()


def imex_stable_time_step_routine(implicit_simulation: Pulse,
                                  explicit_simulation: Pulse,
                                  tol: float = 1e-5,
                                  outputfile: Path = None,
                                  **log):

    # Define common solver configuration
    IMP = implicit_simulation.cfg
    EXP = explicit_simulation.cfg

    time = SynchronizedIMEXTimeRoutine(IMP, EXP)

    # Set filenames for output
    implicit_simulation.set_filenames(**log)
    explicit_simulation.set_filenames(**log)

    imp_bc = InterfaceBC(fields=None)
    exp_bc = InterfaceBC(fields=None)

    IMP.bcs['interface'] = imp_bc
    EXP.bcs['interface'] = exp_bc

    # Initialize the solver
    IMP.fem.initialize_finite_element_spaces()
    IMP.fem.initialize_trial_and_test_functions()
    IMP.fem.initialize_gridfunctions()
    IMP.fem.initialize_time_scheme_gridfunctions()

    EXP.fem.initialize_finite_element_spaces()
    EXP.fem.initialize_trial_and_test_functions()
    EXP.fem.initialize_gridfunctions()
    EXP.fem.initialize_time_scheme_gridfunctions()

    # Get solution fields
    Uhi = IMP.get_all_solution_fields()
    implicit_simulation.set_solution_fields(Uhi)
    exp_bc.fields = Uhi

    Uhe = EXP.get_all_solution_fields()
    explicit_simulation.set_solution_fields(Uhe)
    imp_bc.fields = Uhe

    IMP.fem.set_boundary_conditions()
    IMP.fem.initialize_symbolic_forms()

    EXP.fem.set_boundary_conditions()
    EXP.fem.initialize_symbolic_forms()

    if IMP.io.vtk.enable:
        implicit_simulation.set_vtk_stream()
        explicit_simulation.set_vtk_stream()

    if IMP.io.sensor.enable:
        implicit_simulation.set_sensor_stream()

    process_routine = None
    if outputfile is not None:
        import csv
        outputfile.parent.mkdir(parents=True, exist_ok=True)
        file = outputfile.open(mode='w')
        writer = csv.writer(file)
        writer.writerow(["time_step", "is_stable"])
        file.flush()

        def process_routine(dt, is_stable):
            writer.writerow([dt, is_stable])
            file.flush()

    # Solve the system
    with ngs.TaskManager():
        for _ in time.find_stable_time_step(tol=tol, process_routine=process_routine):

            IMP.fem.set_initial_conditions()
            EXP.fem.set_initial_conditions()

            # Need this, because the mass matrix changes with time step size
            IMP.fem.scheme.mass.Assemble()

    if outputfile is not None:
        file.close()


# %%
if __name__ == "__main__":
    from ngsolve.webgui import Draw

    mesh, imp, exp = get_single_mesh(40, 2, 0.01)

    Draw(mesh)
    Draw(imp)
    Draw(exp)

# %%
