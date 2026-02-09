# %%
import numpy as np
import ngsolve as ngs
from ngsolve.webgui import Draw
from dream.bla import fixpoint_iteration
from dream.mesh import get_structured_cylinder_mesh
from dream.compressible import CompressibleFlowSolver, FarField, Initial, flowfields,  AdiabaticWall, InterfaceBC
from dream.io import BoundarySensor
from dream.time import SynchronizedIMEXTimeRoutine
from time import time as clock
from pathlib import Path


def get_single_mesh(r, phi, curve_all=True) -> ngs.Mesh:

    domains = []
    for i in range(r.size - 1):
        r0, r1 = r[i], r[i+1]
        domains.append((f'domain_{i}', (np.array([r0, r1]), phi)))

    boundaries = [(f'wall', (r[0], phi))]
    for i in range(1, r.size - 1):
        boundaries.append((f'boundary_{i}', (r[i], phi)))
    boundaries.append((f'farfield', (r[-1], phi)))

    return get_structured_cylinder_mesh(domains, boundaries, curve_all=curve_all)


def get_imex_mesh_from_single_mesh(mesh: ngs.Mesh, Ni: int, Nr: int) -> tuple[ngs.Mesh, ngs.Mesh]:

    mesh = ngs.Mesh(mesh.ngmesh.Copy())

    # Netgen is 1 based indexing
    for i in range(1, Ni+1):
        mesh.ngmesh.SetMaterial(i, 'implicit')

    for i in range(Ni+1, Nr+1):
        mesh.ngmesh.SetMaterial(i, 'explicit')

    mesh.ngmesh.SetBCName(Ni, "interface")

    implicit_mesh = mesh.ngmesh.GetSubMesh('implicit', 'implicit')
    explicit_mesh = mesh.ngmesh.GetSubMesh('explicit', 'explicit')

    return ngs.Mesh(implicit_mesh), ngs.Mesh(explicit_mesh)


def get_geometrical_coordinates(Nr, Nphi, dr0=None, dphi0=None, Ro=100.0, Ri=0.5) -> ngs.Mesh:

    if dr0 is None:
        r = np.linspace(Ri, Ro, Nr + 1)
    else:

        dr = (Ro - Ri)

        def fp_geometrical(x0):
            return np.power(1+(dr/dr0)*x0, 1/Nr) - 1

        geom_r = fixpoint_iteration(0.3, fp_geometrical, it=100, tol=1e-16)
        drs = dr0*np.power(1+geom_r, np.arange(Nr))

        if not np.allclose(np.sum(drs), dr):
            raise ValueError("Could not create desired mesh spacing!")

        r = Ri * np.ones(Nr+1)
        r[1:] = Ri + np.cumsum(drs)

    if dphi0 is None:
        phi = np.linspace(0, 2*np.pi, Nphi + 1)
    else:

        dphi = np.pi

        def fp_geometrical(x0):
            return np.power(1+(dphi/dphi0)*x0, 1/(Nphi//2)) - 1

        geom_phi = fixpoint_iteration(0.3, fp_geometrical, it=100, tol=1e-16)

        dphis = dphi0*np.power(1+geom_phi, np.arange(Nphi//2))

        phi = np.zeros(Nphi + 1)
        phi[1:Nphi//2+1] = np.cumsum(dphis)
        phi[Nphi//2+1:] = 2*np.pi - phi[:Nphi//2][::-1]

    return r, phi


def get_twosided_geometrical_coordinates(Nr, Ni, Nphi, dri0, dre0, dphi0=None, Ro=100.0, Ri=0.5) -> ngs.Mesh:

    Ne = Nr - Ni

    r = np.zeros(Nr + 1)
    r[:Ni+1] = Ri

    if Ni != 0:
        geom_i = np.power(dre0/dri0, 1/Ni)
        dri = dri0 * np.power(geom_i, np.arange(Ni))
        r[1:Ni+1] += np.cumsum(dri)
    else:
        dre0 = dri0

    dRe = Ro - r[Ni]

    def fp_geometrical(x0):
        return np.power(1+(dRe/dre0)*x0, 1/Ne) - 1

    geom_e = fixpoint_iteration(0.3, fp_geometrical, it=1000, tol=1e-16)
    dre = dre0 * np.power(1+geom_e, np.arange(Ne))
    r[Ni+1:] = r[Ni]
    r[Ni+1:] += np.cumsum(dre)

    if dphi0 is None:
        phi = np.linspace(0, 2*np.pi, Nphi + 1)
    else:

        dphi = np.pi

        def fp_geometrical(x0):
            return np.power(1+(dphi/dphi0)*x0, 1/(Nphi//2)) - 1

        geom_phi = fixpoint_iteration(0.3, fp_geometrical, it=100, tol=1e-16)

        dphis = dphi0*np.power(1+geom_phi, np.arange(Nphi//2))

        phi = np.zeros(Nphi + 1)
        phi[1:Nphi//2+1] = np.cumsum(dphis)
        phi[Nphi//2+1:] = 2*np.pi - phi[:Nphi//2][::-1]

    return r, phi


def get_hyperbolic_tangent_coordinates(Nr, Nphi, dro=0.08, dri=0.02) -> ngs.Mesh:

    phi = np.linspace(0, 2*np.pi, Nphi + 1)
    dr = np.linspace(-np.pi, np.pi, Nr)
    dr = 0.5*(dro - dri) * np.tanh(dr) + 0.5*(dro + dri)
    dr[0] = dri
    dr[-1] = dro

    return dr, phi


def get_logarithmic_coordinates(Nr, Nphi, dro=0.08, dri=0.02, Ri=0.5, Ro=50.0) -> ngs.Mesh:

    phi = np.linspace(0, 2*np.pi, Nphi + 1)
    dr = np.linspace(Ri, Ro, Nr)
    dr = (dro - dri) * np.log(dr/Ri)/np.log(Ro/Ri) + dri

    return dr, phi


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
    'fem.solver.method.convergence_criterion': 1e-20,
    'fem.solver.method.damping_factor': 1.0,
    'fem.bonus_int_order': 6,
    'fem.viscous_treatment': 'mixed_strain_temperature_gradient',
    'io.path': 'cylinder',
    'io.vtk.enable': False,
    'io.vtk.rate': 1,
    'io.vtk.subdivision': 2,
    # 'io.log.enable': False,
}

PSEUDO_STATIONARY = {
    'reynolds_number': 150.0,
    'prandtl_number': 0.72,
    'mach_number': 0.2,
    'equation_of_state': 'ideal',
    'equation_of_state.heat_capacity_ratio': 1.4,
    'riemann_solver': 'upwind',
    'dynamic_viscosity': 'constant',
    'scaling': 'aerodynamic',
    'time': 'pseudo_time_stepping',
    'time.timer.interval': (0.0, 150.0),
    'time.timer.step': 0.1,
    'time.max_time_step': 1,
    'fem': 'conservative_hdg',
    'fem.scheme': 'implicit_euler',
    'fem.scheme.compile': {'realcompile': False, 'wait': False, 'keep_files': False},
    'fem.order': 3,
    'fem.solver': 'direct',
    'fem.solver.method': 'newton',
    'fem.solver.method.max_iterations': 100,
    'fem.solver.method.convergence_criterion': 1e-6,
    'fem.solver.method.damping_factor': 1.0,
    'fem.bonus_int_order': 6,
    'fem.viscous_treatment': 'mixed_strain_temperature_gradient',
    'io.path': 'cylinder',
    'io.vtk.enable': False,
    'io.vtk.rate': 1,
    'io.vtk.subdivision': 2,
    # 'io.log.enable': False,
}


class Cylinder:

    def __init__(self,
                 cfg: dict,
                 filename: str | None = None,
                 domain: str | None = None,
                 boundaries: str = "wall|farfield"):

        if filename is None:
            filename = f"{str(self)}"

        self._cfg = cfg
        self.filename = filename
        self.domain = domain
        self.boundaries = boundaries

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

    def set_conditions(self, cfg: CompressibleFlowSolver, **cfgs):
        self.cfg = cfg
        self._cfg.update(**cfgs)
        cfg.update(self._cfg)

        cfg.bcs.clear()
        cfg.dcs.clear()

        self.Uinf = self.cfg.get_farfield_fields((1, 0))

        domain = self.domain
        if domain is None:
            domain = "|".join(cfg.dcs)

        cfg.dcs[domain] = Initial(fields=self.Uinf, bonus_int_order=10)

        bcs = {'wall': AdiabaticWall(), 'farfield': FarField(fields=self.Uinf)}
        for bnd in self.boundaries.split('|'):
            cfg.bcs[bnd] = bcs[bnd]

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

        if 'wall' in self.boundaries.split('|'):

            c_d = self.cfg.drag_coefficient(Uh, Uh, self.Uinf)
            c_l = self.cfg.lift_coefficient(Uh, Uh, self.Uinf)

            fields = {'c_d': c_d, 'c_l': c_l}

            sensor = BoundarySensor(fields, self.cfg.mesh, 'wall',
                                    f'aerodynamic_coefficients_{self.sensor_name}', integration_order=order + 10)
            self.cfg.io.sensor.add(sensor)

    def set_vtk_stream(self):
        export = ['density', 'velocity', 'pressure', 'temperature', 'Ma', 'energy', 'thermodynamic_entropy']
        fields = {f"{key}_h": value for key, value in self.Uh.items() if key in export}
        self.cfg.io.vtk.fields = fields

    def __str__(self):
        return self.__class__.__name__


def load_initial_solution(initial_cfg: CompressibleFlowSolver):
    fem = initial_cfg.fem

    try:
        gfu = fem.gfu
    except:
        fem.initialize_finite_element_spaces()
        fem.initialize_trial_and_test_functions()
        fem.initialize_gridfunctions()
        gfu = fem.gfu

    initial_cfg.io.gfu.load_gridfunction(gfu)

    return gfu


def load_initial_solution_to_hdg(
        initial_cfg: CompressibleFlowSolver, hdg_cfg: CompressibleFlowSolver, filename: str = None):

    fem = hdg_cfg.fem
    try:
        gfu = fem.gfu
    except:
        fem.initialize_finite_element_spaces()
        fem.initialize_trial_and_test_functions()
        fem.initialize_gridfunctions()
        gfu = fem.gfu

    igfu = load_initial_solution(initial_cfg)

    gfu.components[0].Set(igfu.components[0], bonus_intorder=10)

    Uhat = hdg_cfg.fem.spaces['Uhat']
    uhat, vhat = Uhat.TnT()

    blf = ngs.BilinearForm(Uhat)
    blf += uhat * vhat * ngs.dx(element_boundary=True, bonus_intorder=10)
    blf.Assemble()

    lf = ngs.LinearForm(Uhat)
    lf += igfu.components[0] * vhat * ngs.dx(element_boundary=True, bonus_intorder=10)
    lf.Assemble()

    gfu.components[1].vec.data = blf.mat.Inverse(Uhat.FreeDofs()) * lf.vec

    if fem.viscous_treatment.name == 'mixed_strain_temperature_gradient':
        gfu.components[2].Set(igfu.components[2], bonus_intorder=10)

    if filename is not None:
        hdg_cfg.io.gfu.save_gridfunction(gfu, filename)

    return gfu


def load_initial_solution_to_dg(initial_cfg: CompressibleFlowSolver,
                                dg_cfg: CompressibleFlowSolver, filename: str = None):

    fem = dg_cfg.fem
    try:
        gfu = fem.gfu
    except:
        fem.initialize_finite_element_spaces()
        fem.initialize_trial_and_test_functions()
        fem.initialize_gridfunctions()
        gfu = fem.gfu

    igfu = load_initial_solution(initial_cfg)

    gfu.Set(igfu.components[0], bonus_intorder=10)

    if filename is not None:
        dg_cfg.io.gfu.save_gridfunction(gfu, filename)

    return gfu


def single_transient_routine(
        simulation: Cylinder, initial_cfg: CompressibleFlowSolver = None, test: bool = False, **log):

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

    if initial_cfg is not None and cfg.fem.name == 'conservative_hdg':
        load_initial_solution_to_hdg(initial_cfg, cfg)
        cfg.fem.scheme.set_initial_conditions()
    elif initial_cfg is not None and cfg.fem.name == 'conservative_dg':
        load_initial_solution_to_dg(initial_cfg, cfg)
        cfg.fem.scheme.set_initial_conditions()
    else:
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
        time = {}
        for _ in cfg.time.start_timing_solution_routine(time):
            pass

    csv = {}
    info = {}
    for key, value in time.items():
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


def imex_transient_routine(implicit_simulation: Cylinder,
                           explicit_simulation: Cylinder,
                           initial_cfg: CompressibleFlowSolver,
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

    EXP.fem.initialize_finite_element_spaces()
    EXP.fem.initialize_trial_and_test_functions()
    EXP.fem.initialize_gridfunctions()
    EXP.fem.initialize_time_scheme_gridfunctions()

    load_initial_solution_to_hdg(initial_cfg, IMP)
    load_initial_solution_to_dg(initial_cfg, EXP)

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
        time = {}

        for _ in time_routine.start_timing_solution_routine(time):
            pass

    csv = {}
    info = {}
    for key, value in time.items():
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


def single_stable_time_step_routine(
        simulation: Cylinder, initial_cfg: CompressibleFlowSolver, outputfile: Path = None, **log):

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
        for _ in cfg.time.find_stable_time_step(tol=1e-5, process_routine=process_routine):

            if initial_cfg is not None and cfg.fem.name == 'conservative_hdg':
                load_initial_solution_to_hdg(initial_cfg, cfg)
                cfg.fem.scheme.set_initial_conditions()
            elif initial_cfg is not None and cfg.fem.name == 'conservative_dg':
                load_initial_solution_to_dg(initial_cfg, cfg)
                cfg.fem.scheme.set_initial_conditions()

    if outputfile is not None:
        file.close()


def imex_stable_time_step_routine(implicit_simulation: Cylinder,
                                  explicit_simulation: Cylinder,
                                  initial_cfg: CompressibleFlowSolver,
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
        for _ in time.find_stable_time_step(tol=1e-5, process_routine=process_routine):

            load_initial_solution_to_dg(initial_cfg, EXP)
            load_initial_solution_to_hdg(initial_cfg, IMP)

            # Need this, because the mass matrix changes with time step size
            IMP.fem.scheme.mass.Assemble()

            IMP.fem.scheme.set_initial_conditions()
            EXP.fem.scheme.set_initial_conditions()

    if outputfile is not None:
        file.close()


# %%
if __name__ == "__main__":
    from ngsolve.webgui import Draw

    r, phi = get_geometrical_coordinates(Nr=64, Nphi=32, dr0=0.05, dphi0=np.pi/32)
    r, phi = get_twosided_geometrical_coordinates(64, 4, 32, 0.001, 0.05, dphi0=np.pi/32)
    mesh = get_single_mesh(r, phi, curve_all=True)
    imp, exp = get_imex_mesh_from_single_mesh(mesh, Ni=4, Nr=64)

    mesh.Curve(3)
    imp.Curve(3)
    exp.Curve(3)

    Draw(mesh)
    Draw(imp)
    Draw(exp)

# %%
if __name__ == "__main__":
    from ngsolve.webgui import Draw
    import matplotlib.pyplot as plt

    r0, _ = get_geometrical_coordinates(Nr=64, Nphi=32, dr0=0.05, dphi0=np.pi/32)
    dr0 = r0[1:] - r0[:-1]
    s0 = dr0 / dr0[0]

    rn = []
    drn = []
    sn = []
    for Ni in [0, 4, 8, 12, 16]:
        r, _ = get_twosided_geometrical_coordinates(64, Ni, 32, 0.001, 0.05, dphi0=np.pi/32)
        rn.append(r)
        drn.append(r[1:] - r[:-1])
        sn.append(drn[-1] / drn[-1][0])

    fig, axes = plt.subplots(nrows=3, figsize=(15, 15))

    ax = axes[0]
    ax.plot(r0, label=r"$\Delta r_0^{0.05}$", marker='o')
    for i, r in enumerate(rn):
        ax.semilogy(r, label=rf"$\Delta r_{i}^{0.001}$", marker='o')
    ax.set_ylabel(r"$r$")
    ax.grid(which='both')

    ax = axes[1]
    ax.plot(dr0, label=r"$\Delta r_0^{0.05}$", marker='o')
    for i, r in enumerate(drn):
        ax.semilogy(r, label=rf"$\Delta r_{i}^{{{0.001}}}$", marker='o')
    ax.set_ylabel(r"$\Delta r$")
    ax.grid(which='both')

    ax = axes[2]
    ax.plot(s0, label=r"$\Delta r_0^{0.05}$", marker='o')
    for i, r in enumerate(sn):
        ax.semilogy(r, label=rf"$\Delta r_{i}^{{{0.001}}}$", marker='o')
    ax.set_ylabel(r"$\frac{\Delta r}{\Delta r_0}$")
    ax.grid(which='both')
    ax.legend()


    


# %%
