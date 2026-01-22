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


def get_connected_mesh(ri, re, phi, curve_all=True) -> ngs.Mesh:

    if not np.allclose(ri.max(), re.min()):
        raise ValueError("Inner and outer radial meshes do not align!")

    domains = (('implicit', (ri, phi)),
               ('explicit', (re, phi)),
               )

    boundaries = (('wall', (ri.min(), phi)),
                  ('farfield', (re.max(), phi)),
                  ('interface', (ri.max(), phi)),
                  )

    return get_structured_cylinder_mesh(domains, boundaries, curve_all=curve_all)


def get_imex_meshes(ri, re, phi, curve_all=True) -> tuple[ngs.Mesh, ngs.Mesh, ngs.Mesh]:
    domains = (('explicit', (re, phi)),
               )

    boundaries = (('farfield', (re.max(), phi)),
                  ('interface', (re.min(), phi)),
                  )

    explicit_mesh = get_structured_cylinder_mesh(domains, boundaries, curve_all=curve_all)

    domains = (('implicit', (ri, phi)),)

    boundaries = (('wall', (ri.min(), phi)),
                  ('interface', (ri.max(), phi)),
                  )

    implicit_mesh = get_structured_cylinder_mesh(domains, boundaries, curve_all=curve_all)

    return implicit_mesh, explicit_mesh


def get_geometrical_coordinates(Nr, Nphi, Ni, dr0=0.08, Ro=25.0, Ri=0.5, wake = None) -> ngs.Mesh:

    dr = (Ro - Ri)

    def fp_geometrical(x0):
        return np.power(1+(dr/dr0)*x0, 1/Nr) - 1

    geom = fixpoint_iteration(0.3, fp_geometrical, it=100, tol=1e-16)
    drs = dr0*np.power(1+geom, np.arange(0, Nr))

    if not np.allclose(np.sum(drs), dr):
        raise ValueError("Could not create desired mesh spacing!")

    r = Ri * np.ones(Nr+1)
    r[1:] = Ri + np.cumsum(drs)

    ri = r[:Ni+1]
    re = r[Ni:]
    phi = np.linspace(0, 2*np.pi, Nphi + 1)
    if wake is not None:
        phi = np.union1d( phi, wake)

    return ri, re, phi


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
    'fem.viscous_treatment': None,
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
                 domain: str = "explicit|implicit",
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

        cfg.dcs[self.domain] = Initial(fields=self.Uinf, bonus_int_order=10)

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

    if fem.viscous_treatment == 'mixed_strain_temperature_gradient':
        gfu.components[2].Set(igfu.components[2], bonus_intorder=10)

    if filename is not None:
        hdg_cfg.io.gfu.save_gridfunction(gfu, filename)

    return gfu


def load_initial_solution_to_dg(
        initial_cfg: CompressibleFlowSolver, dg_cfg: CompressibleFlowSolver, filename: str = None):

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


def single_transient_routine(simulation: Cylinder, initial_cfg: CompressibleFlowSolver = None, **log):

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
    # Uh = cfg.get_solution_fields('eps')
    Uh = cfg.get_all_solution_fields()
    simulation.set_solution_fields(Uh)

    if cfg.io.vtk.enable:
        simulation.set_vtk_stream()

    if cfg.io.sensor.enable:
        simulation.set_sensor_stream()

    # Solve the system
    with ngs.TaskManager():
        cfg.fem.scheme.assemble()

        start = clock()
        for _ in cfg.time.start_solution_routine(False):
            pass
        end = clock()

    with cfg.io.path.joinpath(f"runtime_{simulation.filename}.txt").open("a") as file:
        file.write(f"{cfg.fem.scheme.name} {cfg.time.timer.interval} {cfg.time.timer.step.Get()}: {end - start}\n")


def imex_transient_routine(implicit_simulation: Cylinder,
                           explicit_simulation: Cylinder,
                           initial_cfg: CompressibleFlowSolver,
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

    # Solve the system
    with ngs.TaskManager():
        IMP.fem.scheme.assemble()
        EXP.fem.scheme.assemble()

        start = clock()
        for _ in time.start_solution_routine(False):
            pass
        end = clock()

    with IMP.io.path.joinpath(f"runtime_{implicit_simulation.filename}.txt").open("a") as file:
        file.write(f"{IMP.fem.scheme.name} {IMP.time.timer.interval} {IMP.time.timer.step.Get()}: {end - start}\n")


def single_stable_time_step_routine(simulation: Cylinder, initial: dict, **log):

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

    # Solve the system
    dts = []
    with ngs.TaskManager():
        for dt in cfg.time.find_stable_time_step(tol=1e-6):
            cfg.io.gfu.load_gridfunction(cfg.fem.gfu, **initial)
            dts.append(dt)

    return dts


def imex_stable_time_step_routine(implicit_simulation: Cylinder,
                                  explicit_simulation: Cylinder,
                                  implicit_initial: dict,
                                  explicit_initial: dict,
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
    IMP.fem.set_boundary_conditions()
    IMP.fem.initialize_symbolic_forms()

    EXP.fem.initialize_finite_element_spaces()
    EXP.fem.initialize_trial_and_test_functions()
    EXP.fem.initialize_gridfunctions()
    EXP.fem.initialize_time_scheme_gridfunctions()
    EXP.fem.set_boundary_conditions()
    EXP.fem.initialize_symbolic_forms()

    # Get solution fields
    Uhi = IMP.get_all_solution_fields()
    IMP.set_solution_fields(Uhi)
    exp_bc.fields = Uhi

    Uhe = EXP.get_all_solution_fields()
    EXP.set_solution_fields(Uhe)
    imp_bc.fields = Uhe

    if IMP.io.vtk.enable:
        IMP.set_vtk_stream()
        EXP.set_vtk_stream()

    if IMP.io.sensor.enable:
        IMP.set_sensor_stream()

    # Solve the system
    dts = []
    with ngs.TaskManager():
        for dt in time.find_stable_time_step(tol=1e-6):
            IMP.io.gfu.load_gridfunction(IMP.fem.gfu, **implicit_initial)
            EXP.io.gfu.load_gridfunction(EXP.fem.gfu, **explicit_initial)
            dts.append(dt)

    return dts


# %%
if __name__ == "__main__":
    ri, re, phi = get_geometrical_coordinates(Nr=32, Nphi=32, Ni=8, dr0=0.04, Ro=50.0)
    mesh = get_connected_mesh(ri, re, phi, False)
    implicit_mesh, explicit_mesh = get_imex_meshes(ri, re, phi, False)
    mesh.Curve(3)

    Draw(mesh)
    Draw(implicit_mesh)
    Draw(explicit_mesh)
    print(mesh.ne, implicit_mesh.ne, explicit_mesh.ne)
    print(implicit_mesh.ne/mesh.ne, explicit_mesh.ne/mesh.ne)

