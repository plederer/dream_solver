from __future__ import annotations
import ngsolve as ngs
import logging
import typing

from math import isnan

from .mesh import is_mesh_periodic, Periodic, Initial, BoundaryConditions, DomainConditions
from .config import is_notebook, UniqueConfiguration, InterfaceConfiguration, interface, configuration, unique, ngsdict
from .time import StationaryConfig, TransientConfig, PseudoTimeSteppingConfig
from .io import IOConfiguration

logger = logging.getLogger(__name__)


class BonusIntegrationOrder(UniqueConfiguration):

    @configuration(default=0)
    def vol(self, vol):
        return int(vol)

    @configuration(default=0)
    def bnd(self, bnd):
        return int(bnd)

    @configuration(default=0)
    def bbnd(self, bbnd):
        return int(bbnd)

    @configuration(default=0)
    def bbbnd(self, bbbnd):
        return int(bbbnd)

    vol: int
    bnd: int
    bbnd: int
    bbbnd: int


class Compile(UniqueConfiguration):

    @configuration(default=True)
    def realcompile(self, flag: bool):
        return bool(flag)

    @configuration(default=False)
    def wait(self, flag: bool):
        return bool(flag)

    @configuration(default=False)
    def keep_files(self, flag: bool):
        return bool(flag)

    realcompile: bool
    wait: bool
    keep_files: bool


class Optimizations(UniqueConfiguration):

    @unique(default=Compile)
    def compile(self, compile):
        return compile

    @configuration(default=False)
    def static_condensation(self, static_condensation):
        return bool(static_condensation)

    @unique(default=BonusIntegrationOrder)
    def bonus_int_order(self, dict_):
        return dict_

    compile: Compile
    static_condensation: bool
    bonus_int_order: BonusIntegrationOrder


class Solver(InterfaceConfiguration, is_interface=True):

    cfg: SolverConfiguration

    def inverse(self, blf: ngs.BilinearForm, fes: ngs.FESpace, freedofs: ngs.BitArray = None, **kwargs):
        raise NotImplementedError()

    def initialize(self, blf: ngs.BilinearForm, lf: ngs.LinearForm, gfu: ngs.GridFunction, **kwargs):
        ...

    def log_iteration_error(self, it: int | None = None, t: float | None = None):
        msg = f"residual: {self.error:8e}"
        if it is not None:
            msg += f" | iteration: {it}"
        if t is not None:
            msg += f" | t: {t}"
        logger.info(msg)

    def solve(self, t: float | None = None) -> typing.Generator[int | None, None, None]:
        raise NotImplementedError()


# ------- Linear Solvers ------- #


class LinearSolver(Solver, is_interface=True):

    cfg: SolverConfiguration


class DirectLinearSolver(LinearSolver, skip=True):

    def inverse(self, blf: ngs.BilinearForm, fes: ngs.FESpace, freedofs: ngs.BitArray = None, **kwargs):
        if freedofs is None:
            freedofs = fes.FreeDofs(blf.condense)
        return blf.mat.Inverse(freedofs, inverse=self.name)


class UmfpackLinearSolver(DirectLinearSolver):

    name = "umfpack"


class PardisoLinearSolver(DirectLinearSolver):

    name = "pardiso"


# ------- Nonlinear Solvers ------- #


class NonlinearMethod(InterfaceConfiguration, is_interface=True):

    cfg: SolverConfiguration

    def initialize(self, gfu: ngs.GridFunction, du):
        self.gfu = gfu
        self.du = du

    def update_solution(self):
        raise NotImplementedError()


class NewtonsMethod(NonlinearMethod):

    name: str = "newton"

    @configuration(default=1)
    def damping_factor(self, damping_factor: float):
        return float(damping_factor)

    def update_solution(self):
        self.gfu.vec.data -= self.damping_factor * self.du

    damping_factor: float


class NonlinearSolver(Solver, is_interface=True):

    cfg: SolverConfiguration

    @interface(default=NewtonsMethod)
    def method(self, method: NonlinearMethod):
        return method

    @configuration(default=10)
    def max_iterations(self, max_iterations: int):
        return int(max_iterations)

    @configuration(default=1e-8)
    def convergence_criterion(self, convergence_criterion: float):
        if convergence_criterion <= 0:
            raise ValueError("Convergence Criterion must be greater zero!")
        return float(convergence_criterion)

    def log_iteration_error(self, it: int | None = None, t: float | None = None):
        msg = f"residual: {self.error:8e}"
        if it is not None:
            msg += f" | iteration: {it}"
        if t is not None:
            msg += f" | t: {t}"
        logger.info(msg)

    def reset_status(self):
        self.is_converged = False
        self.is_nan = False

    def set_iteration_error(self, du: ngs.BaseVector, residual: ngs.BaseVector):
        error = ngs.sqrt(ngs.InnerProduct(du, residual)**2)

        self.is_converged = error < self.convergence_criterion
        self.error = error

        if isnan(error):
            self.is_nan = True
            logger.error("Solution process diverged!")

    def solve(self, t: float | None = None) -> typing.Generator[int | None, None, None]:

        self.reset_status()

        for it in range(self.max_iterations):
            yield it

            self.solve_update_step()
            self.log_iteration_error(it, t)

            if self.is_nan:
                break

            self.method.update_solution()

            if self.is_converged:
                break

    def solve_update_step(self):
        raise NotImplementedError()

    method: NonlinearMethod
    max_iterations: int
    convergence_criterion: float


class DirectNonlinearSolver(NonlinearSolver, skip=True):

    def inverse(self, blf: ngs.BilinearForm, fes: ngs.FESpace, freedofs: ngs.BitArray = None, **kwargs):
        if freedofs is None:
            freedofs = fes.FreeDofs(blf.condense)
        return blf.mat.Inverse(freedofs=freedofs, inverse=self.name)

    def initialize(self, blf: ngs.BilinearForm, rhs: ngs.BaseVector, gfu: ngs.GridFunction, **kwargs):
      
        if not isinstance(rhs, ngs.BaseVector) and rhs is not None:
            raise TypeError("Input rhs must be of type either ngs.BaseVector or None.")
      
        self.gfu = gfu
        self.fes = gfu.space
        self.blf = blf
        self.rhs = rhs

        self.residual = gfu.vec.CreateVector()
        self.temporary = gfu.vec.CreateVector()

        self.method.initialize(self.gfu, self.temporary)

    def solve_update_step(self):

        self.blf.Apply(self.gfu.vec, self.residual)
        if self.rhs is not None:
            self.residual.data -= self.rhs
        self.blf.AssembleLinearization(self.gfu.vec)

        inv = self.blf.mat.Inverse(freedofs=self.fes.FreeDofs(self.blf.condense), inverse=self.name)
        if self.blf.condense:
            self.residual.data += self.blf.harmonic_extension_trans * self.residual
            self.temporary.data = inv * self.residual
            self.temporary.data += self.blf.harmonic_extension * self.temporary
            self.temporary.data += self.blf.inner_solve * self.residual
        else:
            self.temporary.data = inv * self.residual

        self.set_iteration_error(self.temporary, self.residual)


class UmfpackNonlinearSolver(DirectNonlinearSolver):

    name = "umfpack"


class PardisoNonlinearSolver(DirectNonlinearSolver):

    name = "pardiso"

# ------- Finite Element Method ------- #


class FiniteElementMethod(InterfaceConfiguration, is_interface=True):

    cfg: SolverConfiguration

    @configuration(default=2)
    def order(self, order):
        return int(order)

    def add_finite_element_spaces(self, spaces: dict[str, ngs.FESpace]):
        raise NotImplementedError("Overload this method in derived class!")

    def add_symbolic_spatial_forms(self,
                                   blf: dict[str, ngs.comp.SumOfIntegrals],
                                   lf: dict[str, ngs.comp.SumOfIntegrals]):
        raise NotImplementedError("Overload this method in derived class!")

    def add_symbolic_temporal_forms(self,
                                    blf: dict[str, ngs.comp.SumOfIntegrals],
                                    lf: dict[str, ngs.comp.SumOfIntegrals]):
        raise NotImplementedError("Overload this method in derived class!")

    def get_temporal_integrators(self) -> dict[str, ngs.comp.DifferentialSymbol]:
        raise NotImplementedError("Overload this method in derived class!")

    def get_fields(self, quantities: dict[str, bool]) -> ngsdict:
        raise NotImplementedError("Overload this method in derived class!")

    def set_initial_conditions(self) -> None:
        if not self.cfg.dcs.has_condition(Initial):
            logger.debug("No initial conditions set!")
            return None

    def set_boundary_conditions(self) -> None:
        raise NotImplementedError("Overload this method in derived class!")

    order: int


# ------- Solver Configuration ------- #


class SolverConfiguration(InterfaceConfiguration, is_interface=True):

    def __init__(self, mesh: ngs.Mesh, bcs: BoundaryConditions, dcs: DomainConditions, **kwargs) -> None:
        self.mesh = mesh
        self.mesh.normal = ngs.specialcf.normal(mesh.dim)
        self.mesh.tangential = ngs.specialcf.tangential(mesh.dim)
        self.mesh.meshsize = ngs.specialcf.mesh_size
        self.mesh.is_periodic = is_mesh_periodic(mesh)

        self.bcs = bcs
        self.dcs = dcs

        super().__init__(cfg=self, mesh=self.mesh, **kwargs)

    @property
    def fem(self) -> FiniteElementMethod:
        raise NotImplementedError("Overload this configuration in derived class!")

    @interface(default=StationaryConfig)
    def time(self, time):
        return time

    @interface(default=UmfpackLinearSolver)
    def linear_solver(self, solver: LinearSolver):
        return solver

    @interface(default=UmfpackNonlinearSolver)
    def nonlinear_solver(self, solver: NonlinearSolver):
        return solver

    @unique(default=Optimizations)
    def optimizations(self, optimizations):
        return optimizations

    @unique(default=IOConfiguration)
    def io(self, io):
        return io

    @configuration(default=None)
    def info(self, info=None):

        if info is None:
            info = {}

        if not isinstance(info, dict):
            raise ValueError("Info must be a dictionary!")

        return info

    def initialize(self) -> None:
        self.initialize_finite_element_spaces()
        self.initialize_trial_and_test_functions()
        self.initialize_gridfunctions()
        self.set_boundary_conditions()

        self.time.initialize()
        self.time.set_initial_conditions()

        self.initialize_symbolic_forms()

    def initialize_finite_element_spaces(self) -> None:

        self.spaces = {}
        self.fem.add_finite_element_spaces(self.spaces)

        if not self.spaces:
            raise ValueError("Spaces container is empty!")

        fes = list(self.spaces.values())

        for space in fes[1:]:
            fes[0] *= space

        self.fes: ngs.ProductSpace = fes[0]

    def initialize_trial_and_test_functions(self) -> None:

        if len(self.spaces) > 1:
            self.TnT = {label: (tr, te) for label, tr, te in zip(
                self.spaces, self.fes.TrialFunction(), self.fes.TestFunction())}
        else:
            self.TnT = {label: self.fes.TnT() for label in self.spaces}

    def initialize_gridfunctions(self) -> None:
        self.gfu = ngs.GridFunction(self.fes)

        if len(self.spaces) > 1:
            self.gfus = {label: gfu for label, gfu in zip(self.spaces, self.gfu.components)}
        else:
            self.gfus = {label: self.gfu for label in self.spaces}

    def initialize_symbolic_forms(self) -> None:
        self.blf = {}
        self.lf = {}

        self.fem.add_symbolic_spatial_forms(self.blf, self.lf)
        self.time.add_symbolic_temporal_forms(self.blf, self.lf)

    def get_fields(self, **quantities: bool) -> ngsdict:
        fields = self.fem.get_fields(quantities)

        for quantity in quantities:
            logger.info(f"Quantity {quantity} not predefined!")

        return fields

    def set_boundary_conditions(self) -> None:
        if self.mesh.is_periodic and not self.bcs.has_condition(Periodic):
            raise ValueError("Mesh has periodic boundaries, but no periodic boundary conditions are set!")

        self.fem.set_boundary_conditions()

    def set_grid_deformation(self):
        grid_deformation = self.dcs.get_grid_deformation_function()
        self.mesh.SetDeformation(grid_deformation)

    def set_solver_documentation(self, doc: str):
        if isinstance(doc, str):
            ...
        elif isinstance(doc, typing.Sequence):
            doc = "\n".join(doc)
        else:
            raise ValueError("Documentation must be a string or sequence!")

        self.doc = doc

    def solve(self, reassemble: bool = True):
        for t in self.time.start_solution_routine(reassemble):
            pass


    time: StationaryConfig | TransientConfig | PseudoTimeSteppingConfig
    linear_solver: DirectLinearSolver
    nonlinear_solver: DirectNonlinearSolver
    optimizations: Optimizations
    io: IOConfiguration

    bcs: BoundaryConditions
    dcs: DomainConditions
