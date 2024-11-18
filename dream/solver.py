from __future__ import annotations
import ngsolve as ngs
import logging
import typing

from math import isnan

from .compressible import CompressibleFlowConfiguration
from .mesh import is_mesh_periodic
from .config import UniqueConfiguration, InterfaceConfiguration, interface, configuration, unique
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


class Inverse(InterfaceConfiguration, is_interface=True):

    cfg: SolverConfiguration

    def get_inverse(self, blf: ngs.BilinearForm, fes: ngs.FESpace, pre: ngs.BilinearForm = None):
        NotImplementedError()


class Direct(Inverse):

    name: str = "direct"

    @configuration(default="umfpack")
    def solver(self, solver: str):
        return solver

    def get_inverse(self, blf: ngs.BilinearForm, fes: ngs.FESpace, pre: ngs.BilinearForm = None):
        return blf.mat.Inverse(fes.FreeDofs(blf.condense), inverse=self.solver)

    solver: str


class NonlinearMethod(InterfaceConfiguration, is_interface=True):

    cfg: SolverConfiguration

    def reset_status(self):
        self.is_converged = False
        self.is_nan = False

    def set_solution_routine_attributes(self):
        raise NotImplementedError()

    def solve_update_step(self):
        raise NotImplementedError()

    def update_solution(self):
        raise NotImplementedError()

    def set_iteration_error(self, it: int | None = None, t: float | None = None):
        raise NotImplementedError()

    def log_iteration_error(self, error: float, it: int | None = None, t: float | None = None):
        msg = f"residual: {error:8e}"
        if it is not None:
            msg += f" | iteration: {it}"
        if t is not None:
            msg += f" | t: {t}"
        logger.info(msg)


class NewtonsMethod(NonlinearMethod):

    name: str = "newton"

    @configuration(default=1)
    def damping_factor(self, damping_factor: float):
        return float(damping_factor)

    def set_solution_routine_attributes(self):
        self.fes = self.cfg.pde.fes
        self.gfu = self.cfg.pde.gfu
        self.inverse = self.cfg.solver.inverse

        self.blf = self.cfg.solver.blf
        self.lf = self.cfg.solver.lf

        self.residual = self.gfu.vec.CreateVector()
        self.temporary = self.gfu.vec.CreateVector()

    def solve_update_step(self):

        self.blf.Apply(self.gfu.vec, self.residual)
        self.residual.data -= self.lf.vec
        self.blf.AssembleLinearization(self.gfu.vec)

        inv = self.cfg.solver.inverse.get_inverse(self.blf, self.fes)
        if self.blf.condense:
            self.residual.data += self.blf.harmonic_extension_trans * self.residual
            self.temporary.data = inv * self.residual
            self.temporary.data += self.blf.harmonic_extension * self.temporary
            self.temporary.data += self.blf.inner_solve * self.residual
        else:
            self.temporary.data = inv * self.residual

    def update_solution(self):
        self.gfu.vec.data -= self.damping_factor * self.temporary

    def set_iteration_error(self, it: int | None = None, t: float | None = None):
        error = ngs.sqrt(ngs.InnerProduct(self.temporary, self.residual)**2)

        self.is_converged = error < self.cfg.solver.convergence_criterion
        self.log_iteration_error(error, it, t)
        self.error = error

        if isnan(error):
            self.is_nan = True
            logger.error("Solution process diverged!")

    damping_factor: float


class Solver(InterfaceConfiguration, is_interface=True):

    cfg: SolverConfiguration

    @interface(default=Direct)
    def inverse(self, inverse: Inverse):
        return inverse

    def initialize(self):

        self.fes = self.cfg.pde.fes
        condense = self.cfg.optimizations.static_condensation
        compile = self.cfg.optimizations.compile

        self.blf = ngs.BilinearForm(self.fes, condense=condense)
        self.lf = ngs.LinearForm(self.fes)

        for name, cf in self.cfg.pde.blf.items():
            logger.debug(f"Adding {name} to the BilinearForm!")

            if compile.realcompile:
                self.blf += cf.Compile(**compile)
            else:
                self.blf += cf

        for name, cf in self.cfg.pde.lf.items():
            logger.debug(f"Adding {name} to the LinearForm!")

            if compile.realcompile:
                self.lf += cf.Compile(**compile)
            else:
                self.lf += cf

    inverse: Direct


class LinearSolver(Solver):

    name: str = "linear"

    def solve(self):

        self.blf.Assemble()

        if hasattr(self, 'dirichlet'):
            self.dirichlet.data = self.blf.mat * self.gfu.vec

        for t in self.cfg.time.start_solution_routine():
            self.lf.Assemble()

            if hasattr(self, 'dirichlet'):
                self.lf.vec.data -= self.proj * self.dirichlet

            self.cfg.pde.gfu.vec.data += self.inverse.get_inverse(self.blf, self.cfg.pde.fes) * self.lf.vec

    def set_dirichlet_vector(self):

        periodic = self.cfg.pde.bcs.get_periodic_boundaries(True)
        dofs = ~self.fes.FreeDofs() & ~self.fes.GetDofs(self.cfg.mesh.Boundaries(periodic))

        if any(dofs):
            self.proj = ngs.Projector(dofs, False)   # dirichlet mask
            self.dirichlet = self.gfu.vec.CreateVector()
            self.dirichlet[:] = 0

    def initialize(self):
        super().initialize()
        self.gfu = self.cfg.pde.gfu
        self.set_dirichlet_vector()


class NonlinearSolver(Solver):

    name: str = "nonlinear"

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

    def initialize(self):
        super().initialize()
        self.method.set_solution_routine_attributes()

    def solve(self):

        for t in self.cfg.time.start_solution_routine():

            self.method.reset_status()

            for it in range(self.max_iterations):

                self.cfg.time.solver_iteration_update(it)

                self.method.solve_update_step()
                self.method.set_iteration_error(it, t)

                if self.method.is_nan:
                    break

                self.method.update_solution()

                if self.method.is_converged:
                    break

            if self.method.is_nan:
                break

    method: NonlinearMethod
    max_iterations: int
    convergence_criterion: float


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


class SolverConfiguration(UniqueConfiguration):

    name: str = 'cfg'

    def __init__(self, mesh: ngs.Mesh, **kwargs) -> None:
        self.mesh = mesh
        self.mesh.normal = ngs.specialcf.normal(mesh.dim)
        self.mesh.tangential = ngs.specialcf.tangential(mesh.dim)
        self.mesh.meshsize = ngs.specialcf.mesh_size
        self.mesh.is_periodic = is_mesh_periodic(mesh)

        super().__init__(cfg=self, mesh=self.mesh, **kwargs)

    @interface(default=CompressibleFlowConfiguration)
    def pde(self, pde):
        return pde

    @interface(default=LinearSolver)
    def solver(self, solver: Solver):
        return solver

    @interface(default=StationaryConfig)
    def time(self, time):
        return time

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

    def set_solver_documentation(self, doc: str):
        if isinstance(doc, str):
            ...
        elif isinstance(doc, typing.Sequence):
            doc = "\n".join(doc)
        else:
            raise ValueError("Documentation must be a string or sequence!")

        self.doc = doc

    def set_grid_deformation(self):
        grid_deformation = self.pde.dcs.get_grid_deformation_function()
        self.mesh.SetDeformation(grid_deformation)

    solver: LinearSolver | NonlinearSolver
    pde: CompressibleFlowConfiguration
    time: StationaryConfig | TransientConfig | PseudoTimeSteppingConfig
    optimizations: Optimizations
    io: IOConfiguration
