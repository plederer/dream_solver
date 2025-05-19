from __future__ import annotations
import ngsolve as ngs
import logging
import typing

from math import isnan

from .mesh import is_mesh_periodic, Periodic, Initial, BoundaryConditions, DomainConditions
from .config import Configuration, dream_configuration, ngsdict, Integrals
from .time import StationaryRoutine, TransientRoutine, PseudoTimeSteppingRoutine, TimeRoutine, Scheme, TimeSchemes
from .io import IOConfiguration

logger = logging.getLogger(__name__)


class BonusIntegrationOrder(Configuration):

    def __init__(self, mesh, root=None, **default):

        DEFAULT = {
            "vol": 0,
            "bnd": 0,
            "bbnd": 0,
            "bbbnd": 0
        }

        DEFAULT.update(default)

        super().__init__(mesh, root, **DEFAULT)

    @dream_configuration
    def vol(self) -> int:
        return self._vol

    @vol.setter
    def vol(self, vol: int):
        if vol < 0:
            raise ValueError("Integration order must be non-negative!")
        self._vol = int(vol)

    @dream_configuration
    def bnd(self) -> int:
        return self._bnd

    @bnd.setter
    def bnd(self, bnd: int):
        if bnd < 0:
            raise ValueError("Integration order must be non-negative!")
        self._bnd = int(bnd)

    @dream_configuration
    def bbnd(self) -> int:
        return self._bbnd

    @bbnd.setter
    def bbnd(self, bbnd: int):
        if bbnd < 0:
            raise ValueError("Integration order must be non-negative!")
        self._bbnd = int(bbnd)

    @dream_configuration
    def bbbnd(self) -> int:
        return self._bbbnd

    @bbbnd.setter
    def bbbnd(self, bbbnd: int):
        if bbbnd < 0:
            raise ValueError("Integration order must be non-negative!")
        self._bbbnd = int(bbbnd)


class Compile(Configuration):

    def __init__(self, mesh, root=None, **default):

        DEFAULT = {
            "realcompile": False,
            "wait": False,
            "keep_files": False
        }

        DEFAULT.update(default)
        super().__init__(mesh, root, **DEFAULT)

    @dream_configuration
    def realcompile(self) -> bool:
        return self._realcompile

    @realcompile.setter
    def realcompile(self, flag: bool):
        self._realcompile = bool(flag)

    @dream_configuration
    def wait(self) -> bool:
        return self._wait

    @wait.setter
    def wait(self, flag: bool):
        self._wait = bool(flag)

    @dream_configuration
    def keep_files(self) -> bool:
        return self._keep_files

    @keep_files.setter
    def keep_files(self, flag: bool):
        self._keep_files = bool(flag)


class Optimizations(Configuration):

    def __init__(self, mesh, root=None, **default):

        DEFAULT = {
            "compile": Compile(mesh, root),
            "static_condensation": False,
            "bonus_int_order": BonusIntegrationOrder(mesh, root)

        }
        DEFAULT.update(default)

        super().__init__(mesh, root, **DEFAULT)

    @dream_configuration
    def compile(self) -> Compile:
        return self._compile

    @compile.setter
    def compile(self, compile: Compile):
        if not isinstance(compile, Compile):
            raise TypeError("Compile must be of type Compile!")
        self._compile = compile

    @dream_configuration
    def static_condensation(self) -> bool:
        return self._static_condensation

    @static_condensation.setter
    def static_condensation(self, static_condensation: bool):
        self._static_condensation = bool(static_condensation)

    @dream_configuration
    def bonus_int_order(self) -> BonusIntegrationOrder:
        return self._bonus_int_order

    @bonus_int_order.setter
    def bonus_int_order(self, bonus_int_order: BonusIntegrationOrder):
        if not isinstance(bonus_int_order, BonusIntegrationOrder):
            raise TypeError("BonusIntegrationOrder must be of type BonusIntegrationOrder!")
        self._bonus_int_order = bonus_int_order


class Solver(Configuration, is_interface=True):

    root: SolverConfiguration

    def inverse(self, blf: ngs.BilinearForm, fes: ngs.FESpace, freedofs: ngs.BitArray = None, **kwargs):
        raise NotImplementedError()

    def initialize(self, blf: ngs.BilinearForm, rhs: ngs.LinearForm, gfu: ngs.GridFunction, **kwargs):
        ...

    def solve(self, t: float | None = None) -> typing.Generator[int | None, None, None]:
        raise NotImplementedError()


# ------- Linear Solvers ------- #


class LinearSolver(Solver, is_interface=True):

    root: SolverConfiguration

    def initialize(self, blf: ngs.BilinearForm, rhs: ngs.LinearForm, gfu: ngs.GridFunction, **kwargs):
        self.blf = blf
        self.rhs = rhs
        self.gfu = gfu

        self.blf.Assemble()
        rhs.vec.data -= blf.mat * gfu.vec

    def solve(self, t: float | None = None) -> typing.Generator[int | None, None, None]:
        yield None
        self.gfu.vec.data += self.inverse(self.blf, self.gfu.space) * self.rhs.vec


class DirectLinearSolver(LinearSolver):

    def inverse(self, blf: ngs.BilinearForm, fes: ngs.FESpace, freedofs: ngs.BitArray = None, **kwargs):
        if freedofs is None:
            freedofs = fes.FreeDofs(blf.condense)
        return blf.mat.Inverse(freedofs, inverse=self.name)


class UmfpackLinearSolver(DirectLinearSolver):

    name = "umfpack"


class PardisoLinearSolver(DirectLinearSolver):

    name = "pardiso"


# ------- Nonlinear Solvers ------- #


class NonlinearMethod(Configuration, is_interface=True):

    root: SolverConfiguration

    def initialize(self, gfu: ngs.GridFunction, du):
        self.gfu = gfu
        self.du = du

    def update_solution(self):
        raise NotImplementedError()


class NewtonsMethod(NonlinearMethod):

    name: str = "newton"

    def __init__(self, mesh, root=None, **default):

        DEFAULT = {
            "damping_factor": 1
        }
        DEFAULT.update(default)

        super().__init__(mesh, root, **DEFAULT)

    @dream_configuration
    def damping_factor(self) -> float:
        return self._damping_factor

    @damping_factor.setter
    def damping_factor(self, damping_factor: float):
        if damping_factor <= 0:
            raise ValueError("Damping factor must be greater than zero!")
        self._damping_factor = float(damping_factor)

    def update_solution(self):
        self.gfu.vec.data -= self.damping_factor * self.du


class NonlinearSolver(Solver, is_interface=True):

    root: SolverConfiguration

    def __init__(self, mesh, root=None, **default):

        DEFAULT = {
            "method": NewtonsMethod(mesh, root),
            "max_iterations": 10,
            "convergence_criterion": 1e-8
        }
        DEFAULT.update(default)

        super().__init__(mesh, root, **DEFAULT)

    @dream_configuration
    def method(self) -> NewtonsMethod:
        return self._method

    @method.setter
    def method(self, method: str | NonlinearMethod):
        OPTIONS = [NewtonsMethod]
        self._method = self._get_configuration_option(method, OPTIONS, NonlinearMethod)

    @dream_configuration
    def max_iterations(self) -> int:
        return self._max_iterations

    @max_iterations.setter
    def max_iterations(self, max_iterations: int):
        if max_iterations <= 0:
            raise ValueError("Max iterations must be greater than zero!")
        self._max_iterations = int(max_iterations)

    @dream_configuration
    def convergence_criterion(self) -> float:
        return self._convergence_criterion

    @convergence_criterion.setter
    def convergence_criterion(self, convergence_criterion: float):
        if convergence_criterion <= 0:
            raise ValueError("Convergence Criterion must be greater than zero!")
        self._convergence_criterion = float(convergence_criterion)

    def log_iteration_error(self, it: int | None = None, t: float | None = None, s: int | None = None):
        msg = f"residual: {self.error:8e}"
        if s is not None:
            msg += f" | stage: {s}"
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

    def solve(self, t: float | None = None, s: int | None = None) -> typing.Generator[int | None, None, None]:

        self.reset_status()

        for it in range(self.max_iterations):
            yield it

            self.solve_update_step()
            self.log_iteration_error(it, t, s)

            if self.is_nan:
                break

            self.method.update_solution()

            if self.is_converged:
                break

    def solve_update_step(self):
        raise NotImplementedError()


class DirectNonlinearSolver(NonlinearSolver):

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


class FiniteElementMethod(Configuration, is_interface=True):

    root: SolverConfiguration

    def __init__(self, mesh, root=None, **default):
        DEFAULT = {
            "order": 2,
        }

        DEFAULT.update(default)

        super().__init__(mesh, root, **DEFAULT)

    @dream_configuration
    def order(self) -> int:
        return self._order

    @order.setter
    def order(self, order: int):
        self._order = int(order)

    @property
    def scheme(self) -> Scheme | TimeSchemes:
        raise NotImplementedError("Overload this configuration in derived class!")

    def initialize(self) -> None:
        self.initialize_finite_element_spaces()
        self.initialize_trial_and_test_functions()
        self.initialize_gridfunctions()
        self.initialize_time_scheme_gridfunctions()

        self.set_boundary_conditions()
        self.set_initial_conditions()

        self.initialize_symbolic_forms()

    def initialize_finite_element_spaces(self) -> None:

        self.spaces = {}
        self.add_finite_element_spaces(self.spaces)

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

    def initialize_time_scheme_gridfunctions(self, *spaces: str) -> None:
        if isinstance(self.scheme, TimeSchemes):
            gfus = {space: gfu for space, gfu in self.gfus.items() if space in spaces}
            self.scheme.initialize_gridfunctions(gfus)

    def initialize_symbolic_forms(self) -> None:
        self.blf: Integrals = {label: {} for label in self.spaces}
        self.lf: Integrals = {label: {} for label in self.spaces}

        self.add_symbolic_spatial_forms(self.blf, self.lf)

        if isinstance(self.scheme, TimeSchemes):
            self.scheme.add_symbolic_temporal_forms(self.blf, self.lf)

    def set_initial_conditions(self) -> None:
        # Make sure to call the upper class method after having set the initial conditions
        if isinstance(self.scheme, TimeSchemes):
            self.scheme.set_initial_conditions()

    def add_finite_element_spaces(self, spaces: dict[str, ngs.FESpace]):
        raise NotImplementedError("Overload this method in derived class!")

    def add_symbolic_spatial_forms(self, blf: Integrals, lf: Integrals):
        raise NotImplementedError("Overload this method in derived class!")

    def get_solution_fields(self) -> ngsdict:
        raise NotImplementedError("Overload this method in derived class!")

    def set_boundary_conditions(self) -> None:
        raise NotImplementedError("Overload this method in derived class!")


# ------- Solver Configuration ------- #


class SolverConfiguration(Configuration, is_interface=True):

    def __init__(self, mesh, bcs: BoundaryConditions, dcs: DomainConditions, **default):

        mesh.normal = ngs.specialcf.normal(mesh.dim)
        mesh.tangential = ngs.specialcf.tangential(mesh.dim)
        mesh.meshsize = ngs.specialcf.mesh_size
        mesh.is_periodic = is_mesh_periodic(mesh)

        DEFAULT = {
            "time": StationaryRoutine(mesh, self),
            "linear_solver": UmfpackLinearSolver(mesh, self),
            "nonlinear_solver": UmfpackNonlinearSolver(mesh, self),
            "optimizations": Optimizations(mesh, self),
            "io": IOConfiguration(mesh, self),
            "info": {},
        }
        DEFAULT.update(default)

        self.bcs = bcs
        self.dcs = dcs

        super().__init__(mesh, self, **DEFAULT)

    @property
    def fem(self) -> FiniteElementMethod:
        raise NotImplementedError("Overload this configuration in derived class!")

    @dream_configuration
    def time(self) -> TimeRoutine:
        return self._time

    @time.setter
    def time(self, time: str | TimeRoutine):
        OPTIONS = [StationaryRoutine, TransientRoutine, PseudoTimeSteppingRoutine]
        self._time = self._get_configuration_option(time, OPTIONS, TimeRoutine)

    @dream_configuration
    def linear_solver(self) -> UmfpackLinearSolver | PardisoLinearSolver:
        return self._linear_solver

    @linear_solver.setter
    def linear_solver(self, solver: str | LinearSolver):
        OPTIONS = [UmfpackLinearSolver, PardisoLinearSolver]
        self._linear_solver = self._get_configuration_option(solver, OPTIONS, LinearSolver)

    @dream_configuration
    def nonlinear_solver(self) -> UmfpackNonlinearSolver | PardisoNonlinearSolver:
        return self._nonlinear_solver

    @nonlinear_solver.setter
    def nonlinear_solver(self, solver: str | NonlinearSolver):
        OPTIONS = [UmfpackNonlinearSolver, PardisoNonlinearSolver]
        self._nonlinear_solver = self._get_configuration_option(solver, OPTIONS, NonlinearSolver)

    @dream_configuration
    def optimizations(self) -> Optimizations:
        return self._optimizations

    @optimizations.setter
    def optimizations(self, optimizations: Optimizations):
        if not isinstance(optimizations, Optimizations):
            raise TypeError("Optimizations must be of type Optimizations!")
        self._optimizations = optimizations

    @dream_configuration
    def io(self) -> IOConfiguration:
        return self._io

    @io.setter
    def io(self, io: IOConfiguration):
        if not isinstance(io, IOConfiguration):
            raise TypeError("IOConfiguration must be of type IOConfiguration!")
        self._io = io

    @dream_configuration
    def info(self) -> dict:
        return self._info

    @info.setter
    def info(self, info: dict):

        if not isinstance(info, dict):
            raise ValueError("Info must be a dictionary!")

        self._info = info

    def initialize(self) -> None:

        if self.mesh.is_periodic and not self.root.bcs.has_condition(Periodic):
            raise ValueError("Mesh has periodic boundaries, but no periodic boundary conditions are set!")

        self.fem.initialize()

    def get_solution_fields(self, *fields) -> ngsdict:

        uh = self.fem.get_solution_fields()

        fields_ = type(uh)()
        for field_ in fields:
            if field_ in uh:
                fields_[field_] = uh[field_]
            elif hasattr(uh, field_):
                fields_[field_] = getattr(uh, field_)
            else:
                logger.info(f"Field {field_} not predefined!")

        return fields_

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
