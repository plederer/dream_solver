from __future__ import annotations
import ngsolve as ngs
import logging
import typing

from math import isnan

from .mesh import is_mesh_periodic, Periodic, BoundaryConditions, DomainConditions
from .config import Configuration, dream_configuration, ngsdict, Integrals
from .time import StationaryRoutine, TransientRoutine, PseudoTimeSteppingRoutine, TimeRoutine, Scheme, TimeSchemes
from .io import IOConfiguration

logger = logging.getLogger(__name__)


# ------- Nonlinear Methods ------- #
class NonlinearMethod(Configuration, is_interface=True):

    root: SolverConfiguration

    def __init__(self, mesh, root=None, **default):

        DEFAULT = {
            "max_iterations": 10,
            "convergence_criterion": 1e-8
        }
        DEFAULT.update(default)

        super().__init__(mesh, root, **DEFAULT)

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

    def update_solution(self, gfu: ngs.GridFunction, update: ngs.BaseVector):
        raise NotImplementedError("Overload this configuration in derived class!")


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

    def update_solution(self, gfu: ngs.GridFunction, update: ngs.BaseVector):
        gfu.vec.data -= self.damping_factor * update

# ------- Solvers ------- #


class Solver(Configuration, is_interface=True):

    root: SolverConfiguration

    def __init__(self, mesh, root=None, **default):

        DEFAULT = {
            'method': NewtonsMethod(mesh, root=root),
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

    def initialize_nonlinear_routine(self,
                                     blf: ngs.BilinearForm,
                                     gfu: ngs.GridFunction,
                                     rhs: ngs.BaseVector = None,
                                     **kwargs):

        if rhs is not None:
            if isinstance(rhs, ngs.GridFunction):
                rhs = rhs.vec
            elif isinstance(rhs, ngs.BaseVector):
                ...
            else:
                raise TypeError("Input rhs must be of type ngs.GridFunction or ngs.BaseVector, or None.")

        self.blf = blf
        self.gfu = gfu
        self.fes = gfu.space
        self.rhs = rhs

        self.res = gfu.vec.CreateVector()
        self.res[:] = 0.0
        self.du = gfu.vec.CreateVector()
        self.du[:] = 0.0

    def solve_nonlinear_system(self) -> typing.Generator[dict[str, float | int], None, None]:
        """ Solves a nonlinear system using the specified method.

        Yields a dictionary containing the current iteration number and the error of the update step.
        """

        for it in range(self.method.max_iterations):

            self.solve_update_step()
            error = self.get_iteration_error(self.du, self.res)

            if isnan(error):
                logger.error("Solution process diverged!")
                break

            self.method.update_solution(self.gfu, self.du)

            yield {'it': it, 'error': error}

            if error < self.method.convergence_criterion:
                logger.debug(f"Solution process converged!")
                break

        if it + 1 == self.method.max_iterations:
            logger.warning(f"Solution process did not converge after {self.method.max_iterations} iterations!")

    def get_iteration_error(self, update: ngs.BaseVector, residual: ngs.BaseVector) -> float:
        return ngs.sqrt(ngs.InnerProduct(update, residual)**2)

    def get_inverse(self, blf: ngs.BilinearForm, fes: ngs.FESpace, freedofs: ngs.BitArray = None, **kwargs):
        raise NotImplementedError("Overload this configuration in derived class!")

    def solve_linear_system(self, blf: ngs.BilinearForm, gfu: ngs.GridFunction, rhs: ngs.BaseVector, **kwargs):
        raise NotImplementedError("Overload this configuration in derived class!")

    def solve_update_step(self) -> None:
        raise NotImplementedError("Overload this configuration in derived class!")


class DirectSolver(Solver):

    name: str = "direct"

    def __init__(self, mesh, root=None, **default):

        DEFAULT = {
            "inverse": "",
        }
        DEFAULT.update(default)

        super().__init__(mesh, root, **DEFAULT)

    @dream_configuration
    def inverse(self) -> str:
        return self._inverse

    @inverse.setter
    def inverse(self, inverse: str):
        OPTIONS = ["", "umfpack", "pardiso", "mumps", "sparsecholesky"]

        if inverse not in OPTIONS:
            raise ValueError(f"Inverse must be one of {OPTIONS}!")

        self._inverse = inverse

    def get_inverse(self, blf: ngs.BilinearForm, fes: ngs.FESpace, freedofs: ngs.BitArray = None, **kwargs):

        inverse = self.inverse
        if inverse in kwargs:
            inverse = kwargs[inverse]

        if freedofs is None:
            freedofs = fes.FreeDofs(blf.condense)

        return blf.mat.Inverse(freedofs, inverse=inverse)

    def solve_linear_system(self, blf: ngs.BilinearForm,
                            gfu: ngs.GridFunction,
                            rhs: ngs.BaseVector,
                            operator: str = "=",
                            **kwargs):

        inv = self.get_inverse(blf, gfu.space, **kwargs)
        if blf.condense:
            ext = ngs.IdentityMatrix() + blf.harmonic_extension
            extT = ngs.IdentityMatrix() + blf.harmonic_extension_trans
            inv = ext @ inv @ extT + blf.inner_solve

        match operator:
            case '=':
                gfu.vec.data = inv * rhs
            case '+=':
                gfu.vec.data += inv * rhs
            case '-=':
                gfu.vec.data -= inv * rhs
            case _:
                raise ValueError(f"Operator {operator} not supported!")

    def solve_update_step(self):

        self.blf.Apply(self.gfu.vec, self.res)
        if self.rhs is not None:
            self.res.data -= self.rhs
        self.blf.AssembleLinearization(self.gfu.vec)

        inv = self.blf.mat.Inverse(freedofs=self.fes.FreeDofs(self.blf.condense), inverse=self.inverse)
        if self.blf.condense:
            self.res.data += self.blf.harmonic_extension_trans * self.res
            self.du.data = inv * self.res
            self.du.data += self.blf.harmonic_extension * self.du
            self.du.data += self.blf.inner_solve * self.res
        else:
            self.du.data = inv * self.res


# ------- Finite Element Method ------- #


class FiniteElementMethod(Configuration, is_interface=True):

    root: SolverConfiguration

    SOLVERS = (DirectSolver,)

    def __init__(self, mesh, root=None, **default):

        self._bonus_int_order = {}

        DEFAULT = {
            'order': 2,
            'solver': DirectSolver(mesh, root=root, method='newton'),
            'static_condensation': False,
            'bonus_int_order': 0,
        }

        DEFAULT.update(default)

        super().__init__(mesh, root, **DEFAULT)

    @dream_configuration
    def order(self) -> int:
        return self._order

    @order.setter
    def order(self, order: int):
        self._order = int(order)

    @dream_configuration
    def solver(self) -> Solver:
        return self._solver

    @solver.setter
    def solver(self, solver: int):
        self._solver = self._get_configuration_option(solver, self.SOLVERS, Solver)

    @dream_configuration
    def static_condensation(self) -> bool:
        return self._static_condensation

    @static_condensation.setter
    def static_condensation(self, static_condensation: bool):
        self._static_condensation = bool(static_condensation)

    @dream_configuration
    def bonus_int_order(self) -> dict[str, dict[str, int]]:
        return self._bonus_int_order

    @bonus_int_order.setter
    def bonus_int_order(self, order: int) -> None:

        vorb = ('vol', 'bnd')

        if isinstance(order, int):
            for term in self._bonus_int_order:
                self._bonus_int_order[term].update(dict.fromkeys(vorb, order))

        elif isinstance(order, dict):
            self._bonus_int_order.update(order)

        elif isinstance(order, (tuple, list)):
            self._bonus_int_order = {term: dict.fromkeys(vorb, 0) for term in order}

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
    def time(self) -> StationaryRoutine | TransientRoutine | PseudoTimeSteppingRoutine:
        return self._time

    @time.setter
    def time(self, time: str | TimeRoutine):
        OPTIONS = [StationaryRoutine, TransientRoutine, PseudoTimeSteppingRoutine]
        self._time = self._get_configuration_option(time, OPTIONS, TimeRoutine)

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
