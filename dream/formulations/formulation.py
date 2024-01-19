from __future__ import annotations
import typing
import logging

from collections import UserDict
from functools import wraps

import ngsolve as ngs

from dream import bla
from dream import mesh as dmesh
from dream.config import MultipleConfiguration, State, InterfaceConfiguration
from dream.time_schemes import TransientGridfunction

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from dream.solver import SolverConfiguration

# ------- State -------- #


# ------- Equations ------- #

def equation(func):
    """ Equation is a decorator that wraps a function which takes as first argument a state.

        The name of the function should ressemble a physical quantity like 'density' or 'velocity'.

        When the decorated function get's called the wrapper checks first if the quantity is already defined
        and returns it. 
        If the quantity is not defined, the wrapper executes the decorated function, which should
        return a valid value. Otherwise a ValueError is thrown.
    """

    @wraps(func)
    def state(self, state: State, *args, **kwargs):

        _state = state.data.get(func.__name__, None)

        if _state is not None:
            name = " ".join(func.__name__.split("_")).capitalize()
            logger.debug(f"{name} set by user! Returning it.")
            return _state

        _state = func(self, state, *args, **kwargs)

        if _state is None:
            raise ValueError(f"Can not determine {func.__name__} from given state!")

        return _state

    return state


class Equations(InterfaceConfiguration, is_interface=True):
    
    logger: logging.Logger


# ------- Boundary Conditions ------- #


class BoundaryConditions:

    @classmethod
    def list(cls):
        return [key for key, bc in vars(cls).items() if isinstance(bc, type) and issubclass(bc, dmesh.Boundary)]


# ------- Domain Conditions ------- #


class DomainConditions:

    @classmethod
    def list(cls):
        return [key for key, dc in vars(cls).items() if isinstance(dc, type) and issubclass(dc, dmesh.Domain)]


# ------- Formulations ------- #

class Space:

    @property
    def fes(self):
        return self._fes

    @property
    def gfu(self):
        return self._gfu

    @property
    def trial(self):
        return self._TnT[0]

    @property
    def test(self):
        return self._TnT[1]

    def __init__(self, cfg: SolverConfiguration, dmesh: dmesh.DreamMesh):
        self.cfg = cfg
        self.dmesh = dmesh

    def initialize(self):
        self._fes = self.get_space()
        self._TnT = self.get_TnT()
        self._gfu = self.get_gridfunction()

    def get_space(self) -> ngs.FESpace:
        raise NotImplementedError()

    def get_TnT(self) -> tuple[ngs.CF, ngs.CF]:
        return self.fes.TnT()

    def get_gridfunction(self) -> ngs.GridFunction:
        return ngs.GridFunction(self.fes, str(self))

    def get_state_from_variable(self, gfu: ngs.GridFunction = None) -> State:
        raise NotImplementedError()

    def get_variable_from_state(self, state: State) -> ngs.CF:
        raise NotImplementedError()

    def get_trial_as_state(self) -> State:
        self.trial_state = self.get_state_from_variable(self.trial)

    def add_mass_bilinearform(self, blf: ngs.BilinearForm, dx=ngs.dx, **dx_kwargs):
        blf += bla.inner(self.trial, self.test) * dx(**dx_kwargs)

    def add_mass_linearform(self, state: State, lf: ngs.LinearForm, dx=ngs.dx, **dx_kwargs):
        f = self.get_variable_from_state(state)
        lf += bla.inner(f, self.test) * dx(**dx_kwargs)

    def __str__(self):
        return self.__class__.__name__.lower()


class TransientSpace(Space):

    @property
    def dt(self) -> TransientGridfunction | None:
        return self._dt

    def get_transient_gridfunction(self) -> TransientGridfunction | None:
        raise NotImplementedError()

    def initialize_time_derivative(self):
        self._dt = self.get_transient_gridfunction()

    def update_time_step(self):
        if self.dt:
            self.dt.update_time_step()

    def update_initial(self):
        if self.dt:
            self.dt.update_initial()


class Spaces(UserDict):

    @property
    def fes(self):
        return self._fes

    @property
    def gfu(self):
        return self._gfu

    @property
    def trial(self):
        return self._TnT[0]

    @property
    def test(self):
        return self._TnT[1]

    def initialize(self) -> None:
        self._fes = self.get_space()
        self._TnT = self._fes.TnT()
        self._gfu = ngs.GridFunction(self._fes, )
        self._broadcast_components()

    def get_space(self) -> ngs.ProductSpace:
        fes = [space.get_space() for space in self.values()]

        if not fes:
            raise ValueError("Spaces container is empty!")

        for space in fes[1:]:
            fes[0] *= space

        return fes[0]

    def initialize_time_derivative(self):
        for space in self.values():
            space.initialize_time_derivative()

    def update_time_step(self):
        for space in self.values():
            space.update_time_step()

    def update_initial(self):
        for space in self.values():
            space.update_initial()

    def add_mass_bilinearform(self, blf: ngs.BilinearForm, **dx_kwargs):
        for space in self.values():
            space.add_mass_bilinearform(blf, **dx_kwargs)

    def add_mass_linearform(self, state: State, lf: ngs.LinearForm, **dx_kwargs):
        for space in self.values():
            space.add_mass_linearform(state, lf, **dx_kwargs)

    def _broadcast_components(self):

        spaces = list(self.values())

        match spaces:

            case [space]:
                space._fes = self._fes
                space._TnT = self._TnT
                space._gfu = self._gfu

            case _:

                for space, fes_, trial, test, gfu_ in zip(
                        spaces, self.fes.components, *self.TnT, self.gfu.components, strict=True):
                    space._fes = fes_
                    space._TnT = (trial, test)
                    space._gfu = gfu_

    def values(self) -> typing.ValuesView[Space]:
        return super().values()

    def __setitem__(self, key, item) -> None:
        if item is None:
            return

        if not isinstance(item, Space):
            raise TypeError(f"Not of type '{Space}'!")

        return super().__setitem__(key, item)

    def __getitem__(self, key) -> None:
        space = self.data.get(key, None)

        if space is None:
            raise ValueError(f"Space '{key}' not set for current configuration!")

        return space


class Formulation(InterfaceConfiguration, is_interface=True):

    def initialize(self, cfg: SolverConfiguration, mesh: ngs.Mesh | dmesh.DreamMesh) -> None:

        if isinstance(mesh, ngs.Mesh):
            mesh = dmesh.DreamMesh(mesh)

        self._cfg = cfg
        self._mesh = mesh
        self._spaces = None

    # def initialize(self):
    #     self._spaces = self.get_space()
    #     self.spaces.initialize()

    #     self.normal = ngs.specialcf.normal(mesh.dim)
    #     self.tangential = ngs.specialcf.tangential(mesh.dim)
    #     self.mesh_size = ngs.specialcf.mesh_size

    @property
    def dmesh(self) -> dmesh.DreamMesh:
        return self._mesh

    @property
    def mesh(self) -> ngs.Mesh:
        return self.dmesh.ngsmesh

    @property
    def cfg(self) -> SolverConfiguration:
        return self._cfg

    @property
    def spaces(self) -> Spaces | Space:
        return self._spaces

    def get_space(self) -> Spaces | Space:
        raise NotImplementedError()

    def get_system(self, blf: list[ngs.comp.SumOfIntegrals], lf: list[ngs.comp.SumOfIntegrals]):
        raise NotImplementedError()


# ------- Configuration ------- #


class PDEConfiguration(MultipleConfiguration, is_interface=True):

    bcs: BoundaryConditions
    dcs: DomainConditions

    @property
    def formulation(self):
        raise NotImplementedError('Override formulation')
