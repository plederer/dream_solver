from __future__ import annotations
import typing
import logging

from collections import UserDict
from functools import wraps

import ngsolve as ngs

from dream import bla
from dream import mesh as dmesh
from dream.config import BaseConfig, Formatter, STATE
from dream.time_schemes import TransientGridfunction

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from dream.solver import SolverConfiguration

# ------- State -------- #


# ------- Dynamic Configuration ------- #


# ------- Dynamic Equations ------- #


class DynamicEquations:

    types: dict[str, DynamicEquations]
    formatter: Formatter

    def __init_subclass__(cls, labels: list[str] = []):

        if not labels:
            cls.types = {}
            cls.logger = logger.getChild(cls.__name__)
            cls.formatter = Formatter()

        for label in labels:
            cls.types[label] = cls

    def __str__(self) -> str:
        return self.__class__.__name__.upper()


# ------- Equations ------- #

def equation(func):

    @wraps(func)
    def state(self, state: STATE, *args, **kwargs):

        _state = getattr(state, func.__name__, None)

        if _state is not None:
            name = " ".join(func.__name__.split("_")).capitalize()
            logger.debug(f"{name} set by user! Returning it.")
            return _state

        _state = func(self, state, *args, **kwargs)

        if _state is None:
            raise NotImplementedError(f"Can not determine {func.__name__} from given state!")

        return _state

    return state


class FlowEquations:

    logger = logger.getChild("Equations")

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return str(self)


# ------- Boundary Conditions ------- #


class FormulationBC:

    @classmethod
    def list(cls):
        return [key for key, bc in vars(cls).items() if isinstance(bc, type) and issubclass(bc, dmesh.Boundary)]


# ------- Domain Conditions ------- #


class FormulationDC:

    @classmethod
    def list(cls):
        return [key for key, dc in vars(cls).items() if isinstance(dc, type) and issubclass(dc, dmesh.Domain)]


# ------- Formulations ------- #

class SpaceInterface:

    @property
    def fes(self):
        return self._fes

    @property
    def gfu(self):
        return self._fes

    @property
    def trial(self):
        return self._TnT[0]

    @property
    def test(self):
        return self._TnT[1]

    @property
    def dt(self) -> TransientGridfunction | None:
        return self._dt

    def initialize(self):
        self._fes = self.get_space()
        self._TnT = self.get_TnT()
        self._gfu = self.get_gridfunction()

    def initialize_time_derivative(self):
        self._dt = self.get_transient_gridfunction()

    def get_space(self) -> ngs.FESpace:
        raise NotImplementedError()

    def get_TnT(self) -> tuple[ngs.CF, ngs.CF]:
        return self.fes.TnT()

    def get_gridfunction(self) -> ngs.GridFunction:
        return ngs.GridFunction(self.fes, str(self))

    def get_transient_gridfunction(self) -> TransientGridfunction | None:
        return None

    def update_time_step(self):
        if self.dt:
            self.dt.update_time_step()

    def update_initial(self):
        if self.dt:
            self.dt.update_initial()

    def add_mass_bilinearform(self, blf: ngs.BilinearForm, dx=ngs.dx, **dx_kwargs):
        blf += bla.inner(self.trial, self.test) * dx(**dx_kwargs)


class Space(SpaceInterface):

    def __init__(self, cfg: SolverConfiguration, dmesh: dmesh.DreamMesh):
        self.cfg = cfg
        self.dmesh = dmesh

    def get_state_from_variable(self, gfu: ngs.GridFunction = None) -> STATE:
        raise NotImplementedError()

    def get_variable_from_state(self, state: STATE) -> ngs.CF:
        raise NotImplementedError()

    def add_mass_linearform(self, state: STATE, lf: ngs.LinearForm, dx=ngs.dx, **dx_kwargs):
        f = self.get_variable_from_state(state)
        lf += bla.inner(f, self.test) * dx(**dx_kwargs)

    def __str__(self):
        return self.__class__.__name__.lower()


class Spaces(SpaceInterface, UserDict):

    def initialize(self) -> None:
        super().initialize()
        self._broadcast_space_components()

    def get_space(self) -> ngs.ProductSpace:
        fes = [space.get_space() for space in self.values()]

        if len(fes) == 0:
            raise ValueError("Spaces container is empty!")

        for space in fes[1:]:
            fes[0] *= space

        return fes

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

    def add_mass_linearform(self, state: STATE, lf: ngs.LinearForm, **dx_kwargs):
        for space in self.values():
            space.add_mass_linearform(state, lf, **dx_kwargs)

    def _broadcast_space_components(self):

        spaces = list(self.values())

        match spaces:

            case []:
                raise ValueError("Spaces container is empty!")

            case [space]:
                space.fes = self.fes
                space.TnT = self.TnT
                space.gfu = self.gfu

            case _:

                for space, fes_, trial, test, gfu_ in zip(spaces, self.fes.components, *self.TnT, self.gfu.components):
                    space.fes = fes_
                    space.TnT = (trial, test)
                    space.gfu = gfu_

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


SPACE = typing.TypeVar("SPACE", bound=Space | Spaces)


class Formulation:

    types: dict[str, Formulation]

    def __init_subclass__(cls, label: str = ""):

        if not label:
            cls.types = {}

        if label:
            cls.types[label] = cls

    def __init__(self, cfg: SolverConfiguration, mesh: ngs.Mesh | dmesh.DreamMesh) -> None:

        if isinstance(mesh, ngs.Mesh):
            mesh = dmesh.DreamMesh(mesh)

        self._cfg = cfg
        self._mesh = mesh
        self._spaces = None

        self.normal = ngs.specialcf.normal(mesh.dim)
        self.tangential = ngs.specialcf.tangential(mesh.dim)
        self.mesh_size = ngs.specialcf.mesh_size

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

    def assemble_system(self, blf: ngs.BilinearForm, lf: ngs.LinearForm):
        raise NotImplementedError()

    def initialize(self):
        self._spaces = self.get_space()
        self.spaces.initialize()

    def __str__(self) -> str:
        return self.__class__.__name__


# ------- Configuration ------- #


class FlowConfig(BaseConfig):

    types: dict[str, FlowConfig] = {}
    formulation: Formulation
    bcs: FormulationBC
    dcs: FormulationDC

    def __init_subclass__(cls, label: str) -> None:
        cls.types[label] = cls

    def __init__(self, equations) -> None:
        self._equations = equations
        self.formulation = list(type(self).formulation.types.keys())[0]

    @property
    def formulation(self) -> str:
        return self._formulation

    @formulation.setter
    def formulation(self, formulation: str) -> str:
        self._formulation = self._is_type(formulation, type(self).formulation)

    def get_formulation(self, cfg: SolverConfiguration, mesh: ngs.Mesh | dmesh.DreamMesh, *args, **kwargs) -> Formulation:
        return self._get_type(self.formulation, type(self).formulation, cfg=cfg, mesh=mesh, *args, **kwargs)

    @property
    def _formulation_type(self) -> Formulation:
        raise NotImplementedError()

    def __repr__(self) -> str:
        return str(self)
