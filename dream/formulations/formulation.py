from __future__ import annotations
import typing
import logging
from typing import ValuesView

import ngsolve as ngs

from dream import bla
from dream.state import DescriptorData, Descriptor, State, equation
from dream.config import BaseConfig, Formatter
from dream.mesh import DreamMesh, Boundary, Domain
from dream.time_schemes import TransientGridfunction

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from dream.solver import SolverConfiguration


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


class FlowEquations:

    logger = logger.getChild("Equations")

    @equation()
    def velocity(self, state: State):

        rho = state.density
        rho_u = state.momentum

        if State.is_set(rho, rho_u):
            self.logger.debug("Returning velocity from density and momentum.")

            if bla.is_zero(rho_u) and bla.is_zero(rho):
                return bla.as_vector((0.0 for _ in range(rho_u.dim)))

            return rho_u/rho

    @equation()
    def momentum(self, state: State):

        rho = state.density
        u = state.velocity

        if State.is_set(rho, u):
            self.logger.debug("Returning momentum from density and velocity.")
            return rho * u

    @equation()
    def inner_energy(self, state: State):

        rho_E = state.energy
        rho_Ek = state.kinetic_energy

        if State.is_set(rho_E, rho_Ek):
            self.logger.debug("Returning bla.inner energy from energy and kinetic energy.")
            return rho_E - rho_Ek

    @equation()
    def specific_inner_energy(self, state: State):

        rho = state.density
        rho_Ei = state.inner_energy

        Ek = state.specific_kinetic_energy
        E = state.specific_energy

        if State.is_set(rho, rho_Ei):
            self.logger.debug("Returning specific bla.inner energy from bla.inner energy and density.")
            return rho_Ei/rho

        elif State.is_set(E, Ek):
            self.logger.debug("Returning specific bla.inner energy from specific energy and specific kinetic energy.")
            return E - Ek

    @equation()
    def kinetic_energy(self, state: State):

        rho = state.density
        rho_u = state.momentum
        u = state.velocity

        rho_E = state.energy
        rho_Ei = state.inner_energy

        Ek = state.specific_kinetic_energy

        if State.is_set(rho, u):
            self.logger.debug("Returning kinetic energy from density and velocity.")
            return 0.5 * rho * bla.inner(u, u)

        elif State.is_set(rho, rho_u):
            self.logger.debug("Returning kinetic energy from density and momentum.")
            return 0.5 * bla.inner(rho_u, rho_u)/rho

        elif State.is_set(rho_E, rho_Ei):
            self.logger.debug("Returning kinetic energy from energy and bla.inner energy.")
            return rho_E - rho_Ei

        elif State.is_set(rho, Ek):
            self.logger.debug("Returning kinetic energy from density and specific kinetic energy.")
            return rho * Ek

    @equation()
    def specific_kinetic_energy(self, state: State):

        rho = state.density
        rho_u = state.momentum
        u = state.velocity

        E = state.specific_energy
        Ei = state.specific_inner_energy
        rho_Ek = state.kinetic_energy

        if State.is_set(u):
            self.logger.debug("Returning specific kinetic energy from velocity.")
            return 0.5 * bla.inner(u, u)

        elif State.is_set(rho, rho_u):
            self.logger.debug("Returning specific kinetic energy from density and momentum.")
            return 0.5 * bla.inner(rho_u, rho_u)/rho**2

        elif State.is_set(rho, rho_Ek):
            self.logger.debug("Returning specific kinetic energy from density and kinetic energy.")
            return rho_Ek/rho

        elif State.is_set(E, Ei):
            self.logger.debug("Returning specific kinetic energy from specific energy and speicific bla.inner energy.")
            return E - Ei

    @equation()
    def energy(self, state: State):

        rho = state.density
        E = state.specific_energy

        rho_Ei = state.inner_energy
        rho_Ek = state.kinetic_energy

        if State.is_set(rho, E):
            self.logger.debug("Returning energy from density and specific energy.")
            return rho * E

        elif State.is_set(rho_Ei, rho_Ek):
            self.logger.debug("Returning energy from bla.inner energy and kinetic energy.")
            return rho_Ei + rho_Ek

    @equation()
    def specific_energy(self, state: State):

        rho = state.density
        rho_E = state.energy

        Ei = state.specific_inner_energy
        Ek = state.specific_kinetic_energy

        if State.is_set(rho, rho_E):
            self.logger.debug("Returning specific energy from density and energy.")
            return rho_E/rho

        elif State.is_set(Ei, Ek):
            self.logger.debug("Returning specific energy from specific bla.inner energy and specific kinetic energy.")
            return Ei + Ek

    @equation()
    def enthalpy(self, state: State):

        rho = state.density
        H = state.specific_enthalpy

        rho_E = state.energy
        p = state.pressure

        if State.is_set(rho_E, p):
            self.logger.debug("Returning enthalpy from energy and pressure.")
            return rho_E + p

        elif State.is_set(rho, H):
            self.logger.debug("Returning enthalpy from density and specific enthalpy.")
            return rho * H

    @equation()
    def specific_enthalpy(self, state: State):

        rho = state.density
        rho_H = state.enthalpy

        rho_E = state.energy
        E = state.specific_energy
        p = state.pressure

        if State.is_set(rho, rho_H):
            self.logger.debug("Returning specific enthalpy from density and enthalpy.")
            return rho_H/rho

        elif State.is_set(rho, rho_E, p):
            self.logger.debug("Returning specific enthalpy from specific energy, density and pressure.")
            return E + p/rho

    @equation()
    def velocity_gradient(self, state: State):

        rho = state.density
        rho_u = state.momentum

        grad_rho = state.density_gradient
        grad_rho_u = state.momentum_gradient

        if State.is_set(rho, rho_u, grad_rho, grad_rho_u):
            self.logger.debug("Returning velocity gradient from density and momentum.")
            return grad_rho_u/rho - bla.outer(rho_u, grad_rho)/rho**2

    @equation()
    def momentum_gradient(self, state: State):

        rho = state.density
        u = state.velocity

        grad_rho = state.density_gradient
        grad_u = state.velocity_gradient

        if State.is_set(rho, u, grad_rho, grad_u):
            self.logger.debug("Returning momentum gradient from density and momentum.")
            return rho * grad_u + bla.outer(u, grad_rho)

    @equation()
    def energy_gradient(self, state: State):

        grad_rho_Ei = state.energy_gradient
        grad_rho_Ek = state.kinetic_energy_gradient

        if State.is_set(grad_rho_Ei, grad_rho_Ek):
            self.logger.debug("Returning energy gradient from bla.inner energy and kinetic energy.")
            return grad_rho_Ei + grad_rho_Ek

    @equation()
    def specific_energy_gradient(self, state: State):

        grad_Ei = state.specific_energy_gradient
        grad_Ek = state.specific_kinetic_energy_gradient

        if State.is_set(grad_Ei, grad_Ek):
            self.logger.debug(
                "Returning specific energy gradient from specific bla.inner energy and specific kinetic energy.")
            return grad_Ei + grad_Ek

    @equation()
    def inner_energy_gradient(self, state: State):

        grad_rho_E = state.energy_gradient
        grad_rho_Ek = state.kinetic_energy_gradient

        if State.is_set(grad_rho_E, grad_rho_Ek):
            self.logger.debug("Returning bla.inner energy gradient from energy and kinetic energy.")
            return grad_rho_E - grad_rho_Ek

    @equation()
    def specific_inner_energy_gradient(self, state: State):

        grad_E = state.specific_energy_gradient
        grad_Ek = state.specific_kinetic_energy_gradient

        if State.is_set(grad_E, grad_Ek):
            self.logger.debug(
                "Returning specific bla.inner energy gradient from specific energy and specific kinetic energy.")
            return grad_E - grad_Ek

    @equation()
    def kinetic_energy_gradient(self, state: State):

        grad_rho_E = state.energy_gradient
        grad_rho_Ei = state.inner_energy_gradient

        rho = state.density
        grad_rho = state.density_gradient
        u = state.velocity
        grad_u = state.velocity_gradient

        rho_u = state.momentum
        grad_rho_u = state.momentum_gradient

        if State.is_set(grad_rho_E, grad_rho_Ei):
            self.logger.debug("Returning kinetic energy gradient from energy and bla.inner energy.")
            return grad_rho_E - grad_rho_Ei

        elif State.is_set(rho, u, grad_rho, grad_u):
            self.logger.debug("Returning kinetic energy gradient from density and velocity.")
            return rho * (grad_u.trans * u) + 0.5 * grad_rho * bla.inner(u, u)

        elif State.is_set(rho, rho_u, grad_rho, grad_rho_u):
            self.logger.debug("Returning kinetic energy gradient from density and momentum.")
            return (grad_rho_u.trans * rho_u)/rho - 0.5 * grad_rho * bla.inner(rho_u, rho_u)/rho**2

    @equation()
    def specific_kinetic_energy_gradient(self, state: State):

        grad_E = state.specific_energy_gradient
        grad_Ei = state.specific_inner_energy_gradient

        rho = state.density
        grad_rho = state.density_gradient
        u = state.velocity
        grad_u = state.velocity_gradient

        rho_u = state.momentum
        grad_rho_u = state.momentum_gradient

        if State.is_set(grad_E, grad_Ei):
            self.logger.debug(
                "Returning specific kinetic energy gradient from specific energy and specific bla.inner energy.")
            return grad_E - grad_Ei

        elif State.is_set(u, grad_u):
            self.logger.debug("Returning specific kinetic energy gradient from velocity.")
            return grad_u.trans * u

        elif State.is_set(rho, rho_u, grad_rho, grad_rho_u):
            self.logger.debug("Returning specific kinetic energy gradient from density and momentum.")
            return (grad_rho_u.trans * rho_u)/rho**2 - grad_rho * bla.inner(rho_u, rho_u)/rho**3

    @equation()
    def enthalpy_gradient(self, state: State):

        grad_rho_E = state.energy_gradient
        grad_p = state.pressure_gradient

        if State.is_set(grad_rho_E, grad_p):
            self.logger.debug("Returning enthalpy gradient from energy and pressure.")
            return grad_rho_E + grad_p

    @equation()
    def strain_rate_tensor(self, state: State):

        grad_u = state.velocity_gradient

        if State.is_set(grad_u):
            self.logger.debug("Returning strain rate tensor from velocity.")
            return 0.5 * (grad_u + grad_u.trans)

    @equation()
    def deviatoric_stress_tensor(self, state: State):

        mu = state.viscosity
        EPS = state.strain_rate_tensor

        if State.is_set(mu, EPS):
            self.logger.debug("Returning deviatoric stress tensor from strain rate tensor and viscosity.")
            return 2 * mu * EPS

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return str(self)


# ------- Boundary Conditions ------- #


class EquationBC:

    @classmethod
    def list(cls):
        return [key for key, bc in vars(cls).items() if isinstance(bc, type) and issubclass(bc, Boundary)]


# ------- Domain Conditions ------- #


class EquationDC:

    @classmethod
    def list(cls):
        return [key for key, dc in vars(cls).items() if isinstance(dc, type) and issubclass(dc, Domain)]


# ------- Formulations ------- #


class Space:

    @property
    def trial(self):
        return self.TnT[0]

    @property
    def test(self):
        return self.TnT[1]

    @property
    def equations(self):
        return self.cfg.flow.equations

    @property
    def order_policy(self):

        order_policy = ngs.ORDER_POLICY.CONSTANT
        if self.dmesh.dcs.psponge_layers:
            order_policy = ngs.ORDER_POLICY.VARIABLE

        return order_policy

    def __init__(self, cfg: SolverConfiguration, dmesh: DreamMesh):
        self.cfg = cfg
        self.dmesh = dmesh

        self.fes: ngs.FESpace = None
        self.TnT = None
        self.gfu: ngs.GridFunction = None

        self.gfu_transient: TransientGridfunction = None

        self.set_flags()

    def set_flags(self,
                  has_time_derivative: bool = False):

        self.has_time_derivative = has_time_derivative

    def get_space(self) -> ngs.FESpace:
        raise NotImplementedError()

    def get_state_from_gridfunction(self, cf: ngs.CF = None) -> State:
        raise NotImplementedError()

    def get_gridfunction_from_state(self, state: State) -> ngs.CF:
        raise NotImplementedError()

    def get_transient_gridfunction(self) -> TransientGridfunction:
        return self.cfg.simulation.scheme.get_transient_gridfunction(self.gfu)

    def _reduce_psponge_layers_order_elementwise(self, space: ngs.L2 | ngs.VectorL2) -> ngs.L2 | ngs.VectorL2:

        if not isinstance(space, (ngs.L2, ngs.VectorL2)):
            raise TypeError("Can not reduce element order of non L2-spaces!")

        psponge_layers = self.dmesh.dcs.psponge_layers
        order = self.cfg.fem.order

        if psponge_layers:
            for domain, bc in psponge_layers.items():

                bc.check_fem_order(order)

                domain = self.dmesh.domain(domain)

                for el in domain.Elements():
                    space.SetOrder(ngs.NodeId(ngs.ELEMENT, el.nr), bc.high_order)

            space.UpdateDofTables()

        return space

    def _reduce_psponge_layers_order_facetwise(self, space: ngs.FacetFESpace) -> ngs.FacetFESpace:

        if not isinstance(space, ngs.FacetFESpace):
            raise TypeError("Can not reduce element order of non FacetFESpace-spaces!")

        psponge_layers = self.dmesh.dcs.psponge_layers
        order = self.cfg.fem.order

        if psponge_layers:

            if self.dmesh.dim != 2:
                raise NotImplementedError("3D PSpongeLayer not implemented for the moment!")

            sponge_region = self.dmesh.domain(self.dmesh.pattern(psponge_layers.keys()))
            vhat_dofs = ~space.GetDofs(sponge_region)

            for domain, bc in psponge_layers.items():
                bc.check_fem_order(order)

                domain = self.dmesh.domain(domain)
                domain_dofs = space.GetDofs(domain)
                for i in range(bc.high_order + 1, order + 1, 1):
                    domain_dofs[i::order + 1] = 0

                vhat_dofs |= domain_dofs

            space = ngs.Compress(space, vhat_dofs)

        return space

    def __str__(self):
        return self.__class__.__name__.lower()


class SpaceHolder(Descriptor):
    """
    Space is a descriptor that sets private variables to an instance with an underscore.

    If None is passed, the attribute is not set!
    """

    def __get__(self, spaces: Spaces, objtype: Spaces) -> Space:
        space = spaces.data.get(self.name, None)
        if space is None:
            raise ValueError(f"Space {self.name} not set for current configuration!")
        return space

    def __set__(self, spaces: Spaces, value) -> None:
        if value is not None:
            spaces.data[self.name] = value


class Spaces(DescriptorData):

    def initialize_fes(self) -> ngs.FESpace:

        fes = [space.get_space() for space in self.values()]

        if len(fes) == 0:
            raise ValueError("Spaces container is empty!")

        for space in fes[1:]:
            fes[0] *= space

        return fes

    def initialize_gridfunction(self, fes: ngs.FESpace, name: str = "gfu"):
        gfu = ngs.GridFunction(fes, name=name)
        self._set_space_components(fes, gfu)

        return gfu

    def initialize_transient_gridfunctions(self) -> list[TransientGridfunction]:

        gfus = []
        for space in self.values():
            if space.has_time_derivative:

                gfu = space.get_transient_gridfunction()
                space.gfu_transient = gfu
                gfus.append(gfu)

        return gfus

    def _set_space_components(self, fes: ngs.FESpace, gfu: ngs.GridFunction):

        spaces = list(self.values())

        match spaces:

            case []:
                raise ValueError("Spaces container is empty!")

            case [space]:
                space.fes = fes
                space.TnT = fes.TnT()
                space.gfu = gfu

            case _:

                for space, fes_, trial, test, gfu_ in zip(spaces, fes.components, *fes.TnT(), gfu.components):
                    space.fes = fes_
                    space.TnT = (trial, test)
                    space.gfu = gfu_

    def values(self) -> ValuesView[Space]:
        return super().values()

class Formulation:

    types: dict[str, Formulation]

    def __init_subclass__(cls, label: str = ""):

        if not label:
            cls.types = {}

        if label:
            cls.types[label] = cls

    def __init__(self, cfg: SolverConfiguration, mesh: ngs.Mesh | DreamMesh) -> None:

        if isinstance(mesh, ngs.Mesh):
            mesh = DreamMesh(mesh)

        self._cfg = cfg
        self._mesh = mesh

        self._fes = None
        self._gfu = None
        self._spaces = None
        self._gfu_transient = []

        self.normal = ngs.specialcf.normal(mesh.dim)
        self.tangential = ngs.specialcf.tangential(mesh.dim)
        self.mesh_size = ngs.specialcf.mesh_size

    @property
    def dmesh(self) -> DreamMesh:
        return self._mesh

    @property
    def mesh(self) -> ngs.Mesh:
        return self.dmesh.ngsmesh

    @property
    def cfg(self) -> SolverConfiguration:
        return self._cfg

    @property
    def fes(self) -> ngs.FESpace:
        return self._fes

    @property
    def gfu(self) -> ngs.GridFunction:
        return self._gfu
    
    @property
    def gfu_transient(self) -> list[TransientGridfunction]:
        return self._gfu_transient

    @property
    def spaces(self) -> Spaces:
        return self._spaces

    def get_spaces(self) -> Spaces:
        raise NotImplementedError()
    
    def assemble_system(self, blf: ngs.BilinearForm, lf: ngs.LinearForm):
        raise NotImplementedError()
    
    def check_configuration(self):
        ...

    def initialize(self):
        self.check_configuration()
        self.initialize_spaces()

    def initialize_spaces(self):

        spaces = self.get_spaces()

        self._fes = spaces.initialize_fes()
        self._gfu = spaces.initialize_gridfunction(self.fes, str(self))
        self._gfu_transient = spaces.initialize_transient_gridfunctions()

    def update_time_step(self):
        for gfu in self.gfu_transient:
            gfu.update_time_step()

    def update_initial(self):
        for gfu in self.gfu_transient:
            gfu.update_initial()

    def __str__(self) -> str:
        return self.__class__.__name__


# ------- Configuration ------- #


class FlowConfig(BaseConfig):

    types: dict[str, FlowConfig] = {}
    formulation: Formulation
    bcs: EquationBC
    dcs: EquationDC

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

    def get_formulation(self, cfg: SolverConfiguration, mesh: ngs.Mesh | DreamMesh, *args, **kwargs) -> Formulation:
        return self._get_type(self.formulation, type(self).formulation, cfg=cfg, mesh=mesh, *args, **kwargs)

    @property
    def _formulation_type(self) -> Formulation:
        raise NotImplementedError()

    def __repr__(self) -> str:
        return str(self)
