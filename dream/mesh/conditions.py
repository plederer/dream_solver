from __future__ import annotations
from ngsolve import *
from collections import UserDict
from typing import NamedTuple
import numpy as np

from .buffer import SpongeFunction, GridDeformationFunction
from .._configuration import State
from ..utils import DreAmLogger

logger = DreAmLogger.get_logger("Condition")

class Condition:

    def __init__(self, state: State = None) -> None:
        self.state = state

    def _check_value(self, val, name: str):
        if val is None:
            raise ValueError(f"{name.capitalize()} can not be {None}")

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return repr(self.state)


class DomainConditions(UserDict):

    def __init__(self, domains) -> None:
        super().__init__({domain: {bc: None for bc in self._Domain.conditions} for domain in set(domains)})

    @property
    def pattern(self) -> str:
        return "|".join(self)

    @property
    def force(self) -> dict[str, Force]:
        return self._get_condition(self.Force)

    @property
    def initial_conditions(self) -> dict[str, Initial]:
        condition = self._get_condition(self.Initial)
        for domain in set(self).difference(set(condition)):
            logger.warn(f"Initial condition for '{domain}' has not been set!")
        return condition

    @property
    def sponge_layers(self) -> dict[str, SpongeLayer]:
        return self._get_condition(self.SpongeLayer)

    @property
    def psponge_layers(self) -> dict[str, PSpongeLayer]:
        psponge = self._get_condition(self.PSpongeLayer)
        psponge = dict(sorted(psponge.items(), key=lambda x: x[1].order.high))
        return psponge

    @property
    def grid_deformation(self) -> dict[str, GridDeformation]:
        return self._get_condition(self.GridDeformation)

    @property
    def perturbations(self) -> dict[str, Perturbation]:
        return self._get_condition(self.Perturbation)

    @property
    def pmls(self) -> dict[str, PML]:
        return self._get_condition(self.PML)

    class _Domain(Condition):

        conditions: list = []

        def __init_subclass__(cls) -> None:
            cls.conditions.append(cls)
            return super().__init_subclass__()

    class Force(_Domain):
        def __init__(self, state: State = None):
            super().__init__(state)

    class Initial(_Domain):
        def __init__(self, state: State):
            self._check_value(state.density, "density")
            self._check_value(state.velocity, "velocity")
            if state.is_none_thermodynamic:
                raise ValueError("A Thermodynamic quantity is required!")
            super().__init__(state)

    class SpongeLayer(_Domain):

        fes: FESpace = L2
        fes_order: int = 0

        def __init__(self,
                     state: State,
                     *sponges: SpongeFunction) -> None:

            self._check_value(state.density, "density")
            self._check_value(state.velocity, "velocity")
            if state.is_none_thermodynamic:
                raise ValueError("A Thermodynamic quantity is required!")
            super().__init__(state)

            self.sponges = sponges

        @property
        def bonus_int_order(self):
            return max([sponge.order for sponge in self.sponges])

        @property
        def weight_function(self):
            return sum([sponge.weight_function for sponge in self.sponges])

    class PSpongeLayer(_Domain):

        class SpongeOrder(NamedTuple):
            high: int
            low: int

        fes: FESpace = L2
        fes_order: int = 0

        def __init__(self,
                     high_order: int,
                     low_order: int,
                     *sponges: SpongeFunction,
                     state: State = None) -> None:

            if high_order < 0 or low_order < 0:
                raise ValueError("Negative polynomial order!")

            if high_order == low_order:
                if state is None:
                    raise ValueError("For equal order polynomials a state is required")
                else:
                    self._check_value(state.density, "density")
                    self._check_value(state.velocity, "velocity")
                    if state.is_none_thermodynamic:
                        raise ValueError("A Thermodynamic quantity is required!")
            elif not high_order > low_order:
                raise ValueError("Low Order must be smaller than High Order")

            super().__init__(state)

            self.order = self.SpongeOrder(int(high_order), int(low_order))
            self.sponges = sponges

        @property
        def is_equal_order(self) -> bool:
            return self.order.high == self.order.low

        @property
        def bonus_int_order(self) -> int:
            return max([sponge.order for sponge in self.sponges])

        @property
        def weight_function(self) -> CF:
            return sum([sponge.weight_function for sponge in self.sponges])

        @classmethod
        def range(cls, highest, lowest: int = 0, step: int = 1) -> tuple[SpongeOrder, ...]:
            range = np.arange(highest, lowest - 2*step, -step)
            range[range < lowest] = lowest
            return tuple(cls.SpongeOrder(int(high), int(low)) for high, low in zip(range[:-1], range[1:]))

        def __repr__(self) -> str:
            return f"(High: {self.order.high}, Low: {self.order.low}, State: {self.state})"

    class GridDeformation(_Domain):

        fes: FESpace = VectorH1
        fes_order: int = 1

        def __init__(self, mapping: GridDeformationFunction) -> None:
            if not isinstance(mapping, GridDeformationFunction):
                raise TypeError()
            self.mapping = mapping

        @property
        def bonus_int_order(self) -> int:
            return max([x.order for x in self.mapping])

        def deformation_function(self, dim: int) -> CF:
            deformation = tuple(map.deformation for map in self.mapping)
            return CF(deformation[:dim])

        def __repr__(self) -> str:
            return repr(self.mapping)

    class Perturbation(_Domain):
        ...

    class PML(_Domain):
        ...

    def set(self, condition: _Domain, domain: str = None):
        if not isinstance(condition, self._Domain):
            raise TypeError(f"Domain Condition must be instance of {self._Domain}")

        if domain is None:
            domains = tuple(self)
        elif isinstance(domain, str):
            domain = domain.split("|")

            domains = set(self).intersection(domain)
            missed = set(domain).difference(domains)

            for miss in missed:
                logger.warning(f"Domain {miss} does not exist! {condition} can not be set!")
        else:
            domains = tuple(domain)

        for domain in domains:
            self[domain][type(condition)] = condition

    def set_from_dict(self, other: dict[str, _Domain]):
        for key, bc in other.items():
            self.set(bc, key)

    def _get_condition(self, type: _Domain) -> dict[str, _Domain]:
        return {domain: bc[type] for domain, bc in self.items() if isinstance(bc[type], type)}


class BoundaryConditions(UserDict):

    def __init__(self, boundaries) -> None:
        super().__init__({boundary: None for boundary in set(boundaries)})

    @property
    def pattern(self) -> str:
        keys = [key for key, bc in self.items() if isinstance(bc, self._Boundary) and not isinstance(bc, self.Periodic)]
        return "|".join(keys)

    @property
    def nscbc(self) -> dict[str, Outflow_NSCBC]:
        return {name: bc for name, bc in self.items() if isinstance(bc, self.Outflow_NSCBC)}

    class _Boundary(Condition):
        ...

    class Dirichlet(_Boundary):
        def __init__(self, state: State):
            self._check_value(state.density, "density")
            self._check_value(state.velocity, "velocity")
            if state.is_none_thermodynamic:
                raise ValueError("A Thermodynamic quantity is required!")
            super().__init__(state)

    class FarField(_Boundary):
        def __init__(self, state: State):
            self._check_value(state.density, "density")
            self._check_value(state.velocity, "velocity")
            if state.is_none_thermodynamic:
                raise ValueError("A Thermodynamic quantity is required!")
            super().__init__(state)

    class Outflow(_Boundary):
        def __init__(self, pressure: float):
            state = State(pressure=pressure)
            self._check_value(state.pressure, "pressure")
            super().__init__(state)

    class Outflow_NSCBC(_Boundary):

        def __init__(self,
                     pressure: float,
                     sigma: float = 0.25,
                     reference_length: float = 1,
                     tangential_convective_fluxes: bool = True,
                     tangential_viscous_fluxes: bool = True,
                     normal_viscous_fluxes: bool = False) -> None:

            state = State(pressure=pressure)
            self._check_value(state.pressure, "pressure")
            super().__init__(state)

            self.sigma = sigma
            self.reference_length = reference_length
            self.tang_conv_flux = tangential_convective_fluxes
            self.tang_visc_flux = tangential_viscous_fluxes
            self.norm_visc_flux = normal_viscous_fluxes

    class InviscidWall(_Boundary):
        def __init__(self) -> None:
            super().__init__()

    class Symmetry(_Boundary):
        def __init__(self) -> None:
            super().__init__()

    class IsothermalWall(_Boundary):
        def __init__(self, temperature: CF) -> None:
            state = State(temperature=temperature)
            self._check_value(state.temperature, "temperature")
            super().__init__(state)

    class AdiabaticWall(_Boundary):
        def __init__(self) -> None:
            super().__init__()

    class Periodic(_Boundary):
        def __init__(self) -> None:
            super().__init__()

    class Custom(_Boundary):
        def __init__(self, state: State) -> None:
            super().__init__(state)

    def set(self, condition: _Boundary, boundary: str):
        if not isinstance(condition, self._Boundary):
            raise TypeError(f"Boundary Condition must be instance of {self._Boundary}")

        if isinstance(boundary, str):
            boundary = boundary.split("|")

        boundaries = set(self).intersection(boundary)
        missed = set(boundary).difference(boundaries)

        for miss in missed:
            logger.warning(f"Boundary {miss} does not exist! {condition} can not be set!")

        for key in boundaries:
            self[key] = condition

    def set_from_dict(self, other: dict[str, _Boundary]):
        for key, bc in other.items():
            self.set(bc, key)
