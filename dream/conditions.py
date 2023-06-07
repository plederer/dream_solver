from __future__ import annotations
import enum
import dataclasses
from collections import UserDict
from typing import Optional, TYPE_CHECKING, ItemsView
from ngsolve import CF, Parameter
from .utils import IdealGasCalculator

if TYPE_CHECKING:
    from .configuration import SolverConfiguration


def extract_boundaries_from_pattern(pattern, facets):
    if isinstance(pattern, str):
        pattern = pattern.split("|")
    intersection = set(facets).intersection(pattern)

    return tuple(intersection)


@dataclasses.dataclass
class DomainConditionsContainer:
    initial: Optional[Initial] = None
    sponge: Optional[SpongeLayer] = None
    pml: Optional[PML] = None


class DomainConditions(UserDict):

    def __init__(self, domains, solver_configuration: SolverConfiguration) -> None:
        super().__init__({domain: DomainConditionsContainer() for domain in set(domains)})
        self.cfg = solver_configuration

    @property
    def domains(self) -> tuple[str, ...]:
        return tuple(self)

    @property
    def pattern(self) -> str:
        return "|".join(self.domains)

    @property
    def has_initial_condition(self) -> bool:
        return bool([True for _, condition in self.items() if condition.initial is not None])

    def set_initial(self,
                    velocity: tuple[float, ...],
                    density: Optional[float] = None,
                    pressure: Optional[float] = None,
                    temperature: Optional[float] = None,
                    energy: Optional[float] = None,
                    domain: Optional[str] = None):

        if domain is None:
            domains = self.domains
        else:
            domains = extract_boundaries_from_pattern(domain, self.domains)

        calc = IdealGasCalculator(self.cfg.heat_capacity_ratio)
        initial = Initial(*calc.determine_missing(CF(velocity), density, pressure, temperature, energy))

        for domain in domains:
            self[domain].initial = initial

    def set_sponge_layer(self,
                         domain: str,
                         weight_function: CF,
                         velocity: tuple[float, ...],
                         density: float = None,
                         pressure: float = None,
                         temperature: float = None,
                         energy: float = None,
                         weight_function_order: int = 3):

        domains = extract_boundaries_from_pattern(domain, self.domains)

        calc = IdealGasCalculator(self.cfg.heat_capacity_ratio)
        quantities = calc.determine_missing(CF(velocity), density, pressure, temperature, energy)
        sponge = SpongeLayer(weight_function, *quantities, weight_function_order=weight_function_order)

        for domain in domains:
            self[domain].sponge = sponge

    def __getitem__(self, key: str) -> DomainConditionsContainer:
        return super().__getitem__(key)

    def items(self) -> ItemsView[str, DomainConditionsContainer]:
        return super().items()


class BoundaryConditions(UserDict):

    def __init__(self, boundaries, solver_configuration: SolverConfiguration) -> None:
        super().__init__({boundary: None for boundary in set(boundaries)})
        self.cfg = solver_configuration

    @property
    def boundaries(self) -> tuple[str, ...]:
        return tuple(self)

    @property
    def pattern(self) -> str:
        return "|".join([boundary for boundary, bc in self.items() if isinstance(bc, _Boundary)])

    def set_dirichlet(self,
                      boundary: str,
                      velocity: tuple[float, ...],
                      density: float = None,
                      pressure: float = None,
                      temperature: float = None,
                      energy: float = None):

        boundaries = extract_boundaries_from_pattern(boundary, self.boundaries)

        calc = IdealGasCalculator(self.cfg.heat_capacity_ratio)
        dirichlet = Dirichlet(*calc.determine_missing(CF(velocity), density, pressure, temperature, energy))

        for boundary in boundaries:
            self[boundary] = dirichlet

    def set_farfield(self,
                     boundary: str,
                     velocity: tuple[float, ...],
                     density: float = None,
                     pressure: float = None,
                     temperature: float = None,
                     energy: float = None):

        boundaries = extract_boundaries_from_pattern(boundary, self.boundaries)

        calc = IdealGasCalculator(self.cfg.heat_capacity_ratio)
        farfield = FarField(*calc.determine_missing(CF(velocity), density, pressure, temperature, energy))

        for boundary in boundaries:
            self[boundary] = farfield

    def set_outflow(self,
                    boundary: str,
                    pressure: float):

        boundaries = extract_boundaries_from_pattern(boundary, self.boundaries)

        for boundary in boundaries:
            self[boundary] = Outflow(CF(pressure))

    def set_nonreflecting_outflow(self,
                                  boundary: str,
                                  pressure: float,
                                  type: str = "perfect",
                                  sigma: float = 0.25,
                                  reference_length: float = 1,
                                  tangential_convective_fluxes: bool = True,
                                  tangential_viscous_fluxes: bool = True,
                                  normal_viscous_fluxes: bool = False):

        boundaries = extract_boundaries_from_pattern(boundary, self.boundaries)

        for boundary in boundaries:
            self[boundary] = NRBC_Outflow(pressure, type, sigma, reference_length,
                                          tangential_convective_fluxes, tangential_viscous_fluxes,
                                          normal_viscous_fluxes)

    def set_nonreflecting_inflow(self,
                                 boundary: str,
                                 density: float,
                                 velocity: tuple[float, ...],
                                 temperature: float = None,
                                 pressure: float = None,
                                 energy: float = None,
                                 type: str = "perfect",
                                 sigma: float = 0.25,
                                 reference_length: float = 1,
                                 tangential_convective_fluxes: bool = True,
                                 tangential_viscous_fluxes: bool = True,
                                 normal_viscous_fluxes: bool = False):

        gamma = self.cfg.heat_capacity_ratio
        boundaries = extract_boundaries_from_pattern(boundary, self.boundaries)

        for boundary in boundaries:
            self[boundary] = NRBC_Inflow(density,
                                         velocity,
                                         temperature,
                                         pressure,
                                         energy,
                                         type,
                                         sigma,
                                         reference_length,
                                         gamma,
                                         tangential_convective_fluxes,
                                         tangential_viscous_fluxes,
                                         normal_viscous_fluxes)

    def set_inviscid_wall(self, boundary: str):

        boundaries = extract_boundaries_from_pattern(boundary, self.boundaries)

        for boundary in boundaries:
            self[boundary] = InviscidWall()

    def set_periodic(self, boundary: str):

        boundaries = extract_boundaries_from_pattern(boundary, self.boundaries)

        for boundary in boundaries:
            self[boundary] = Periodic()

    def set_symmetry(self, boundary: str):

        boundaries = extract_boundaries_from_pattern(boundary, self.boundaries)

        for boundary in boundaries:
            self[boundary] = InviscidWall()

    def set_isothermal_wall(self, boundary: str, temperature: float):

        boundaries = extract_boundaries_from_pattern(boundary, self.boundaries)
        for boundary in boundaries:
            self[boundary] = IsothermalWall(CF(temperature))

    def set_adiabatic_wall(self, boundary: str):

        boundaries = extract_boundaries_from_pattern(boundary, self.boundaries)
        for boundary in boundaries:
            self[boundary] = AdiabaticWall()

    def set_custom(self, boundary):
        boundaries = extract_boundaries_from_pattern(boundary, self.boundaries)
        for boundary in boundaries:
            self[boundary] = _Boundary()


class Condition:

    def __str__(self) -> str:
        return self.__class__.__name__


class _Boundary(Condition):
    ...


@dataclasses.dataclass
class Dirichlet(_Boundary):

    velocity: CF
    density: CF
    pressure: CF
    temperature: CF
    energy: CF

    @property
    def momentum(self):
        return self.density * self.velocity


@dataclasses.dataclass
class FarField(_Boundary):

    velocity: CF
    density: CF
    pressure: CF
    temperature: CF
    energy: CF

    @property
    def momentum(self):
        return self.density * self.velocity


@dataclasses.dataclass
class Outflow(_Boundary):

    pressure: CF


class NRBC_Outflow(_Boundary):
    class TYPE(enum.Enum):
        PERFECT = "perfect"
        PARTIALLY = "partially"
        PIROZZOLI = "pirozzoli"

    def __init__(self,
                 pressure,
                 type: str = "perfect",
                 sigma: int = 0.25,
                 reference_length: float = 1,
                 tangential_convective_fluxes: bool = True,
                 tangential_viscous_fluxes: bool = True,
                 normal_viscous_fluxes: bool = False) -> None:

        self.pressure = CF(pressure)
        self.type = self.TYPE(type)
        self.sigma = Parameter(sigma)
        self.reference_length = Parameter(reference_length)
        self.tang_conv_flux = tangential_convective_fluxes
        self.tang_visc_flux = tangential_viscous_fluxes
        self.norm_visc_flux = normal_viscous_fluxes


class NRBC_Inflow(_Boundary):

    class TYPE(enum.Enum):
        PERFECT = "perfect"
        PARTIALLY = "partially"

    def __init__(self,
                 density: float,
                 velocity: tuple[float, ...],
                 temperature: Optional[float] = None,
                 pressure: Optional[float] = None,
                 energy: Optional[float] = None,
                 type: str = "perfect",
                 sigma: float = 0.25,
                 reference_length: float = 1,
                 gamma: float = 1.4,
                 tangential_convective_fluxes: bool = True,
                 tangential_viscous_fluxes: bool = True,
                 normal_viscous_fluxes: bool = False) -> None:

        self.type = self.TYPE(type)
        self.sigma = Parameter(sigma)
        self.reference_length = Parameter(reference_length)

        density = CF(density)
        velocity = CF(velocity)

        pressure, temperature, energy = convert_to_pressure_temperature_energy(
            density, velocity, temperature, pressure, energy, gamma)

        self.density = density
        self.velocity = velocity
        self.momentum = density * velocity
        self.temperature = temperature
        self.pressure = pressure
        self.energy = energy

        self.tang_conv_flux = tangential_convective_fluxes
        self.tang_visc_flux = tangential_viscous_fluxes
        self.norm_visc_flux = normal_viscous_fluxes


@dataclasses.dataclass
class InviscidWall(_Boundary):
    ...


@dataclasses.dataclass
class IsothermalWall(_Boundary):

    temperature: CF


@dataclasses.dataclass
class AdiabaticWall(_Boundary):
    ...


@dataclasses.dataclass
class _Domain(Condition):

    velocity: CF
    density: CF
    pressure: CF
    temperature: CF
    energy: CF

    @property
    def momentum(self):
        return self.density * self.velocity


class Initial(_Domain):
    ...


class Perturbation(_Domain):
    ...


class SpongeLayer(_Domain):

    def __init__(self,
                 weight_function: CF,
                 velocity: CF,
                 density: CF,
                 pressure: CF,
                 temperature: CF,
                 energy: CF,
                 weight_function_order: int = 3) -> None:

        super().__init__(velocity, density, pressure, temperature, energy)
        self.weight_function = weight_function
        self.weight_function_order = weight_function_order


class PML(_Domain):
    ...


class Periodic(Condition):
    ...
