from enum import Enum
from ngsolve import Parameter
from typing import Optional

from formulations import CompressibleFormulations, MixedMethods
from time_schemes import TimeSchemes


class Simulation(Enum):
    STATIONARY = "steady"
    TRANSIENT = "transient"


class DynamicViscosity(Enum):
    INVISCID = "inviscid"
    CONSTANT = "constant"
    SUTHERLAND = "sutherland"


class SolverConfiguration:

    def __init__(self,
                 formulation: str = "conservative",
                 mixed_method: str = None,
                 dynamic_viscosity: str = "inviscid",
                 Mach_number: Optional[float] = None,
                 Reynold_number: Optional[float] = None,
                 Prandtl_number: float = 0.72,
                 heat_capacity_ratio: float = 1.4,
                 simulation: str = "transient",
                 time_scheme: str = "IE",
                 time_step: float = 1e-4,
                 order: int = 2,
                 static_condensation: bool = True,
                 bonus_int_order_vol: int = 0,
                 bonus_int_order_bnd: int = 0,
                 compile_flag: bool = True
                 ) -> None:

        # Formulation options
        self.formulation = formulation
        self.dynamic_viscosity = dynamic_viscosity
        self.mixed_method = mixed_method

        # Flow properties
        self.Mach_number = Mach_number
        self.heat_capacity_ratio = heat_capacity_ratio
        self.Reynold_number = Reynold_number
        self.Prandtl_number = Prandtl_number

        # Solver options
        self.simulation = simulation
        self.time_scheme = time_scheme
        self.time_step = time_step
        self.static_condensation = static_condensation
        self.order = order
        self.bonus_int_order_vol = bonus_int_order_vol
        self.bonus_int_order_bnd = bonus_int_order_bnd
        self.compile_flag = compile_flag

    @property
    def Reynold_number(self) -> Parameter:
        """ Represents the ratio between inertial and viscous forces """
        return self._Re

    @Reynold_number.setter
    def Reynold_number(self, Reynold_number: Optional[float]):

        if self.dynamic_viscosity is DynamicViscosity.INVISCID or Reynold_number is None:
            self._Re = None
        elif Reynold_number <= 0:
            raise ValueError("Invalid Reynold number. Value has to be > 0!")
        else:
            self._Re = Parameter(Reynold_number)

    @property
    def Mach_number(self) -> Parameter:
        return self._Ma

    @Mach_number.setter
    def Mach_number(self, Mach_number: Optional[float]):

        if Mach_number is None:
            self._Ma = None
        elif Mach_number <= 0:
            raise ValueError("Invalid Mach number. Value has to be > 0!")
        else:
            self._Ma = Parameter(Mach_number)

    @property
    def Prandtl_number(self) -> Parameter:
        return self._Pr

    @Prandtl_number.setter
    def Prandtl_number(self, Prandtl_number: Optional[float]):

        if self.dynamic_viscosity is DynamicViscosity.INVISCID:
            self._Pr = None
        elif Prandtl_number <= 0:
            raise ValueError("Invalid Prandtl_number. Value has to be > 0!")
        else:
            self._Pr = Parameter(Prandtl_number)

    @property
    def heat_capacity_ratio(self) -> Parameter:
        return self._gamma

    @heat_capacity_ratio.setter
    def heat_capacity_ratio(self, heat_capacity_ratio: Optional[float]):

        if heat_capacity_ratio is None:
            self._gamma = None
        elif heat_capacity_ratio <= 1:
            raise ValueError("Invalid heat capacity ratio. Value has to be > 1!")
        else:
            self._gamma = Parameter(heat_capacity_ratio)

    @property
    def formulation(self) -> CompressibleFormulations:
        return self._formulation

    @formulation.setter
    def formulation(self, value: str):

        try:
            self._formulation = CompressibleFormulations(value.lower())

        except ValueError:
            options = [enum.value for enum in CompressibleFormulations]
            raise ValueError(f"'{value.capitalize()}' is not a valid formulation. Possible alternatives: {options}")

    @property
    def dynamic_viscosity(self) -> DynamicViscosity:
        return self._mu

    @dynamic_viscosity.setter
    def dynamic_viscosity(self, value: str):

        try:
            self._mu = DynamicViscosity(value.lower())

        except ValueError:
            options = [enum.value for enum in DynamicViscosity]
            raise ValueError(f"'{value.capitalize()}' is not a valid viscosity. Possible alternatives: {options}")

    @property
    def mixed_method(self) -> MixedMethods:
        return self._mixed_method

    @mixed_method.setter
    def mixed_method(self, mixed_method: Optional[str]):

        if isinstance(mixed_method, str):
            mixed_method = mixed_method.lower()

        try:
            self._mixed_method = MixedMethods(mixed_method)

        except ValueError:
            options = [enum.value for enum in MixedMethods]
            raise ValueError(
                f"'{mixed_method.capitalize()}' is not a valid mixed variant. Possible alternatives: {options}")

    @property
    def simulation(self) -> Simulation:
        return self._simulation

    @simulation.setter
    def simulation(self, simulation: str):
        self._simulation = Simulation(simulation)

    @property
    def time_step(self) -> Parameter:
        return self._time_step

    @time_step.setter
    def time_step(self, time_step: str):
        self._time_step = Parameter(time_step)

    @property
    def time_scheme(self) -> TimeSchemes:
        return self._time_scheme

    @time_scheme.setter
    def time_scheme(self, time_scheme: str):
        self._time_scheme = TimeSchemes(time_scheme.upper())

    @property
    def order(self) -> int:
        return self._order

    @order.setter
    def order(self, order: int):
        self._order = int(order)

    @property
    def static_condensation(self) -> bool:
        return self._static_condensation

    @static_condensation.setter
    def static_condensation(self, static_condensation: bool):
        self._static_condensation = bool(static_condensation)

    @property
    def bonus_int_order_vol(self) -> int:
        return self._bonus_int_order_vol

    @bonus_int_order_vol.setter
    def bonus_int_order_vol(self, bonus_int_order_vol: int):
        self._bonus_int_order_vol = int(bonus_int_order_vol)

    @property
    def bonus_int_order_bnd(self) -> int:
        return self._bonus_int_order_bnd

    @bonus_int_order_bnd.setter
    def bonus_int_order_bnd(self, bonus_int_order_bnd: int):
        self._bonus_int_order_bnd = int(bonus_int_order_bnd)

    @property
    def compile_flag(self) -> bool:
        return self._compile_flag

    @compile_flag.setter
    def compile_flag(self, compile_flag: bool):
        self._compile_flag = bool(compile_flag)
