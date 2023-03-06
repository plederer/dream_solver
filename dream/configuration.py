from __future__ import annotations
from enum import Enum
from ngsolve import Parameter
from typing import Optional
import numpy as np

from . import io
from .formulations import CompressibleFormulations, MixedMethods, RiemannSolver
from .time_schemes import TimeSchemes, TimePeriod
from .viscosity import DynamicViscosity
from .utils.formatter import Formatter


class Simulation(Enum):
    STATIONARY = "stationary"
    TRANSIENT = "transient"


class SolverConfiguration:

    def __init__(self) -> None:

        # Formulation Settings
        self.formulation = "conservative"
        self.dynamic_viscosity = None
        self.mixed_method = None
        self.riemann_solver = 'roe'

        # Flow Settings'
        self._Mach_number = Parameter(0.3)
        self._Reynolds_number = Parameter(1)
        self._Prandtl_number = Parameter(0.72)
        self._heat_capacity_ratio = Parameter(1.4)
        self._farfield_temperature = Parameter(293.15)

        # Solver options
        self.simulation = "transient"
        self.time_scheme = "IE"
        self._time_step = Parameter(1e-4)
        self.time_period = (0, 1)
        self.time_step_max = 1
        self.order = 2
        self.static_condensation = True
        self.bonus_int_order_vol = 0
        self.bonus_int_order_bnd = 0
        self.compile_flag = True
        self.max_iterations = 10
        self.convergence_criterion = 1e-8
        self.damping_factor = 1
        self.linear_solver = "pardiso"

        # Domain properties
        self.periodic = False

        # Output options
        self.save_iteration = False
        self.save_state = False

    @property
    def Reynolds_number(self) -> Parameter:
        """ Represents the ratio between inertial and viscous forces """
        if self.dynamic_viscosity is DynamicViscosity.INVISCID:
            raise Exception("Inviscid solver configuration: Reynolds number not applicable")
        return self._Reynolds_number

    @Reynolds_number.setter
    def Reynolds_number(self, Reynolds_number: Optional[float]):
        if Reynolds_number <= 0:
            raise ValueError("Invalid Reynold number. Value has to be > 0!")
        else:
            self._Reynolds_number.Set(Reynolds_number)

    @property
    def Mach_number(self) -> Parameter:
        return self._Mach_number

    @Mach_number.setter
    def Mach_number(self, Mach_number: Optional[float]):
        if Mach_number <= 0:
            raise ValueError("Invalid Mach number. Value has to be > 0!")
        else:
            self._Mach_number.Set(Mach_number)

    @property
    def Prandtl_number(self) -> Parameter:
        if self.dynamic_viscosity is DynamicViscosity.INVISCID:
            raise Exception("Inviscid solver configuration: Prandtl number not applicable")
        return self._Prandtl_number

    @Prandtl_number.setter
    def Prandtl_number(self, Prandtl_number: Optional[float]):
        if Prandtl_number <= 0:
            raise ValueError("Invalid Prandtl_number. Value has to be > 0!")
        else:
            self._Prandtl_number.Set(Prandtl_number)

    @property
    def heat_capacity_ratio(self) -> Parameter:
        return self._heat_capacity_ratio

    @heat_capacity_ratio.setter
    def heat_capacity_ratio(self, heat_capacity_ratio: Optional[float]):
        if heat_capacity_ratio <= 1:
            raise ValueError("Invalid heat capacity ratio. Value has to be > 1!")
        else:
            self._heat_capacity_ratio.Set(heat_capacity_ratio)

    @property
    def farfield_temperature(self) -> Parameter:
        """ Defines the farfield temperature needed for Sutherland's law"""
        if self.dynamic_viscosity is not DynamicViscosity.SUTHERLAND:
            raise Exception(f"Farfield temperature requires {DynamicViscosity.SUTHERLAND}")
        return self._farfield_temperature

    @farfield_temperature.setter
    def farfield_temperature(self, farfield_temperature: Optional[float]):
        if farfield_temperature <= 0:
            raise ValueError("Invalid farfield temperature. Value has to be > 1!")
        else:
            self._farfield_temperature.Set(farfield_temperature)

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
    def dynamic_viscosity(self, dynamic_viscosity: str):

        if isinstance(dynamic_viscosity, str):
            dynamic_viscosity = dynamic_viscosity.lower()

        try:
            self._mu = DynamicViscosity(dynamic_viscosity)

        except ValueError:
            options = [enum.value for enum in DynamicViscosity]
            raise ValueError(
                f"'{dynamic_viscosity.capitalize()}' is not a valid viscosity. Possible alternatives: {options}")

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
    def riemann_solver(self) -> MixedMethods:
        return self._riemann_solver

    @riemann_solver.setter
    def riemann_solver(self, riemann_solver: str):

        try:
            self._riemann_solver = RiemannSolver(riemann_solver.lower())
        except ValueError:
            options = [enum.value for enum in RiemannSolver]
            raise ValueError(
                f"'{riemann_solver.capitalize()}' is not a valid Riemann Solver. Possible alternatives: {options}")

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
    def time_step(self, time_step: float):
        self._time_step.Set(time_step)
        io.DreAmLogger._time_step_digit = int(np.abs(np.log10(time_step)))

    @property
    def time_period(self) -> TimePeriod:
        return self._time_period

    @time_period.setter
    def time_period(self, value: tuple[float, float]):
        if len(value) != 2:
            raise ValueError("Time period must be a container of length 2!")

        start, end = value

        if start > end:
            raise ValueError("Start time is greater than end time!")

        self._time_period = TimePeriod(start, end, self.time_step)

    @property
    def time_step_max(self) -> float:
        return self._time_step_max

    @time_step_max.setter
    def time_step_max(self, time_step_max: float):
        self._time_step_max = float(time_step_max)

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

    @property
    def max_iterations(self) -> int:
        return self._max_iterations

    @max_iterations.setter
    def max_iterations(self, max_iterations: int):
        self._max_iterations = int(max_iterations)

    @property
    def convergence_criterion(self) -> float:
        return self._convergence_criterion

    @convergence_criterion.setter
    def convergence_criterion(self, convergence_criterion: float):
        self._convergence_criterion = float(convergence_criterion)

    @property
    def damping_factor(self) -> float:
        return self._damping_factor

    @damping_factor.setter
    def damping_factor(self, damping_factor: float):
        self._damping_factor = float(damping_factor)

    @property
    def linear_solver(self) -> str:
        return self._linear_solver

    @linear_solver.setter
    def linear_solver(self, linear_solver: str):
        self._linear_solver = str(linear_solver).lower()

    def __repr__(self) -> str:

        formatter = Formatter()
        formatter.header('Solver Configuration').newline()
        formatter.subheader("Formulation Settings").newline()
        formatter.entry("Formulation", self._formulation.name)
        formatter.entry("Viscosity", self._mu.name)
        formatter.entry("Mixed Method", self._mixed_method.name)
        formatter.entry("Riemann Solver", self.riemann_solver.name)
        formatter.newline()

        formatter.subheader('Flow Settings').newline()
        formatter.entry("Mach Number", self.Mach_number.Get())
        if self._mu is not DynamicViscosity.INVISCID:
            formatter.entry("Reynolds Number", self.Reynolds_number.Get())
            formatter.entry("Prandtl Number", self.Mach_number.Get())
        formatter.entry("Heat Capacity Ratio", self.heat_capacity_ratio.Get())
        if self._mu is DynamicViscosity.SUTHERLAND:
            formatter.entry("Farfield Temperature", self.farfield_temperature.Get())
        formatter.newline()

        formatter.subheader('Finite Element Settings').newline()
        formatter.entry('Polynomial Order', self._order)
        formatter.entry('Static Condensation', str(self._static_condensation))
        formatter.entry('Bonus Integration Order BND', self._bonus_int_order_bnd)
        formatter.entry('Bonus Integration Order VOL', self._bonus_int_order_vol)
        formatter.newline()

        formatter.subheader('Solution Routine Settings').newline()
        formatter.entry('Simulation', self.simulation.name)
        formatter.entry('Time Scheme', self.time_scheme.name)
        formatter.entry('Time Step', self.time_step.Get())
        if self.simulation is Simulation.TRANSIENT:
            formatter.entry('Time Period', f"({self.time_period.start}, {self.time_period.end}]")
        elif self.simulation is Simulation.STATIONARY:
            formatter.entry('Max Time Step', self.time_step_max)
        formatter.entry('Linear Solver', self._linear_solver)
        formatter.entry('Damping Factor', self._damping_factor)
        formatter.entry('Convergence Criterion', self._convergence_criterion)
        formatter.entry('Maximal Iterations', self._max_iterations)
        formatter.newline()

        formatter.subheader('Various Settings').newline()
        formatter.entry('Compile Flag', str(self._compile_flag))
        formatter.newline()

        return formatter.output
