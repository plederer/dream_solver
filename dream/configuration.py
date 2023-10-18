from __future__ import annotations
import numpy as np
from ngsolve import Parameter, CF
from ngsolve.comp import VorB
from typing import Optional, Any, NamedTuple
from numbers import Number
from pathlib import Path

from .formulations import CompressibleFormulations, MixedMethods, RiemannSolver, Scaling
from .time_schemes import TimeSchemes, Simulation
from .utils import Formatter


import logging
logger = logging.getLogger('DreAm.Configuration')


class ResultsDirectoryTree:

    def __init__(self,
                 directory_name: str = "results",
                 state_directory_name: str = "states",
                 sensor_directory_name: str = "sensor",
                 vtk_directory_name: str = "vtk",
                 parent_path: Optional[Path] = None) -> None:

        self.parent_path = parent_path
        self.directory_name = directory_name
        self.state_directory_name = state_directory_name
        self.sensor_directory_name = sensor_directory_name
        self.vtk_directory_name = vtk_directory_name

    @property
    def main_path(self) -> Path:
        return self.parent_path.joinpath(self.directory_name)

    @property
    def state_path(self) -> Path:
        return self.main_path.joinpath(self.state_directory_name)

    @property
    def sensor_path(self) -> Path:
        return self.main_path.joinpath(self.sensor_directory_name)

    @property
    def vtk_path(self) -> Path:
        return self.main_path.joinpath(self.vtk_directory_name)

    @property
    def parent_path(self) -> Path:
        return self._parent_path

    @parent_path.setter
    def parent_path(self, parent_path: Path):

        if parent_path is None:
            self._parent_path = Path.cwd()

        elif isinstance(parent_path, (str, Path)):
            parent_path = Path(parent_path)

            if not parent_path.exists():
                raise ValueError(f"Path {parent_path} does not exist!")

            self._parent_path = parent_path

        else:
            raise ValueError(f"Type {type(parent_path)} not supported!")

    def find_directory_paths(self, pattern: str = "") -> tuple[Path]:
        return tuple(dir for dir in self.parent_path.glob(pattern + "*") if dir.is_dir())

    def find_directory_names(self, pattern: str = "") -> tuple[str]:
        return tuple(dir.name for dir in self.parent_path.glob(pattern + "*") if dir.is_dir())

    def update(self, tree: ResultsDirectoryTree):
        for key, bc in vars(tree).items():
            setattr(self, key, bc)

    def __repr__(self) -> str:
        formatter = Formatter()
        formatter.COLUMN_RATIO = (0.2, 0.8)
        formatter.header("Results Directory Tree").newline()
        formatter.entry("Path", str(self.parent_path))
        formatter.entry("Main", f"{self.parent_path.stem}/{self.directory_name}")
        formatter.entry("State", f"{self.parent_path.stem}/{self.directory_name}/{self.state_directory_name}")
        formatter.entry("Sensor", f"{self.parent_path.stem}/{self.directory_name}/{self.sensor_directory_name}")

        return formatter.output


class DreAmLogger:

    _iteration_error_digit: int = 8
    _time_step_digit: int = 1

    @classmethod
    def set_time_step_digit(cls, time_step: float):

        if isinstance(time_step, Parameter):
            time_step = time_step.Get()

        if isinstance(time_step, float):
            digit = len(str(time_step).split(".")[1])
            cls._time_step_digit = digit

    def __init__(self, log_to_terminal: bool = False, log_to_file: bool = False) -> None:
        self.logger = logging.getLogger("DreAm")
        self.tree = ResultsDirectoryTree()

        self.stream_handler = logging.NullHandler()
        self.file_handler = logging.NullHandler()
        self.formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        self.filename = "log.txt"

        self.log_to_terminal = log_to_terminal
        self.log_to_file = log_to_file

    @property
    def filepath(self):
        if not self.tree.main_path.exists():
            self.tree.main_path.mkdir(parents=True)
        return self.tree.main_path.joinpath(self.filename)

    def set_level(self, level):
        self.logger.setLevel(level)

    def silence_logger(self):
        self.log_to_file = False
        self.log_to_terminal = False

    @property
    def log_to_terminal(self):
        return self._log_to_terminal

    @log_to_terminal.setter
    def log_to_terminal(self, value: bool):
        if value:
            self.stream_handler = logging.StreamHandler()
            self.stream_handler.setLevel(self.logger.level)
            self.stream_handler.setFormatter(self.formatter)
            self.logger.addHandler(self.stream_handler)
        else:
            self.logger.removeHandler(self.stream_handler)
            self.stream_handler = logging.NullHandler()
        self._log_to_terminal = value

    @property
    def log_to_file(self):
        return self._log_to_file

    @log_to_file.setter
    def log_to_file(self, value: bool):
        if value:
            self.file_handler = logging.FileHandler(self.filepath, delay=True)
            self.file_handler.setLevel(self.logger.level)
            self.file_handler.setFormatter(self.formatter)
            self.logger.addHandler(self.file_handler)
        else:
            self.logger.removeHandler(self.file_handler)
            self.file_handler = logging.NullHandler()
        self._log_to_file = value


class BaseConfiguration:

    __slots__ = ()

    @staticmethod
    def _get_enum(value: str, enum, variable: str):
        try:
            value = enum(value)
        except ValueError:
            msg = f"'{str(value).capitalize()}' is not a valid {variable}. "
            msg += f"Possible alternatives: {[e.value for e in enum]}"
            raise ValueError(msg) from None
        return value

    @staticmethod
    def _get_dict(key: Any, dictionary, variable: str):
        try:
            value = dictionary[key]
        except KeyError:
            msg = f"'{str(key).capitalize()}' is not a valid {variable}. "
            msg += f"Possible alternatives: {[key for key in dictionary]}"
            raise KeyError(msg) from None
        return value

    def to_dict(self) -> dict[str, Any]:
        cfg = {}
        for key in self.__slots__:
            value = getattr(self, key)
            if isinstance(value, BaseConfiguration):
                value = value.to_dict()
            cfg[key] = value

        return cfg

    def update(self, cfg: dict[str, Any]):
        if isinstance(cfg, type(self)):
            cfg = cfg.to_dict()
        elif isinstance(cfg, dict):
            pass
        else:
            raise TypeError(f'Update requires dictionary or type {str(self)}')

        for key, value in cfg.items():
            if key.startswith("_"):
                key = key[1:]
            try:
                setattr(self, key, value)
                continue
            except AttributeError:
                msg = f"Trying to set '{key}' attribute in {str(self)}. "
                msg += "It is either deprecated or not supported"
                logger.warning(msg)

    def __str__(self) -> str:
        return self.__class__.__name__


class TimeConfiguration(BaseConfiguration):

    class Interval(NamedTuple):
        start: float
        end: float

    __slots__ = ("_simulation",
                 "_scheme",
                 "_interval",
                 "_step",
                 "_max_step",
                 "_t")

    def __init__(self) -> None:
        self.simulation = "transient"
        self.scheme = "IE"
        self.interval = self.Interval(0, 1)
        self.max_step = 1
        self._step = Parameter(1e-4)
        self._t = Parameter(0)

    @property
    def simulation(self) -> Simulation:
        return self._simulation

    @simulation.setter
    def simulation(self, simulation: str):

        if isinstance(simulation, str):
            simulation = simulation.lower()

        self._simulation = self._get_enum(simulation, Simulation, "Simulation")

    @property
    def interval(self) -> Interval:
        return self._interval

    @interval.setter
    def interval(self, interval) -> None:
        if isinstance(interval, Number):
            interval = (0, abs(interval))

        elif isinstance(interval, tuple):
            if len(interval) == 2:
                interval = tuple(sorted(interval))
            else:
                raise ValueError("Time period must be a container of length 2!")

        self._interval = self.Interval(*interval)

    @property
    def step(self) -> Parameter:
        return self._step

    @step.setter
    def step(self, step: float):
        if isinstance(step, Parameter):
            step = step.Get()

        self._step.Set(step)
        DreAmLogger.set_time_step_digit(step)

    @property
    def max_step(self) -> float:
        return self._max_step

    @max_step.setter
    def max_step(self, max_step: float):
        self._max_step = float(max_step)

    @property
    def scheme(self) -> TimeSchemes:
        return self._scheme

    @scheme.setter
    def scheme(self, scheme: str):
        if isinstance(scheme, str):
            scheme = scheme.upper()

        self._scheme = self._get_enum(scheme, TimeSchemes, "Time Scheme")

    @property
    def t(self) -> Parameter:
        return self._t

    @t.setter
    def t(self, t) -> None:
        if isinstance(t, Parameter):
            t = t.Get()

        self._t.Set(t)

    def __iter__(self):
        for t in self.range(step=1):
            self.t.Set(t)
            yield t

    def range(self, step: int = 1):
        for t in self.to_array()[::step]:
            yield t

    def to_array(self, include_start_time: bool = False) -> np.ndarray:
        start, end = self.interval
        dt = self.step.Get()
        num = round((end - start)/dt) + 1

        interval = np.linspace(start, end, num)
        if not include_start_time:
            interval = interval[1:]
        return interval.round(DreAmLogger._time_step_digit)

    def __repr__(self):
        formatter = Formatter()
        formatter.subheader('Time Configuration').newline()
        formatter.entry('Simulation', self.simulation.name)
        formatter.entry('Time Scheme', self.scheme.name)
        formatter.entry('Time Step', self.step.Get())
        if self.simulation is Simulation.TRANSIENT:
            start, end = self.interval
            formatter.entry('Time Period', f"({start}, {end}]")
        elif self.simulation is Simulation.STATIONARY:
            formatter.entry('Max Time Step', self.max_step)

        return formatter.output

    def __str__(self) -> str:
        start, end = self.interval
        return f"Interval: ({start}, {end}], Time Step: {self.step.Get()}"


class SolverConfiguration(BaseConfiguration):

    __slots__ = ("_Mach_number",
                 "_Reynolds_number",
                 "_Prandtl_number",
                 "_heat_capacity_ratio",
                 "_dynamic_viscosity",
                 "_formulation",
                 "_scaling",
                 "_mixed_method",
                 "_riemann_solver",
                 "_order",
                 "_static_condensation",
                 "_bonus_int_order",
                 "_time",
                 "_compile_flag",
                 "_max_iterations",
                 "_convergence_criterion",
                 "_damping_factor",
                 "_linear_solver",
                 "_save_state",
                 "_info")

    def __init__(self) -> None:

        # Flow Configuration
        self._Mach_number = Parameter(0.3)
        self._Reynolds_number = Parameter(1)
        self._Prandtl_number = Parameter(0.72)
        self._heat_capacity_ratio = Parameter(1.4)
        self.dynamic_viscosity = 'inviscid'

        # Formulation Configuration
        self.formulation = "conservative"
        self.scaling = "aerodynamic"
        self.mixed_method = None
        self.riemann_solver = 'roe'

        # Finite Element Configuration
        self.order = 2
        self.static_condensation = True
        self._bonus_int_order = {VorB.VOL: 0, VorB.BND: 0, VorB.BBND: 0, VorB.BBBND: 0}

        # Time Configuration
        self._time = TimeConfiguration()

        # Solution routine Configuration
        self.compile_flag = True
        self.max_iterations = 10
        self.convergence_criterion = 1e-8
        self.damping_factor = 1
        self.linear_solver = "pardiso"

        # Output Configuration
        self.save_state = False

        # Simulation Info
        self._info = {}

    @property
    def Reynolds_number(self) -> Parameter:
        """ Represents the ratio between inertial and viscous forces """
        if self.dynamic_viscosity.is_inviscid:
            raise Exception("Inviscid solver configuration: Reynolds number not applicable")
        return self._Reynolds_number

    @Reynolds_number.setter
    def Reynolds_number(self, Reynolds_number: float):
        if isinstance(Reynolds_number, Parameter):
            Reynolds_number = Reynolds_number.Get()

        if Reynolds_number <= 0:
            raise ValueError("Invalid Reynold number. Value has to be > 0!")
        else:
            self._Reynolds_number.Set(Reynolds_number)

    @property
    def Mach_number(self) -> Parameter:
        return self._Mach_number

    @Mach_number.setter
    def Mach_number(self, Mach_number: float):
        if isinstance(Mach_number, Parameter):
            Mach_number = Mach_number.Get()

        if Mach_number < 0:
            raise ValueError("Invalid Mach number. Value has to be >= 0!")
        else:
            self._Mach_number.Set(Mach_number)

    @property
    def Prandtl_number(self) -> Parameter:
        if self.dynamic_viscosity.is_inviscid:
            raise Exception("Inviscid solver configuration: Prandtl number not applicable")
        return self._Prandtl_number

    @Prandtl_number.setter
    def Prandtl_number(self, Prandtl_number: float):
        if isinstance(Prandtl_number, Parameter):
            Prandtl_number = Prandtl_number.Get()

        if Prandtl_number <= 0:
            raise ValueError("Invalid Prandtl_number. Value has to be > 0!")
        else:
            self._Prandtl_number.Set(Prandtl_number)

    @property
    def heat_capacity_ratio(self) -> Parameter:
        return self._heat_capacity_ratio

    @heat_capacity_ratio.setter
    def heat_capacity_ratio(self, heat_capacity_ratio: float):
        if isinstance(heat_capacity_ratio, Parameter):
            heat_capacity_ratio = heat_capacity_ratio.Get()

        if heat_capacity_ratio <= 1:
            raise ValueError("Invalid heat capacity ratio. Value has to be > 1!")
        else:
            self._heat_capacity_ratio.Set(heat_capacity_ratio)

    @property
    def formulation(self) -> CompressibleFormulations:
        return self._formulation

    @formulation.setter
    def formulation(self, formulation: str):

        if isinstance(formulation, str):
            formulation = formulation.lower()

        self._formulation = self._get_enum(formulation, CompressibleFormulations, "Compressible Formulation")

    @property
    def scaling(self) -> Scaling:
        return self._scaling

    @scaling.setter
    def scaling(self, scaling: str):

        if isinstance(scaling, str):
            scaling = scaling.lower()

        self._scaling = self._get_enum(scaling, Scaling, "Scaling")

    @property
    def dynamic_viscosity(self) -> DynamicViscosity:
        return self._dynamic_viscosity

    @dynamic_viscosity.setter
    def dynamic_viscosity(self, dynamic_viscosity: DynamicViscosity):
        if isinstance(dynamic_viscosity, str):
            obj = self._get_dict(dynamic_viscosity.lower(), DynamicViscosity.TYPES, "Dynamic Viscosity")
            dynamic_viscosity = obj(solver_configuration=self)
        elif isinstance(dynamic_viscosity, DynamicViscosity):
            dynamic_viscosity.cfg = self
        else:
            raise TypeError(f'Only type {DynamicViscosity} or {str} allowed!')
        
        self._dynamic_viscosity = dynamic_viscosity

    @property
    def mixed_method(self) -> MixedMethods:
        return self._mixed_method

    @mixed_method.setter
    def mixed_method(self, mixed_method: Optional[str]):

        if isinstance(mixed_method, str):
            mixed_method = mixed_method.lower()

        self._mixed_method = self._get_enum(mixed_method, MixedMethods, "Mixed Method")

    @property
    def riemann_solver(self) -> RiemannSolver:
        return self._riemann_solver

    @riemann_solver.setter
    def riemann_solver(self, riemann_solver: str):

        if isinstance(riemann_solver, str):
            riemann_solver = riemann_solver.lower()

        self._riemann_solver = self._get_enum(riemann_solver, RiemannSolver, "Riemann Solver")

    @property
    def time(self) -> TimeConfiguration:
        return self._time

    @time.setter
    def time(self, time: dict[str, Any]) -> None:
        self._time.update(time)

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
    def bonus_int_order(self) -> dict[VorB, int]:
        return self._bonus_int_order

    @bonus_int_order.setter
    def bonus_int_order(self, bonus_int_order: dict[VorB, int]):
        if isinstance(bonus_int_order, dict):
            bonus_int_order = {key: value for key, value in bonus_int_order.items() if key in self._bonus_int_order}

        elif isinstance(bonus_int_order, (tuple, list)):
            bonus_int_order = {key: value for key, value in bonus_int_order if key in self._bonus_int_order}

        else:
            options = {key: 0 for key in self._bonus_int_order}
            raise TypeError(f"Bonus Integration Order must be a dictionary of format {options}")

        self._bonus_int_order.update(bonus_int_order)

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
        convergence_criterion = float(convergence_criterion)
        if convergence_criterion <= 0:
            raise ValueError("Convergence Criterion must be greater zero!")
        self._convergence_criterion = convergence_criterion

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

    @property
    def save_state(self) -> bool:
        return self._save_state

    @save_state.setter
    def save_state(self, value: bool):
        self._save_state = bool(value)

    @property
    def info(self) -> dict[str, Any]:
        """ Info returns a dictionary reserved for the storage of user defined parameters. """
        return self._info

    @info.setter
    def info(self, info: dict[str, Any]):
        self._info.update(info)

    def __repr__(self) -> str:

        formatter = Formatter()
        formatter.header('Solver Configuration').newline()
        formatter.subheader("Formulation Configuration").newline()
        formatter.entry("Formulation", self.formulation.name)
        formatter.entry("Scaling", self.scaling.name)
        formatter.entry("Mixed Method", self.mixed_method.name)
        formatter.entry("Riemann Solver", self.riemann_solver.name)
        formatter.newline()

        formatter.subheader('Flow Configuration').newline()
        formatter.entry("Mach Number", self.Mach_number.Get())
        if not self.dynamic_viscosity.is_inviscid:
            formatter.entry("Reynolds Number", self.Reynolds_number.Get())
            formatter.entry("Prandtl Number", self.Prandtl_number.Get())
        formatter.newline()

        formatter.add(self.dynamic_viscosity).newline()

        formatter.subheader('Finite Element Configuration').newline()
        formatter.entry('Polynomial Order', self._order)
        formatter.entry('Static Condensation', str(self._static_condensation))
        formatter.entry('Bonus Integration Order', str(
            {key.name: value for key, value in self._bonus_int_order.items()}))
        formatter.newline()

        formatter.add(self.time).newline()

        formatter.subheader('Solution Routine Configuration').newline()
        formatter.entry('Linear Solver', self._linear_solver)
        formatter.entry('Damping Factor', self._damping_factor)
        formatter.entry('Convergence Criterion', self._convergence_criterion)
        formatter.entry('Maximal Iterations', self._max_iterations)
        formatter.newline()

        formatter.subheader('Various Configuration').newline()
        formatter.entry('Compile Flag', str(self._compile_flag))
        formatter.entry('Save State', str(self._save_state))
        formatter.newline()

        if self.info:
            formatter.subheader('Simulation Info').newline()

            for key, value in self.info.items():
                if hasattr(value, '__str__'):
                    value = str(value)
                elif hasattr(value, '__repr__'):
                    value = repr(value)
                else:
                    value = f"Type: {type(value)} not displayable"

                formatter.entry(str(key), value)
            formatter.newline()

        return formatter.output


class DynamicViscosity:

    TYPES: dict[str, DynamicViscosity] = {}

    def __init_subclass__(cls) -> None:
        label = cls.__name__.lower()
        cls.TYPES[label] = cls

    def __init__(self, solver_configuration: SolverConfiguration = None) -> None:
        self.cfg = solver_configuration

    @property
    def is_inviscid(self):
        return isinstance(self, Inviscid)

    def get(self, temperature: CF) -> CF:
        raise NotImplementedError("Implement .get(temperature) member function!")

    def get_gradient(self, temperature: CF, temperature_gradient: CF) -> CF:
        raise NotImplementedError("Implement .get_gradient(temperature) member function!")

    def __call__(self, temperature: CF) -> CF:
        return self.get(temperature)

    def __repr__(self):
        formatter = Formatter()
        formatter.subheader('Dynamic Viscosity').newline()
        formatter.entry('Type', str(self))
        return formatter.output

    def __str__(self) -> str:
        return self.__class__.__name__.capitalize()


class Inviscid(DynamicViscosity):

    def get(self, temperature: CF) -> CF:
        raise TypeError("Can not get dynamic viscosity from inviscid setting!")

    def get_gradient(self, temperature: CF, temperature_gradient: CF) -> CF:
        raise TypeError("Can not get dynamic viscosity gradient from inviscid setting!")


class Constant(DynamicViscosity):

    def get(self, temperature: CF) -> CF:
        return 1

    def get_gradient(self, temperature: CF, temperature_gradient: CF) -> CF:
        return CF(tuple(0 for _ in range(temperature_gradient.dim)))


class Sutherland(DynamicViscosity):

    def __init__(self,
                 temperature_ref: float = 293.15,
                 temperature_0: float = 110.4,
                 viscosity_0: float = 1.716e-5,
                 solver_configuration: SolverConfiguration = None) -> None:

        self._temperature_ref = Parameter(temperature_ref)
        self._temperature_0 = Parameter(temperature_0)
        self._viscosity_0 = Parameter(viscosity_0)
        super().__init__(solver_configuration)

    def get(self, temperature: CF) -> CF:
        M = self.cfg.Mach_number
        gamma = self.cfg.heat_capacity_ratio

        T_ = temperature
        T_ref = self.temperature_ref
        S0 = self.temperature_0

        S_ = S0/(T_ref * (gamma - 1) * M**2)
        T_ref_ = 1/((gamma - 1) * M**2)

        return (T_/T_ref_)**(3/2) * (T_ref_ + S_)/(T_ + S_)

    def get_gradient(self, temperature: CF, temperature_gradient: CF) -> CF:
        return super().get_gradient(temperature, temperature_gradient)

    @property
    def temperature_ref(self) -> Parameter:
        return self._temperature_ref

    @temperature_ref.setter
    def temperature_ref(self, temperature_ref):
        self._temperature_ref.Set(temperature_ref)

    @property
    def temperature_0(self) -> Parameter:
        return self._temperature_0

    @temperature_0.setter
    def temperature_0(self, temperature_0):
        self._temperature_0.Set(temperature_0)

    @property
    def viscosity_0(self) -> Parameter:
        return self._viscosity_0

    @viscosity_0.setter
    def viscosity_0(self, viscosity_0):
        self._viscosity_0.Set(viscosity_0)

    def __repr__(self):
        formatter = Formatter()
        formatter.output += super().__repr__()
        formatter.entry('Reference Temperature', self.temperature_ref.Get())
        formatter.entry('Law Reference Temperature', self.temperature_0.Get())
        formatter.entry('Law Reference Viscosity', self.viscosity_0.Get())

        return formatter.output
