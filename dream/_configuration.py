from __future__ import annotations
import numpy as np

from ngsolve import *
from .utils import Formatter

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .configuration import SolverConfiguration


class State:

    @classmethod
    def FarField(cls,
                 solver_configuration: SolverConfiguration,
                 velocity_direction: tuple[float, ...],
                 normalize_direction: bool = True,
                 as_float: bool = False) -> State:

        cfg = solver_configuration

        M = cfg.Mach_number
        gamma = cfg.heat_capacity_ratio

        if as_float:
            M = M.Get()
            gamma = gamma.Get()

        if normalize_direction:
            vec = np.array(velocity_direction)
            velocity_direction = tuple(vec/np.sqrt(np.sum(np.square(vec))))

        rho_factor = cfg.scaling.get_density_factor(M)
        u_factor = cfg.scaling.get_velocity_factor(M)
        T_factor = cfg.scaling.get_temperature_factor(M)
        p_factor = cfg.scaling.get_pressure_factor(M)
        c_factor = cfg.scaling.get_speed_of_sound_factor(M)

        density = 1 * rho_factor
        velocity = tuple(comp * u_factor for comp in velocity_direction)
        pressure = p_factor/gamma
        temperature = T_factor/(gamma - 1)
        energy = pressure/(gamma - 1) + density * sum(u**2 for u in velocity)/2
        speed_of_sound = 1 * c_factor

        if not as_float:
            velocity = CF(velocity)

        return State(density, velocity, pressure, temperature, energy, speed_of_sound)

    def __init__(self, 
                 density: CF = None, 
                 velocity: CF = None, 
                 pressure: CF = None, 
                 temperature: CF = None, 
                 energy: CF = None,
                 speed_of_sound: CF = None):
        
        self.density = density
        self.velocity = velocity
        self.pressure = pressure
        self.temperature = temperature
        self.energy = energy
        self.speed_of_sound = speed_of_sound

    @property
    def momentum(self) -> CF:
        if self.velocity is None or self.density is None:
            raise ValueError("Can not determine momentum if velocity and density are None!")
        return CF(tuple(self.density * u for u in self.velocity))

    @property
    def is_none_thermodynamic(self) -> bool:
        return all([val is None for val in (self.pressure, self.temperature, self.energy)])

    def __repr__(self) -> str:
        variables = ", ".join([f"{key}: {value}" for key, value in vars(self).items() if value is not None])
        return f"State(" + variables + ")"


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
        return self.__class__.__name__.upper()


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


class Scaling:

    TYPES: dict[str, Scaling] = {}

    def __init_subclass__(cls) -> None:
        label = cls.__name__.lower()
        cls.TYPES[label] = cls

    def get_Reynolds_number_scaled(self, Reynolds_number, Mach_number):
        return Reynolds_number/self.get_velocity_factor(Mach_number)

    def get_density_factor(self, Mach_number):
        return 1

    def get_velocity_factor(self, Mach_number):
        raise NotImplementedError()

    def get_speed_of_sound_factor(self, Mach_number):
        raise NotImplementedError()

    def get_pressure_factor(self, Mach_number):
        return self.get_speed_of_sound_factor(Mach_number)**2

    def get_temperature_factor(self, Mach_number):
        return self.get_speed_of_sound_factor(Mach_number)**2

    def __str__(self) -> str:
        return self.__class__.__name__.upper()


class Aerodynamic(Scaling):

    def get_velocity_factor(self,  Mach_number):
        return 1

    def get_speed_of_sound_factor(self, Mach_number):
        return 1/Mach_number


class Acoustic(Scaling):

    def get_velocity_factor(self,  Mach_number):
        return Mach_number

    def get_speed_of_sound_factor(self, Mach_number):
        return 1


class Aeroacoustic(Scaling):

    def get_velocity_factor(self, Mach_number):
        return Mach_number/(1 + Mach_number)

    def get_speed_of_sound_factor(self, Mach_number):
        return 1/(1 + Mach_number)
