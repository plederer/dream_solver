from __future__ import annotations
from .utils import Formatter
from ngsolve import *

DYNAMIC_VISCOSITY = {}


def dynamic_viscosity_factory(viscosity, **kwargs):
    if isinstance(viscosity, _DynamicViscosity):
        return viscosity
    elif isinstance(viscosity, str):
        viscosity = viscosity.lower()
        try:
            mu = DYNAMIC_VISCOSITY[viscosity]
        except KeyError:
            msg = f"'{viscosity.capitalize()}' is not a valid Dynamic Viscosity. "
            msg += f"Possible alternatives: {[key for key in DYNAMIC_VISCOSITY]}"
            raise ValueError(msg) from None
        return mu(**kwargs)
    else:
        raise TypeError(f'Dynamic Viscosity must inherit from {_DynamicViscosity}')


class _DynamicViscosity:

    def __init_subclass__(self) -> None:
        label = self.__name__.lower()
        DYNAMIC_VISCOSITY[label] = self

    @property
    def is_inviscid(self):
        return isinstance(self, Inviscid)

    def __repr__(self):
        formatter = Formatter()
        formatter.subheader('Dynamic Viscosity').newline()
        formatter.entry('Type', str(self))
        return formatter.output

    def __str__(self) -> str:
        return self.__class__.__name__.capitalize()


class Inviscid(_DynamicViscosity):
    ...


class Constant(_DynamicViscosity):
    ...


class Sutherland(_DynamicViscosity):

    def __init__(self, temperature_ref: float = 293.15, temperature_0: float = 110.4, viscosity_0: float = 1.716e-5) -> None:
        self._temperature_ref = Parameter(temperature_ref)
        self._temperature_0 = Parameter(temperature_0)
        self._viscosity_0 = Parameter(viscosity_0)

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
