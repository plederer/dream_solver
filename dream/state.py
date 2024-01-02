from __future__ import annotations
import logging

from collections import UserDict
from functools import wraps

from dream.bla import as_vector, as_matrix, as_scalar, ngs

logger = logging.getLogger(__name__)


def equation(throw: bool = False):

    def wrapper(func):

        @wraps(func)
        def state(self, state: DescriptorData, *args, **kwargs):

            _state = getattr(state, func.__name__, None)

            if _state is not None:
                name = " ".join(func.__name__.split("_")).capitalize()
                logger.debug(f"{name} set by user! Returning it.")
                return _state

            _state = func(self, state, *args, **kwargs)

            if throw and _state is None:
                raise NotImplementedError(f"Can not determine {func.__name__} from given state!")

            return _state

        return state

    return wrapper


class Descriptor:

    __slots__ = ("name",)

    def __set_name__(self, owner: DescriptorData, name: str):
        self.name = name

    def __get__(self, dict: DescriptorData, objtype) -> ngs.CF:
        value = dict.data.get(self.name, None)
        if value is None:
            logger.debug(f"{self.name.capitalize()} called, but not set!")
        return value

    def __set__(self, dict: DescriptorData, value) -> None:
        if value is not None:
            dict.data[self.name] = value


class Variable(Descriptor):
    """
    Variable is a descriptor that sets private variables to an instance with an underscore.

    Since it ressembles a scalar, a vector or a matrix it is possible to pass a cast function
    to the constructor such that every time an attribute is set, the value gets cast to the appropriate
    tensor dimensions.

    By default, if an instance does not hold an attribute of a given name, the descriptor returns None.
    """

    __slots__ = ("cast",)

    def __init__(self, cast=None) -> None:
        self.cast = cast

    def __set__(self, dict: DescriptorData, value) -> None:
        if value is not None:
            if self.cast is not None:
                value = self.cast(value)
            dict.data[self.name] = value


class DescriptorData(UserDict):

    def __init_subclass__(cls) -> None:
        cls.descriptors = tuple(key for key, des in vars(cls).items() if isinstance(des, Descriptor))

    @staticmethod
    def is_set(*args) -> bool:
        return all([arg is not None for arg in args])

    def __setitem__(self, label: str, item) -> None:
        if not isinstance(label, str):
            raise TypeError(f"Label is required to be of type string!")

        if label in self.descriptors:
            setattr(self, label, item)
        else:
            logger.info(f"{label.capitalize()} is not predefined!")

    def __getitem__(self, label: str):
        if label in self.descriptors:
            return getattr(self, label)
        else:
            return super().__getitem__(label)

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"{str(self)}{super().__repr__()}"


class State(DescriptorData):

    density = Variable(as_scalar)
    velocity = Variable(as_vector)
    momentum = Variable(as_vector)
    pressure = Variable(as_scalar)
    temperature = Variable(as_scalar)
    energy = Variable(as_scalar)
    specific_energy = Variable(as_scalar)
    inner_energy = Variable(as_scalar)
    specific_inner_energy = Variable(as_scalar)
    kinetic_energy = Variable(as_scalar)
    specific_kinetic_energy = Variable(as_scalar)
    enthalpy = Variable(as_scalar)
    specific_enthalpy = Variable(as_scalar)
    speed_of_sound = Variable(as_scalar)

    viscosity = Variable(as_scalar)
    strain_rate_tensor = Variable(as_matrix)
    deviatoric_stress_tensor = Variable(as_matrix)
    heat_flux = Variable(as_vector)

    density_gradient = Variable(as_vector)
    velocity_gradient = Variable(as_matrix)
    momentum_gradient = Variable(as_matrix)
    pressure_gradient = Variable(as_vector)
    temperature_gradient = Variable(as_vector)
    energy_gradient = Variable(as_vector)
    specific_energy_gradient = Variable(as_vector)
    inner_energy_gradient = Variable(as_vector)
    specific_inner_energy_gradient = Variable(as_vector)
    kinetic_energy_gradient = Variable(as_vector)
    specific_kinetic_energy_gradient = Variable(as_vector)
    enthalpy_gradient = Variable(as_vector)
    specific_enthalpy_gradient = Variable(as_vector)

    @staticmethod
    def merge_state(*states: State) -> State:
        merge = {}
        for state in states:
            for label, var in state.items():
                if label in merge:
                    raise ValueError(f"Merge conflict! '{label.capitalize()}' included multiple times!")
                merge[label] = var
        return State(**merge)


class ScalingState(DescriptorData):

    length = Variable()
    density = Variable()
    momentum = Variable()
    velocity = Variable()
    speed_of_sound = Variable()
    temperature = Variable()
    pressure = Variable()
    energy = Variable()
    inner_energy = Variable()
    kinetic_energy = Variable()
