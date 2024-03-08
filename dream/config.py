from __future__ import annotations
import logging
import typing
import textwrap
import collections
import abc
import functools

import ngsolve as ngs

logger = logging.getLogger(__name__)


# ------- Decorators ------- #


def equation(func):
    """ Equation is a decorator that wraps a function which takes as first argument a state.

        The name of the function should ressemble a physical quantity like 'density' or 'velocity'.

        When the decorated function get's called the wrapper checks first if the quantity is already defined
        and returns it. 
        If the quantity is not defined, the wrapper executes the decorated function, which should
        return a valid value. Otherwise a ValueError is thrown.
    """

    @functools.wraps(func)
    def state(self, state: State, *args, **kwargs):

        _state = state.data.get(func.__name__, None)

        if _state is not None:
            name = " ".join(func.__name__.split("_")).capitalize()
            logger.debug(f"{name} set by user! Returning it.")
            return _state

        _state = func(self, state, *args, **kwargs)

        if _state is None:
            raise ValueError(f"Can not determine {func.__name__} from given state!")

        return _state

    return state


# ------- Descriptors ------- #


class descriptor:

    __slots__ = ("label", )

    def __set_name__(self, owner: DescriptorDict, label: str):
        self.label = label


class variable(descriptor):
    """
    Variable is a descriptor that mimics a physical quantity.

    Since it ressembles a scalar, a vector or a matrix it is possible to pass a cast function
    to the constructor such that every time an attribute is set, the value gets cast to the appropriate
    tensor dimensions.

    By default, if an instance does not hold an attribute of a given name, the descriptor returns None.
    """

    __slots__ = ("cast",)

    def __init__(self, cast: typing.Callable) -> None:
        self.cast = cast

    def __get__(self, state: State, objtype) -> ngs.CF:
        return state.data.get(self.label, None)

    def __set__(self, state: State, value) -> None:
        if value is not None:
            value = self.cast(value)
            state.data[self.label] = value

    def __delete__(self, state: State):
        del state.data[self.label]


class standard_configuration(descriptor):

    __slots__ = ('_default', 'fset_', 'fget_', '__doc__')

    @property
    def default(self):
        return self._default

    def __init__(self, default, fset_=None, fget_=None, doc: str = None):
        self._default = default
        self.fset_ = fset_
        self.fget_ = fget_
        self.__doc__ = doc

        if doc is None and fset_ is not None:
            doc = fset_.__doc__

        self.__doc__ = doc
        self.label = ''

    @typing.overload
    def __get__(self, instance: None, owner: type[object]) -> standard_configuration:
        """ Called when an attribute is accessed via class not an instance """

    @typing.overload
    def __get__(self, instance: SingleConfiguration, owner: type[object]) -> typing.Any:
        """ Called when an attribute is accessed on an instance variable """

    def __get__(self, cfg: SingleConfiguration, owner: type[SingleConfiguration]):
        if cfg is None:
            return self

        if self.fget_ is not None:
            self.fget_(cfg)

        return cfg.data[self.label]

    def __set__(self, cfg: SingleConfiguration, value):
        if self.fset_ is not None:
            value = self.fset_(cfg, value)

        cfg.data[self.label] = value

    def __delete__(self, cfg: SingleConfiguration):
        del cfg.data[self.label]

    def reset(self, cfg: SingleConfiguration):
        self.__set__(cfg, self.default)

    def getter_check(self, fget_: typing.Callable[[typing.Any, typing.Any], None]) -> standard_configuration:
        prop = type(self)(self._default, self.fset_, fget_, self.__doc__)
        prop.label = self.label
        return prop

    def set_check(self, fset_: typing.Callable[[typing.Any, typing.Any], None]) -> standard_configuration:
        prop = type(self)(self._default, fset_, self.fget_, self.__doc__)
        prop.label = self.label
        return prop

    def __call__(self, fset_: typing.Callable[[typing.Any, typing.Any], None]) -> standard_configuration:
        return self.set_check(fset_)


class parameter_configuration(standard_configuration):

    def __set__(self, cfg: SingleConfiguration, value: float) -> None:

        if isinstance(value, ngs.Parameter):
            value = value.Get()

        if self.fset_ is not None:
            value = self.fset_(cfg, value)

        current = cfg.data.get(self.label, None)
        if current is None:
            cfg.data[self.label] = ngs.Parameter(value)
        else:
            cfg.data[self.label].Set(value)


class single_configuration(standard_configuration):

    _default: SingleConfiguration

    @property
    def default(self):
        return self._default()

    def __set__(self, cfg: SingleConfiguration, value: str | SingleConfiguration) -> None:

        if isinstance(value, self._default):
            cfg.data[self.label] = value
            return

        if not isinstance(value, dict):
            msg = "Dictionary is required!"
            raise TypeError(msg)

        if self.fset_ is not None:
            value = self.fset_(cfg, value)

        cfg.data[self.label].update(**value)


class interface_configuration(standard_configuration):

    _default: InterfaceConfiguration

    @property
    def options(self) -> OptionHolder:
        return self._default.options

    def __set__(self, cfg: SingleConfiguration, value: str | InterfaceConfiguration) -> None:
        if self.fset_ is not None:
            value = self.fset_(cfg, value)

        cfg.data[self.label] = self.options[value]


class multiple_configuration(standard_configuration):

    _default: MultipleConfiguration

    @property
    def default(self) -> str:
        return self._default.label

    @property
    def options(self) -> OptionHolder:
        return self._default.options

    def __set__(self, cfg: MultipleConfiguration, value: str | MultipleConfiguration | dict) -> None:

        if isinstance(value, self.options.interface):
            cfg.data[self.label] = value
            return

        if isinstance(value, str) or value is None:
            value = self.options.label_to_dict(value)

        if self.fset_ is not None:
            value = self.fset_(cfg, value)

        if not isinstance(value, dict) and 'type' in value:
            msg = "Don't know which type to instantiate!"
            msg += "Dictionary with key 'type' required!"
            raise TypeError(msg)

        type_ = self.options[value.pop('type')]
        current = cfg.data.get(self.label, None)

        if isinstance(current, type_):
            current.update(**value)
        else:
            cfg.data[self.label] = type_(**value)


# ------- Configuration Metaclasses ------- #

class ConfigMeta(abc.ABCMeta):

    @classmethod
    def get_configurations(cls, attrs: dict) -> list[standard_configuration]:
        return [cfg_ for cfg_ in attrs.values() if isinstance(cfg_, standard_configuration)]


class SingleConfigurationMeta(ConfigMeta):

    cfgs: list[standard_configuration]

    def __new__(cls, clsname, bases, attrs):

        if 'cfgs' not in attrs:
            attrs['cfgs'] = cls.get_configurations(attrs)

        for base in bases:
            if hasattr(base, 'cfgs'):
                attrs['cfgs'] += base.cfgs

        return ConfigMeta.__new__(cls, clsname, bases, attrs)


class InterfaceMeta(ConfigMeta):

    label: str
    aliases: tuple[str]
    options: OptionHolder

    def __new__(cls, clsname, bases, attrs, is_interface: bool = False):

        label = attrs.get('label', clsname)
        attrs['label'] = label.lower()

        if 'aliases' not in attrs:
            attrs['aliases'] = ()

        cls = ConfigMeta.__new__(cls, clsname, bases, attrs)

        if is_interface:
            cls.options = OptionHolder(cls)
        else:
            cls.options[cls.label] = cls
            for alias in cls.aliases:
                cls.options[alias] = cls

        return cls


class MultipleConfigurationMeta(SingleConfigurationMeta, InterfaceMeta):

    label: str
    aliases: tuple[str]
    options: OptionHolder
    cfgs: list[standard_configuration]

    def __new__(cls, clsname, bases, attrs, is_interface: bool = False):
        dict_ = SingleConfigurationMeta.__new__(cls, clsname, bases, attrs)
        return InterfaceMeta.__new__(cls, clsname, bases, attrs, is_interface)


class OptionHolder(collections.UserDict):

    @classmethod
    def option_to_dict(cls, option: InterfaceConfiguration, **kwargs) -> dict[str, typing.Any]:
        return cls.label_to_dict(option.label, **kwargs)

    @classmethod
    def label_to_dict(cls, label, **kwargs) -> dict[str, typing.Any]:
        dict_ = {'type': label}
        dict_.update(**kwargs)
        return dict_

    def __init__(self, interface: InterfaceConfiguration):
        self.interface = interface
        super().__init__()

    def __getitem__(self, alias):
        if isinstance(alias, self.interface):
            return type(alias)
        elif isinstance(alias, type) and issubclass(alias, self.interface):
            return alias
        elif alias in self:
            return super().__getitem__(alias)
        else:
            msg = f"Invalid type '{alias}' for class '{self.interface}'.\n"
            msg += f"Allowed options: {list(self)}!"
            raise TypeError(msg)

    def __contains__(self, key: object) -> bool:
        if isinstance(key, self.interface):
            return True
        elif isinstance(key, type) and issubclass(key, self.interface):
            return True
        return super().__contains__(key)


# ------- Abstract Classes ------- #


class DescriptorDict(collections.UserDict):

    def __setitem__(self, key: str, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def __delitem__(self, key) -> None:
        delattr(self, key)

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"{str(self)}{self.data}"


class State(DescriptorDict):

    @staticmethod
    def is_set(*args) -> bool:
        return all([arg is not None for arg in args])

    def to_python(self) -> State:
        """ Returns the current state represented by pure python objects. 
        """
        mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=1))
        mip = mesh()
        return State(**{key: value(mip) if isinstance(value, ngs.CF) else value for key, value in self.items()})

    def merge_states(self, *states: State, overwrite: bool = False, inplace: bool = False) -> State:
        merge = State()
        if inplace:
            merge = self

        for state in states:

            duplicates = set(state).intersection(merge)
            if duplicates:

                if not overwrite:
                    raise ValueError(f"Merge conflict! '{duplicates}' included multiple times!")

                logger.warning(f"Found duplicates '{duplicates}'! Overwriting state!")

            merge.update(**state)

        return merge


class InterfaceConfiguration(metaclass=InterfaceMeta, is_interface=True):

    label: str
    aliases: tuple[str]
    options: OptionHolder

    def __str__(self):
        return self.__class__.label

    def __repr__(self) -> str:
        return str(self)


class SingleConfiguration(DescriptorDict, metaclass=SingleConfigurationMeta):

    cfgs: list[standard_configuration]

    def __init__(self, **kwargs):
        self.clear()
        self.update(**kwargs)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            self[key] = value

            if key not in self:
                msg = f"'{key}' is not predefined for {type(self)}. "
                msg += "It is either deprecated or set manually!"
                logger.info(msg)

    def clear(self) -> None:
        self.data = {}
        for value in self.cfgs:
            value.reset(self)

    def flatten(self) -> dict[str, typing.Any]:
        return {key: value.flatten() if isinstance(value, SingleConfiguration) else value for key, value in self.items()}

    def __repr__(self) -> str:
        cfg = f"{str(self)}" + "{\n"
        cfg += "".join([f"  {key}: {repr(value)} \n" for key, value in self.items()])
        cfg += '}'
        return cfg


class MultipleConfiguration(SingleConfiguration, metaclass=MultipleConfigurationMeta, is_interface=True):

    label: str
    aliases: tuple[str]
    options: OptionHolder

    def flatten(self) -> dict[str, typing.Any]:
        return self.options.option_to_dict(self, **super().flatten())


# ------- Concrete Classes ------- #

class BonusIntegrationOrder(SingleConfiguration):

    @standard_configuration(default=0)
    def VOL(self, VOL):
        return int(VOL)

    @standard_configuration(default=0)
    def BND(self, BND):
        return int(BND)

    @standard_configuration(default=0)
    def BBND(self, BBND):
        return int(BBND)

    @standard_configuration(default=0)
    def BBBND(self, BBBND):
        return int(BBBND)

    VOL: int
    BND: int
    BBND: int
    BBBND: int


class FiniteElementConfig(SingleConfiguration):

    @standard_configuration(default=2)
    def order(self, order):
        return int(order)

    @standard_configuration(default=True)
    def static_condensation(self, static_condensation):
        return bool(static_condensation)

    @single_configuration(default=BonusIntegrationOrder)
    def bonus_int_order(self, dict_):
        return dict_

    def format(self):
        formatter = self.formatter.new()
        formatter.subheader('Finite Element Configuration').newline()
        formatter.entry('Polynomial Order', self._order)
        formatter.entry('Static Condensation', str(self._static_condensation))
        formatter.add_config(self.bonus_int_order)
        return formatter.output

    order: int
    static_condensation: bool
    bonus_int_order: BonusIntegrationOrder


# ------ Formatter ------- #

class Formatter:

    TERMINAL_WIDTH: int = 80
    TEXT_INDENT = 5
    COLUMN_RATIO = (0.5, 0.5)

    @classmethod
    def new(cls):
        return cls()

    def __init__(self) -> None:
        self.reset()

    def header(self, text: str):
        text = f" {text.upper()} "
        header = f"{text:─^{self.TERMINAL_WIDTH}}" + "\n"
        self.output += header
        return self

    def subheader(self, text: str):
        text = '─── ' + text.upper() + ' ───'
        subheader = f"{text:^{self.TERMINAL_WIDTH}}" + "\n"
        self.output += subheader
        return self

    def entry(self, text: str, value, equal: str = ":"):

        if isinstance(value, ngs.Parameter):
            value = value.Get()

        txt_width, value_width = tuple(int(self.TERMINAL_WIDTH * ratio) for ratio in self.COLUMN_RATIO)
        entry = f"{text + equal:>{txt_width}} {value:<{value_width}}" + "\n"
        self.output += entry
        return self

    def text(self, text: str):
        width = self.TERMINAL_WIDTH - 2*self.TEXT_INDENT
        indent = self.TEXT_INDENT * ' '
        for line in text.split('\n'):
            wrap_line = textwrap.fill(line, width, initial_indent=indent, subsequent_indent=indent,
                                      replace_whitespace=False, drop_whitespace=False)
            self.output += wrap_line + "\n"
        return self

    def newline(self):
        self.output += "\n"
        return self

    def reset(self):
        self.output = ""

    def add_config(self, object):
        self.output += object.format()
        return self

    def __repr__(self) -> str:
        return self.output
