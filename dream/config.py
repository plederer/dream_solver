from __future__ import annotations
import logging
import typing
import textwrap
import collections
import abc

import ngsolve as ngs

logger = logging.getLogger(__name__)

# ------- Descriptors ------- #


class descriptor:

    __slots__ = ("label", )

    def __set_name__(self, owner: DescriptorDict, label: str):
        self.label = label


class variable(descriptor):
    """
    Variable is a descriptor that sets private variables to an instance with an underscore.

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


class cfg(descriptor):

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
    def __get__(self, instance: None, owner: type[object]) -> cfg:
        """ Called when an attribute is accessed via class not an instance """

    @typing.overload
    def __get__(self, instance: DictConfig, owner: type[object]) -> typing.Any:
        """ Called when an attribute is accessed on an instance variable """

    def __get__(self, cfg: DictConfig, owner: type[DictConfig]):
        if cfg is None:
            return self

        if self.fget_ is not None:
            self.fget_(cfg)

        return cfg.data[self.label]

    def __set__(self, cfg: DictConfig, value):
        if self.fset_ is not None:
            value = self.fset_(cfg, value)

        cfg.data[self.label] = value

    def __delete__(self, cfg: DictConfig):
        del cfg.data[self.label]

    def reset(self, cfg: DictConfig):
        self.__set__(cfg, self.default)

    def get_check(self, fget_: typing.Callable[[typing.Any, typing.Any], None]) -> cfg:
        prop = type(self)(self._default, self.fset_, fget_, self.__doc__)
        prop.label = self.label
        return prop

    def set_check(self, fset_: typing.Callable[[typing.Any, typing.Any], None]) -> cfg:
        prop = type(self)(self._default, fset_, self.fget_, self.__doc__)
        prop.label = self.label
        return prop

    def __call__(self, fset_: typing.Callable[[typing.Any, typing.Any], None]) -> cfg:
        return self.set_check(fset_)


class parameter(cfg):

    def __set__(self, cfg: DictConfig, value: float) -> None:

        if isinstance(value, ngs.Parameter):
            value = value.Get()

        if self.fset_ is not None:
            value = self.fset_(cfg, value)

        current = cfg.data.get(self.label, None)
        if current is None:
            cfg.data[self.label] = ngs.Parameter(value)
        else:
            cfg.data[self.label].Set(value)


class cfgdict(cfg):

    @property
    def default(self):
        return self._default()

    def __set__(self, cfg: DictConfig, value: str | DictConfig) -> None:

        if isinstance(value, self._default):
            cfg.data[self.label] = value
            return

        if not isinstance(value, dict):
            msg = "Dictionary is required!"
            raise TypeError(msg)

        if self.fset_ is not None:
            value = self.fset_(cfg, value)

        cfg.data[self.label].update(**value)


class options(cfg):

    @property
    def options(self) -> OptionHolder:
        return self._default.options

    def __set__(self, cfg: DictConfig, value: str | OptionConfig) -> None:
        if self.fset_ is not None:
            value = self.fset_(cfg, value)

        cfg.data[self.label] = self.options[value]


class optionsdict(cfg):

    @property
    def default(self) -> str:
        return self._default.label

    @property
    def options(self) -> OptionHolder:
        return self._default.options

    def __set__(self, cfg: OptionDictConfig, value: str | OptionDictConfig | dict) -> None:

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

    logger: logging.Logger

    def __new__(cls, clsname, bases, attrs):

        if 'logger' not in attrs:
            attrs['logger'] = logger.getChild(clsname)

        return abc.ABCMeta.__new__(cls, clsname, bases, attrs)

    @classmethod
    def get_configurations(cls, attrs: dict) -> list[cfg]:
        return [cfg_ for cfg_ in attrs.values() if isinstance(cfg_, cfg)]


class DictConfigMeta(ConfigMeta):

    cfgs: list[cfg]

    def __new__(cls, clsname, bases, attrs):

        if 'cfgs' not in attrs:
            attrs['cfgs'] = cls.get_configurations(attrs)

        return ConfigMeta.__new__(cls, clsname, bases, attrs)


class OptionConfigMeta(ConfigMeta):

    label: str
    aliases: tuple[str]
    options: OptionHolder

    def __new__(cls, clsname, bases, attrs, is_interface: bool = False):

        label = attrs.get('label', clsname)
        attrs['label'] = label.lower()

        if 'aliases' not in attrs:
            attrs['aliases'] = ()

        if not is_interface:
            interface = bases[0]
            attrs['logger'] = interface.logger

        cls = ConfigMeta.__new__(cls, clsname, bases, attrs)

        if is_interface:
            cls.options = OptionHolder(cls)
        else:
            cls.options[cls.label] = cls
            for alias in cls.aliases:
                cls.options[alias] = cls

        return cls


class OptionDictConfigMeta(DictConfigMeta, OptionConfigMeta):

    label: str
    aliases: tuple[str]
    options: OptionHolder
    logger: logging.Logger
    cfgs: list[cfg]

    def __new__(cls, clsname, bases, attrs, is_interface: bool = False):

        if not is_interface:
            interface = bases[0]
            attrs['logger'] = interface.logger

        dict_ = DictConfigMeta.__new__(cls, clsname, bases, attrs)

        if not is_interface:
            interface = bases[0]
            attrs['cfgs'] += interface.cfgs

        return OptionConfigMeta.__new__(cls, clsname, bases, attrs, is_interface)


class OptionHolder(collections.UserDict):

    @classmethod
    def type_to_dict(cls, type_, **kwargs) -> dict[str, typing.Any]:
        return cls.label_to_dict(type_.label, **kwargs)

    @classmethod
    def label_to_dict(cls, label, **kwargs) -> dict[str, typing.Any]:
        dict_ = {'type': label}
        dict_.update(**kwargs)
        return dict_

    def __init__(self, interface):
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

class OptionConfig(metaclass=OptionConfigMeta, is_interface=True):

    label: str
    aliases: tuple[str]
    options: OptionHolder

    def __str__(self):
        return self.__class__.label

    def __repr__(self) -> str:
        return str(self)


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

    logger = logger.getChild("State")

    @staticmethod
    def is_set(*args) -> bool:
        return all([arg is not None for arg in args])

    def to_python(self) -> State:
        """ Casts a state to a base state in which every NGSolve Coefficientfunction is cast back to a python object.
        """
        mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=1))
        mip = mesh()
        return State(**{key: value(mip) if isinstance(value, ngs.CF) else value for key, value in self.items()})

    def merge(self, *states: State, inplace: bool = False) -> State:
        merge = State()
        if inplace:
            merge = self

        for state in states:

            duplicates = set(state).intersection(merge)
            if duplicates:
                raise ValueError(f"Merge conflict! '{duplicates}' included multiple times!")

            merge.update(**state)

        return merge


class DictConfig(DescriptorDict, metaclass=DictConfigMeta):

    logger: logging.Logger
    cfgs: list[cfg]

    def __init__(self, **kwargs):
        self.clear()
        self.update(**kwargs)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            self[key] = value

            if key not in self:
                msg = f"'{key}' is not predefined for {type(self)}. "
                msg += "It is either deprecated or set manually!"
                self.logger.info(msg)

    def clear(self) -> None:
        self.__dict__.clear()
        self.data = {}
        for value in self.cfgs:
            value.reset(self)

    def flatten(self) -> dict[str, typing.Any]:
        return {key: value.flatten() if isinstance(value, DictConfig) else value for key, value in self.items()}

    def __repr__(self) -> str:
        cfg = f"{str(self)}" + "{\n"
        cfg += "".join([f"  {key}: {repr(value)} \n" for key, value in self.items()])
        cfg += '}'
        return cfg


class OptionDictConfig(DictConfig, metaclass=OptionDictConfigMeta, is_interface=True):

    label: str
    aliases: tuple[str]
    options: OptionHolder

    def flatten(self) -> dict[str, typing.Any]:
        return self.options.type_to_dict(self, **super().flatten())


# ------- Concrete Classes ------- #

class BonusIntegrationOrder(DictConfig):

    @cfg(default=0)
    def VOL(self, VOL):
        return int(VOL)

    @cfg(default=0)
    def BND(self, BND):
        return int(BND)

    @cfg(default=0)
    def BBND(self, BBND):
        return int(BBND)

    @cfg(default=0)
    def BBBND(self, BBBND):
        return int(BBBND)

    VOL: int
    BND: int
    BBND: int
    BBBND: int


class FiniteElementConfig(DictConfig):

    @cfg(default=2)
    def order(self, order):
        return int(order)

    @cfg(default=True)
    def static_condensation(self, static_condensation):
        return bool(static_condensation)

    @cfgdict(default=BonusIntegrationOrder)
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
