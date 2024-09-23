from __future__ import annotations
import logging
import typing
import textwrap
import collections
import functools

import ngsolve as ngs

logger = logging.getLogger(__name__)


# ------- State ------- #

def equation(func):
    """ Equation is a decorator that wraps a function which takes as first argument a state.

        The name of the function should ressemble a physical quantity like 'density' or 'velocity'.

        When the decorated function get's called the wrapper checks first if the quantity is already defined
        and returns it. 
        If the quantity is not defined, the wrapper executes the decorated function, which should
        return a valid value. Otherwise a ValueError is thrown.
    """

    @functools.wraps(func)
    def state(self, U: State, *args, **kwargs):

        _state = U.data.get(func.__name__, None)

        if _state is not None:
            name = " ".join(func.__name__.split("_")).capitalize()
            logger.debug(f"{name} set by user! Returning it.")
            return _state

        _state = func(self, U, *args, **kwargs)

        if _state is None:
            raise ValueError(f"Can not determine {func.__name__} from given state!")

        return _state

    return state


# ------- Descriptors ------- #


class descriptor:

    __slots__ = ("name", )

    def __set_name__(self, owner, name: str):
        self.name = name


class variable(descriptor):
    """
    Variable is a descriptor that mimics a physical quantity.

    Since it ressembles a scalar, a vector or a matrix it is possible to pass a cast function
    to the constructor such that every time an attribute is set, the value gets cast to the appropriate
    tensor dimensions.
    """

    __slots__ = ("cast", "label")

    def __init__(self, cast: typing.Callable, label: str) -> None:
        self.cast = cast
        self.label = label

    def __get__(self, state: State, objtype) -> ngs.CF:
        return state.data.get(self.label, None)

    def __set__(self, state: State, value) -> None:
        if value is not None:
            value = self.cast(value)
            state.data[self.label] = value

    def __delete__(self, state: State):
        del state.data[self.label]


class configuration(descriptor):

    __slots__ = ('_default', 'fset_', 'fget_', '__doc__')

    def __init__(self, default, fset_=None, fget_=None, doc: str = None):
        self.default = default
        self.fset_ = fset_
        self.fget_ = fget_
        self.__doc__ = doc

        if doc is None and fset_ is not None:
            doc = fset_.__doc__

        self.__doc__ = doc

    @typing.overload
    def __get__(self, instance: None, owner: type[object]) -> configuration:
        """ Called when an attribute is accessed via class not an instance """

    @typing.overload
    def __get__(self, instance: MultipleConfiguration, owner: type[object]) -> typing.Any:
        """ Called when an attribute is accessed on an instance variable """

    def __get__(self, cfg: MultipleConfiguration, owner: type[MultipleConfiguration]):
        if cfg is None:
            return self

        if self.fget_ is not None:
            self.fget_(cfg)

        return cfg.data[self.name]

    def __delete__(self, cfg: MultipleConfiguration):
        del cfg.data[self.name]

    def getter_check(self, fget_: typing.Callable[[typing.Any, typing.Any], None]) -> configuration:
        prop = type(self)(self.default, self.fset_, fget_, self.__doc__)
        return prop

    def setter_check(self, fset_: typing.Callable[[typing.Any, typing.Any], None]) -> configuration:
        prop = type(self)(self.default, fset_, self.fget_, self.__doc__)
        return prop

    def __call__(self, fset_: typing.Callable[[typing.Any, typing.Any], None]) -> configuration:
        return self.setter_check(fset_)


class any(configuration):

    def __set__(self, cfg: MultipleConfiguration, value):
        if self.fset_ is not None:
            value = self.fset_(cfg, value)

        cfg.data[self.name] = value

    def reset(self, cfg: MultipleConfiguration):
        self.__set__(cfg, self.default)


class parameter(configuration):

    def __set__(self, cfg: MultipleConfiguration, value: float) -> None:

        if isinstance(value, ngs.Parameter):
            value = value.Get()

        if self.fset_ is not None:
            value = self.fset_(cfg, value)

        cfg.data[self.name].Set(value)

    def reset(self, cfg: MultipleConfiguration):

        value = self.default
        if self.fset_ is not None:
            value = self.fset_(cfg, value)

        cfg.data[self.name] = ngs.Parameter(value)


class unique(configuration):

    default: UniqueConfiguration

    def __set__(self, cfg: MultipleConfiguration, value: str | MultipleConfiguration) -> None:

        if isinstance(value, self.default):
            cfg.data[self.name] = value

        elif isinstance(value, str):

            if value == self.name:
                cfg.data[self.name] = self.default()

            if self.fset_ is not None:
                value = self.fset_(cfg, cfg.data[self.name])

        elif isinstance(value, dict):
            cfg.data[self.name].update(**value)

        else:
            msg = f""" Can not set variable of type {type(value)}!

            To set '{self.name}' pass either:
            1. An instance of type {self.default.leafs.root}.
            2. A keyword from available options: {list(self.default.leafs)}.

            To update the current instance pass a dictionary with following keywords:
            {list(cfg.data[self.name].data.keys())}
            """
            raise TypeError(msg)

        cfg.data[self.name].parent = cfg

    def reset(self, cfg: MultipleConfiguration):
        self.__set__(cfg, self.default())


class multiple(unique):

    default: MultipleConfiguration

    def __set__(self, cfg: MultipleConfiguration, value: str | MultipleConfiguration) -> None:

        if isinstance(value, self.default.leafs.root):
            cfg.data[self.name] = value

        elif isinstance(value, str):
            cfg_ = self.default.leafs[value]
            if not isinstance(cfg.data[self.name], cfg_):
                cfg.data[self.name] = cfg_()

            if self.fset_ is not None:
                value = self.fset_(cfg, cfg.data[self.name])

        elif isinstance(value, dict):
            cfg.data[self.name].update(**value)

        else:
            msg = f""" Can not set variable of type {type(value)}!

            To set '{self.name}' pass either:
            1. An instance of type {self.default.leafs.root}.
            2. A keyword from available options: {list(self.default.leafs)}.

            To update the current instance pass a dictionary with following keywords:
            {list(cfg.data[self.name].data.keys())}
            """
            raise TypeError(msg)

        cfg.data[self.name].parent = cfg


# ------- Abstract Classes ------- #

class InheritanceTree(collections.UserDict):

    def __init__(self, root: UniqueConfiguration):
        self.root = root
        super().__init__()

    def __getitem__(self, alias) -> UniqueConfiguration | MultipleConfiguration:
        if isinstance(alias, self.root):
            return type(alias)
        elif isinstance(alias, type) and issubclass(alias, self.root):
            return alias
        elif alias in self:
            return super().__getitem__(alias)
        else:
            msg = f"Invalid type '{alias}' for class '{self.root}'.\n"
            msg += f"Allowed options: {list(self)}!"
            raise TypeError(msg)

    def __contains__(self, key: object) -> bool:
        if isinstance(key, self.root):
            return True
        elif isinstance(key, type) and issubclass(key, self.root):
            return True
        return super().__contains__(key)


class State(collections.UserDict):

    def __init_subclass__(cls) -> None:
        cls.alias_map = {value.label: var for var, value in vars(cls).items() if isinstance(value, variable)}
        return super().__init_subclass__()

    def __setitem__(self, key: str, value):
        if key in self.alias_map:
            key = self.alias_map[key]
        setattr(self, key, value)

    def __getitem__(self, key):
        if key in self.alias_map:
            key = self.alias_map[key]
        return getattr(self, key)

    def __delitem__(self, key) -> None:
        if key in self.alias_map:
            key = self.alias_map[key]
        delattr(self, key)

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"{str(self)}{self.data}"

    @staticmethod
    def is_set(*args) -> bool:
        return all([arg is not None for arg in args])

    def to_float(self) -> State:
        """ Returns the current state represented by pure python objects. 
        """
        mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=1))
        return {key: value(mesh()) if isinstance(value, ngs.CF) else value for key, value in self.items()}

    def __repr__(self) -> str:
        data = {key: value for key, value in self.data.items() if value is not None}
        return f"{str(self)}{data}"

    def __str__(self):
        return self.__class__.__name__


class UniqueConfiguration(collections.UserDict):

    name: str
    cfgs: list[configuration]

    def __init_subclass__(cls) -> None:

        if not hasattr(cls, "cfgs"):
            cls.cfgs = [cfg_ for cfg_ in vars(cls).values() if isinstance(cfg_, configuration)]

        if not hasattr(cls, "name"):
            cls.name = cls.__name__

        cls.name = cls.name.lower()

        super().__init_subclass__()

    def __init__(self, dict=None, /, **kwargs):
        self.clear()
        self.update(dict, **kwargs)

    def update(self, dict=None, **kwargs):

        if dict is not None:
            kwargs.update(dict)

        for key in list(kwargs):

            match key.split("."):

                case [key_] if key_ in self:
                    self[key_] = kwargs.pop(key)
                case [key_, *args] if key_ not in kwargs and key_ in self:
                    kwargs[key_] = {".".join(args): kwargs.pop(key)}
                case [key_, *args] if key_ in kwargs and key_ in self:
                    kwargs[key_][".".join(args)] = kwargs.pop(key)
                case _:
                    kwargs.pop(key)
                    msg = f"'{key}' is not predefined for {type(self)}. "
                    msg += "It is either deprecated or set manually!"
                    logger.info(msg)

        for key, value in kwargs.items():
            self[key].update(value)

    def to_flat_dict(self, data: dict = None, root: str = "") -> dict[str, typing.Any]:

        if data is None:
            data = {}

        for key, value in self.items():

            if root:
                key = f"{root}.{key}"

            match value:

                case UniqueConfiguration():
                    data[key] = value.name
                    value.to_flat_dict(data, key)
                case ngs.Parameter():
                    data[key] = value.Get()
                case _:
                    data[key] = value

        return data

    def clear(self) -> None:
        self.data = {}
        for value in self.cfgs:
            value.reset(self)

    def __setitem__(self, key: str, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def __delitem__(self, key) -> None:
        delattr(self, key)

    def __repr__(self) -> str:
        return str(self.to_flat_dict())


class MultipleConfiguration(UniqueConfiguration):

    name: str
    leafs: InheritanceTree
    cfgs: list[configuration]
    aliases: tuple[str]

    def __init_subclass__(cls, is_interface: bool = False) -> None:

        if is_interface:
            cls.leafs = InheritanceTree(cls)
            cls.aliases = ()
            cls.cfgs = [cfg for cfg in vars(cls).values() if isinstance(cfg, configuration)] 

        else:
            for name in [cls.name, *cls.aliases]:
                name = name.lower()
                if name in cls.leafs:
                    raise ValueError(f"Concrete Class of type {cls.leafs.root} with name {name} already included!")
                cls.leafs[name] = cls

            cls.cfgs = [cfg for cfg in vars(cls).values() if isinstance(cfg, configuration)] + cls.cfgs

        super().__init_subclass__()


# ------- Concrete Classes ------- #

class BonusIntegrationOrder(UniqueConfiguration):

    @any(default=0)
    def vol(self, vol):
        return int(vol)

    @any(default=0)
    def bnd(self, bnd):
        return int(bnd)

    @any(default=0)
    def bbnd(self, bbnd):
        return int(bbnd)

    @any(default=0)
    def bbbnd(self, bbbnd):
        return int(bbbnd)

    vol: int
    bnd: int
    bbnd: int
    bbbnd: int


class FiniteElementConfig(UniqueConfiguration):

    @any(default=2)
    def order(self, order):
        return int(order)

    @any(default=True)
    def static_condensation(self, static_condensation):
        return bool(static_condensation)

    @unique(default=BonusIntegrationOrder)
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
