from __future__ import annotations
import logging
import typing
import functools
import pathlib

import ngsolve as ngs

logger = logging.getLogger(__name__)

# ------ Aliases ------ #
Integrals: typing.TypeAlias = dict[str, dict[str, ngs.comp.SumOfIntegrals]]

# ------ Helper Functions ------ #


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True
        elif shell == 'TerminalInteractiveShell':
            return False
        else:
            return False
    except NameError:
        return False


# ------- State ------- #

def equation(func):
    """ Equation is a decorator that wraps a function which takes states as arguments.

        The wrapper executes the decorated function, which should
        return a valid value. Otherwise a ValueError is thrown.
    """

    @functools.wraps(func)
    def fields(self, *args: ngsdict, **kwargs):

        quantity = func(self, *args, **kwargs)

        if quantity is None:
            raise ValueError(f"Can not determine {func.__name__} from given fields!")

        return quantity

    return fields


class quantity:

    __slots__ = ('symbol', 'name', '__doc__')

    def __set_name__(self, owner, symbol: str):
        self.symbol = symbol

    def __init__(self, name: str, symbol: str = None) -> None:
        self.name = name
        if symbol is not None:
            self.__doc__ = f"{name} :math:`{symbol}`"

    def __get__(self, fields: ngsdict, objtype) -> ngs.CF:
        if fields is None:
            return self

        if self.name in fields:
            return fields[self.name]
        elif self.symbol in fields:
            return fields[self.symbol]
        else:
            return None

    def __set__(self, fields: ngsdict, value) -> None:
        if value is not None:
            fields[self.name] = value

    def __delete__(self, fields: ngsdict):
        del fields[self.name]


class ngsdict(dict):

    def __init__(self, other: ngsdict = None, **kwargs):
        super().__init__()
        self.update(other, **kwargs)

    def update(self, other: ngsdict = None, **kwargs):

        if other is None:
            other = {}

        for key in list(other):
            if not isinstance(other[key], ngs.CF):
                other[key] = ngs.CF(other[key])

        for key in list(kwargs):
            if not isinstance(kwargs[key], ngs.CF):
                kwargs[key] = ngs.CF(kwargs[key])

        super().update(other, **kwargs)

    def __setitem__(self, key: str, value):
        if not isinstance(value, ngs.CF):
            value = ngs.CF(value)
        super().__setitem__(key, value)

    def copy(self):
        return type(self)(**super().copy())

    def to_py(self, mesh: ngs.Mesh) -> dict:
        """ Returns the current fields represented by pure python objects. 
        """
        return {key: value(mesh()) for key, value in self.items()}


# ------- User configuration ------- #


class dream_configuration(property):
    ...


class Configuration:

    CONFIG: dict[str, dream_configuration] = {}
    name: str = "configuration"

    def _get_configuration_option(self, value, OPTIONS: list[Configuration], INTERFACE: Configuration, **default):

        OPTIONS = {cls.name: cls for cls in OPTIONS}

        if isinstance(value, str) and value in OPTIONS:
            return OPTIONS[value](self.mesh, self.root, **default)
        elif isinstance(value, INTERFACE):
            value.root = self.root
            return value
        else:
            raise ValueError(f"{INTERFACE.__name__} '{value}' not available! Available configurations: {list(OPTIONS.keys())}")

    def __init_subclass__(cls, is_interface: bool = False) -> None:

        CONFIG = {name: cfg for name, cfg in vars(cls).items() if isinstance(cfg, dream_configuration)}

        if is_interface:
            cls.CONFIG = CONFIG
        else:
            cls.CONFIG = cls.CONFIG.copy()
            cls.CONFIG.update(CONFIG)

        if not hasattr(cls, "name"):
            cls.name = cls.__name__
        cls.name = cls.name.lower()

    def __init__(self, mesh: ngs.Mesh, root: Configuration = None, **default):

        if root is None:
            root = self

        self.mesh = mesh
        self.root = root

        self.update(**default)

    def update(self, dict=None, **kwargs):

        if dict is not None:
            kwargs.update(dict)

        for key in list(kwargs):

            match key.split("."):

                case [key_] if key_ in self.CONFIG:
                    setattr(self, key, kwargs.pop(key))
                case [key_, *args] if key_ in self.CONFIG:
                    getattr(self, key_).update({".".join(args): kwargs.pop(key)})
                case _:
                    kwargs.pop(key)
                    msg = f"'{key}' is not predefined for {type(self).__name__}. "
                    msg += "It is either deprecated or set manually!"
                    logger.info(msg)

    def to_dict(self, dict: dict | None = None, root: str = "") -> dict[str, typing.Any]:

        if dict is None:
            dict = {}

        for key in self.CONFIG:

            try:
                value = getattr(self, key)
            except Exception:
                continue

            if root:
                key = f"{root}.{key}"

            match value:

                case Configuration():
                    dict[key] = value.name
                    value.to_dict(dict, key)
                case ngs.Parameter():
                    dict[key] = value.Get()
                case pathlib.Path():
                    dict[key] = str(value)
                case _:
                    dict[key] = value

        return dict

    def clear(self) -> None:
        self.__init__(self.mesh, self.root)

    def __str__(self) -> str:
        return self.name
    
    def __repr__(self) -> str:
        return "\n".join([f"{key}: {value}" for key, value in self.to_dict().items()])
