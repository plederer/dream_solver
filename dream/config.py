from __future__ import annotations
import logging
import typing
import functools
import pathlib

import ngsolve as ngs

logger = logging.getLogger(__name__)

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
    def state(self, *args: ngsdict, **kwargs):

        quantity = func(self, *args, **kwargs)

        if quantity is None:
            raise ValueError(f"Can not determine {func.__name__} from given state!")

        return quantity

    return state


    """
    Variable is a descriptor that mimics a physical quantity.

    Since it ressembles a scalar, a vector or a matrix it is possible to pass a cast function
    to the constructor such that every time an attribute is set, the value gets cast to the appropriate
    tensor dimensions.
    """

class quantity:


    __slots__ = ('symbol', 'name', '__doc__')

    def __set_name__(self, owner, symbol: str):
        self.symbol = symbol

    def __init__(self, name: str, symbol: str = None) -> None:
        self.name = name
        if symbol is not None:
            self.__doc__ = f"{name} :math:`{symbol}`"

    def __get__(self, state: ngsdict, objtype) -> ngs.CF:
        if state is None:
            return self
        return state.get(self.name, None)

    def __set__(self, state: ngsdict, value) -> None:
        if value is not None:
            state[self.name] = value

    def __delete__(self, state: ngsdict):
        del state[self.name]


class ngsdict(typing.MutableMapping):

    symbols: dict[str, str] = {}

    def to_py(self) -> dict:
        """ Returns the current state represented by pure python objects. 
        """
        mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=1))
        return {key: value(mesh()) for key, value in self.items()}

    def __init_subclass__(cls) -> None:
        symbols = {symbol: value.name for symbol, value in vars(cls).items() if isinstance(value, quantity)}
        if hasattr(cls, "symbols"):
            symbols.update(cls.symbols)
        cls.symbols = symbols
        return super().__init_subclass__()

    def __init__(self, *args, **kwargs):
        self.data = {}
        self.update(*args, **kwargs)

    def __setitem__(self, key: str, value):
        if not isinstance(value, ngs.CF):
            value = ngs.CF(value)

        if key in self.symbols:
            key = self.symbols[key]

        self.data[key] = value

    def __getitem__(self, key):
        return self.data[key]

    def __delitem__(self, key):
        del self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return "".join([f"{key}: {str(value)}" for key, value in self.items()])


# ------- User configuration ------- #

VALUE = typing.TypeVar('VALUE')


class configuration(typing.Generic[VALUE]):

    __slots__ = ('default', 'fset', 'fget', '__name__', '__doc__', '__annotations__')

    def __set_name__(self, owner, name):
        self.__name__ = name

    def __init__(self, default, fset=None, fget=None, doc: str | None = None):

        if doc is None and fset is not None:
            doc = fset.__doc__

        self.default = default
        self.fset = fset
        self.fget = fget
        self.__doc__ = doc

    @typing.overload
    def __get__(self, cfg: None, owner: type[CONFIG]) -> configuration:
        """ Called when an attribute is accessed via class not an instance """

    @typing.overload
    def __get__(self, cfg: CONFIG, owner: type[CONFIG]) -> VALUE:
        """ Called when an attribute is accessed on an instance variable """

    def __get__(self, cfg: CONFIG | None, owner: type[CONFIG]) -> VALUE | configuration:
        if cfg is None:
            return self

        if self.fget is not None:
            self.fget(cfg)

        return cfg.data[self.__name__]

    def __set__(self, cfg: CONFIG, value):
        if self.fset is not None:
            value = self.fset(cfg, value)

        cfg.data[self.__name__] = value

    def __delete__(self, cfg: UniqueConfiguration):
        del cfg.data[self.__name__]

    def getter_check(self, fget: typing.Callable[[CONFIG], None]):
        return type(self)(self.default, self.fset, fget, self.__doc__)

    def setter_check(self, fset: typing.Callable[[CONFIG, typing.Any], VALUE]):
        return type(self)(self.default, fset, self.fget, self.__doc__)

    def __call__(self, fset: typing.Callable[[CONFIG, typing.Any], typing.Any]) -> configuration:
        return self.setter_check(fset)

    def reset(self, cfg: UniqueConfiguration):
        self.__set__(cfg, self.default)


class parameter(configuration):

    default: float

    def __set__(self, cfg: CONFIG, value) -> None:

        if isinstance(value, ngs.Parameter):
            value = value.Get()

        if self.fset is not None:
            value = self.fset(cfg, value)

        cfg.data[self.__name__].Set(value)

    def reset(self, cfg: UniqueConfiguration):

        value = self.default
        if self.fset is not None:
            value = self.fset(cfg, value)

        cfg.data[self.__name__] = ngs.Parameter(value)


class unique(configuration):

    default: type[UniqueConfiguration]

    def __set__(self, cfg: CONFIG, value) -> None:

        if isinstance(value, self.default):
            value.cfg = cfg.cfg
            value.mesh = cfg.mesh
            self.update_subconfigurations_recursively(value)

            # Set subconfiguration in parent configuration after all subconfigurations are updated
            cfg.data[self.__name__] = value

        elif isinstance(value, str):

            if value == self.default.name:
                cfg.data[self.__name__] = self.default(cfg=cfg.cfg, mesh=cfg.mesh)
            else:
                msg = f""" Invalid type '{value}'!
                           Allowed option: '{self.default.name}!'
                """
                raise TypeError(msg)

        elif isinstance(value, dict):
            cfg.data[self.__name__].update(**value)

        else:
            msg = f""" Can not set variable of type {type(value)}!

            To set '{self.__name__}' pass either:
            1. An instance of type {self.default}.
            2. A keyword: {list(self.default.name)}.

            To update the current instance pass a dictionary with following keywords:
            {list(cfg.data[self.__name__].data.keys())}
            """
            raise TypeError(msg)

        if self.fset is not None:
            value = self.fset(cfg, cfg.data[self.__name__])

    def reset(self, cfg: CONFIG):
        self.__set__(cfg, self.default(cfg=cfg.cfg, mesh=cfg.mesh))

    def update_subconfigurations_recursively(self, parent: CONFIG):

        for cfg in parent.data.values():
            if isinstance(cfg, UniqueConfiguration):
                cfg.cfg = parent.cfg
                cfg.mesh = parent.mesh
                self.update_subconfigurations_recursively(cfg)


class interface(unique):

    default: type[InterfaceConfiguration]

    def __set__(self, cfg: CONFIG, value) -> None:

        if isinstance(value, self.default.tree.root):
            value.cfg = cfg.cfg
            value.mesh = cfg.mesh
            self.update_subconfigurations_recursively(value)

            # Set subconfiguration in parent configuration after all subconfigurations are updated
            cfg.data[self.__name__] = value

        elif isinstance(value, str):
            cfg_ = self.default.tree[value]
            if not isinstance(cfg.data[self.__name__], cfg_):
                cfg.data[self.__name__] = cfg_(cfg=cfg.cfg, mesh=cfg.mesh)

        elif isinstance(value, dict):
            cfg.data[self.__name__].update(**value)

        else:
            msg = f""" Can not set variable of type {type(value)}!

            To set '{self.__name__}' pass either:
            1. An instance of type {self.default.tree.root}.
            2. A keyword from available options: {list(self.default.tree)}.

            To update the current instance pass a dictionary with following keywords:
            {list(cfg.data[self.__name__].data.keys())}
            """
            raise TypeError(msg)

        if self.fset is not None:
            value = self.fset(cfg, cfg.data[self.__name__])


class InterfaceTree(typing.MutableMapping):

    def __init__(self, root: UniqueConfiguration):
        self.root = root
        self.leafs = {}

    def __getitem__(self, alias) -> UniqueConfiguration | InterfaceConfiguration:
        if isinstance(alias, self.root):
            return type(alias)
        elif isinstance(alias, type) and issubclass(alias, self.root):
            return alias
        elif alias in self:
            return self.leafs[alias]
        else:
            msg = f"Invalid type '{alias}' for class '{self.root}'.\n"
            msg += f"Allowed options: {list(self)}!"
            raise TypeError(msg)

    def __contains__(self, key: object) -> bool:
        if isinstance(key, self.root):
            return True
        elif isinstance(key, type) and issubclass(key, self.root):
            return True
        return key in self.leafs

    def __setitem__(self, key: str, value):
        self.leafs[key] = value

    def __delitem__(self, key):
        del self.leafs[key]

    def __iter__(self):
        return iter(self.leafs)

    def __len__(self):
        return len(self.leafs)


class UniqueConfiguration(typing.MutableMapping):

    name: str
    mesh: ngs.Mesh

    @classmethod
    def get_configurations(cls, type: configuration) -> list[configuration]:
        return [cfg for cfg in vars(cls).values() if isinstance(cfg, type)]

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

    def to_tree(self, data: dict | None = None, root: str = "") -> dict[str, typing.Any]:

        if data is None:
            data = {}

        for key in self:

            try:
                value = self[key]
            except Exception:
                continue

            if root:
                key = f"{root}.{key}"

            match value:

                case UniqueConfiguration():
                    data[key] = value.name
                    value.to_tree(data, key)
                case ngs.Parameter():
                    data[key] = value.Get()
                case pathlib.Path():
                    data[key] = str(value)
                case _:
                    data[key] = value

        return data

    def clear(self) -> None:
        self.data = {}
        for value in self.cfgs:
            value.reset(self)

    def __init_subclass__(cls) -> None:

        cfgs = cls.get_configurations(configuration)
        if hasattr(cls, "cfgs"):
            cfgs = cfgs + cls.cfgs
        cls.cfgs = cfgs

        if not hasattr(cls, "name"):
            cls.name = cls.__name__
        cls.name = cls.name.lower()

        super().__init_subclass__()

    def __init__(self, cfg=None, mesh=None, **kwargs):

        if cfg is None:
            cfg = self

        self.cfg = cfg
        self.mesh = mesh

        self.clear()
        self.update(**kwargs)

    def __setitem__(self, key: str, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def __delitem__(self, key) -> None:
        delattr(self, key)

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __contains__(self, key: object) -> bool:
        return key in self.data

    def __repr__(self) -> str:
        return "\n".join([f"{key}: {value}" for key, value in self.to_tree(root=self.name).items()])


class InterfaceConfiguration(UniqueConfiguration):

    name: str
    tree: InterfaceTree
    cfgs: list[configuration]
    aliases: tuple[str]

    def __init_subclass__(cls, is_interface: bool = False, skip: bool = False) -> None:

        if not skip:

            if is_interface:
                cls.tree = InterfaceTree(cls)
                cls.aliases = ()

            else:
                for name in [cls.name, *cls.aliases]:
                    name = name.lower()
                    if name in cls.tree:
                        raise ValueError(f"Concrete Class of type {cls.tree.root} with name {name} already included!")
                    cls.tree[name] = cls

        super().__init_subclass__()


CONFIG = typing.TypeVar('CONFIG', bound=UniqueConfiguration)
