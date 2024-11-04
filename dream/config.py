from __future__ import annotations
import logging
import typing
import functools

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


class quantity:
    """
    Variable is a descriptor that mimics a physical quantity.

    Since it ressembles a scalar, a vector or a matrix it is possible to pass a cast function
    to the constructor such that every time an attribute is set, the value gets cast to the appropriate
    tensor dimensions.
    """

    __slots__ = ('symbol', 'name')

    def __set_name__(self, owner, symbol: str):
        self.symbol = symbol

    def __init__(self, name: str) -> None:
        self.name = name

    def __get__(self, state: ngsdict, objtype) -> ngs.CF:
        if state is None:
            return self
        return state.get(self.name, None)

    def __set__(self, state: ngsdict, value) -> None:
        if value is not None:
            state[self.name] = value

    def __delete__(self, state: ngsdict):
        del state[self.name]


class ngsdict(typing.MutableMapping, dict):

    symbols: dict[str, str] = {}

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

    def to_py(self) -> dict:
        """ Returns the current state represented by pure python objects. 
        """
        mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=1))
        return {key: value(mesh()) if isinstance(value, ngs.CF) else value for key, value in self.items()}

    def __repr__(self):
        return str(self.to_py())


# ------- User configuration ------- #
class configuration:

    __slots__ = ('name', 'default', 'fset', 'fget', '__doc__')

    def __set_name__(self, owner, name: str):
        self.name = name

    def __init__(self, default, fset=None, fget=None, doc: str = None):
        self.default = default
        self.fset = fset
        self.fget = fget
        self.__doc__ = doc

        if doc is None and fset is not None:
            doc = fset.__doc__

        self.__doc__ = doc

    @typing.overload
    def __get__(self, instance: None, owner: type[object]) -> configuration:
        """ Called when an attribute is accessed via class not an instance """

    @typing.overload
    def __get__(self, instance: InterfaceConfiguration, owner: type[object]) -> typing.Any:
        """ Called when an attribute is accessed on an instance variable """

    def __get__(self, cfg: InterfaceConfiguration, owner: type[InterfaceConfiguration]):
        if cfg is None:
            return self

        if self.fget is not None:
            self.fget(cfg)

        return cfg.data[self.name]

    def __delete__(self, cfg: InterfaceConfiguration):
        del cfg.data[self.name]

    def getter_check(self, fget_: typing.Callable[[typing.Any, typing.Any], None]) -> configuration:
        prop = type(self)(self.default, self.fset, fget_, self.__doc__)
        return prop

    def setter_check(self, fset_: typing.Callable[[typing.Any, typing.Any], None]) -> configuration:
        prop = type(self)(self.default, fset_, self.fget, self.__doc__)
        return prop

    def __call__(self, fset_: typing.Callable[[typing.Any, typing.Any], None]) -> configuration:
        return self.setter_check(fset_)

    def __set__(self, cfg: InterfaceConfiguration, value):
        if self.fset is not None:
            value = self.fset(cfg, value)

        cfg.data[self.name] = value

    def reset(self, cfg: InterfaceConfiguration):
        self.__set__(cfg, self.default)


class parameter(configuration):

    def __set__(self, cfg: InterfaceConfiguration, value: float) -> None:

        if isinstance(value, ngs.Parameter):
            value = value.Get()

        if self.fset is not None:
            value = self.fset(cfg, value)

        cfg.data[self.name].Set(value)

    def reset(self, cfg: InterfaceConfiguration):

        value = self.default
        if self.fset is not None:
            value = self.fset(cfg, value)

        cfg.data[self.name] = ngs.Parameter(value)


class unique(configuration):

    default: UniqueConfiguration

    def __set__(self, cfg: InterfaceConfiguration, value: str | InterfaceConfiguration) -> None:

        if isinstance(value, self.default):
            cfg.data[self.name] = value

        elif isinstance(value, str):

            if value == self.default.name:
                cfg.data[self.name] = self.default(cfg=cfg.cfg)
            else:
                msg = f""" Invalid type '{value}'!
                           Allowed option: '{self.default.name}!'
                """
                raise TypeError(msg)

            if self.fset is not None:
                value = self.fset(cfg, cfg.data[self.name])

        elif isinstance(value, dict):
            cfg.data[self.name].update(**value)

        else:
            msg = f""" Can not set variable of type {type(value)}!

            To set '{self.name}' pass either:
            1. An instance of type {self.default}.
            2. A keyword: {list(self.default.name)}.

            To update the current instance pass a dictionary with following keywords:
            {list(cfg.data[self.name].data.keys())}
            """
            raise TypeError(msg)

    def reset(self, cfg: InterfaceConfiguration):
        self.__set__(cfg, self.default(cfg=cfg.cfg))


class interface(unique):

    default: InterfaceConfiguration

    def __set__(self, cfg: InterfaceConfiguration, value: str | InterfaceConfiguration) -> None:

        if isinstance(value, self.default.tree.root):
            cfg.data[self.name] = value

        elif isinstance(value, str):
            cfg_ = self.default.tree[value]
            if not isinstance(cfg.data[self.name], cfg_):
                cfg.data[self.name] = cfg_(cfg=cfg.cfg)

            if self.fset is not None:
                value = self.fset(cfg, cfg.data[self.name])

        elif isinstance(value, dict):
            cfg.data[self.name].update(**value)

        else:
            msg = f""" Can not set variable of type {type(value)}!

            To set '{self.name}' pass either:
            1. An instance of type {self.default.tree.root}.
            2. A keyword from available options: {list(self.default.tree)}.

            To update the current instance pass a dictionary with following keywords:
            {list(cfg.data[self.name].data.keys())}
            """
            raise TypeError(msg)


class InterfaceTree(typing.MutableMapping, dict):

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


class UniqueConfiguration(typing.MutableMapping, dict):

    name: str
    cfgs: list[configuration]

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

    def to_tree(self, data: dict = None, root: str = "") -> dict[str, typing.Any]:

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
                case _:
                    data[key] = value

        return data

    def clear(self) -> None:
        self.data = {}
        for value in self.cfgs:
            value.reset(self)

    def __init_subclass__(cls) -> None:

        if not hasattr(cls, "cfgs"):
            cls.cfgs = [cfg_ for cfg_ in vars(cls).values() if isinstance(cfg_, configuration)]

        if not hasattr(cls, "name"):
            cls.name = cls.__name__

        cls.name = cls.name.lower()

        super().__init_subclass__()

    def __init__(self, cfg: UniqueConfiguration = None, **kwargs):

        if cfg is None:
            cfg = self
        self.cfg = cfg

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

    def __init_subclass__(cls, is_interface: bool = False) -> None:

        if is_interface:
            cls.tree = InterfaceTree(cls)
            cls.aliases = ()
            cls.cfgs = [cfg for cfg in vars(cls).values() if isinstance(cfg, configuration)]

        else:
            for name in [cls.name, *cls.aliases]:
                name = name.lower()
                if name in cls.tree:
                    raise ValueError(f"Concrete Class of type {cls.tree.root} with name {name} already included!")
                cls.tree[name] = cls

            cls.cfgs = cls.cfgs + [cfg for cfg in vars(cls).values() if isinstance(cfg, configuration)]

        super().__init_subclass__()
