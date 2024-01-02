from __future__ import annotations
import logging
import typing
import textwrap

from ngsolve import Parameter


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

        if isinstance(value, Parameter):
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


class BaseConfig:

    logger = logging.getLogger(__name__)
    formatter = Formatter()

    @classmethod
    def _get_enum(cls, value: str, enum, variable: str):
        try:
            value = enum(value)
        except ValueError:
            msg = f"'{str(value).capitalize()}' is not a valid {variable}. "
            msg += f"Possible alternatives: {[e.value for e in enum]}"
            raise ValueError(msg) from None
        return value

    @classmethod
    def _is_type(cls, _type, _class):
        if not hasattr(_class, "types"):
            raise ValueError(f"Class '{_class}' needs to have subclass dictionary 'types'")

        if not isinstance(_type, str) and _type in _class.types:
            types = list(_class.types.keys())
            raise TypeError(f"Type '{_type}' invalid for '{_class}'. Allowed types: {types}!")

        return _type

    @classmethod
    def _get_type(cls, _type, _class, **init_kwargs):

        if not isinstance(_type, _class):
            _type = cls._is_type(_type, _class)
            _type = _class.types[_type](**init_kwargs)
        else:
            for key, value in init_kwargs:
                setattr(_type, key, value)

        return _type

    def to_dict(self) -> dict[str, typing.Any]:
        return {key: value for key, value in self.items()}

    def update(self, cfg: dict[str, typing.Any]):
        cfg_ = vars(self)

        if isinstance(cfg, (dict, type(self))):
            for key, value in cfg.items():
                if key.startswith("_"):
                    key = key[1:]

                setattr(self, key, value)

                if key not in cfg_:
                    msg = f"Trying to set '{key}' attribute in {str(self)}. "
                    msg += "It is either deprecated or not supported"
                    self.logger.warning(msg)

        elif isinstance(cfg, BaseConfig):
            sub_cfg = [value for key, value in self.items() if isinstance(value, type(cfg))]

            for value in sub_cfg:
                value.update(cfg)

            if not sub_cfg:
                msg = f"Subconfiguration {type(cfg)} can not updated!"
                msg += "It is either not included or you have to set it manually!"
                self.logger.warning(msg)

        else:
            raise TypeError(f"Update requires dictionary or type '{str(self)}'")

    def items(self) -> tuple[str, typing.Any]:
        for key in vars(self):
            yield key, getattr(self, key)

    def format(self):
        raise NotImplementedError()

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return self.format()


class BonusIntegrationOrder(BaseConfig):

    def __init__(self) -> None:
        self.VOL = 0
        self.BND = 0
        self.BBND = 0
        self.BBBND = 0

    @property
    def VOL(self):
        return self._VOL

    @VOL.setter
    def VOL(self, value):
        self._VOL = int(value)

    @property
    def BND(self):
        return self._BND

    @BND.setter
    def BND(self, value):
        self._BND = int(value)

    @property
    def BBND(self):
        return self._BBND

    @BBND.setter
    def BBND(self, value):
        self._BBND = int(value)

    @property
    def BBBND(self):
        return self._BBBND

    @BBBND.setter
    def BBBND(self, value):
        self._BBBND = int(value)

    def format(self):
        formatter = self.formatter.new()
        formatter.entry('Bonus Integration Order', str({key[1:]: value for key, value in self.items()}))
        return formatter.output


class FiniteElementConfig(BaseConfig):

    def __init__(self) -> None:
        self.order = 2
        self.static_condensation = True
        self._bonus_int_order = BonusIntegrationOrder()

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
    def bonus_int_order(self) -> BonusIntegrationOrder:
        return self._bonus_int_order

    @bonus_int_order.setter
    def bonus_int_order(self, bonus_int_order: BonusIntegrationOrder):
        self._bonus_int_order.update(bonus_int_order)

    def format(self):
        formatter = self.formatter.new()
        formatter.subheader('Finite Element Configuration').newline()
        formatter.entry('Polynomial Order', self._order)
        formatter.entry('Static Condensation', str(self._static_condensation))
        formatter.add_config(self.bonus_int_order)
        return formatter.output
