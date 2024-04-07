from __future__ import annotations
import numpy as np
import ngsolve as ngs
import logging

from dream.config import MultipleConfiguration, UniqueConfiguration, parameter, any
from collections import UserDict


logger = logging.getLogger(__name__)


class Timer(UniqueConfiguration):

    @any(default=0.0)
    def start(self, start):
        return start

    @any(default=1.0)
    def end(self, end):
        return end

    @parameter(default=1e-4)
    def step(self, step):
        step = float(step)
        self._set_time_step_digit(step)
        return step

    @parameter(default=0.0)
    def t(self, t):
        return t

    def __call__(self, **kwargs):
        self.update(**kwargs)
        for t in self.to_iterator(step=1):
            self.t = t
            yield t
        
    def to_iterator(self, step: int = 1):
        for t in self.to_array()[::step]:
            yield t

    def to_array(self, include_start: bool = False) -> np.ndarray:
        num = round((self.end - self.start)/self.step.Get()) + 1

        interval = np.linspace(self.start, self.end, num)
        if not include_start:
            interval = interval[1:]

        return interval.round(self._step_digit)

    def _set_time_step_digit(self, step: float):

        if isinstance(step, ngs.Parameter):
            step = step.Get()

        digit = f"{step:.16f}".split(".")[1]
        self._step_digit = len(digit.rstrip("0"))


class SimulationConfig(MultipleConfiguration, is_interface=True):

    @property
    def is_stationary(self) -> bool:
        return isinstance(self, StationaryConfig)


class StationaryConfig(SimulationConfig):

    label: str = "stationary"

    def format(self):
        formatter = self.formatter.new()
        formatter.subheader('Stationary Configuration').newline()
        return formatter.output


class TransientConfig(SimulationConfig):

    label: str = "transient"

    def __init__(self):
        self._scheme = "IE"
        self._timer = Timer()

    @property
    def timer(self) -> Timer:
        return self._timer

    @timer.setter
    def timer(self, timer: Timer):
        self._timer.set(timer.start, timer.end, timer.step)

    @property
    def scheme(self) -> TimeSchemes:
        return self._scheme

    @scheme.setter
    def scheme(self, scheme: str):
        self._scheme = self._get_type(scheme, TimeSchemes, self)

    def format(self):
        formatter = self.formatter.new()
        formatter.subheader('Transient Configuration').newline()
        formatter.entry('Time Scheme', self.scheme)
        formatter.entry('Timer', repr(self.timer))
        return formatter.output


class PseudoTimeSteppingConfig(TransientConfig):

    label: str = "pseudo"

    def __init__(self):
        super().__init__()
        self.max_step = 1

    @property
    def max_step(self) -> float:
        return self._max_step

    @max_step.setter
    def max_step(self, max_step: float):
        self._max_step = float(max_step)

    def format(self) -> str:
        formatter = self.formatter.new()
        formatter.subheader('Pseudo Time Stepping Configuration').newline()
        formatter.entry('Time Scheme', self.scheme)
        formatter.entry('Timer', repr(self.timer))
        formatter.entry('Max Time Step', self.max_step)
        return formatter.output


class TransientGridfunction(UserDict):

    def swap_level(self, gfu: ngs.GridFunction, level='n+1') -> TransientGridfunction:
        swap = self.copy()
        swap[level] = gfu
        return swap

    def update_time_step(self):
        gfus = list(self.values()).reverse()
        for new, old in zip(gfus[1:], gfus[:-1]):
            old.vec.data = new.vec

    def update_initial(self):
        gfus = list(self.values())
        new = gfus[0]
        for old in gfus[1:]:
            old.vec.data = new.vec


class TimeSchemes:

    types: dict[str, TimeSchemes] = {}
    time_levels: tuple[str]
    is_implicit: bool

    @classmethod
    def get_transient_gridfunction(cls, gfu: ngs.GridFunction) -> TransientGridfunction:
        gfus = [gfu] + [ngs.GridFunction(gfu.space, name=f"{gfu.name}_{label}") for label in cls.time_levels[1:]]
        return TransientGridfunction({level: gfu for level, gfu in zip(cls.time_levels, gfus)})

    def __init_subclass__(cls, labels: str) -> None:
        if isinstance(labels, str):
            labels = [labels]

        for label in labels:
            cls.types[label] = cls

    def __init__(self, cfg: TransientConfig = None) -> None:
        if cfg is None:
            cfg = TransientConfig()
        self.cfg = cfg

    def scheme(self, gfu: TransientGridfunction) -> ngs.CF:
        return self.nominator(gfu)/self.denominator()

    def nominator(self, gfu: TransientGridfunction) -> ngs.CF:
        raise NotImplementedError()

    def denominator(self):
        raise NotImplementedError()


class ImplicitEuler(TimeSchemes, labels=["IE", "implicit_euler"]):

    time_levels = ('n+1', 'n')
    is_implicit = True

    def nominator(self, gfu: TransientGridfunction) -> ngs.CF:
        return gfu['n+1'] - gfu['n']

    def denominator(self) -> ngs.CF:
        return self.cfg.timer.step


class BDF2(TimeSchemes, labels=["BDF2"]):

    time_levels = ('n+1', 'n', 'n-1')
    is_implicit = True

    def nominator(self, gfu: TransientGridfunction) -> ngs.CF:
        return 3*gfu['n+1'] - 4 * gfu['n'] + gfu['n-1']

    def denominator(self) -> ngs.CF:
        return 2*self.cfg.timer.step

# %%
