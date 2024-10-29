from __future__ import annotations
import typing

from dream.config import MultipleConfiguration, any
from dream.compressible.config import CompressibleState

if typing.TYPE_CHECKING:
    from dream.solver import SolverConfiguration


class DynamicViscosity(MultipleConfiguration, is_interface=True):

    cfg: SolverConfiguration

    @property
    def is_inviscid(self) -> bool:
        return isinstance(self, Inviscid)

    def viscosity(self, U: CompressibleState):
        raise NotImplementedError()


class Inviscid(DynamicViscosity):

    name: str = "inviscid"
    aliases = ('euler', )

    def viscosity(self, U: CompressibleState):
        raise TypeError("Inviscid Setting! Dynamic Viscosity not defined!")


class Constant(DynamicViscosity):

    name: str = "constant"

    def viscosity(self, U: CompressibleState):
        return 1


class Sutherland(DynamicViscosity):

    name: str = "sutherland"

    @any(default=110.4)
    def measurement_temperature(self, value: float) -> float:
        return value

    @any(default=1.716e-5)
    def measurement_viscosity(self, value: float) -> float:
        return value

    def viscosity(self, U: CompressibleState):

        T = U.T

        if U.is_set(T):

            REF = self.cfg.pde.equations.get_reference_state()
            INF = self.cfg.pde.equations.get_farfield_state()

            T0 = self.measurement_temperature/REF.T

            return (T/INF.T)**(3/2) * (INF.T + T0)/(T + T0)

