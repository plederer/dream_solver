from __future__ import annotations
import abc
import numpy as np
from typing import NamedTuple

from ngsolve import *
from ..utils import DreAmLogger

logger = DreAmLogger.get_logger("BufferRegion")


class BufferCoordinate(NamedTuple):

    start: float
    end: float
    coord: CF
    offset: float
    dim: int = 1

    @classmethod
    def x(cls, x0, xn, offset: float = 0.0):
        return cls(x0, xn, x, offset)

    @classmethod
    def y(cls, y0, yn, offset: float = 0.0):
        return cls(y0, yn, y, offset)

    @classmethod
    def z(cls, z0, zn, offset: float = 0.0):
        return cls(z0, zn, z, offset)

    @classmethod
    def xy(cls, xy0, xyn, offset: float = (0.0, 0.0)):
        return cls(xy0, xyn, (x, y), offset, dim=2)

    @classmethod
    def yz(cls, yz0, yzn, offset: float = (0.0, 0.0)):
        return cls(yz0, yzn, (y, z), offset, dim=2)

    @classmethod
    def zx(cls, zx0, zxn, offset: float = (0.0, 0.0)):
        return cls(zx0, zxn, (z, x), offset, dim=2)

    @classmethod
    def polar(cls,
              r0,
              rn,
              loc: tuple[float, float, float] = (0.0, 0.0, 0.0),
              axis: tuple[int, int, int] = (0, 0, 1)):

        loc = tuple(p for p, ax in zip(loc, axis) if ~bool(ax))
        coord = tuple(x_ for x_, ax in zip((x, y, z), axis) if ~bool(ax))

        r = sum([(x-x0)**2 for x, x0 in zip(coord, loc)])
        return cls(r0, rn, sqrt(r), loc, dim=1)

    @classmethod
    def spherical(cls,
                  r0,
                  rn,
                  loc: tuple[float, float, float] = (0.0, 0.0, 0.0)):

        r = sum([(a-a0)**2 for a, a0 in zip((x, y, z), loc)])
        return cls(r0, rn, sqrt(r), dim=1)

    @property
    def length(self):
        if self.dim == 1:
            return abs(self.end - self.start)
        else:
            return tuple([abs(end - start) for end, start in zip(self.end, self.start)])

    def get(self, mirror: bool = False) -> CF:
        if self.dim == 1:

            # Geometry Tolerance!
            # Important to avoid steep downfall from 1 to 0

            geo_tol = 1e-3
            x = (self.coord - self.start)/(self.end - self.start)
            x = IfPos(x, IfPos(x-(1+geo_tol), 0, x), 0)

            if mirror:
                x += self.mirror().get(mirror=False)

            return x

        elif self.dim == 2:

            x = BufferCoordinate(self.start[0], self.end[0], self.coord[0], self.offset[0]).get(mirror)
            y = BufferCoordinate(self.start[1], self.end[1], self.coord[1], self.offset[1]).get(mirror)

            return x + y - x*y

        else:
            raise NotImplementedError()

    def mirror(self) -> BufferCoordinate:

        if self.start < self.end:
            if self.offset <= self.start:
                new_start = self.start - 2*(self.start - self.offset)
                new_end = self.end - 2*(self.end - self.offset)
            else:
                raise ValueError("Offset has to be smaller than Start")

        elif self.start > self.end:
            if self.offset >= self.start:
                new_start = self.start + 2*(self.offset - self.start)
                new_end = self.end + 2*(self.offset - self.end)
            else:
                raise ValueError("Offset has to be bigger than Start")

        return BufferCoordinate(new_start, new_end, self.coord, self.offset)

    def __call__(self, mirror: bool = False) -> CF:
        return self.get(mirror)


class SpongeWeight:

    @staticmethod
    def target_damping(dB: float, sponge_length: float, Mach_number: float, function_integral: float):
        if not dB < 0:
            raise ValueError("Target Dezibel must be smaller zero!")
        return float(dB*(1-Mach_number**2)/(-40 * np.log10(np.exp(1))) * 1/(sponge_length * function_integral))

    @classmethod
    def constant(cls, sponge_length: float, Mach_number: float, dB: float = -40):
        return cls.target_damping(dB, sponge_length, Mach_number, 1)

    @classmethod
    def quadratic(cls, sponge_length: float, Mach_number: float, dB: float = -40):
        return cls.target_damping(dB, sponge_length, Mach_number, 1/3)

    @classmethod
    def cubic(cls, sponge_length: float, Mach_number: float, dB: float = -40):
        return cls.target_damping(dB, sponge_length, Mach_number, 1/4)

    @classmethod
    def penta(cls, sponge_length: float, Mach_number: float, dB: float = -40):
        return cls.target_damping(dB, sponge_length, Mach_number, 1/6)

    @classmethod
    def penta_smooth(cls, sponge_length: float, Mach_number: float, dB: float = -40):
        return cls.target_damping(dB, sponge_length, Mach_number, 0.5)


class SpongeFunction(NamedTuple):

    weight_function: CF
    order: int
    repr: str = u"\u03C3 f(x)"

    @classmethod
    def constant(cls, weight: float = 1):
        """ SpongeFunction: \u03C3 """
        return cls(CF(weight), 0, u"\u03C3")

    @classmethod
    def quadratic(cls, coord: BufferCoordinate, weight: float = 1, mirror: bool = False):
        """ SpongeFunction: \u03C3 x² """
        x = coord.get(mirror)
        func = weight * x**2
        order = 2 * coord.dim
        return cls(func, order, u"\u03C3 x²")

    @classmethod
    def cubic(cls, coord: BufferCoordinate, weight: float = 1, mirror: bool = False):
        """ SpongeFunction: \u03C3 x³ """
        x = coord.get(mirror)
        func = weight * x**3
        order = 3 * coord.dim
        return cls(func, order, u"\u03C3 x³")

    @classmethod
    def penta(cls, coord: BufferCoordinate, weight: float = 1, mirror: bool = False):
        """ SpongeFunction: \u03C3 x\u2075 """
        x = coord.get(mirror)
        func = weight * x**5
        order = 5 * coord.dim
        return cls(func, order, u"\u03C3 x\u2075")

    @classmethod
    def penta_smooth(cls, coord: BufferCoordinate, weight: float = 1, mirror: bool = False):
        """ SpongeFunction: \u03C3 (6x\u2075 - 15x\u2074 + 10x³) """
        x = coord.get(mirror)
        func = weight * (6*x**5 - 15 * x**4 + 10 * x**3)
        order = 5 * coord.dim
        return cls(func, order, u"\u03C3 (6x\u2075 - 15x\u2074 + 10x³)")

    def __repr__(self) -> str:
        return f"{self.repr}"


class GridDeformationFunction(NamedTuple):

    class _Mapping(abc.ABC):

        def __init__(self, coord: BufferCoordinate, mirror: bool = False, order: int = 1) -> None:

            if isinstance(coord, BufferCoordinate):
                if coord.dim != 1:
                    raise ValueError("Buffercoordinate needs to be 1-dimensional")

            self.coord = coord
            self.mirror = mirror
            self.order = order

            self._deformation = None

        @property
        def deformation(self):
            if self._deformation is None:
                self._deformation = self.get_deformation(self.mirror)
            return self._deformation

        @abc.abstractmethod
        def get_deformation(self, mirror: bool = False) -> CF: ...

        @abc.abstractmethod
        def mapping(self, x, mirror: bool = False) -> float: ...

        def deformed_length(self, coord: BufferCoordinate) -> float:
            start = self(coord.start)
            end = self(coord.end)
            return abs(end - start)

        @staticmethod
        def fixed_point_iteration(x0, func, iterations: int = 100, print_error: bool = False):
            for i in range(iterations):

                xn = func(x0)
                err = abs(xn - x0)

                if print_error:
                    logger.info(f"Fixpoint - It: {i:3d} - n+1: {xn:.5e} - n: {x0:.5e} - err: {err:.5e}")

                x0 = xn

                if err < 1e-16:
                    break
            return x0

        def __call__(self, x) -> float:
            return self.mapping(x)

    class Zero(_Mapping):

        def __init__(self, coord: BufferCoordinate) -> None:
            super().__init__(coord, False, order=0)

        def get_deformation(self, mirror: bool = False) -> CF:
            return 0

        def mapping(self, x) -> float:
            return x

        def __repr__(self) -> str:
            return "0"

    class Linear(_Mapping):

        def __init__(self, factor: float, coord: BufferCoordinate, mirror: bool = False) -> None:
            """ Linear mapping: factor * (x - x0) + x0 """

            if not factor >= 1:
                raise ValueError(f"Thickness has to be >= 1")

            self.factor = factor
            super().__init__(coord, mirror, order=1)

        def get_deformation(self, mirror: bool = False) -> CF:

            def one_sided(coord: BufferCoordinate):
                D = coord.end - coord.start
                x_ = coord.get()
                return D * x_ * (self.factor - 1)

            def_ = one_sided(self.coord)
            if mirror:
                def_ += one_sided(self.coord.mirror())

            return def_

        def mapping(self, x, mirror: bool = False) -> float:
            coord = self.coord
            if mirror:
                coord = self.coord.mirror()
            return self.factor * (x - coord.start) + coord.start

        def __repr__(self) -> str:
            return "a*x  - x"

    class ExponentialThickness(_Mapping):

        def __init__(self,
                     factor: int,
                     coord: BufferCoordinate,
                     mirror: bool = False,
                     order: int = 5) -> None:

            if not 1 < factor:
                raise ValueError(f"Choose factor > 1")

            self.factor = factor

            self._constants = None
            self._mirror_constants = None

            super().__init__(coord, mirror, order)

        @property
        def constants(self):
            if self._constants is None:
                self._constants = self._determine_deformation_constants(self.coord)
            return self._constants

        @property
        def mirror_constants(self):
            if self._mirror_constants is None:
                self._mirror_constants = self._determine_deformation_constants(self.coord.mirror())
            return self._mirror_constants

        def _determine_deformation_constants(self,
                                             coord: BufferCoordinate,
                                             iterations: int = 100,
                                             print_error: bool = False):
            D = coord.end - coord.start
            a = self.fixed_point_iteration(-D, lambda x: -D/(np.log(1 - self.factor*D/x)), iterations, print_error)
            c = -D/a
            return a, c

        def get_deformation(self, mirror: bool = False):

            def one_sided(coord: BufferCoordinate, constants):
                x_ = coord.get()
                D = coord.end - coord.start
                a, c = constants
                return a * (1 - exp(c*x_)) - D*x_

            def_ = one_sided(self.coord, self.constants)

            if mirror:
                def_ += one_sided(self.coord.mirror(), self.mirror_constants)

            return def_

        def mapping(self, x, mirror: bool = False) -> float:

            coord = self.coord
            a, c = self.constants
            if mirror:
                coord = self.coord.mirror()
                a, c = self.mirror_constants

            x_ = (x - coord.start)/(coord.end - coord.start)
            return float(a * (1 - np.exp(c * x_)) + coord.start)

        def __repr__(self) -> str:
            return "a exp(c x) - x"

    class ExponentialJacobian(_Mapping):

        def __init__(self,
                     endpoint_jacobian: float,
                     coord: BufferCoordinate,
                     mirror: bool = False,
                     order: int = 5) -> None:

            if not 0 < endpoint_jacobian < 1:
                raise ValueError(f"Endpoint Jacobian has to be 0 < p < 1")

            self.endpoint_jacobian = endpoint_jacobian

            self._constants = None
            self._mirror_constants = None

            super().__init__(coord, mirror, order)

        @property
        def constants(self):
            if self._constants is None:
                self._constants = self._determine_deformation_constants(self.coord)
            return self._constants

        @property
        def mirror_constants(self):
            if self._mirror_constants is None:
                self._mirror_constants = self._determine_deformation_constants(self.coord.mirror())
            return self._mirror_constants

        def _determine_deformation_constants(self, coord: BufferCoordinate):
            k = self.endpoint_jacobian/(self.endpoint_jacobian-1)
            D = coord.end - coord.start
            a = D * k
            c = -1/k
            return a, c

        def get_deformation(self, mirror: bool = False):

            def one_sided(coord: BufferCoordinate, constants):
                x_ = coord.get()
                D = coord.end - coord.start
                a, c = constants
                return a*(1 - exp(c*x_)) - D*x_

            def_ = one_sided(self.coord, self.constants)

            if mirror:
                def_ += one_sided(self.coord.mirror(), self.mirror_constants)

            return def_

        def mapping(self, x, mirror: bool = False) -> float:

            coord = self.coord
            a, c = self.constants
            if mirror:
                coord = self.coord.mirror()
                a, c = self.mirror_constants

            x_ = (x - coord.start)/(coord.end - coord.start)
            return float(a * (1 - np.exp(c * x_)) + coord.start)

        def __repr__(self) -> str:
            return "a exp(c x) - x"

    class TangensThickness(_Mapping):

        def __init__(self,
                     factor: int,
                     coord: BufferCoordinate,
                     mirror: bool = False,
                     order: int = 5) -> None:

            if not 1 < factor:
                raise ValueError(f"Choose factor > 1")

            self.factor = factor

            self._constants = None
            self._mirror_constants = None

            super().__init__(coord, mirror, order)

        @property
        def constants(self):
            if self._constants is None:
                self._constants = self._determine_deformation_constants(self.coord)
            return self._constants

        @property
        def mirror_constants(self):
            if self._mirror_constants is None:
                self._mirror_constants = self._determine_deformation_constants(self.coord.mirror())
            return self._mirror_constants

        def _determine_deformation_constants(self,
                                             coord: BufferCoordinate,
                                             iterations: int = 100,
                                             print_error: bool = False):
            D = coord.end - coord.start
            a = self.fixed_point_iteration(D, lambda x: D/np.arctan(self.factor * D/x), iterations, print_error)
            c = D/a
            return a, c

        def get_deformation(self, mirror: bool = False):

            def one_sided(coord: BufferCoordinate, constants):
                x_ = coord.get()
                D = coord.end - coord.start
                a, c = constants
                return a * tan(c * x_) - D*x_

            def_ = one_sided(self.coord, self.constants)

            if mirror:
                def_ += one_sided(self.coord.mirror(), self.mirror_constants)

            return def_

        def mapping(self, x, mirror: bool = False) -> float:

            coord = self.coord
            a, c = self.constants
            if mirror:
                coord = self.coord.mirror()
                a, c = self.mirror_constants

            x_ = (x - coord.start)/(coord.end - coord.start)
            return float(a * np.tan(c * x_) + coord.start)

        def __repr__(self) -> str:
            return "a tan(c x) - x"

    class TangensJacobian(_Mapping):

        def __init__(self,
                     endpoint_jacobian: float,
                     coord: BufferCoordinate,
                     mirror: bool = False,
                     order: int = 5) -> None:

            if not 0 < endpoint_jacobian < 1:
                raise ValueError(f"Endpoint Jacobian has to be 0 < p < 1")

            self.endpoint_jacobian = endpoint_jacobian

            self._constants = None
            self._mirror_constants = None

            super().__init__(coord, mirror, order)

        @property
        def constants(self):
            if self._constants is None:
                self._constants = self._determine_deformation_constants(self.coord)
            return self._constants

        @property
        def mirror_constants(self):
            if self._mirror_constants is None:
                self._mirror_constants = self._determine_deformation_constants(self.coord.mirror())
            return self._mirror_constants

        def _determine_deformation_constants(self, coord: BufferCoordinate):
            k = np.sqrt(self.endpoint_jacobian/(1-self.endpoint_jacobian))
            D = coord.end - coord.start
            a = D * k
            c = -1/k
            return a, c

        def get_deformation(self, mirror: bool = False):

            def one_sided(coord: BufferCoordinate, constants):
                x_ = coord.get()
                D = coord.end - coord.start
                a, c = constants
                return a * tan(c * x_) - D*x_

            def_ = one_sided(self.coord, self.constants)

            if mirror:
                def_ += one_sided(self.coord.mirror(), self.mirror_constants)

            return def_

        def mapping(self, x, mirror: bool = False) -> float:

            coord = self.coord
            a, c = self.constants
            if mirror:
                coord = self.coord.mirror()
                a, c = self.mirror_constants

            x_ = (x - coord.start)/(coord.end - coord.start)
            return float(a * np.tan(c * x_) + coord.start)

        def __repr__(self) -> str:
            return "a tan(c x) - x"

    x: _Mapping = Zero(x)
    y: _Mapping = Zero(y)
    z: _Mapping = Zero(z)
