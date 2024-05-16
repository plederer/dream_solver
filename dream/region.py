from __future__ import annotations
from ngsolve import *
from collections import UserDict
from typing import Any, NamedTuple
import abc
import numpy as np

from .state import State

import logging
logger = logging.getLogger("DreAm.Regions")


class DreamMesh:

    def __init__(self, mesh: Mesh) -> None:

        self._mesh = mesh
        self._bcs = BoundaryConditions(self.mesh.GetBoundaries())
        self._dcs = DomainConditions(self.mesh.GetMaterials())
        self._is_periodic = bool(mesh.GetPeriodicNodePairs(VERTEX))

    @property
    def mesh(self) -> Mesh:
        return self._mesh

    @property
    def dim(self) -> int:
        return self.mesh.dim

    @property
    def bcs(self) -> BoundaryConditions:
        return self._bcs

    @property
    def dcs(self) -> DomainConditions:
        return self._dcs

    @property
    def boundary_names(self) -> tuple[str]:
        return tuple(self.bcs)

    @property
    def domain_names(self) -> tuple[str]:
        return tuple(self.dcs)

    @property
    def is_grid_deformation(self) -> bool:
        return bool(self.dcs.grid_deformation)

    @property
    def is_periodic(self) -> bool:
        return self._is_periodic

    @property
    def highest_order_psponge(self) -> int:
        return max([sponge.order.high for sponge in self.dcs.psponge_layers.values()], default=0)

    def boundary(self, region: str) -> Region:
        return self.mesh.Boundaries(region)

    def domain(self, region: str) -> Region:
        return self.mesh.Materials(region)

    def pattern(self, sequence: list) -> str:
        return "|".join(sequence)

    def set_grid_deformation(self):
        grid = self.get_grid_deformation_function()
        self.mesh.SetDeformation(grid)

    def get_grid_deformation_function(self) -> GridFunction:
        return self._get_buffer_grid_function(self.dcs.GridDeformation)

    def get_sponge_weight_function(self) -> GridFunction:
        return self._get_buffer_grid_function(self.dcs.SpongeLayer)

    def get_psponge_weight_function(self) -> GridFunction:
        return self._get_buffer_grid_function(self.dcs.PSpongeLayer)

    def _get_buffer_grid_function(self, type) -> GridFunction:

        fes = type.fes(self.mesh, order=type.fes_order)
        if type is self.dcs.GridDeformation and self.is_periodic:
            fes = Periodic(fes)

        u, v = fes.TnT()
        buffer = GridFunction(fes)

        domains = self.dcs._get_condition(type)

        if domains:

            blf = BilinearForm(fes)
            blf += InnerProduct(u, v) * dx

            lf = LinearForm(fes)
            for domain, bc in domains.items():

                domain = self.domain(domain)

                if isinstance(bc, self.dcs.GridDeformation):
                    lf += InnerProduct(bc.deformation_function(self.dim),
                                       v) * dx(definedon=domain, bonus_intorder=bc.bonus_int_order)
                else:
                    lf += bc.weight_function * v * dx(definedon=domain, bonus_intorder=bc.bonus_int_order)

            blf.Assemble()
            lf.Assemble()

            buffer.vec.data = blf.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * lf.vec

        else:
            buffer.vec[:] = 0
            logger.warning(f"{type.__name__} has not been set in domain conditions! Returning zero GridFunction.")

        return buffer


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


class Condition:

    def __init__(self, state: State = None) -> None:
        self.state = state

    def _check_value(self, val, name: str):
        if val is None:
            raise ValueError(f"{name.capitalize()} can not be {None}")

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return repr(self.state)


class DomainConditions(UserDict):

    def __init__(self, domains) -> None:
        super().__init__({domain: {bc: None for bc in self._Domain.conditions} for domain in set(domains)})

    @property
    def pattern(self) -> str:
        return "|".join(self)

    @property
    def force(self) -> dict[str, Force]:
        return self._get_condition(self.Force)

    @property
    def initial_conditions(self) -> dict[str, Initial]:
        condition = self._get_condition(self.Initial)
        for domain in set(self).difference(set(condition)):
            logger.warn(f"Initial condition for '{domain}' has not been set!")
        return condition

    @property
    def sponge_layers(self) -> dict[str, SpongeLayer]:
        return self._get_condition(self.SpongeLayer)

    @property
    def psponge_layers(self) -> dict[str, PSpongeLayer]:
        psponge = self._get_condition(self.PSpongeLayer)
        psponge = dict(sorted(psponge.items(), key=lambda x: x[1].order.high))
        return psponge

    @property
    def grid_deformation(self) -> dict[str, GridDeformation]:
        return self._get_condition(self.GridDeformation)

    @property
    def perturbations(self) -> dict[str, Perturbation]:
        return self._get_condition(self.Perturbation)

    @property
    def pmls(self) -> dict[str, PML]:
        return self._get_condition(self.PML)

    class _Domain(Condition):

        conditions: list = []

        def __init_subclass__(cls) -> None:
            cls.conditions.append(cls)
            return super().__init_subclass__()

    class Force(_Domain):
        def __init__(self, state: State = None):
            super().__init__(state)

    class Initial(_Domain):
        def __init__(self, state: State):
            self._check_value(state.density, "density")
            self._check_value(state.velocity, "velocity")
            if state.all_thermodynamic_none:
                raise ValueError("A Thermodynamic quantity is required!")
            super().__init__(state)

    class SpongeLayer(_Domain):

        fes: FESpace = L2
        fes_order: int = 0

        def __init__(self,
                     state: State,
                     *sponges: SpongeFunction) -> None:

            self._check_value(state.density, "density")
            self._check_value(state.velocity, "velocity")
            if state.all_thermodynamic_none:
                raise ValueError("A Thermodynamic quantity is required!")
            super().__init__(state)

            self.sponges = sponges

        @property
        def bonus_int_order(self):
            return max([sponge.order for sponge in self.sponges])

        @property
        def weight_function(self):
            return sum([sponge.weight_function for sponge in self.sponges])

    class PSpongeLayer(_Domain):

        class SpongeOrder(NamedTuple):
            high: int
            low: int

        fes: FESpace = L2
        fes_order: int = 0

        def __init__(self,
                     high_order: int,
                     low_order: int,
                     *sponges: SpongeFunction,
                     state: State = None) -> None:

            if high_order < 0 or low_order < 0:
                raise ValueError("Negative polynomial order!")

            if high_order == low_order:
                if state is None:
                    raise ValueError("For equal order polynomials a state is required")
                else:
                    self._check_value(state.density, "density")
                    self._check_value(state.velocity, "velocity")
                    if state.all_thermodynamic_none:
                        raise ValueError("A Thermodynamic quantity is required!")
            elif not high_order > low_order:
                raise ValueError("Low Order must be smaller than High Order")

            super().__init__(state)

            self.order = self.SpongeOrder(int(high_order), int(low_order))
            self.sponges = sponges

        @property
        def is_equal_order(self) -> bool:
            return self.order.high == self.order.low

        @property
        def bonus_int_order(self) -> int:
            return max([sponge.order for sponge in self.sponges])

        @property
        def weight_function(self) -> CF:
            return sum([sponge.weight_function for sponge in self.sponges])

        @classmethod
        def range(cls, highest, lowest: int = 0, step: int = 1) -> tuple[SpongeOrder, ...]:
            range = np.arange(highest, lowest - 2*step, -step)
            range[range < lowest] = lowest
            return tuple(cls.SpongeOrder(int(high), int(low)) for high, low in zip(range[:-1], range[1:]))

        def __repr__(self) -> str:
            return f"(High: {self.order.high}, Low: {self.order.low}, State: {self.state})"

    class GridDeformation(_Domain):

        fes: FESpace = VectorH1
        fes_order: int = 1

        def __init__(self, mapping: GridDeformationFunction) -> None:
            if not isinstance(mapping, GridDeformationFunction):
                raise TypeError()
            self.mapping = mapping

        @property
        def bonus_int_order(self) -> int:
            return max([x.order for x in self.mapping])

        def deformation_function(self, dim: int) -> CF:
            deformation = tuple(map.deformation for map in self.mapping)
            return CF(deformation[:dim])

        def __repr__(self) -> str:
            return repr(self.mapping)

    class Perturbation(_Domain):
        ...

    class PML(_Domain):
        ...

    def set(self, condition: _Domain, domain: str = None):
        if not isinstance(condition, self._Domain):
            raise TypeError(f"Domain Condition must be instance of {self._Domain}")

        if domain is None:
            domains = tuple(self)
        elif isinstance(domain, str):
            domain = domain.split("|")

            domains = set(self).intersection(domain)
            missed = set(domain).difference(domains)

            for miss in missed:
                logger.warning(f"Domain {miss} does not exist! {condition} can not be set!")
        else:
            domains = tuple(domain)

        for domain in domains:
            self[domain][type(condition)] = condition

    def set_from_dict(self, other: dict[str, _Domain]):
        for key, bc in other.items():
            self.set(bc, key)

    def _get_condition(self, type: _Domain) -> dict[str, _Domain]:
        return {domain: bc[type] for domain, bc in self.items() if isinstance(bc[type], type)}


class BoundaryConditions(UserDict):

    def __init__(self, boundaries) -> None:
        super().__init__({boundary: None for boundary in set(boundaries)})

    @property
    def pattern(self) -> str:
        keys = [key for key, bc in self.items() if isinstance(bc, self._Boundary) and not isinstance(bc, self.Periodic)]
        return "|".join(keys)

    @property
    def nscbc(self) -> dict[str, NSCBC]:
        return {name: bc for name, bc in self.items() if isinstance(bc, self.NSCBC)}

    class _Boundary(Condition):
        glue: bool = False

    class GFarField(_Boundary):
        def __init__(self, state: State, sigma:float = 1, pressure_relaxation: bool = False, tangential_flux: bool = False, glue: bool = False):
            self._check_value(state.density, "density")
            self._check_value(state.velocity, "velocity")
            if state.all_thermodynamic_none:
                raise ValueError("A Thermodynamic quantity is required!")
            super().__init__(state)
            self.sigma = sigma
            self.pressure_relaxation = pressure_relaxation
            self.tangential_flux = tangential_flux
            self.glue = glue

    class FarField(_Boundary):
        def __init__(self, state: State, theta_0: float = 0, Qform: bool = False):
            self._check_value(state.density, "density")
            self._check_value(state.velocity, "velocity")
            if state.all_thermodynamic_none:
                raise ValueError("A Thermodynamic quantity is required!")
            super().__init__(state)

            self.theta_0 = theta_0
            self.Qform = Qform

    class Outflow(_Boundary):
        def __init__(self, pressure: float):
            state = State(pressure=pressure)
            self._check_value(state.pressure, "pressure")
            super().__init__(state)

    class NSCBC(_Boundary):

        def __init__(self,
                     state: State,
                     type: str = "poinsot",
                     sigmas: State = State(pressure=0.278, velocity=0.278, temperature=0.278),
                     reference_length: float = 1,
                     tangential_flux: bool = True,
                     glue: bool = False) -> None:

            self._check_value(state.density, "density")
            self._check_value(state.velocity, "velocity")
            if state.all_thermodynamic_none:
                raise ValueError("A Thermodynamic quantity is required!")
            super().__init__(state)

            self.type = type
            self.sigmas = sigmas
            self.reference_length = reference_length
            self.tangential_flux = tangential_flux
            self.glue = glue

    class InviscidWall(_Boundary):
        def __init__(self) -> None:
            super().__init__()

    class Symmetry(_Boundary):
        def __init__(self) -> None:
            super().__init__()

    class IsothermalWall(_Boundary):
        def __init__(self, temperature: CF) -> None:
            state = State(temperature=temperature)
            self._check_value(state.temperature, "temperature")
            super().__init__(state)

    class AdiabaticWall(_Boundary):
        def __init__(self) -> None:
            super().__init__()

    class Periodic(_Boundary):
        def __init__(self) -> None:
            super().__init__()

    class Custom(_Boundary):
        def __init__(self, state: State) -> None:
            super().__init__(state)

    def set(self, condition: _Boundary, boundary: str):
        if not isinstance(condition, self._Boundary):
            raise TypeError(f"Boundary Condition must be instance of {self._Boundary}")

        if isinstance(boundary, str):
            boundary = boundary.split("|")

        boundaries = set(self).intersection(boundary)
        missed = set(boundary).difference(boundaries)

        for miss in missed:
            logger.warning(f"Boundary {miss} does not exist! {condition} can not be set!")

        for key in boundaries:
            self[key] = condition

    def set_from_dict(self, other: dict[str, _Boundary]):
        for key, bc in other.items():
            self.set(bc, key)


if __name__ == "__main__":
    ...
