from __future__ import annotations
import logging
from collections import UserDict
from typing import Callable, Sequence

import ngsolve as ngs

from dream import bla
from dream.state import State

logger = logging.getLogger(__name__)


def pattern(sequence: Sequence) -> str:
    if isinstance(sequence, str):
        return sequence
    return "|".join(sequence)


class DreamMesh:

    def __init__(self, mesh: ngs.Mesh) -> None:
        self._mesh = mesh
        self._bcs = BoundaryConditions(mesh.GetBoundaries())
        self._dcs = DomainConditions(mesh.GetMaterials())
        self._is_periodic = bool(mesh.GetPeriodicNodePairs(ngs.VERTEX))

    @property
    def ngsmesh(self) -> ngs.Mesh:
        return self._mesh

    @property
    def dim(self) -> int:
        return self.ngsmesh.dim

    @property
    def bcs(self) -> BoundaryConditions:
        return self._bcs

    @property
    def dcs(self) -> DomainConditions:
        return self._dcs

    @property
    def boundaries(self) -> tuple[str]:
        return self.bcs.regions

    @property
    def domains(self) -> tuple[str]:
        return self.dcs.regions

    @property
    def is_periodic(self) -> bool:
        return self._is_periodic

    def boundary(self, boundary: Sequence) -> ngs.Region:
        boundary = pattern(boundary)
        return self.ngsmesh.Boundaries(boundary)

    def domain(self, domain: Sequence) -> ngs.Region:
        domain = pattern(domain)
        return self.ngsmesh.Materials(domain)

    def get_grid_deformation(self, grid_deformation: dict[str, GridDeformation] = None):
        if grid_deformation is None:
            grid_deformation = self.dcs.grid_deformation
        return self._buffer_function_(grid_deformation, GridDeformation)

    def get_sponge_function(self, sponge_layers: dict[str, SpongeLayer] = None):
        if sponge_layers is None:
            sponge_layers = self.dcs.sponge_layers
        return self._buffer_function_(sponge_layers, SpongeLayer)

    def get_psponge_function(self, psponge_layers:  dict[str, PSpongeLayer] = None):
        if psponge_layers is None:
            psponge_layers = self.dcs.psponge_layers
        return self._buffer_function_(psponge_layers, PSpongeLayer)

    def _buffer_function_(self, domains: dict[str, Buffer] = None, buffer_type: Buffer = None):

        if not domains:
            logger.warning(f"{buffer_type} has not been set in domain conditions!")
            return None

        fes = buffer_type.space(domains, self)
        u, v = fes.TnT()

        blf = ngs.BilinearForm(fes)
        blf += u * v * ngs.dx

        lf = ngs.LinearForm(fes)
        for region, bc in domains.items():

            if not isinstance(bc, buffer_type):
                raise TypeError(f"Only domains of type '{buffer_type}' allowed!")

            lf += bc.buffer_function * v * ngs.dx(definedon=self.domain(region))

        gfu = ngs.GridFunction(fes, name=buffer_type.__name__)
        blf.Assemble()
        lf.Assemble()

        gfu.vec.data = blf.mat.Inverse(inverse="sparsecholesky") * lf.vec

        return gfu


class BufferCoord(ngs.CF):
    r""" One-dimensional coordinate used in buffer layers.

        The physical coordinate is truncated at the starting and end point.

        If the coordinate system underwent a translation, it can be set by the shift parameter.
        This is useful e.g. when dealing with radial or spherical coordinates.

        .. math::
            \tilde{x} = \begin{cases} 0 & x <= x_0 \\ \frac{x - x_0}{x_n - x_0} & x_0 < x < x_n \\ 1 & x > x_n \end{cases}

    """

    @classmethod
    def x(cls, x0: float, xn: float, shift: float = 0):
        return cls(ngs.x - shift, x0, xn, shift)

    @classmethod
    def y(cls, y0: float, yn: float, shift: float = 0):
        return cls(ngs.y - shift, y0, yn, shift)

    @classmethod
    def z(cls, z0: float, zn: float, shift: float = 0):
        return cls(ngs.z - shift, z0, zn, shift)

    @classmethod
    def polar(cls, r0: float, rn: float, shift: tuple = (0, 0)):
        r = ngs.sqrt(sum([(x - x0)**2 for x, x0 in zip([ngs.x, ngs.y], shift)]))
        return cls(r, r0, rn, shift)

    @classmethod
    def spherical(cls, r0: float, rn: float, shift: tuple = (0, 0, 0)):
        r = ngs.sqrt(sum([(x - x0)**2 for x, x0 in zip([ngs.x, ngs.y, ngs.z], shift)]))
        return cls(r, r0, rn, shift)

    def __init__(self, x: ngs.CF, x0: bla.SCALAR, xn: bla.SCALAR, shift: tuple[float]):
        self._x0 = x0
        self._xn = xn
        self._shift = shift
        super().__init__(bla.interval((x - x0)/(xn - x0), 0, 1) * (xn - x0) + x0)

    @property
    def x0(self):
        return self._x0

    @property
    def xn(self):
        return self._xn

    @property
    def shift(self):
        return self._shift

    @property
    def length(self):
        return bla.abs(self.xn - self.x0)

    def get_normalised_coordinate(self) -> ngs.CF:
        return (self - self.x0)/(self.xn - self.x0)


class SpongeFunction(ngs.CF):
    """ Defines some predefined sponge function in a buffer layer
    """

    @classmethod
    def constant(cls, weight: float):
        return cls(weight)

    @classmethod
    def polynomial(cls, weight: float, x: BufferCoord, order: int = 3):
        if not isinstance(x, BufferCoord):
            raise TypeError(f"Coordinate has to be of type '{BufferCoord}'")

        x = x.get_normalised_coordinate()
        return cls(weight * x**order)

    @classmethod
    def penta(cls, weight: float, x: BufferCoord):
        if not isinstance(x, BufferCoord):
            raise TypeError(f"Coordinate has to be of type '{BufferCoord}'")

        x = x.get_normalised_coordinate()
        return cls(weight * (6*x**5 - 15 * x**4 + 10 * x**3))


class GridMapping:
    """ Mapping used for mesh deformation purposes.

        One-dimensional buffer coordinates are mapped from computational to physical buffer coordinates.
    """

    @classmethod
    def none(cls, coordinate: ngs.CF):
        """ Returns a zero grid mapping.

            This is mainly used as consistency between mappings in different coordinates.
            It can be seen as the python equivalent None.
        """
        def map(x_): return x_
        return cls(coordinate, map)

    @classmethod
    def linear(cls, scale: float, coordinate: BufferCoord):
        """ Returns a linear grid mapping. 

        The thickness of the buffer layer is scaled by the factor 'scale'.

        .. math::
            f(x) = scale * (x - x_0) + x_0
        """
        if not isinstance(coordinate, BufferCoord):
            raise TypeError(f"Coordinate has to be of type '{BufferCoord}'")

        if not scale > 1:
            raise ValueError(f"Buffer scale has to be greater 1, otherwise the mapping is meaningless!")

        start = coordinate.x0
        def map(x_): return scale * (x_ - start) + start

        return cls(coordinate, map)

    @classmethod
    def exponential(cls, scale: float, coordinate: BufferCoord):
        """ Returns an exponential grid mapping. 

        The thickness of the buffer layer is scaled by the factor 'scale'.

        The constants c_0 and c_1 are determined by a fixpoint iteration.

        .. math::
            f(x) = c_0 * (1 - \exp^{c_1 (x - x_0)}) + x_0
        """
        if not isinstance(coordinate, BufferCoord):
            raise TypeError(f"Coordinate has to be of type '{BufferCoord}'")

        if not scale > 1:
            raise ValueError(f"Buffer scale has to be greater 1, otherwise the mapping is meaningless!")

        start = coordinate.x0
        L = coordinate.xn - coordinate.x0

        c1 = bla.fixpoint(L, lambda c: ngs.log(1 + scale * L * c)/L, it=100, tol=1e-16)
        c0 = -1/c1

        def map(x_): return c0 * (1 - ngs.exp(c1 * (x_ - start))) + start

        return cls(coordinate, map)

    @classmethod
    def tangential(cls, scale: float, coordinate: BufferCoord):
        """ Returns a tangential grid mapping. 

        The thickness of the buffer layer is scaled by the factor 'scale'.

        The constants c_0 and c_1 are determined by a fixpoint iteration.

        .. math::
            f(x) = c_0 * (\tan{c_1 (x - x_0)}) + x_0
        """
        if not isinstance(coordinate, BufferCoord):
            raise TypeError(f"Coordinate has to be of type '{BufferCoord}'")

        if not scale > 1:
            raise ValueError(f"Buffer scale has to be greater 1, otherwise the mapping is meaningless!")

        start = coordinate.x0
        L = coordinate.xn - coordinate.x0

        c1 = bla.fixpoint(L, lambda c: ngs.atan(scale * L * c)/L, it=100, tol=1e-16)
        c0 = 1/c1

        def map(x_): return c0 * ngs.tan(c1 * (x_ - start)) + start

        return cls(coordinate, map)

    def __init__(self,
                 x: ngs.CF,
                 map: Callable[[bla.SCALAR], bla.SCALAR]) -> None:

        self.x = x
        self.map = map

    def polar_to_cartesian(self) -> tuple[GridMapping, GridMapping]:
        """ Automates the transformation from a polar mapping to a cartesian one """
        r = self.x
        x = ngs.x - r.shift[0]
        y = ngs.y - r.shift[1]

        cos_theta = x/ngs.sqrt(x**2 + y**2)
        sin_theta = y/ngs.sqrt(x**2 + y**2)

        def map_x(x_): return self(r) * cos_theta
        def map_y(y_): return self(r) * sin_theta

        x = BufferCoord(r * cos_theta, r.x0 * cos_theta, r.xn * cos_theta, r.shift[0])
        y = BufferCoord(r * sin_theta, r.x0 * sin_theta, r.xn * sin_theta, r.shift[1])

        mapping_x = GridMapping(x, map_x)
        mapping_y = GridMapping(y, map_y)

        return mapping_x, mapping_y

    @property
    def deformation(self) -> bla.SCALAR:
        return self(self.x) - self.x

    @property
    def length(self) -> float:
        return bla.abs(self(self.x.xn) - self(self.x.x0))

    def __call__(self, x_: bla.SCALAR) -> bla.SCALAR:
        return self.map(x_)


class Condition:

    def __init__(self, state: State = None) -> None:
        if state is None:
            state = State()
        self.state = state

    def __str__(self) -> str:
        return self.__class__.__name__

    def __hash__(self) -> int:
        return id(self)


class ConditionDescriptor:

    __slots__ = ("condition", "log")

    def __init__(self, condition: Condition, log: bool = False) -> None:
        self.condition = condition
        self.log = log

    def __get__(self, dict: ConditionContainer, objtype) -> Condition:
        value = dict.get(self.condition)

        if self.log:
            for domain in set(dict.regions).difference(value):
                logger.warning(f"{self.condition.__name__} condition for '{domain}' has not been set!")

        return value


class ConditionContainer(UserDict):

    _condition: Condition

    def __init__(self, regions) -> None:
        self.regions = tuple(dict.fromkeys(regions))
        super().__init__({})

    def set(self, condition, regions):

        if not isinstance(condition, self._condition):
            raise TypeError(f"{self._condition.__name__} condition must be instance of '{self._condition}'")

        regions = self._filter_region(regions)

        if regions:

            conditions = self.get(condition)

            if not conditions:
                self[condition.__class__.__name__] = conditions

            for region in regions:
                conditions[region] = condition

    def get(self, condition) -> dict[str, Condition]:
        if isinstance(condition, type) and issubclass(condition, self._condition):
            key = condition.__name__
        elif isinstance(condition, self._condition):
            key = condition.__class__.__name__
        else:
            raise TypeError(f"Condition not of type '{self._condition}'")
        return super().get(key, {})

    def set_from_dict(self, other: dict[str, Condition]):
        for key, bc in other.items():
            self.set(bc, key)

    def to_unique_pattern(self):
        regions = type(self)(self.regions)

        for label, condition in self.items():
            conditions = set(condition.values())
            data = {pattern([key for key, value in condition.items() if value == a]): a for a in conditions}
            regions[label] = data

        return regions

    def _filter_region(self, region) -> tuple[str]:

        if isinstance(region, str):
            region = region.split("|")

        regions = tuple(r for r in region if r in self.regions)

        for miss in set(region).difference(regions):
            logger.warning(f"{self._condition.__name__} {miss} does not exist! Condition can not be set!")

        return regions

    def __repr__(self):
        return repr({key: pattern(val) for key, val in self.items()})


class Domain(Condition):
    ...


class Initial(Domain):
    ...


class Perturbation(Domain):
    ...


class Force(Domain):
    ...


class Buffer(Domain):

    def __init__(self, buffer_function: ngs.CF, state: State = None, order: int = 0) -> None:
        self.buffer_function = buffer_function
        self.order = int(order)
        super().__init__(state)

    @staticmethod
    def space(domains: dict[str, Buffer], dmesh: DreamMesh) -> ngs.FESpace:
        raise NotImplementedError()


class GridDeformation(Buffer):

    @staticmethod
    def _check_mapping(map: GridMapping, coordinate: ngs.CF) -> GridMapping:
        if map is None:
            map = GridMapping.none(coordinate)

        if not isinstance(map, GridMapping):
            raise TypeError(f"Map has to be of type '{GridMapping}'")

        return map

    @staticmethod
    def space(domains: dict[str, GridDeformation], dmesh: DreamMesh) -> ngs.VectorH1:
        orders = [dc.order for dc in domains.values()]

        fes = ngs.VectorH1(dmesh.ngsmesh, order=max(orders))
        fes = ngs.Compress(fes, active_dofs=fes.GetDofs(dmesh.domain(domains)))

        if dmesh.is_periodic:
            fes = ngs.Periodic(fes)

        return fes

    def __init__(self, x: GridMapping = None, y: GridMapping = None, z: GridMapping = None, dim: int = 2, order: int = 2) -> None:
        x = self._check_mapping(x, ngs.x)
        y = self._check_mapping(y, ngs.y)
        z = self._check_mapping(z, ngs.z)

        deformation = ngs.CF(tuple(map.deformation for map in (x, y, z)[:dim]))
        super().__init__(deformation, order=order)


class SpongeLayer(Buffer):

    @staticmethod
    def space(domains: dict[str, SpongeLayer], dmesh: DreamMesh) -> ngs.L2:

        orders = [dc.order for dc in domains.values()]
        is_constant_order = all([order == orders[0] for order in orders])

        order_policy = ngs.ORDER_POLICY.CONSTANT
        if not is_constant_order:
            order_policy = ngs.ORDER_POLICY.VARIABLE

        fes = ngs.L2(dmesh.ngsmesh, order=max(orders), order_policy=order_policy)

        if not is_constant_order:

            for region, bc in domains.items():
                region = dmesh.domain(region)

                for el in region.Elements():
                    fes.SetOrder(ngs.NodeId(ngs.ELEMENT, el.nr), bc.order)

        fes.UpdateDofTables()
        fes = ngs.Compress(fes, active_dofs=fes.GetDofs(dmesh.domain(domains)))

        return fes

    def __init__(self, sponge_function: ngs.CF, reference_state: State, order: int = 0) -> None:
        super().__init__(sponge_function, reference_state, order)


class PSpongeLayer(Buffer):

    @staticmethod
    def space(domains: dict[str, PSpongeLayer], dmesh: DreamMesh) -> ngs.L2:
        return SpongeLayer.space(domains, dmesh)

    @staticmethod
    def polynomial_order_range(highest, lowest: int = 0, step: int = 1) -> tuple[tuple[int, int]]:
        orders = [order for order in range(highest, lowest, -step)] + [lowest, lowest]
        return tuple((high, low) for high, low in zip(orders[:-1], orders[1:]))

    def __init__(self,
                 high_order: int,
                 low_order: int,
                 sponge_function: ngs.CF,
                 reference_state: State,
                 order: int = 0) -> None:

        if high_order < 0 or low_order < 0:
            raise ValueError("Negative polynomial order!")

        if not high_order >= low_order:
            raise ValueError("Low order must be less equal high order")

        self.high_order = int(high_order)
        self.low_order = int(low_order)
        super().__init__(sponge_function, reference_state, order)

    @property
    def is_equal_order(self) -> bool:
        return self.high_order == self.low_order

    def check_fem_order(self, order: int):
        if self.high_order > order:
            raise ValueError("Polynomial sponge order higher than polynomial discretization order")


class DomainConditions(ConditionContainer):

    _condition = Domain

    initial: dict[str, Initial] = ConditionDescriptor(Initial, True)
    perturbation: dict[str, Perturbation] = ConditionDescriptor(Perturbation)
    force: dict[str, Force] = ConditionDescriptor(Force)
    grid_deformation: dict[str, GridDeformation] = ConditionDescriptor(GridDeformation)
    sponge_layer: dict[str, SpongeLayer] = ConditionDescriptor(SpongeLayer)
    psponge_layer: dict[str, PSpongeLayer] = ConditionDescriptor(PSpongeLayer)

    def set(self, condition: Domain, domains: str = "default"):
        super().set(condition, domains)


class Boundary(Condition):
    ...


class Periodic(Boundary):
    ...


class BoundaryConditions(ConditionContainer):

    _condition = Boundary

    periodic: dict[str, Periodic] = ConditionDescriptor(Periodic)

    def set(self, condition: Boundary, boundaries: str):
        super().set(condition, boundaries)
