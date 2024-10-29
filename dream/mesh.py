# %%
from __future__ import annotations
import logging
import ngsolve as ngs

from collections import UserDict
from typing import Callable, Sequence, Any, TypeVar

from dream import bla
from dream.config import State, MultipleConfiguration, any

logger = logging.getLogger(__name__)


def is_mesh_periodic(mesh: ngs.Mesh) -> bool:
    return bool(mesh.GetPeriodicNodePairs(ngs.VERTEX))


def get_pattern_from_sequence(sequence: Sequence) -> str:
    if isinstance(sequence, str):
        return sequence
    return "|".join(sequence)


def get_regions_from_pattern(regions, pattern) -> tuple[str]:

    if isinstance(pattern, str):
        pattern = pattern.split("|")

    regions = tuple(region for region in pattern if region in regions)

    return regions


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

        c1 = bla.fixpoint_iteration(L, lambda c: ngs.log(1 + scale * L * c)/L, it=100, tol=1e-16)
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

        c1 = bla.fixpoint_iteration(L, lambda c: ngs.atan(scale * L * c)/L, it=100, tol=1e-16)
        c0 = 1/c1

        def map(x_): return c0 * ngs.tan(c1 * (x_ - start)) + start

        return cls(coordinate, map)

    def __init__(self, x: ngs.CF, map: Callable[[bla.SCALAR], bla.SCALAR]) -> None:
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


class Condition(MultipleConfiguration, is_interface=True):

    def __hash__(self) -> int:
        return id(self)

    def __repr__(self) -> str:
        return f"{self.name}:\n" + super().__repr__()


class Initial(Condition):

    name = "initial"

    @any(default=None)
    def state(self, state) -> State:
        return state

    @state.getter_check
    def state(self) -> None:
        if self.data['state'] is None:
            raise ValueError("Initial State not set!")

    state: State


class Perturbation(Condition):

    name = "perturbation"


class Force(Condition):

    name = "force"


class Buffer(Condition, is_interface=True):

    @any(default=None)
    def function(self, buffer_function) -> ngs.CF:
        return buffer_function

    @function.getter_check
    def function(self) -> ngs.CF:
        if self.data['function'] is None:
            raise ValueError("Buffer function not set!")

    @any(default=0)
    def order(self, order) -> int:
        return int(order)

    @classmethod
    def get_space(cls, buffers: dict[str, Buffer], mesh: ngs.Mesh) -> ngs.FESpace:
        raise NotImplementedError()

    function: ngs.CF
    order: int


class GridDeformation(Buffer):

    name = "grid_deformation"

    @any(default=2)
    def dim(self, dim) -> int:
        return int(dim)

    @any(default=None)
    def x(self, x) -> GridMapping:
        map = self.check_mapping(x, ngs.x)
        self.set_function(x=x)
        return map

    @any(default=None)
    def y(self, y) -> GridMapping:
        map = self.check_mapping(y, ngs.y)
        self.set_function(y=y)
        return map

    @any(default=None)
    def z(self, z) -> GridMapping:
        map = self.check_mapping(z, ngs.z)
        self.set_function(z=z)
        return map

    def set_function(self, x: GridMapping = None, y: GridMapping = None, z: GridMapping = None) -> None:

        if x is None:
            x = self.data.get("x", GridMapping.none(ngs.x))

        if y is None:
            y = self.data.get("y", GridMapping.none(ngs.y))

        if z is None:
            z = self.data.get("z", GridMapping.none(ngs.z))

        self.function = ngs.CF(tuple(map.deformation for map in (x, y, z)[:self.dim]))

    @classmethod
    def get_space(cls, grid_deformations: dict[str, GridDeformation], mesh: ngs.Mesh) -> ngs.VectorH1:
        orders = [grid_deformation.order for grid_deformation in grid_deformations.values()]

        grid_deformation_pattern = get_pattern_from_sequence(grid_deformations)
        fes = ngs.VectorH1(mesh, order=max(orders))
        fes = ngs.Compress(fes, active_dofs=fes.GetDofs(mesh.Materials(grid_deformation_pattern)))

        if hasattr(mesh, "is_periodic") and mesh.is_periodic or is_mesh_periodic(mesh):
            fes = ngs.Periodic(fes)

        return fes

    @staticmethod
    def check_mapping(map: GridMapping, coordinate: ngs.CF) -> GridMapping:
        if map is None:
            map = GridMapping.none(coordinate)

        if not isinstance(map, GridMapping):
            raise TypeError(f"Map has to be of type '{GridMapping}'")

        return map

    dim: int
    x: GridMapping
    y: GridMapping
    z: GridMapping


class SpongeLayer(Buffer):

    name = "sponge_layer"

    @any(default=None)
    def target_state(self, target_state) -> State:
        return target_state

    @target_state.getter_check
    def target_state(self) -> None:
        if self.data['target_state'] is None:
            raise ValueError("Target State not set!")

    @classmethod
    def get_space(cls, sponge_layers: dict[str, SpongeLayer], mesh: ngs.Mesh):

        orders = [sponge_layer.order for sponge_layer in sponge_layers.values()]
        is_constant_order = all([order == orders[0] for order in orders])

        order_policy = ngs.ORDER_POLICY.CONSTANT
        if not is_constant_order:
            order_policy = ngs.ORDER_POLICY.VARIABLE

        fes = ngs.L2(mesh, order=max(orders), order_policy=order_policy)

        if not is_constant_order:

            for region, bc in sponge_layers.items():
                region = mesh.Materials(region)

                for el in region.Elements():
                    fes.SetOrder(ngs.NodeId(ngs.ELEMENT, el.nr), bc.order)

        sponge_layers_pattern = get_pattern_from_sequence(sponge_layers)

        fes.UpdateDofTables()
        fes = ngs.Compress(fes, active_dofs=fes.GetDofs(mesh.Materials(sponge_layers_pattern)))

        return fes

    target_state: State


class PSpongeLayer(SpongeLayer):

    name = "psponge_layer"

    @any(default=0)
    def high_order(self, high_order) -> int:

        if high_order < 0:
            raise ValueError("Negative polynomial order!")

        low_order = self.data.get("low_order", 0)
        if not high_order >= low_order:
            raise ValueError("Low order must be less equal high order")

        return int(high_order)

    @any(default=0)
    def low_order(self, low_order) -> int:

        if low_order < 0:
            raise ValueError("Negative polynomial order!")

        if not self.high_order >= low_order:
            raise ValueError("Low order must be less equal high order")

        return int(low_order)

    @property
    def is_equal_order(self) -> bool:
        return self.high_order == self.low_order

    @classmethod
    def get_pairs_of_polynomial_orders(cls, highest, lowest: int = 0, step: int = 1) -> tuple[tuple[int, int]]:
        orders = [order for order in range(highest, lowest, -step)] + [lowest, lowest]
        return tuple((high, low) for high, low in zip(orders[:-1], orders[1:]))

    high_order: int
    low_order: int


class Periodic(Condition):

    name = "periodic"


class Conditions(UserDict):

    conditions: dict[str, Condition] = {}

    def __init_subclass__(cls) -> None:
        cls.conditions = {}
        return super().__init_subclass__()

    def __init__(self, regions: list[str], mesh: ngs.Mesh) -> None:
        self.data = {region: [] for region in regions}
        self.mesh = mesh

        if not hasattr(mesh, "is_periodic"):
            self.mesh.is_periodic = is_mesh_periodic(mesh)

    def to_pattern(self, condition_type: Condition | str = Condition) -> dict[str, Condition]:

        if isinstance(condition_type, str):
            condition_type = self.conditions[condition_type]

        pattern = {}
        unique_conditions = set([condition for conditions in self.values()
                                for condition in conditions if isinstance(condition, condition_type)])

        for unique in unique_conditions:
            regions = [region for region in self if unique in self[region]]
            pattern[get_pattern_from_sequence(regions)] = unique

        return pattern

    def clear(self) -> None:
        self.data = {region: [] for region in self}

    @classmethod
    def register_condition(cls, condition: Condition) -> None:
        cls.conditions[condition.name] = condition

    def __setitem__(self, pattern: str, condition: Condition) -> None:

        if isinstance(condition, str):
            if condition not in self.conditions:
                msg = f""" Can not set condition '{condition}'!
                           Valid alternatives are: {self.conditions}"""
                raise ValueError(msg)
            condition = self.conditions[condition]()

        elif not isinstance(condition, Condition):
            raise TypeError(f"Condition must be instance of '{Condition}'")

        if isinstance(pattern, str):
            pattern = pattern.split("|")

        regions = get_regions_from_pattern(self, pattern)

        for miss in set(pattern).difference(regions):
            logger.warning(f"Region '{miss}' does not exist! Condition can not be set!")

        for region in regions:
            self.data[region].append(condition)

            if len(self.data[region]) > 1:
                logger.warning(f"Multiple conditions set for region '{region}': {'|'.join([condition.name for condition in self[region]])}!")

    def __repr__(self) -> str:
        return "\n".join([f"{region}: {'|'.join([condition.name for condition in conditions])}" for region, conditions in self.items()])

class BoundaryConditions(Conditions):

    def __init__(self, mesh: ngs.Mesh) -> None:
        super().__init__(list(dict.fromkeys(mesh.GetBoundaries())), mesh)

    def get_domain_boundaries(self, as_pattern: bool = False) -> list | str:
        """ Returns a list or pattern of the domain boundaries!

            The domain boundaries are deduced by the current set boundary conditions,
            while periodic boundaries are neglected! 
        """

        bnds = [bnd for bnd, bc in self.items() if not isinstance(bc, Periodic) or bc]

        if as_pattern:
            bnds = get_pattern_from_sequence(bnds)

        return bnds


class DomainConditions(Conditions):

    def __init__(self, mesh: ngs.Mesh) -> None:
        super().__init__(list(dict.fromkeys(mesh.GetMaterials())), mesh)

    def get_psponge_layers(self) -> dict[str, PSpongeLayer]:
        return {region: dc for region, dcs in self.items() for dc in dcs if isinstance(dc, PSpongeLayer)}

    def get_sponge_layers(self) -> dict[str, SpongeLayer]:
        return {region: dc for region, dcs in self.items() for dc in dcs if isinstance(dc, SpongeLayer) and not isinstance(dc, PSpongeLayer)}

    def get_grid_deformations(self) -> dict[str, GridDeformation]:
        return {region: dc for region, dcs in self.items() for dc in dcs if isinstance(dc, GridDeformation)}

    def get_grid_deformation_function(self, grid_deformations: dict[str, GridDeformation] = None):
        if grid_deformations is None:
            self.to_pattern(GridDeformation)
        return self.get_buffer_function_(grid_deformations, GridDeformation)

    def get_sponge_layer_function(self, sponge_layers: dict[str, SpongeLayer] = None):
        if sponge_layers is None:
            sponge_layers = self.to_pattern(SpongeLayer)
        return self.get_buffer_function_(sponge_layers, SpongeLayer)

    def get_psponge_layer_function(self, psponge_layers:  dict[str, PSpongeLayer] = None):
        if psponge_layers is None:
            psponge_layers = self.to_pattern(PSpongeLayer)
        return self.get_buffer_function_(psponge_layers, PSpongeLayer)

    def get_buffer_function_(self, domains: dict[str, Buffer], buffer_type: Buffer) -> ngs.GridFunction:

        if not isinstance(buffer_type, type):
            buffer_type = type(buffer_type)

        if not issubclass(buffer_type, Buffer):
            raise TypeError(f"Buffer type has to be a subclass of '{Buffer}'")

        fes = buffer_type.get_space(domains, self.mesh)
        u, v = fes.TnT()

        blf = ngs.BilinearForm(fes)
        blf += u * v * ngs.dx

        lf = ngs.LinearForm(fes)
        for region, bc in domains.items():

            if not isinstance(bc, buffer_type):
                raise TypeError(f"Only domains of type '{buffer_type}' allowed!")

            lf += bc.function * v * ngs.dx(definedon=self.mesh.Materials(region))

        gfu = ngs.GridFunction(fes, name=str(buffer_type))

        with ngs.TaskManager():
            blf.Assemble()
            lf.Assemble()

            gfu.vec.data = blf.mat.Inverse(inverse="sparsecholesky") * lf.vec

        return gfu

    def reduce_psponge_layers_order_elementwise(
            self, space: ngs.L2 | ngs.VectorL2, psponge_layers: dict[str, PSpongeLayer] = None) -> ngs.L2 | ngs.VectorL2:

        if not isinstance(space, (ngs.L2, ngs.VectorL2)):
            raise TypeError("Can not reduce element order of non L2-spaces!")

        if psponge_layers is None:
            psponge_layers = self.to_pattern(PSpongeLayer)

        if psponge_layers:

            space = type(space)(space.mesh, **space.flags, order_policy=ngs.ORDER_POLICY.VARIABLE)

            for domain, bc in psponge_layers.items():

                if bc.high_order > space.globalorder:
                    raise ValueError("Polynomial sponge order higher than polynomial discretization order")

                domain = self.mesh.Materials(domain)

                for el in domain.Elements():
                    space.SetOrder(ngs.NodeId(ngs.ELEMENT, el.nr), bc.high_order)

            space.UpdateDofTables()

        return space

    def reduce_psponge_layers_order_facetwise(
            self, space: ngs.FacetFESpace, psponge_layers: dict[str, PSpongeLayer] = None) -> ngs.FacetFESpace:

        if not isinstance(space, ngs.FacetFESpace):
            raise TypeError("Can not reduce element order of non FacetFESpace-spaces!")

        if psponge_layers is None:
            psponge_layers = self.to_pattern(PSpongeLayer)

        if psponge_layers:

            space = type(space)(space.mesh, **space.flags, order_policy=ngs.ORDER_POLICY.VARIABLE)

            if self.mesh.dim != 2:
                raise NotImplementedError("3D PSpongeLayer not implemented for the moment!")

            psponge_region = self.mesh.Materials(get_pattern_from_sequence(psponge_layers))
            vhat_dofs = ~space.GetDofs(psponge_region)

            for domain, bc in psponge_layers.items():

                if bc.high_order > space.globalorder:
                    raise ValueError("Polynomial sponge order higher than polynomial discretization order")

                domain = self.mesh.Materials(domain)
                domain_dofs = space.GetDofs(domain)
                for i in range(bc.high_order + 1, space.globalorder + 1, 1):
                    domain_dofs[i::space.globalorder + 1] = 0

                vhat_dofs |= domain_dofs

            space = ngs.Compress(space, vhat_dofs)

        return space
