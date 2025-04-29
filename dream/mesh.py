# %%
from __future__ import annotations
import logging
import ngsolve as ngs

from collections import UserDict
from typing import Callable, Sequence

from dream import bla
from dream.config import ngsdict, dream_configuration

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
        r""" Returns a linear grid mapping.

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
        r""" Returns an exponential grid mapping.

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
        r""" Returns a tangential grid mapping.

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


class Condition:

    name: str = "condition"

    def __init_subclass__(cls) -> None:
        if not hasattr(cls, "name"):
            cls.name = cls.__name__
        cls.name = cls.name.lower()

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other) -> bool:
        return self is other

    def __ne__(self, other):
        return self is not other

    def __repr__(self) -> str:
        return f"{self.name}"


class Periodic(Condition):

    name = "periodic"


class Initial(Condition):

    def __init__(self, fields: ngsdict | None = None):
        super().__init__()
        self.fields = fields

    @dream_configuration
    def fields(self) -> ngsdict:
        """ Returns the fields of the initial condition """
        if self._fields is None:
            raise ValueError("Initial fields not set!")
        return self._fields

    @fields.setter
    def fields(self, fields: ngsdict) -> None:
        if isinstance(fields, ngsdict):
            self._fields = fields
        elif isinstance(fields, dict):
            self._fields = ngsdict(**fields)
        elif fields is None:
            self._fields = None
        else:
            raise TypeError(f"Initial fields must be of type '{ngsdict}' or '{dict}'")


class Perturbation(Condition):

    def __init__(self, fields: ngsdict | None = None):
        super().__init__()
        self.fields = fields

    @dream_configuration
    def fields(self) -> ngsdict:
        """ Returns the fields of the initial condition """
        if self._fields is None:
            raise ValueError("Initial fields not set!")
        return self._fields

    @fields.setter
    def fields(self, fields: ngsdict) -> None:
        if isinstance(fields, ngsdict):
            self._fields = fields
        elif isinstance(fields, dict):
            self._fields = ngsdict(**fields)
        elif fields is None:
            self._fields = None
        else:
            raise TypeError(f"Perturbation fields must be of type '{ngsdict}' or '{dict}'")


class Force(Condition):

    def __init__(self, fields: ngsdict | None = None):
        super().__init__()
        self.fields = fields

    @dream_configuration
    def fields(self) -> ngsdict:
        """ Returns the force of the corresponding equation """
        if self._fields is None:
            raise ValueError("Force fields not set!")
        return self._fields

    @fields.setter
    def fields(self, fields: ngsdict) -> None:
        if isinstance(fields, ngsdict):
            self._fields = fields
        elif isinstance(fields, dict):
            self._fields = ngsdict(**fields)
        elif fields is None:
            self._fields = None
        else:
            raise TypeError(f"Fields must be of type '{ngsdict}' or '{dict}'")


class Buffer(Condition):

    def __init__(self, function: ngs.CF | None = None, order: int = 0):
        self.function = function
        self.order = order
        super().__init__()

    @dream_configuration
    def function(self) -> ngs.CF:
        if self._function is None:
            raise ValueError("Buffer function not set!")
        return self._function

    @function.setter
    def function(self, function: ngs.CF) -> None:
        if isinstance(function, ngs.CF):
            self._function = function
        elif function is None:
            self._function = None
        else:
            raise TypeError(f"Buffer function must be of type '{ngs.CF}' or '{None}'")

    @dream_configuration
    def order(self) -> int:
        return self._order

    order.setter

    def order(self, order):
        self._order = int(order)

    @classmethod
    def get_space(cls, buffers: dict[str, Buffer], mesh: ngs.Mesh) -> ngs.FESpace:
        raise NotImplementedError()

    function: ngs.CF
    order: int


class GridDeformation(Buffer):

    name = "grid_deformation"

    def __init__(self,
                 x: GridMapping = None,
                 y: GridMapping = None,
                 z: GridMapping = None,
                 dim: int = 2,
                 order: int = 0) -> None:

        super().__init__(None, order)
        self.dim = dim
        self.x = x
        self.y = y
        self.z = z

    @dream_configuration
    def x(self) -> GridMapping:
        return self._x

    @x.setter
    def x(self, x):
        self._x = self.check_mapping(x, ngs.x)
        self.set_function(x=self._x)

    @dream_configuration
    def y(self) -> GridMapping:
        return self._y

    @y.setter
    def y(self, y):
        self._y = self.check_mapping(y, ngs.y)
        self.set_function(y=self._y)

    @dream_configuration
    def z(self) -> GridMapping:
        return self._z

    @z.setter
    def z(self, z):
        self._z = self.check_mapping(z, ngs.z)
        self.set_function(z=self._z)

    @dream_configuration
    def dim(self) -> int:
        return self._dim

    @dim.setter
    def dim(self, dim):
        self._dim = int(dim)

    def set_function(self, x: GridMapping = None, y: GridMapping = None, z: GridMapping = None) -> None:

        if x is None:
            x = getattr(self, "_x", GridMapping.none(ngs.x))

        if y is None:
            y = getattr(self, "_y", GridMapping.none(ngs.y))

        if z is None:
            z = getattr(self, "_z", GridMapping.none(ngs.z))

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


class SpongeLayer(Buffer):

    name = "sponge_layer"

    def __init__(self,
                 target_state: ngsdict = None,
                 function: ngs.CF = None,
                 order=0):
        super().__init__(function, order)
        self.target_state = target_state

    @dream_configuration
    def target_state(self) -> ngsdict:
        """ Returns the fields of the target state """
        if self._target_state is None:
            raise ValueError("Target state not set!")
        return self._target_state

    @target_state.setter
    def target_state(self, fields: ngsdict) -> None:
        if isinstance(fields, ngsdict):
            self._target_state = fields
        elif isinstance(fields, dict):
            self._target_state = ngsdict(**fields)
        elif fields is None:
            self._target_state = None
        else:
            raise TypeError(f"Target state must be of type '{ngsdict}' or '{dict}'")

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


class PSpongeLayer(SpongeLayer):

    name = "psponge_layer"

    def __init__(self,
                 high_order: int = 0,
                 low_order: int = 0,
                 target_state: ngsdict = None,
                 function: ngs.CF = None,
                 order: int = 0) -> None:

        super().__init__(target_state, function, order)
        self.high_order = high_order
        self.low_order = low_order

    @dream_configuration
    def high_order(self) -> int:
        return self._high_order

    @high_order.setter
    def high_order(self, high_order: int) -> None:

        if high_order < 0:
            raise ValueError("Negative polynomial order!")

        low_order = getattr(self, "_low_order", 0)
        if not high_order >= low_order:
            raise ValueError("Low order must be less equal high order")

        self._high_order = int(high_order)

    @dream_configuration
    def low_order(self) -> int:
        return self._low_order

    @low_order.setter
    def low_order(self, low_order) -> None:

        if low_order < 0:
            raise ValueError("Negative polynomial order!")

        if not self.high_order >= low_order:
            raise ValueError("Low order must be less equal high order")

        self._low_order = int(low_order)

    @property
    def is_equal_order(self) -> bool:
        return self.high_order == self.low_order

    @classmethod
    def get_pairs_of_polynomial_orders(cls, highest, lowest: int = 0, step: int = 1) -> tuple[tuple[int, int]]:
        orders = [order for order in range(highest, lowest, -step)] + [lowest, lowest]
        return tuple((high, low) for high, low in zip(orders[:-1], orders[1:]))

    high_order: int
    low_order: int


class Conditions(UserDict):

    def __init__(self, regions: list[str], mesh: ngs.Mesh, options: list[Condition]) -> None:
        self.data = {region: [] for region in regions}
        self.mesh = mesh
        self.options = {option.name: option for option in options}

    def get_region(self, *condition_types, as_pattern=False) -> str | list[str]:
        reg = [name for name, cond in self.items() if any(isinstance(c, condition_types) for c in cond)]
        if as_pattern:
            reg = get_pattern_from_sequence(reg)
        return reg

    def has_condition(self, condition_type: Condition | str) -> bool:

        if isinstance(condition_type, str):
            condition_type = self.options[condition_type]

        for container in self.values():
            if any(isinstance(condition, condition_type) for condition in container):
                return True

        return False

    def to_pattern(self, condition_type: Condition | str = Condition) -> dict[str, Condition]:

        if isinstance(condition_type, str):
            condition_type = self.options[condition_type]

        pattern = {}
        unique_conditions = set([condition for conditions in self.values()
                                for condition in conditions if isinstance(condition, condition_type)])

        for unique in unique_conditions:
            regions = [region for region in self if unique in self[region]]
            pattern[get_pattern_from_sequence(regions)] = unique

        return pattern

    def clear(self) -> None:
        self.data = {region: [] for region in self}

    def __setitem__(self, pattern: str, condition: Condition) -> None:

        if isinstance(condition, str):
            if condition not in self.options:
                msg = f""" Can not set condition '{condition}'!
                           Valid alternatives are: {self.options}"""
                raise ValueError(msg)
            condition = self.options[condition]()

        elif not isinstance(condition, Condition):
            raise TypeError(f"Condition must be instance of '{Condition}'")

        if isinstance(pattern, str):
            pattern = pattern.split("|")

        regions = get_regions_from_pattern(self, pattern)

        for miss in set(pattern).difference(regions):
            logger.warning(f"Region '{miss}' does not exist! Condition can not be set!")

        for region in regions:
            self.data[region].append(condition)

            # Give mesh access to the condition
            condition.mesh = self.mesh

            if len(self.data[region]) > 1:
                logger.warning(f"""Multiple conditions set for region '{region}': {
                               '|'.join([condition.name for condition in self[region]])}!""")

    def __repr__(self) -> str:
        return "\n".join([f"{region}: {'|'.join([condition.name for condition in conditions])} "for region,
                          conditions in self.items()])


class BoundaryConditions(Conditions):

    def __init__(self, mesh: ngs.Mesh, options: list[Condition]) -> None:
        super().__init__(list(dict.fromkeys(mesh.GetBoundaries())), mesh, options)

    def get_periodic_boundaries(self, as_pattern: bool = False) -> str | list[str]:
        return self.get_region(Periodic, as_pattern=as_pattern)

    def get_domain_boundaries(self, as_pattern: bool = False) -> list | str:
        """ Returns a list or pattern of the domain boundaries!

            The domain boundaries are deduced by the current set boundary conditions,
            while periodic boundaries are neglected! 
        """

        bnds = [bnd for bnd, bcs in self.items() if not any([isinstance(bc, Periodic) for bc in bcs]) and bcs]

        if as_pattern:
            bnds = get_pattern_from_sequence(bnds)

        return bnds


class DomainConditions(Conditions):

    def __init__(self, mesh: ngs.Mesh, options: list[Condition]) -> None:
        super().__init__(list(dict.fromkeys(mesh.GetMaterials())), mesh, options)

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


# --- Predefined Meshes --- #

def get_cylinder_omesh(ri: float,
                       ro: float,
                       n_polar: int,
                       n_radial: int,
                       geom: float = 1,
                       bnd: tuple[str, str, str] = ('cylinder', 'left', 'right')) -> ngs.Mesh:
    """ Generates a ring mesh with a given inner and outer radius.

    :param ri: Inner radius of the ring
    :type ri: float
    :param ro: Outer radius of the ring
    :type ro: float
    :param n_polar: Number of elements in the polar direction
    :type n_polar: int
    :param n_radial: Number of elements in the radial direction
    :type n_radial: int
    :param geom: Geometric factor for the radial direction, defaults to 1
    :type geom: float, optional
    :return: Ring mesh
    :rtype: ngs.Mesh
    """
    from netgen.csg import Plane, Vec, Cylinder, Pnt, CSGeometry
    from netgen.meshing import Element1D, Element2D, MeshPoint, FaceDescriptor, Mesh

    if (n_polar % 4) > 0:
        raise ValueError("Number of elements in polar direction must be a multiplicative of 4!")
    if (n_radial % 2) > 0:
        raise ValueError("Number of elements in radial direction must be a multiplicative of 2!")
    if n_radial > int(n_polar/2):
        raise ValueError("n_radial > n_polar/2!")

    mesh = Mesh()
    mesh.dim = 2

    top = Plane(Pnt(0, 0, 0), Vec(0, 0, 1))
    ring = Cylinder(Pnt(0, 0, 0), Pnt(0, 0, 1), ro)
    inner = Cylinder(Pnt(0, 0, 0), Pnt(0, 0, 1), ri)
    geo = CSGeometry()
    geo.SetBoundingBox(Pnt(-ro, -ro, -ro), Pnt(ro, ro, ro))
    geo.Add(top)
    geo.Add(inner)
    geo.Add(ring)

    mesh.SetGeometry(geo)

    pnums = []
    for j in range(n_radial+1):
        for i in range(n_polar):
            phi = ngs.pi/n_polar * j
            px = ngs.cos(2 * ngs.pi * i/n_polar + phi)
            py = ngs.sin(2 * ngs.pi * i/n_polar + phi)

            r = (ro - ri) * (j/n_radial)**geom + ri

            pnums.append(mesh.Add(MeshPoint(Pnt(r * px, r * py, 0))))

    # print(pnums)
    mesh.Add(FaceDescriptor(surfnr=1, domin=1, bc=1))
    mesh.Add(FaceDescriptor(surfnr=2, domin=1, domout=0, bc=1))
    mesh.Add(FaceDescriptor(surfnr=3, domin=1, domout=0, bc=2))

    idx_dom = 1

    for j in range(n_radial):
        # print("j=",j)
        for i in range(n_polar-1):
            # print("i=",i)
            # offset =
            mesh.Add(
                Element2D(
                    idx_dom, [pnums[i + j * (n_polar)],
                              pnums[i + (j + 1) * (n_polar)],
                              pnums[i + 1 + j * (n_polar)]]))
            mesh.Add(
                Element2D(
                    idx_dom,
                    [pnums[i + (j + 1) * (n_polar)],
                     pnums[i + (j + 1) * (n_polar) + 1],
                     pnums[i + 1 + j * (n_polar)]]))

        mesh.Add(
            Element2D(
                idx_dom,
                [pnums[n_polar - 1 + j * (n_polar)],
                 pnums[n_polar - 1 + (j + 1) * (n_polar)],
                 pnums[j * (n_polar)]]))
        mesh.Add(
            Element2D(
                idx_dom,
                [pnums[0 + j * (n_polar)],
                 pnums[n_polar - 1 + (j + 1) * (n_polar)],
                 pnums[(j + 1) * (n_polar)]]))

    for i in range(n_polar-1):
        mesh.Add(Element1D([pnums[i], pnums[i+1]], [0, 1], 1))
    mesh.Add(Element1D([pnums[n_polar-1], pnums[0]], [0, 1], 1))

    offset = int(-n_radial/2 + n_polar/4)

    for i in range(0, offset):
        mesh.Add(Element1D([pnums[i + n_radial * n_polar], pnums[i + n_radial * n_polar + 1]], [0, 2], index=3))

    for i in range(offset, int(n_polar/2)+offset):
        mesh.Add(Element1D([pnums[i + n_radial * n_polar], pnums[i + n_radial * n_polar + 1]], [0, 2], index=2))

    for i in range(int(n_polar/2)+offset, n_polar-1):
        mesh.Add(Element1D([pnums[i + n_radial * n_polar], pnums[i + n_radial * n_polar + 1]], [0, 2], index=3))
    mesh.Add(Element1D([pnums[n_radial*n_polar], pnums[n_polar - 1 + n_radial * n_polar]], [0, 2], index=3))

    for i, name in enumerate(bnd):
        mesh.SetBCName(i, name)

    return ngs.Mesh(mesh)


def get_cylinder_mesh(radius: float = 0.5,
                      sponge_layer: bool = False,
                      boundary_layer_levels: int = 5,
                      boundary_layer_thickness: float = 0.0,
                      transition_layer_levels: int = 5,
                      transition_layer_growth: float = 1.4,
                      transition_radial_factor: float = 6,
                      farfield_radial_factor: float = 50,
                      sponge_radial_factor: float = 60,
                      wake_maxh: float = 2,
                      farfield_maxh: float = 4,
                      sponge_maxh: float = 4,
                      bnd: tuple[str, str, str] = ('inflow', 'outflow', 'cylinder'),
                      mat: tuple[str, str] = ('sound', 'sponge'),
                      curve_layers: bool = False,
                      grading: float = 0.3):

    import numpy as np
    from netgen.occ import WorkPlane, OCCGeometry, Glue

    if boundary_layer_thickness < 0:
        raise ValueError(f"Boundary Layer Thickness needs to be greater equal Zero!")
    if not sponge_layer:
        sponge_radial_factor = farfield_radial_factor
        sponge_maxh = farfield_maxh
    elif sponge_radial_factor < farfield_radial_factor and sponge_layer:
        raise ValueError("Sponge Radial Factor must be greater than Farfield Radial Factor")

    bl_radius = radius + boundary_layer_thickness
    tr_radius = transition_radial_factor * radius
    ff_radius = farfield_radial_factor * radius
    sp_radius = sponge_radial_factor * radius

    wp = WorkPlane()

    # Cylinder
    cylinder = wp.Circle(radius).Face()
    cylinder.edges[0].name = bnd[2]

    # Viscous regime
    if boundary_layer_thickness > 0:
        bl_maxh = boundary_layer_thickness/boundary_layer_levels
        bl_radial_levels = np.linspace(radius, bl_radius, int(boundary_layer_levels) + 1)
        bl_faces = [wp.Circle(r).Face() for r in np.flip(bl_radial_levels[1:])]
        for bl_face in bl_faces:
            bl_face.maxh = bl_maxh
        boundary_layer = Glue(bl_faces) - cylinder
        for face in boundary_layer.faces:
            face.name = mat[0]

    # Transition regime
    tr_layer_growth = np.linspace(0, 1, transition_layer_levels+1)**transition_layer_growth
    tr_radial_levels = bl_radius + (tr_radius - bl_radius) * tr_layer_growth
    tr_maxh = np.diff(tr_radial_levels)
    tr_faces = [wp.Circle(r).Face() for r in np.flip(tr_radial_levels[1:])]
    for tr_face, maxh in zip(tr_faces, tr_maxh):
        tr_face.maxh = maxh
    transition_regime = Glue(tr_faces) - cylinder
    for face in transition_regime.faces:
        face.name = mat[0]

    # Farfield region
    farfield = wp.MoveTo(0, 0).Circle(ff_radius).Face()
    farfield.maxh = farfield_maxh
    for face in farfield.faces:
        face.name = mat[0]

    # Wake region
    wake_radius = tr_radius + maxh
    wp.MoveTo(0, wake_radius).Direction(-1, 0)
    wp.Arc(wake_radius, 180)
    wp.LineTo(ff_radius, -wake_radius)
    wp.LineTo(ff_radius, wake_radius)
    wp.LineTo(0, wake_radius)
    wake = wp.Face() - transition_regime - cylinder
    wake = wake * farfield
    wake.maxh = wake_maxh
    for face in wake.faces:
        face.name = mat[0]

    # Outer region (if defined)
    wp.MoveTo(0, sp_radius).Direction(-1, 0)
    wp.Arc(sp_radius, 180)
    wp.Arc(sp_radius, 180)
    outer = wp.Face()

    for edge, bc in zip(outer.edges, [bnd[0], bnd[1]]):
        edge.name = bc

    if sponge_layer:
        for face in outer.faces:
            face.name = mat[1]
        outer = outer - farfield
        outer.maxh = sponge_maxh
        outer = Glue([outer, farfield])

    sound = Glue([outer - wake, wake * outer]) - transition_regime - cylinder

    geo = Glue([sound, transition_regime])
    if boundary_layer_thickness > 0:
        geo = Glue([geo, boundary_layer])

    geo = OCCGeometry(geo, dim=2)
    mesh = geo.GenerateMesh(maxh=sponge_maxh, grading=grading)

    if not curve_layers:
        from netgen.meshing import Mesh, FaceDescriptor, Element1D, Element2D

        geo = outer - cylinder
        geo = OCCGeometry(geo, dim=2)

        new_mesh = Mesh()
        new_mesh.dim = 2
        new_mesh.SetGeometry(geo)

        edge_map = set(elem.edgenr for elem in mesh.Elements1D())

        new_mesh.Add(FaceDescriptor(surfnr=1, domin=1, domout=0, bc=1))
        new_mesh.Add(FaceDescriptor(surfnr=2, domin=1, domout=0, bc=2))
        new_mesh.Add(FaceDescriptor(surfnr=3, domin=1, domout=0, bc=3))

        for i, name in enumerate(bnd):
            new_mesh.SetBCName(i, name)

        idx_dom = new_mesh.AddRegion(mat[0], dim=2)
        new_mesh.SetMaterial(idx_dom, mat[0])

        if sponge_layer:
            new_mesh.SetBCName(3, "default")
            sponge_dom = new_mesh.AddRegion(mat[1], dim=2)
            new_mesh.SetMaterial(sponge_dom, mat[1])
            edge_map = {1: (0, 1), 2: (1, 2), 3: (2, 4),  4: (2, 4), 5: (2, 4), max(edge_map): (3, 3)}
        else:
            edge_map = {6: (0, 1), 8: (1, 2), 7: (1, 2), 5: (1, 2), 1: (1, 2), max(edge_map): (2, 3)}

        for point in mesh.Points():
            new_mesh.Add(point)

        for elem in mesh.Elements2D():
            if sponge_layer and elem.index == 1:
                new_mesh.Add(Element2D(sponge_dom, elem.vertices))
            else:
                new_mesh.Add(Element2D(idx_dom, elem.vertices))

        for elem in mesh.Elements1D():
            if elem.edgenr in edge_map:
                edgenr, index = edge_map[elem.edgenr]
                new_mesh.Add(Element1D(elem.points, elem.surfaces, index, edgenr))

        mesh = new_mesh

    return ngs.Mesh(mesh)
