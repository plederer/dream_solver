from __future__ import annotations
from ngsolve import *
from collections import UserDict
from typing import Any, NamedTuple
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
    def is_grid_stretching(self) -> bool:
        return bool(self.dcs.grid_stretching)

    @property
    def is_periodic(self) -> bool:
        return self._is_periodic

    @property
    def highest_order_psponge(self) -> int:
        return max([sponge.high_order for sponge in self.dcs.psponge_layers.values()], default=0)

    def boundary(self, region: str) -> Region:
        return self.mesh.Boundaries(region)

    def domain(self, region: str) -> Region:
        return self.mesh.Materials(region)

    def pattern(self, sequence: list) -> str:
        return "|".join(sequence)

    def get_grid_stretching_function(self) -> GridFunction:
        grids = self.dcs.grid_stretching

        if grids:

            max_order = max([gsf.max_order for gsf in self.dcs.grid_stretching.values()], default=1)

            fes = VectorH1(self.mesh, order=max_order)
            u, v = fes.TnT()

            grid = GridFunction(fes)

            blf = BilinearForm(fes)
            blf += InnerProduct(u, v) * dx

            lf = LinearForm(fes)

            for domain, bc in grids.items():
                domain = self.domain(domain)
                lf += InnerProduct(bc.vec(self.dim), v) * dx(definedon=domain)

            blf.Assemble()
            lf.Assemble()

            grid.vec.data = blf.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * lf.vec

        else:
            fes = VectorH1(self.mesh, order=0)
            grid = GridFunction(fes)
            grid.vec[:] = 0

            logger.warning("Grid Stretching has not been set in domain conditions! Returning zero stretching.")

        return grid

    def set_grid_stretching(self):
        grid = self.get_grid_stretching_function()
        self.mesh.SetDeformation(grid)


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
    def thickness(self):
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


class SpongeFunction(NamedTuple):

    weight_function: CF
    order: int
    repr: str = u"\u03C3 f(x)"

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
        return f"{self.__class__.__name__}: {self.repr}"


class GridStretchFunction(NamedTuple):

    stretch_function: CF
    order: int
    repr: str = "f(x) - x"

    @classmethod
    def exponential_jacobi(cls,
                           coord: BufferCoordinate,
                           value: float = 0.25,
                           mirror: bool = False,
                           order: int = 5) -> GridStretchFunction:

        if not 0 < value < 1:
            raise ValueError(f"Endpoint Jacobian has to be 0 < p < 1")

        if coord.dim != 1:
            raise ValueError("Buffercoordinate needs to be 1-dimensional")

        def stretch_function(start, end, x_, k):
            D = end - start
            return D*k*(1 - exp(-x_/k)) - D*x_

        k = value/(value-1)
        start, end = coord.start, coord.end
        x_ = coord.get()

        sf = stretch_function(start, end, x_, k)

        if mirror:
            m_coord = coord.mirror()
            m_start, m_end = m_coord.start, m_coord.end
            mx_ = m_coord.get()

            sf += stretch_function(m_start, m_end, mx_, k)

        return cls(sf, order, "a exp(c x) - x")

    @classmethod
    def exponential_thickness(cls,
                              coord: BufferCoordinate,
                              factor: int = 5,
                              mirror: bool = False,
                              order: int = 5,
                              print_error: bool = False) -> GridStretchFunction:

        if not 1 < factor:
            raise ValueError(f"Choose factor > 1")

        if coord.dim != 1:
            raise ValueError("Buffercoordinate needs to be 1-dimensional")

        def stretch_function(start, end, x_):

            D = end - start

            a = -D
            def fix(a): return -D/(np.log(1 - factor*D/a))

            for i in range(100):
                new_a = fix(a)
                err = abs(new_a - a)

                if print_error:
                    logger.info(f"Exponential Fixpoint - It: {i:3d} - n+1: {new_a:.5e} - n: {a:.5e} - err: {err:.5e}")

                a = new_a

                if err < 1e-16:
                    break

            c = -D/a

            return a * (1 - exp(c*x_)) - D*x_

        start, end = coord.start, coord.end
        x_ = coord.get()

        sf = stretch_function(start, end, x_)

        if mirror:
            m_coord = coord.mirror()
            m_start, m_end = m_coord.start, m_coord.end
            mx_ = m_coord.get()

            sf += stretch_function(m_start, m_end, mx_)

        return cls(sf, order, "a exp(c x) - x")

    @classmethod
    def tangens_jacobi(cls,
                       coord: BufferCoordinate,
                       endpoint_jacobian: float = 0.33,
                       mirror: bool = False,
                       order: int = 5) -> GridStretchFunction:

        if not 0 < endpoint_jacobian < 1:
            raise ValueError(f"Endpoint Jacobian has to be 0 < p < 1")

        if coord.dim != 1:
            raise ValueError("Buffercoordinate needs to be 1-dimensional")

        def stretch_function(start, end, x_, k):
            D = end - start
            return D*k*tan(x_/k) - D*x_

        k = sqrt(endpoint_jacobian/(1-endpoint_jacobian))
        start, end = coord.start, coord.end
        x_ = coord.get()

        sf = stretch_function(start, end, x_, k)

        if mirror:
            m_coord = coord.mirror()
            m_start, m_end = m_coord.start, m_coord.end
            mx_ = m_coord.get()

            sf += stretch_function(m_start, m_end, mx_, k)

        return cls(sf, order, "a tan(c x) - x")

    @classmethod
    def tangens_thickness(cls,
                          coord: BufferCoordinate,
                          factor: int = 5,
                          mirror: bool = False,
                          order: int = 5,
                          print_error: bool = False) -> GridStretchFunction:

        if not factor > 1:
            raise ValueError(f"Choose factor > 1")

        if coord.dim != 1:
            raise ValueError("Buffercoordinate needs to be 1-dimensional")

        def stretch_function(start, end, x_):

            D = end - start

            a = D
            def fix(a): return D/np.arctan(factor * D/a)

            for i in range(100):
                new_a = fix(a)
                err = abs(new_a - a)

                if print_error:
                    logger.info(f"Tangens Fixpoint - It: {i:3d} - n+1: {new_a:.5e} - n: {a:.5e} - err: {err:.5e}")

                a = new_a

                if err < 1e-16:
                    break

            c = D/a

            return a * tan(c * x_) - D*x_

        start, end = coord.start, coord.end
        x_ = coord.get()

        sf = stretch_function(start, end, x_)

        if mirror:
            m_coord = coord.mirror()
            m_start, m_end = m_coord.start, m_coord.end
            mx_ = m_coord.get()

            sf += stretch_function(m_start, m_end, mx_)

        return cls(sf, order, "a tan(c x) - x")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}: {self.repr}"


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
        psponge = dict(sorted(psponge.items(), key=lambda x: x[1].high_order))
        return psponge

    @property
    def grid_stretching(self) -> dict[str, GridStretching]:
        return self._get_condition(self.GridStretching)

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

        def __init__(self,
                     high_order: int,
                     low_order: int,
                     *sponges: SpongeFunction,
                     state: State = None) -> None:

            self.high_order = high_order

            if self.is_constant:
                low_order = 0

                if state is None:
                    raise ValueError("For constant polynomials a state is required")
                else:
                    self._check_value(state.density, "density")
                    self._check_value(state.velocity, "velocity")
                    if state.all_thermodynamic_none:
                        raise ValueError("A Thermodynamic quantity is required!")
            elif not high_order > low_order:
                raise ValueError("Low Order must be smaller than High Order")
            super().__init__(state)

            self.low_order = low_order
            self.sponges = sponges

        @property
        def is_constant(self) -> bool:
            return self.high_order == 0

        @property
        def bonus_int_order(self) -> int:
            return max([sponge.order for sponge in self.sponges])

        @property
        def weight_function(self) -> CF:
            return sum([sponge.weight_function for sponge in self.sponges])

        def __repr__(self) -> str:
            return f"(High: {self.high_order}, Low: {self.low_order}, State: {self.state})"

    class GridStretching(_Domain):

        def __init__(self, x_: GridStretchFunction = None, y_: GridStretchFunction = None, z_: GridStretchFunction = None) -> None:
            self.x = self._set_zero_if_none(x_)
            self.y = self._set_zero_if_none(y_)
            self.z = self._set_zero_if_none(z_)

        @property
        def max_order(self) -> int:
            return max([x.order for x in (self.x, self.y, self.z)])

        def vec(self, dim: int = 2) -> CF:
            vec = (self.x.stretch_function, self.y.stretch_function, self.z.stretch_function)[:dim]
            return CF(vec)

        def _set_zero_if_none(self, streching: GridStretchFunction):
            if streching is None:
                streching = GridStretchFunction(0, order=0)
            return streching

        def __repr__(self) -> str:
            return repr((self.x, self.y, self.z))

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

    class _Boundary(Condition):
        ...

    class Dirichlet(_Boundary):
        def __init__(self, state: State):
            self._check_value(state.density, "density")
            self._check_value(state.velocity, "velocity")
            if state.all_thermodynamic_none:
                raise ValueError("A Thermodynamic quantity is required!")
            super().__init__(state)

    class FarField(_Boundary):
        def __init__(self, state: State):
            self._check_value(state.density, "density")
            self._check_value(state.velocity, "velocity")
            if state.all_thermodynamic_none:
                raise ValueError("A Thermodynamic quantity is required!")
            super().__init__(state)

    class Outflow(_Boundary):
        def __init__(self, pressure: float):
            state = State(pressure=pressure)
            self._check_value(state.pressure, "pressure")
            super().__init__(state)

    class Outflow_NSCBC(_Boundary):

        def __init__(self,
                     pressure: float,
                     sigma: float = 0.25,
                     reference_length: float = 1,
                     tangential_convective_fluxes: bool = True,
                     tangential_viscous_fluxes: bool = True,
                     normal_viscous_fluxes: bool = False) -> None:

            state = State(pressure=pressure)
            self._check_value(state.pressure, "pressure")
            super().__init__(state)

            self.sigma = sigma
            self.reference_length = reference_length
            self.tang_conv_flux = tangential_convective_fluxes
            self.tang_visc_flux = tangential_viscous_fluxes
            self.norm_visc_flux = normal_viscous_fluxes

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
