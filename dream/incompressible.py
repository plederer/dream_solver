from __future__ import annotations

import logging
import ngsolve as ngs

# Import the necessary modules from dream
from dream.config import (Configuration,
                          dream_configuration,
                          ngsdict,
                          quantity,
                          equation,
                          Integrals)

from dream.time import StationaryScheme, Scheme, StationaryRoutine
from dream.solver import FiniteElementMethod, SolverConfiguration
from dream.mesh import (BoundaryConditions,
                        DomainConditions,
                        Condition,
                        Periodic,
                        Initial,
                        Force)

# Instantiate the logger with the name of the current module
logger = logging.getLogger(__name__)

# --- Implementation --- #

# Define a specific dict for the flow state


class flowfields(ngsdict):

    u = quantity('velocity', r"$u$")
    p = quantity('pressure', r"$p$")
    tau = quantity('deviatoric_stress_tensor', r"$\tau$")
    eps = quantity('strain_rate_tensor', r"\varepsilon")
    grad_u = quantity('velocity_gradient', r"\nabla u")
    g = quantity('gravity', r"g")
    f = quantity('force', r"f")

# Define a boundary conditions


class Inflow(Condition):

    name: str = "inflow"

    def __init__(self, velocity: flowfields | ngs.CF = None):
        self.velocity = velocity
        super().__init__()

    @dream_configuration
    def velocity(self) -> flowfields:
        """ Returns the fields of the farfield condition """
        if self._velocity is None:
            raise ValueError("Velocity is not set!")
        return self._velocity

    @velocity.setter
    def velocity(self, velocity: ngsdict) -> None:
        if velocity is None:
            self._velocity = None
        else:
            self._velocity = ngs.CF(tuple(velocity))


class Outflow(Condition):

    name: str = "outflow"


class Wall(Condition):

    name: str = "wall"


BCS = [Inflow, Outflow, Wall, Periodic]
DCS = [Force, Initial]


class DynamicViscosity(Configuration, is_interface=True):

    root: IncompressibleSolver

    @property
    def is_linear(self):
        return isinstance(self, Constant)

    def shear_rate(self, u: flowfields):
        eps = self.root.strain_rate_tensor(u)
        return 2*ngs.sqrt(ngs.InnerProduct(eps, eps))

    def viscosity(self, u: flowfields):
        raise NotImplementedError()


class Constant(DynamicViscosity):

    name: str = "constant"

    def viscosity(self, u: flowfields):
        return 1.0


class Powerlaw(DynamicViscosity):

    name: str = "powerlaw"

    def __init__(self, mesh, root=None, **default):

        self._powerlaw_exponent = ngs.Parameter(2.0)
        self._viscosity_ratio = ngs.Parameter(1.0)

        DEFAULT = {
            'powerlaw_exponent': self._powerlaw_exponent.Get(),
            'viscosity_ratio': self._viscosity_ratio.Get()
        }
        DEFAULT.update(default)

        super().__init__(mesh, root, **DEFAULT)

    @dream_configuration
    def powerlaw_exponent(self) -> ngs.Parameter:
        """ Returns the power law exponent """
        return self._powerlaw_exponent

    @powerlaw_exponent.setter
    def powerlaw_exponent(self, power_law_exponent: ngs.Parameter) -> None:

        if isinstance(power_law_exponent, ngs.Parameter):
            power_law_exponent = power_law_exponent.Get()

        if not 1 <= power_law_exponent <= 2:
            raise ValueError("Invalid power law exponent. Value has to be in the range of 1 <= r <= 2!")

        self._powerlaw_exponent.Set(power_law_exponent)

    @dream_configuration
    def viscosity_ratio(self) -> ngs.Parameter:
        """ Returns the viscosity ratio """
        return self._viscosity_ratio

    @viscosity_ratio.setter
    def viscosity_ratio(self, viscosity_ratio: ngs.Parameter) -> None:

        if isinstance(viscosity_ratio, ngs.Parameter):
            viscosity_ratio = viscosity_ratio.Get()

        if viscosity_ratio <= 0:
            raise ValueError("Invalid viscosity ratio. Value has to be > 0!")

        self._viscosity_ratio.Set(viscosity_ratio)

    def viscosity(self, u: flowfields):
        return self.viscosity_ratio * self.shear_rate(u)**(self.powerlaw_exponent - 1)


# Define solving schemes
class LinearScheme(StationaryScheme):

    name: str = "stationary_linear"

    def assemble(self):

        self.fem = self.root.fem

        self.blf = ngs.BilinearForm(self.fem.fes)
        self.add_sum_of_integrals(self.blf, self.root.fem.blf)

        self.lf = ngs.LinearForm(self.fem.fes)
        self.add_sum_of_integrals(self.lf, self.root.fem.lf)

        self.blf.Assemble()
        self.lf.Assemble()

        # Add dirichlet boundary conditions
        self.lf.vec.data -= self.blf.mat * self.fem.gfu.vec
        self.inv = self.root.linear_solver.inverse(self.blf, self.fem.fes)

    def solve(self):
        self.fem.gfu.vec.data += self.inv * self.lf.vec



# Define Finite Elements

class IncompressibleFiniteElement(FiniteElementMethod):

    root: IncompressibleSolver

    @dream_configuration    
    def scheme (self) -> LinearScheme:
        """ Returns the scheme for the incompressible flow solver """
        return self._scheme
    
    @scheme.setter
    def scheme (self, scheme: LinearScheme) -> None:
        if isinstance(self.root.time, StationaryRoutine):
            OPTIONS = [LinearScheme]
        else:
            raise ValueError("Invalid scheme for incompressible flow solver. Only stationary schemes are supported!")
        self._scheme = self._get_configuration_option(scheme, OPTIONS, Scheme)

    def add_symbolic_spatial_forms(self, blf: Integrals, lf: Integrals):
        self.add_symbolic_stokes_form(blf, lf)

    def get_solution_fields(self) -> flowfields:
        return self.get_incompressible_state(self.gfu)

    def add_symbolic_stokes_form(self, blf, lf):
        raise NotImplementedError("Overload this method in derived class!")

    def get_incompressible_state(self, u: ngs.CF) -> flowfields:
        raise NotImplementedError("Overload this method in derived class!")

class TaylorHood(IncompressibleFiniteElement):

    name: str = "taylor-hood"

    def add_finite_element_spaces(self, spaces: dict[str, ngs.FESpace]):

        bnds = self.root.bcs.get_region(Wall, Inflow, as_pattern=True)
        U = ngs.VectorH1(self.mesh, order=self.order, dirichlet=bnds)
        P = ngs.H1(self.mesh, order=self.order-1)

        if self.root.bcs.has_condition(Periodic):
            U = ngs.Periodic(U)

        spaces['u'] = U
        spaces['p'] = P

    def add_symbolic_stokes_form(self, blf: Integrals, lf: Integrals):
        bonus = self.root.optimizations.bonus_int_order

        u, v = self.TnT['u']
        p, q = self.TnT['p']

        U = self.get_incompressible_state([u, p])
        V = self.get_incompressible_state([v, q])

        blf['u']['stokes'] = ngs.InnerProduct(U.tau, V.eps) * ngs.dx(bonus_intorder=bonus.vol)
        blf['u']['stokes'] += -ngs.div(V.u) * U.p * ngs.dx
        blf['p']['stokes'] = -ngs.div(U.u) * V.p * ngs.dx

        if not self.root.bcs.has_condition(Outflow):
            blf['p']['stokes'] += -1e-8 * U.p * V.p * ngs.dx

    def get_incompressible_state(self, u: ngs.CF) -> flowfields:

        if isinstance(u, ngs.GridFunction):
            u = u.components

        state = flowfields()
        state.u = u[0]
        state.p = u[1]
        state.grad_u = ngs.Grad(state.u)
        state.eps = self.root.strain_rate_tensor(state)
        state.tau = self.root.deviatoric_stress_tensor(state)

        return state

    def set_boundary_conditions(self) -> None:
        u = self.mesh.BoundaryCF({dom: bc.velocity for dom, bc in self.root.bcs.to_pattern(Inflow).items()})
        self.gfus['u'].Set(u, ngs.BND)


class IncompressibleSolver(SolverConfiguration):

    def __init__(self, mesh=None, **default):
        bcs = BoundaryConditions(mesh, BCS)
        dcs = DomainConditions(mesh, DCS)

        self._reynolds_number = ngs.Parameter(150.0)

        DEFAULT = {
            'reynolds_number': self._reynolds_number.Get(),
            'dynamic_viscosity': Constant(mesh, self),
            'convection': False,
        }
        DEFAULT.update(default)

        super().__init__(mesh=mesh, bcs=bcs, dcs=dcs, **DEFAULT)

    @dream_configuration
    def fem(self) -> TaylorHood:
        """ Returns the finite element method """
        return self._fem

    @fem.setter
    def fem(self, fem: IncompressibleFiniteElement) -> None:
        OPTIONS = [TaylorHood]
        self._fem = self._get_configuration_option(fem, OPTIONS, IncompressibleFiniteElement)

    @dream_configuration
    def reynolds_number(self) -> ngs.Parameter:
        """ Returns the Reynolds number """
        return self._reynolds_number

    @reynolds_number.setter
    def reynolds_number(self, reynolds_number: ngs.Parameter) -> None:
        if isinstance(reynolds_number, ngs.Parameter):
            reynolds_number = reynolds_number.Get()

        if reynolds_number <= 0:
            raise ValueError("Invalid Reynold number. Value has to be > 0!")

        self._reynolds_number.Set(reynolds_number)

    @dream_configuration
    def dynamic_viscosity(self) -> Constant | Powerlaw:
        r""" Sets the dynamic viscosity for the incompressible flow solver.

            :setter: Sets the dynamic viscosity, defaults to Constant
            :getter: Returns the dynamic viscosity
        """
        return self._dynamic_viscosity

    @dynamic_viscosity.setter
    def dynamic_viscosity(self, dynamic_viscosity: DynamicViscosity) -> None:
        OPTIONS = [Constant, Powerlaw]
        self._dynamic_viscosity = self._get_configuration_option(dynamic_viscosity, OPTIONS, DynamicViscosity)

    @dream_configuration
    def convection(self) -> bool:
        """ Returns the convection flag """
        return self._convection

    @convection.setter
    def convection(self, convection: bool) -> None:
        if not isinstance(convection, bool):
            raise TypeError("Convection must be of type 'bool'!")

        self._convection = convection

    def get_solution_fields(self, *fields, default_fields=True):

        if default_fields:
            fields = ('velocity', 'pressure') + fields

        return super().get_solution_fields(*fields)

    @equation
    def velocity(self, u: flowfields):
        if u.u is not None:
            return u.u

    @equation
    def pressure(self, u: flowfields):
        if u.p is not None:
            return u.p

    @equation
    def viscosity(self, u: flowfields):
        return self.dynamic_viscosity.viscosity(u)

    @equation
    def deviatoric_stress_tensor(self, u: flowfields):

        Re = self.reynolds_number

        mu = self.viscosity(u)
        strain = self.strain_rate_tensor(u)

        return 2 * mu/Re * strain

    @equation
    def strain_rate_tensor(self, u: flowfields):
        if u.eps is not None:
            return u.eps
        elif u.grad_u is not None:
            return 0.5 * (u.grad_u + u.grad_u.trans)
