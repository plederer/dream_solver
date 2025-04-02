from __future__ import annotations

import logging
import ngsolve as ngs

# Import the necessary modules from dream
from dream import bla
from dream.config import (configuration,
                          parameter,
                          interface,
                          InterfaceConfiguration,
                          ngsdict,
                          quantity,
                          equation,
                          Integrals)

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

    u = quantity('velocity')
    p = quantity('pressure')
    tau = quantity('deviatoric_stress_tensor')
    eps = quantity('strain_rate_tensor')
    grad_u = quantity('velocity_gradient')
    g = quantity('gravity')
    f = quantity('force')

# Define a boundary conditions


class Inflow(Condition):

    name: str = "inflow"

    @configuration(default=None)
    def fields(self, fields):
        if fields is None:
            return fields
        elif bla.is_vector(fields):
            return flowfields(u=fields)
        else:
            return flowfields(**fields)

    @fields.getter_check
    def fields(self) -> None:
        if self.data['fields'] is None:
            raise ValueError("Inflow fields not set!")


class Outflow(Condition):

    name: str = "outflow"


class Wall(Condition):

    name: str = "wall"


BCS = [Inflow, Outflow, Wall, Periodic]
DCS = [Force, Initial]



class DynamicViscosity(InterfaceConfiguration, is_interface=True):

    cfg: IncompressibleSolver

    @property
    def is_linear(self):
        return isinstance(self, Constant)

    def shear_rate(self, u: flowfields):
        eps = self.cfg.strain_rate_tensor(u)
        return 2*ngs.sqrt(ngs.InnerProduct(eps, eps))

    def viscosity(self, u: flowfields):
        raise NotImplementedError()


class Constant(DynamicViscosity):

    name: str = "constant"

    def viscosity(self, u: flowfields):
        return 1.0


class Powerlaw(DynamicViscosity):

    name: str = "powerlaw"
    aliases = ('ostwald-de-waele',)

    @parameter(default=2.0)
    def powerlaw_exponent(self, power_law_exponent):

        if not 1 <= power_law_exponent <= 2:
            raise ValueError("Invalid power law exponent. Value has to be in the range of 1 <= r <= 2!")

        return power_law_exponent

    @parameter(default=1.0)
    def viscosity_ratio(self, K):
        return K

    def viscosity(self, u: flowfields):
        return self.viscosity_ratio * self.shear_rate(u)**(self.powerlaw_exponent - 1)


# Define Finite Elements

class IncompressibleFiniteElement(FiniteElementMethod, is_interface=True):

    cfg: IncompressibleSolver

    @property
    def TnT(self):
        return self.cfg.TnT
    
    @property
    def gfu(self):
        return self.cfg.gfus

    def add_symbolic_spatial_forms(self, blf: Integrals, lf: Integrals):
        self.add_symbolic_stokes_form(blf, lf)
        
    def add_symbolic_stokes_form(self, blf, lf):
        raise NotImplementedError("Overload in subclass")
    
    def get_incompressible_state(self, u: ngs.CF) -> flowfields:
        raise NotImplementedError()

    def get_fields(self, quantities: dict[str, bool]) -> ngsdict:
        u = self.get_incompressible_state(self.cfg.gfu)

        state = flowfields()
        for symbol, name in u.symbols.items():
            if name in quantities and quantities[name]:
                quantities.pop(name)
                state[symbol] = getattr(self.cfg, name)(u)
            elif symbol in quantities and quantities[symbol]:
                quantities.pop(symbol)
                state[symbol] = getattr(self.cfg, name)(u)

        return state
    
    def set_initial_conditions(self):
        ...

    def set_boundary_conditions(self):
        ...


class TaylorHood(IncompressibleFiniteElement):

    name: str = "taylor-hood"
    aliases = ('th',)

    def add_finite_element_spaces(self, spaces: dict[str, ngs.FESpace]):

        dirichlet = self.cfg.bcs.get_region(Wall, Inflow, as_pattern=True)
        U = ngs.VectorH1(self.mesh, order=self.order, dirichlet=dirichlet)
        P = ngs.H1(self.mesh, order=self.order-1)

        if self.cfg.bcs.has_condition(Periodic):
            U = ngs.Periodic(U)

        spaces['u'] = U
        spaces['p'] = P

    def add_transient_gridfunctions(self, gfus: dict[str, dict[str, ngs.GridFunction]]):
        gfus['u'] = self.cfg.time.scheme.get_transient_gridfunctions(self.u_gfu)

    def add_symbolic_stokes_form(self, blf: Integrals, lf: Integrals):
        bonus = self.cfg.optimizations.bonus_int_order

        u, v = self.TnT['u']
        p, q = self.TnT['p']

        U = self.get_incompressible_state([u, p])
        V = self.get_incompressible_state([v, q])

        blf['u']['stokes'] = ngs.InnerProduct(U.tau, V.eps) * ngs.dx(bonus_intorder=bonus.vol)
        blf['u']['stokes'] += -ngs.div(V.u) * U.p * ngs.dx
        blf['p']['stokes'] =  -ngs.div(U.u) * V.p * ngs.dx

        if not self.cfg.bcs.has_condition(Outflow):
            blf['p']['stokes'] += -1e-8 * U.p * V.p * ngs.dx

    def get_incompressible_state(self, u: ngs.CF) -> flowfields:

        if isinstance(u, ngs.GridFunction):
            u = u.components

        state = flowfields(u=u[0], p=u[1])
        state.grad_u = ngs.Grad(state.u)
        state.eps = self.cfg.strain_rate_tensor(state)
        state.tau = self.cfg.deviatoric_stress_tensor(state)

        return state

    def set_boundary_conditions(self) -> None:
        u = self.mesh.BoundaryCF({dom: bc.fields.u for dom, bc in self.cfg.bcs.to_pattern(Inflow).items()})
        self.gfu['u'].Set(u, ngs.BND)


# Define Solver

class IncompressibleSolver(SolverConfiguration):

    def __init__(self, mesh=None, **kwargs):
        bcs = BoundaryConditions(mesh, BCS)
        dcs = DomainConditions(mesh, DCS)
        super().__init__(mesh=mesh, bcs=bcs, dcs=dcs, **kwargs)

    @interface(default=TaylorHood)
    def fem(self, fem: TaylorHood):
        return fem

    @parameter(default=150)
    def reynolds_number(self, reynolds_number: ngs.Parameter):
        if reynolds_number <= 0:
            raise ValueError("Invalid Reynold number. Value has to be > 0!")

        return reynolds_number

    @interface(default=Constant)
    def dynamic_viscosity(self, dynamic_viscosity):
        r""" Sets the dynamic viscosity for the incompressible flow solver.

            :setter: Sets the dynamic viscosity, defaults to Inviscid
            :getter: Returns the dynamic viscosity
        """
        return dynamic_viscosity
    
    @configuration(default=False)
    def convection(self, convection: bool):
        return convection

    fem: TaylorHood
    reynolds_number: ngs.Parameter
    dynamic_viscosity: Constant | Powerlaw

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
