# %%
from __future__ import annotations

import logging
import ngsolve as ngs

# Import the necessary modules from dream
from dream import bla
from dream.config import (configuration,
                          parameter,
                          unique,
                          interface,
                          UniqueConfiguration,
                          InterfaceConfiguration,
                          ngsdict,
                          quantity,
                          equation)
from dream.pde import PDEConfiguration, FiniteElementMethod
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


class flowstate(ngsdict):

    u = quantity('velocity')
    p = quantity('pressure')
    tau = quantity('deviatoric_stress_tensor')
    eps = quantity('strain_rate_tensor')
    grad_u = quantity('velocity_gradient')
    g = quantity('gravity')
    f = quantity('force')

# %%
# Define a boundary conditions


class Inflow(Condition):

    name: str = "inflow"

    @configuration(default=None)
    def state(self, state):
        if state is None:
            return state
        elif bla.is_vector(state):
            return flowstate(u=state)
        else:
            return flowstate(**state)

    @state.getter_check
    def state(self) -> None:
        if self.data['state'] is None:
            raise ValueError("Inflow state not set!")


class Outflow(Condition):

    name: str = "outflow"


class Wall(Condition):

    name: str = "wall"


class IncompressibleBoundaryConditions(BoundaryConditions):
    ...


# Assign conditions
IncompressibleBoundaryConditions.register_condition(Periodic)
IncompressibleBoundaryConditions.register_condition(Inflow)
IncompressibleBoundaryConditions.register_condition(Outflow)
IncompressibleBoundaryConditions.register_condition(Wall)


class IncompressibleDomainConditions(DomainConditions):
    ...


IncompressibleDomainConditions.register_condition(Force)
IncompressibleDomainConditions.register_condition(Initial)

# %%
# Define properties like viscosity


class DynamicViscosity(InterfaceConfiguration, is_interface=True):

    @property
    def is_linear(self):
        return isinstance(self, Constant)

    def shear_rate(self, u: flowstate):
        eps = self.cfg.pde.strain_rate_tensor(u)
        return 2*ngs.sqrt(ngs.InnerProduct(eps, eps))

    def viscosity(self, u: flowstate):
        raise NotImplementedError()


class Constant(DynamicViscosity):

    name: str = "constant"

    def viscosity(self, u: flowstate):
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

    def viscosity(self, u: flowstate):
        return self.viscosity_ratio * self.shear_rate(u)**(self.powerlaw_exponent - 1)


# %%
# Define Finite Elements

class IncompressibleFiniteElement(FiniteElementMethod, is_interface=True):

    @property
    def u_TnT(self):
        return self.cfg.pde.TnT['u']

    @property
    def p_TnT(self):
        return self.cfg.pde.TnT['u']

    @property
    def u_gfu(self):
        return self.cfg.pde.gfus['u']

    def get_incompressible_state(self, u: ngs.CF) -> flowstate:
        raise NotImplementedError()

    def get_fields(self, quantities: dict[str, bool]) -> ngsdict:
        u = self.get_incompressible_state(self.cfg.pde.gfu)

        state = flowstate()
        for symbol, name in u.symbols.items():
            if name in quantities and quantities[name]:
                quantities.pop(name)
                state[symbol] = getattr(self.cfg.pde, name)(u)
            elif symbol in quantities and quantities[symbol]:
                quantities.pop(symbol)
                state[symbol] = getattr(self.cfg.pde, name)(u)

        return state


class TaylorHood(IncompressibleFiniteElement):

    name: str = "taylor-hood"
    aliases = ('th')

    def add_finite_element_spaces(self, spaces: dict[str, ngs.FESpace]):

        dirichlet = self.cfg.pde.bcs.get_region(Wall, Inflow, as_pattern=True)
        U_space = ngs.VectorH1(self.mesh, order=self.order, dirichlet=dirichlet)
        P_space = ngs.H1(self.mesh, order=self.order-1)

        if self.cfg.pde.bcs.has_condition(Periodic):
            U_space = ngs.Periodic(U_space)

        spaces['u'] = U_space
        spaces['p'] = P_space

    def add_transient_gridfunctions(self, gfus: dict[str, dict[str, ngs.GridFunction]]):
        gfus['u'] = self.cfg.time.scheme.get_transient_gridfunctions(self.u_gfu)

    def add_symbolic_forms(self, blfi: dict[str, ngs.comp.SumOfIntegrals],
                           blfe: dict[str, ngs.comp.SumOfIntegrals],
                           lf: dict[str, ngs.comp.SumOfIntegrals]):

        self.add_stokes_form(blfi, blfe, lf)

    def add_stokes_form(self, blfi: dict[str, ngs.comp.SumOfIntegrals],
                        blfe: dict[str, ngs.comp.SumOfIntegrals],
                        lf: dict[str, ngs.comp.SumOfIntegrals]):

        bonus = self.cfg.optimizations.bonus_int_order

        u, v = self.cfg.pde.fes.TnT()
        u = self.get_incompressible_state(u)
        v = self.get_incompressible_state(v)

        blfi['stokes'] = ngs.InnerProduct(u.tau, v.eps) * ngs.dx(bonus_intorder=bonus.vol)
        blfi['stokes'] += (-ngs.div(u.u) * v.p - ngs.div(v.u) * u.p) * ngs.dx

        if not self.cfg.pde.bcs.has_condition(Outflow):
            blfi['stokes'] += -1e-8 * u.p * v.p * ngs.dx

    def get_incompressible_state(self, u: ngs.CF) -> flowstate:

        if isinstance(u, ngs.GridFunction):
            u = u.components

        state = flowstate(u=u[0], p=u[1])
        state.grad_u = ngs.Grad(state.u)
        state.eps = self.cfg.pde.strain_rate_tensor(state)
        state.tau = self.cfg.pde.deviatoric_stress_tensor(state)

        return state

    def set_initial_conditions(self) -> None:
        raise NotImplementedError()

    def set_boundary_conditions(self) -> None:
        u = self.mesh.BoundaryCF({dom: bc.state.u for dom, bc in self.cfg.pde.bcs.to_pattern(Inflow).items()})
        self.u_gfu.Set(u, ngs.BND)


class IncompressibleFlowConfiguration(PDEConfiguration):

    name = "incompressible"

    def __init__(self, cfg=None, mesh=None, **kwargs):
        super().__init__(cfg, mesh, **kwargs)

        if mesh is not None:
            self.bcs = IncompressibleBoundaryConditions(self.mesh)
            self.dcs = IncompressibleDomainConditions(self.mesh)

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

    fem: TaylorHood
    reynolds_number: ngs.Parameter
    dynamic_viscosity: Constant

    @equation
    def velocity(self, u: flowstate):
        if u.u is not None:
            return u.u

    @equation
    def pressure(self, u: flowstate):
        if u.p is not None:
            return u.p

    @equation
    def viscosity(self, u: flowstate):
        return self.dynamic_viscosity.viscosity(u)

    @equation
    def deviatoric_stress_tensor(self, u: flowstate):

        Re = self.cfg.pde.reynolds_number

        mu = self.viscosity(u)
        strain = self.strain_rate_tensor(u)

        return 2 * mu/Re * strain

    @equation
    def strain_rate_tensor(self, u: flowstate):
        if u.eps is not None:
            return u.eps
        elif u.grad_u is not None:
            return 0.5 * (u.grad_u + u.grad_u.trans)
