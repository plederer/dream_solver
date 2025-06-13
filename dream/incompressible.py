r""" Dimensionless incompressible Navier-Stokes equations

We consider the dimensionless incompressible Navier-Stokes equations

.. math::
    \frac{\partial \vec{u}}{\partial t} + \vec{u} \cdot \nabla \vec{u} - \frac{1}{Re} \div{(\mat{\tau})} + \nabla p = 0

"""
from __future__ import annotations

import logging
import ngsolve as ngs

# Import the necessary modules from dream
import dream.bla as bla
from dream.config import (Configuration,
                          dream_configuration,
                          ngsdict,
                          quantity,
                          equation,
                          Integrals)

from dream.time import StationaryScheme, TimeSchemes, Scheme, StationaryRoutine, TransientRoutine
from dream.solver import FiniteElementMethod, SolverConfiguration
from dream.mesh import (BoundaryConditions,
                        DomainConditions,
                        Condition,
                        Periodic,
                        Initial)

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

# Define a boundary conditions


class Inflow(Condition):

    name: str = "inflow"

    def __init__(self, velocity: flowfields | ngs.CF):
        self.velocity = velocity
        super().__init__()

    @dream_configuration
    def velocity(self) -> flowfields:
        """ Returns the inflow velocity """
        return self._velocity

    @velocity.setter
    def velocity(self, velocity: ngsdict) -> None:
        if isinstance(velocity, flowfields):
            self._velocity = velocity.u
        elif isinstance(velocity, ngs.CF):
            self._velocity = velocity
        else:
            self._velocity = ngs.CF(tuple(velocity))


class Outflow(Condition):

    name: str = "outflow"


class Wall(Condition):

    name: str = "wall"


class Force(Condition):

    name: str = "force"

    def __init__(self, force: flowfields | ngs.CF):
        self.force = force
        super().__init__()

    @dream_configuration
    def force(self) -> flowfields:
        """ Returns the force vector """
        return self._force

    @force.setter
    def force(self, force: ngsdict) -> None:
        if isinstance(force, flowfields):
            self._force = force.u
        elif isinstance(force, ngs.CF):
            self._force = force
        else:
            self._force = ngs.CF(tuple(force))


BCS = [Inflow, Outflow, Wall, Periodic]
DCS = [Initial, Force]


class DynamicViscosity(Configuration, is_interface=True):

    root: IncompressibleSolver

    @property
    def is_linear(self):
        return isinstance(self, Constant)

    def shear_rate(self, u: flowfields):
        eps = self.root.strain_rate_tensor(u)
        return 2*ngs.sqrt(0.5 * ngs.InnerProduct(eps, eps))

    def kinematic_viscosity(self, u: flowfields):
        raise NotImplementedError("Overload this method in derived class!")


class Constant(DynamicViscosity):

    name: str = "constant"

    def kinematic_viscosity(self, u: flowfields):
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

    def kinematic_viscosity(self, u: flowfields):
        return self.viscosity_ratio * self.shear_rate(u)**(self.powerlaw_exponent - 2)


# Define solving schemes
class DirectScheme(StationaryScheme):

    root: IncompressibleSolver

    name: str = "direct"

    def assemble(self):

        fem = self.root.fem

        self.blf = ngs.BilinearForm(fem.fes)
        self.add_sum_of_integrals(self.blf, fem.blf)

        self.lf = ngs.LinearForm(fem.fes)
        self.add_sum_of_integrals(self.lf, fem.lf)

        self.lf.Assemble()

        if not self.root.dynamic_viscosity.is_linear or self.root.convection:
            # self.blf.Apply(fem.gfu.vec, self.lf.vec)
            self.root.nonlinear_solver.initialize(self.blf, self.lf.vec, fem.gfu)

        else:
            # Add dirichlet boundary conditions
            self.blf.Assemble()
            self.lf.vec.data -= self.blf.mat * fem.gfu.vec

    def solve(self):

        fem = self.root.fem

        if not self.root.dynamic_viscosity.is_linear or self.root.convection:
            for it in self.root.nonlinear_solver.solve():
                ...
        else:
            self.inv = self.root.linear_solver.inverse(self.blf, fem.fes)
            fem.gfu.vec.data += self.inv * self.lf.vec


class IMEX(TimeSchemes):

    root: IncompressibleSolver
    name: str = "imex"

    time_levels = ("n+1",)

    def assemble(self) -> None:

        if not self.root.convection:
            raise ValueError("The IMEX scheme only support convection terms!")

        fem = self.root.fem

        self.blf = ngs.BilinearForm(fem.fes)
        self.add_sum_of_integrals(self.blf, fem.blf, 'convection')

        self.stokes = ngs.BilinearForm(fem.fes)
        self.add_sum_of_integrals(self.stokes, fem.blf, 'mass', 'convection')

        self.convection = ngs.BilinearForm(fem.fes, nonassemble=True)
        for forms in fem.blf.values():
            for key, form in forms.items():
                if key == "convection":
                    self.convection += form

        self.lf = ngs.LinearForm(fem.fes)
        self.add_sum_of_integrals(self.lf, fem.lf)
        self.lf.Assemble()

        self.tmp = fem.gfu.vec.CreateVector()
        self.tmp[:] = 0.0

        if not self.root.dynamic_viscosity.is_linear:
            # self.blf.Apply(fem.gfu.vec, self.lf.vec)
            self.root.nonlinear_solver.initialize(self.blf, self.lf.vec, fem.gfu)

        else:
            # Add dirichlet boundary conditions
            self.blf.Assemble()
            self.stokes.Assemble()
            self.inv = self.root.linear_solver.inverse(self.blf, fem.fes)

    def add_symbolic_temporal_forms(self, blf: Integrals, lf: Integrals) -> None:
        u, v = self.root.fem.TnT['u']

        blf['u']['mass'] = ngs.InnerProduct(u/self.dt, v) * ngs.dx

    def solve_current_time_level(self, t: float | None = None):

        fem = self.root.fem

        if not self.root.dynamic_viscosity.is_linear:
            for it in self.root.nonlinear_solver.solve():
                ...
        else:

            self.convection.Apply(fem.gfu.vec, self.tmp)
            self.tmp.data += self.stokes.mat * fem.gfu.vec
            fem.gfu.vec.data -= self.inv * self.tmp

            logger.info(f"IMEX | t: {t}")

        yield None


# Define Finite Elements

class IncompressibleFiniteElement(FiniteElementMethod):

    root: IncompressibleSolver

    @dream_configuration
    def scheme(self) -> DirectScheme:
        """ Returns the scheme for the incompressible flow solver """
        return self._scheme

    @scheme.setter
    def scheme(self, scheme) -> None:
        if isinstance(self.root.time, StationaryRoutine):
            OPTIONS = [DirectScheme]
        elif isinstance(self.root.time, TransientRoutine):
            OPTIONS = [IMEX]
        else:
            raise ValueError("Invalid scheme for incompressible flow solver. Only stationary schemes are supported!")
        self._scheme = self._get_configuration_option(scheme, OPTIONS, Scheme)

    def add_symbolic_spatial_forms(self, blf: Integrals, lf: Integrals):

        self.add_symbolic_stokes_form(blf, lf)

        if self.root.convection:
            self.add_symbolic_convection_form(blf, lf)

        self.add_domain_conditions(blf, lf)

    def initialize_time_scheme_gridfunctions(self):
        super().initialize_time_scheme_gridfunctions('u')

    def get_solution_fields(self) -> flowfields:
        return self.get_incompressible_state(self.gfu)

    def add_symbolic_stokes_form(self, blf, lf):
        raise NotImplementedError("Overload this method in derived class!")

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

    def add_domain_conditions(self, blf, lf):

        _, v = self.TnT['u']

        doms = self.root.dcs.to_pattern()

        for dom, dc in doms.items():

            logger.debug(f"Adding domain condition {dc} on domain {dom}.")

            dom = self.mesh.Materials(dom)

            if isinstance(dc, Force):

                lf['u']['force'] = dc.force * v * ngs.dx(definedon=dom)

    def add_symbolic_convection_form(self, blf, lf):
        raise NotImplementedError("Overload this method in derived class!")

    def tang(self, u: ngs.CF) -> ngs.CF:
        """ Returns the tangential component of a vector field """
        n = self.mesh.normal
        return u - (u * n) * n


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

    def add_symbolic_convection_form(self, blf: Integrals, lf: Integrals):
        bonus = self.root.optimizations.bonus_int_order

        u, v = self.TnT['u']

        blf['u']['convection'] = ngs.InnerProduct(ngs.Grad(u) * u, v) * ngs.dx(bonus_intorder=bonus.vol)

    def set_boundary_conditions(self) -> None:

        inflows = {dom: bc.velocity for dom, bc in self.root.bcs.to_pattern(Inflow).items()}

        if inflows:
            u = self.mesh.BoundaryCF(inflows)
            self.gfus['u'].Set(u, ngs.BND)


class HDivHDG(IncompressibleFiniteElement):

    name: str = "HDivHDG"

    def add_finite_element_spaces(self, spaces: dict[str, ngs.FESpace]):

        bnds = self.root.bcs.get_region(Wall, Inflow, as_pattern=True)

        U = ngs.HDiv(self.mesh, order=self.order, dirichlet=bnds)
        Uhat = ngs.TangentialFacetFESpace(self.mesh, order=self.order, dirichlet=bnds)
        P = ngs.L2(self.mesh, order=self.order-1, lowest_order_wb=True)

        if self.root.bcs.has_condition(Periodic):
            U = ngs.Periodic(U)
            Uhat = ngs.Periodic(Uhat)

        spaces['u'] = U
        spaces['p'] = P
        spaces['uhat'] = Uhat

    def add_symbolic_stokes_form(self, blf: Integrals, lf: Integrals):
        bonus = self.root.optimizations.bonus_int_order

        u, v = self.TnT['u']
        uhat, vhat = self.TnT['uhat']
        p, q = self.TnT['p']

        U = self.get_incompressible_state([u, p])
        V = self.get_incompressible_state([v, q])

        n = self.mesh.normal
        h = self.mesh.meshsize
        alpha = 10

        Re = self.root.reynolds_number
        nu = self.root.kinematic_viscosity(U)
        mu = 2 * nu/Re

        blf['u']['stokes'] = ngs.InnerProduct(U.tau, V.eps) * ngs.dx(bonus_intorder=bonus.vol)
        blf['u']['stokes'] += -ngs.InnerProduct(U.tau * n, self.tang(V.u - vhat)
                                                ) * ngs.dx(element_boundary=True, bonus_intorder=bonus.vol)
        blf['u']['stokes'] += -ngs.InnerProduct(V.tau * n, self.tang(U.u - uhat)
                                                ) * ngs.dx(element_boundary=True, bonus_intorder=bonus.vol)
        blf['u']['stokes'] += mu * alpha * (self.order+1)**2 / h * ngs.InnerProduct(self.tang(V.u - vhat),
                                                                                    self.tang(U.u - uhat)) * ngs.dx(element_boundary=True, bonus_intorder=bonus.vol)

        blf['u']['stokes'] += -ngs.div(V.u) * U.p * ngs.dx
        blf['p']['stokes'] = -ngs.div(U.u) * V.p * ngs.dx

        if not self.root.bcs.has_condition(Outflow):
            blf['p']['stokes'] += -mu * 1e-10 * U.p * V.p * ngs.dx

    def add_symbolic_convection_form(self, blf: Integrals, lf: Integrals):
        bonus = self.root.optimizations.bonus_int_order

        u, v = self.TnT['u']
        p, q = self.TnT['p']
        uhat, vhat = self.TnT['uhat']

        U = self.get_incompressible_state([u, p])
        V = self.get_incompressible_state([v, q])

        n = self.mesh.normal

        Uup = ngs.IfPos(U.u * n, U.u, (U.u*n) * n + self.tang(2*uhat-U.u))

        blf['u']['convection'] = -ngs.InnerProduct(V.grad_u.trans * U.u, U.u) * ngs.dx(bonus_intorder=bonus.vol)
        blf['u']['convection'] += U.u*n * Uup * V.u * ngs.dx(element_boundary=True, bonus_intorder=bonus.vol)
        blf['uhat']['convection'] = self.tang(uhat-U.u) * vhat * ngs.dx(element_boundary=True, bonus_intorder=bonus.vol)
        
    def set_boundary_conditions(self) -> None:
        inflows = {dom: bc.velocity for dom, bc in self.root.bcs.to_pattern(Inflow).items()}

        if inflows:
            u = self.mesh.BoundaryCF(inflows)
            self.gfus['u'].Set(u, ngs.BND)
            self.gfus['uhat'].Set(self.tang(u), ngs.BND)


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
        OPTIONS = [TaylorHood, HDivHDG]
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
            raise ValueError("Invalid Reynolds number. Value has to be > 0!")

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

        self._convection = bool(convection)

    def get_solution_fields(self, *fields, default_fields=True) -> flowfields:

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
    def kinematic_viscosity(self, u: flowfields):
        return self.dynamic_viscosity.kinematic_viscosity(u)

    @equation
    def deviatoric_stress_tensor(self, u: flowfields):

        if u.tau is not None:
            return u.tau
        else:
            Re = self.reynolds_number

            nu = self.kinematic_viscosity(u)
            strain = self.strain_rate_tensor(u)

            return 2 * nu/Re * strain

    @equation
    def strain_rate_tensor(self, u: flowfields):
        if u.eps is not None:
            return u.eps
        elif u.grad_u is not None:
            return 0.5 * (u.grad_u + u.grad_u.trans)

    @equation
    def drag_coefficient(
            self, u: flowfields, uinf: flowfields, drag_direction: tuple[float, ...] = (1, 0),
            aera: float = 1.0) -> float:
        r""" Returns the definition of the drag coefficient. 
             Needs to be integrated over a surface, due to the inclusion of the boundary normal vector :math:`\bm{n}_{bnd}`.

            .. math::
                C_d = \frac{1}{\frac{1}{2} \rho_\infty |\bm{u}_\infty|^2 A} \bm{n}_{drag} \left(\mat{\tau} - p \mat{\I} \right) \bm{n}_{bnd}

            :param U: A dictionary containing the flow quantities
            :type U: flowfields
            :param Uinf: A dictionary containing the reference flow quantities
            :type Uinf: flowfields
            :param drag_direction: A container containing the drag direction :math:`\bm{n}_{drag}`
            :type drag_direction: tuple[float, ...]
            :param aera: The reference area :math:`A`
            :type aera: float
            :return: The drag coefficient
            :rtype: float
        """
        return self._get_aerodynamic_coefficient(u, uinf, drag_direction, aera)

    @equation
    def lift_coefficient(
            self, u: flowfields,  uinf: flowfields, lift_direction: tuple[float, ...] = (0, 1),
            aera: float = 1.0) -> float:
        r""" Returns the definition of the lift coefficient. 
             Needs to be integrated over a surface, due to the inclusion of the boundary normal vector :math:`\bm{n}_{bnd}`.

            .. math::
                C_l = \frac{1}{\frac{1}{2} \rho_\infty |\bm{u}_\infty|^2 A} \bm{n}_{lift} \left(\mat{\tau} - p \mat{\I} \right) \bm{n}_{bnd} 

            :param U: A dictionary containing the flow quantities
            :type U: flowfields
            :param dU: A dictionary containing the gradients of the flow quantities for the evaluation of the viscous stress tensor
            :type dU: flowfields
            :param Uinf: A dictionary containing the reference flow quantities
            :type Uinf: flowfields
            :param lift_direction: A container containing the lift direction :math:`\bm{n}_{lift}`
            :type lift_direction: tuple[float, ...]
            :param aera: The reference area :math:`A`
            :type aera: float
            :return: The drag coefficient
            :rtype: float
        """
        return self._get_aerodynamic_coefficient(u, uinf, lift_direction, aera)

    def _get_aerodynamic_coefficient(
            self, u: flowfields, uref: flowfields, direction: tuple[float, ...],
            aera: float) -> float:

        sigma = -self.pressure(u) * ngs.Id(self.mesh.dim)
        if not self.dynamic_viscosity.is_inviscid:
            sigma += self.deviatoric_stress_tensor(u)

        return bla.inner(sigma * self.mesh.normal, bla.unit_vector(direction))/(0.5 * aera * uref.u**2)
