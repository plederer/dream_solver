from __future__ import annotations
import typing
import logging

import ngsolve as ngs

from dream.config import dream_configuration, Integrals
from dream.time import Scheme, Scheme, StationaryRoutine, TransientRoutine
from dream.solver import FiniteElementMethod

from .config import flowfields, Force, Inflow, Outflow, Wall, Periodic
from .time import StationaryScheme, IMEX

if typing.TYPE_CHECKING:
    from dream.incompressible_flow.solver import IncompressibleFlowSolver

logger = logging.getLogger(__name__)


class IncompressibleFiniteElement(FiniteElementMethod):

    root: IncompressibleFlowSolver

    def __init__(self, mesh, root=None, **default):

        DEFAULT = {
            'bonus_int_order': ('convection', 'diffusion'),
        }

        DEFAULT.update(default)

        super().__init__(mesh, root, **DEFAULT)

    @dream_configuration
    def scheme(self) -> StationaryScheme | IMEX:
        """ Returns the scheme for the incompressible flow solver """
        return self._scheme

    @scheme.setter
    def scheme(self, scheme) -> None:
        if isinstance(self.root.time, StationaryRoutine):
            OPTIONS = [StationaryScheme]
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

        for dom, dc in self.root.dcs.items():

            logger.debug(f"Adding domain condition {dc} on domain {dom}.")

            dom = self.mesh.Materials(dom)

            if isinstance(dc, Force):

                lf['u']['force'] = dc.force * v * ngs.dx(definedon=dom)

    def add_symbolic_convection_form(self, blf, lf):
        raise NotImplementedError("Overload this method in derived class!")

    def tangential_projection(self, u: ngs.CF) -> ngs.CF:
        """ Returns the tangential component of a vector field """
        n = self.mesh.normal
        return u - (u * n) * n


class TaylorHood(IncompressibleFiniteElement):

    name: str = "taylor-hood"

    def __init__(self, mesh, root=None, **default):

        DEFAULT = {
            'static_condensation': True,
        }

        DEFAULT.update(default)

        super().__init__(mesh, root, **DEFAULT)

    def add_finite_element_spaces(self, spaces: dict[str, ngs.FESpace]):

        dirichlet = self.root.bcs.get_boundary_names(Wall, Inflow)
        U = ngs.VectorH1(self.mesh, order=self.order, dirichlet=dirichlet)
        P = ngs.H1(self.mesh, order=self.order-1)

        if Periodic in self.root.bcs:
            U = ngs.Periodic(U)

        spaces['u'] = U
        spaces['p'] = P

    def add_symbolic_stokes_form(self, blf: Integrals, lf: Integrals):
        bonus = self.bonus_int_order['diffusion']

        u, v = self.TnT['u']
        p, q = self.TnT['p']

        U = self.get_incompressible_state([u, p])
        V = self.get_incompressible_state([v, q])

        blf['u']['stokes'] = ngs.InnerProduct(U.tau, V.eps) * ngs.dx(bonus_intorder=bonus['vol'])
        blf['u']['stokes'] += -ngs.div(V.u) * U.p * ngs.dx
        blf['p']['stokes'] = -ngs.div(U.u) * V.p * ngs.dx

        if Outflow not in self.root.bcs:
            blf['p']['stokes'] += -1e-8 * U.p * V.p * ngs.dx

    def add_symbolic_convection_form(self, blf: Integrals, lf: Integrals):
        bonus = self.bonus_int_order['convection']

        u, v = self.TnT['u']

        blf['u']['convection'] = ngs.InnerProduct(ngs.Grad(u) * u, v) * ngs.dx(bonus_intorder=bonus['vol'])

    def set_boundary_conditions(self) -> None:

        inflows = {dom: bc.velocity for dom, bc in self.root.bcs.items(Inflow)}

        if inflows:
            u = self.mesh.BoundaryCF(inflows)
            self.gfus['u'].Set(u, ngs.BND)


class HDivHDG(IncompressibleFiniteElement):

    name: str = "HDivHDG"

    def __init__(self, mesh, root=None, **default):

        DEFAULT = {
            'static_condensation': True,
        }

        DEFAULT.update(default)

        super().__init__(mesh, root, **DEFAULT)

    def add_finite_element_spaces(self, spaces: dict[str, ngs.FESpace]):

        dirichlet = self.root.bcs.get_boundary_names(Wall, Inflow)
        U = ngs.HDiv(self.mesh, order=self.order, dirichlet=dirichlet)
        Uhat = ngs.TangentialFacetFESpace(self.mesh, order=self.order, dirichlet=dirichlet)
        P = ngs.L2(self.mesh, order=self.order-1, lowest_order_wb=True)

        if Periodic in self.root.bcs:
            U = ngs.Periodic(U)
            Uhat = ngs.Periodic(Uhat)

        spaces['u'] = U
        spaces['p'] = P
        spaces['uhat'] = Uhat

    def add_symbolic_stokes_form(self, blf: Integrals, lf: Integrals):

        bonus = self.bonus_int_order['diffusion']

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

        blf['u']['stokes'] = ngs.InnerProduct(U.tau, V.eps) * ngs.dx(bonus_intorder=bonus['vol'])
        blf['u']['stokes'] += -ngs.InnerProduct(U.tau * n, self.tangential_projection(V.u - vhat)
                                                ) * ngs.dx(element_boundary=True, bonus_intorder=bonus['bnd'])
        blf['u']['stokes'] += -ngs.InnerProduct(V.tau * n, self.tangential_projection(U.u - uhat)
                                                ) * ngs.dx(element_boundary=True, bonus_intorder=bonus['bnd'])
        blf['u']['stokes'] += mu * alpha * (self.order+1)**2 / h * ngs.InnerProduct(self.tangential_projection(V.u - vhat),
                                                                                    self.tangential_projection(U.u - uhat)) * ngs.dx(element_boundary=True, bonus_intorder=bonus['bnd'])

        blf['u']['stokes'] += -ngs.div(V.u) * U.p * ngs.dx
        blf['p']['stokes'] = -ngs.div(U.u) * V.p * ngs.dx

        if Outflow not in self.root.bcs:
            blf['p']['stokes'] += -mu * 1e-10 * U.p * V.p * ngs.dx

    def add_symbolic_convection_form(self, blf: Integrals, lf: Integrals):
        bonus = self.bonus_int_order['convection']

        u, v = self.TnT['u']
        p, q = self.TnT['p']
        uhat, vhat = self.TnT['uhat']

        U = self.get_incompressible_state([u, p])
        V = self.get_incompressible_state([v, q])

        n = self.mesh.normal

        Uup = ngs.IfPos(U.u * n, U.u, (U.u*n) * n + self.tangential_projection(2*uhat-U.u))

        blf['u']['convection'] = -ngs.InnerProduct(V.grad_u.trans * U.u, U.u) * ngs.dx(bonus_intorder=bonus['vol'])
        blf['u']['convection'] += U.u*n * Uup * V.u * ngs.dx(element_boundary=True, bonus_intorder=bonus['bnd'])
        blf['uhat']['convection'] = self.tangential_projection(
            uhat-U.u) * vhat * ngs.dx(element_boundary=True, bonus_intorder=bonus['bnd'])

    def set_boundary_conditions(self) -> None:

        inflows = {dom: bc.velocity for dom, bc in self.root.bcs.items(Inflow)}

        if inflows:
            u = self.mesh.BoundaryCF(inflows)
            self.gfus['u'].Set(u, ngs.BND)
            self.gfus['uhat'].Set(self.tangential_projection(u), ngs.BND)
