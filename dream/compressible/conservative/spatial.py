""" Definitions of conservative spatial discretizations """
from __future__ import annotations
import logging
import ngsolve as ngs
import typing
import dream.bla as bla

from dream.time import TransientRoutine, PseudoTimeSteppingRoutine
from dream.config import Configuration, dream_configuration, Integrals
from dream.mesh import SpongeLayer, PSpongeLayer, Periodic, Initial
from dream.compressible.config import (flowfields,
                                       CompressibleFiniteElementMethod,
                                       FarField,
                                       Outflow,
                                       InviscidWall,
                                       Symmetry,
                                       IsothermalWall,
                                       AdiabaticWall,
                                       CBC)

from .time import TimeSchemes, ImplicitEuler, BDF2, SDIRK22, SDIRK33, SDIRK54, DIRK34_LDD, DIRK43_WSO2, ExplicitEuler, SSPRK3, CRK4, IMEXRK_ARS443

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from ..solver import CompressibleFlowSolver


# --- Conservative --- #

class MixedMethod(Configuration, is_interface=True):

    root: CompressibleFlowSolver

    @property
    def fem(self) -> ConservativeFiniteElementMethod:
        return self.root.fem

    @property
    def TnT(self) -> dict[str, tuple[ngs.comp.ProxyFunction, ...]]:
        return self.root.fem.TnT

    @property
    def gfu(self) -> dict[str, ngs.GridFunction]:
        return self.root.fem.gfus

    def add_mixed_finite_element_spaces(self, fes: dict[str, ngs.FESpace]) -> None:
        raise NotImplementedError("Mixed method must implement get_mixed_finite_element_spaces method!")

    def add_mixed_form(self, blf: Integrals, lf: Integrals) -> None:
        pass

    def get_cbc_viscous_terms(self, bc: CBC) -> ngs.CF:
        return ngs.CF(tuple(0 for _ in range(self.mesh.dim + 2)))

    def get_diffusive_stabilisation_matrix(self, U: flowfields) -> bla.MATRIX:
        Re = self.root.scaling.reference_reynolds_number
        Pr = self.root.prandtl_number
        mu = self.root.viscosity(U)

        tau_d = [0] + [1 for _ in range(self.mesh.dim)] + [1/Pr]
        return bla.diagonal(tau_d) * mu / Re


class Inactive(MixedMethod):

    name: str = "inactive"

    def add_mixed_finite_element_spaces(self, fes: dict[str, ngs.FESpace]) -> None:

        if not self.root.dynamic_viscosity.is_inviscid:
            raise TypeError(f"Viscous configuration requires mixed method!")


class StrainHeat(MixedMethod):

    name: str = "strain_heat"

    def add_mixed_finite_element_spaces(self, fes: dict[str, ngs.FESpace]) -> None:

        if self.root.dynamic_viscosity.is_inviscid:
            raise TypeError(f"Inviscid configuration does not require mixed method!")

        dim = 4*self.mesh.dim - 3
        order = self.fem.order

        Q = ngs.L2(self.mesh, order=order)
        Q = self.root.dcs.reduce_psponge_layers_order_elementwise(Q)

        fes['Q'] = Q**dim

    def add_mixed_form(self, blf: Integrals, lf: Integrals) -> None:

        if self.mesh.dim == 3:
            raise NotImplementedError("StrainHeat method is not implemented for domain dimension 3!")

        bonus = self.root.optimizations.bonus_int_order

        U, _ = self.fem.method.TnT['U']
        Uhat, _ = self.fem.method.TnT['Uhat']
        Q, P = self.TnT['Q']

        U = self.fem.method.get_conservative_fields(U)
        Uhat = self.fem.method.get_conservative_fields(Uhat)

        gradient_P = ngs.grad(P)
        Q = self.get_mixed_fields(Q)
        P = self.get_mixed_fields(P)

        dev_zeta = P.eps - bla.trace(P.eps) * ngs.Id(self.mesh.dim)/3
        div_dev_zeta = ngs.CF((gradient_P[0, 0] + gradient_P[1, 1], gradient_P[1, 0] + gradient_P[2, 1]))
        div_dev_zeta -= 1/3 * ngs.CF((gradient_P[0, 0] + gradient_P[2, 0], gradient_P[0, 1] + gradient_P[2, 1]))

        blf['Q']['mixed'] = ngs.InnerProduct(Q.eps, P.eps) * ngs.dx
        blf['Q']['mixed'] += ngs.InnerProduct(U.u, div_dev_zeta) * ngs.dx(bonus_intorder=bonus.vol)
        blf['Q']['mixed'] -= ngs.InnerProduct(Uhat.u, dev_zeta*self.mesh.normal) * \
            ngs.dx(element_boundary=True, bonus_intorder=bonus.bnd)

        div_xi = gradient_P[3, 0] + gradient_P[4, 1]
        blf['Q']['mixed'] += ngs.InnerProduct(Q.grad_T, P.grad_T) * ngs.dx
        blf['Q']['mixed'] += ngs.InnerProduct(U.T, div_xi) * ngs.dx(bonus_intorder=bonus.vol)
        blf['Q']['mixed'] -= ngs.InnerProduct(Uhat.T*self.mesh.normal, P.grad_T) * \
            ngs.dx(element_boundary=True, bonus_intorder=bonus.bnd)

    def get_cbc_viscous_terms(self, bc: CBC):

        if self.mesh.dim == 3:
            raise NotImplementedError("StrainHeat method CBC is not implemented for domain dimension 3!")

        Q, _ = self.TnT['Q']
        U, _ = self.fem.method.TnT['U']

        U = self.fem.method.get_conservative_fields(U)
        Q = self.get_mixed_fields(Q)

        t = self.mesh.tangential
        n = self.mesh.normal

        R = ngs.CF((n, t), dims=(2, 2)).trans

        grad_Q = ngs.grad(Q.Q) * n
        grad_EPS = R.trans * ngs.CF((grad_Q[0], grad_Q[1], grad_Q[1], grad_Q[2]), dims=(2, 2)) * R
        grad_q = ngs.CF((grad_Q[3], grad_Q[4]))

        if bc.target == "outflow":
            grad_q = R.trans * ngs.CF((grad_Q[3], grad_Q[4]))
            grad_EPS = R * ngs.CF((grad_EPS[0], 0, 0, grad_EPS[2]), dims=(2, 2)) * R.trans
            grad_q = R * ngs.CF((0, grad_q[1]))
        else:
            grad_EPS = R * ngs.CF((0, grad_EPS[1], grad_EPS[1], grad_EPS[2]), dims=(2, 2)) * R.trans

        grad_Q = ngs.CF((grad_EPS[0], grad_EPS[1], grad_EPS[2], grad_q[0], grad_q[1]))

        S = self.get_conservative_diffusive_jacobian(U, Q, t) * (ngs.grad(U.U) * t)
        S += self.get_conservative_diffusive_jacobian(U, Q, n) * (ngs.grad(U.U) * n)
        S += self.get_mixed_diffusive_jacobian(U, t) * (ngs.grad(Q.Q) * t)
        S += self.get_mixed_diffusive_jacobian(U, n) * grad_Q

        return S

    def get_mixed_fields(self, Q: ngs.CF):

        dim = self.mesh.dim

        if isinstance(Q, ngs.GridFunction):
            Q = Q.components

        Q_ = flowfields()
        Q_.eps = bla.symmetric_matrix_from_vector(Q[:3*dim - 3])
        Q_.grad_T = Q[3*dim - 3:]

        if isinstance(Q, ngs.comp.ProxyFunction):
            Q_.Q = Q

        return Q_

    def get_conservative_diffusive_jacobian_x(self, U: flowfields, Q: flowfields):

        if self.mesh.dim == 3:
            raise NotImplementedError("StrainHeat method is not implemented for domain dimension 3!")

        rho = self.root.density(U)
        stess_tensor = self.root.deviatoric_stress_tensor(U, Q)
        txx, txy = stess_tensor[0, 0], stess_tensor[0, 1]
        ux, uy = U.u

        A = ngs.CF((
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            -txx*ux/rho - txy*uy/rho, txx/rho, txy/rho, 0
        ), dims=(4, 4))

        return A

    def get_conservative_diffusive_jacobian_y(self, U: flowfields, Q: flowfields):

        if self.mesh.dim == 3:
            raise NotImplementedError("StrainHeat method is not implemented for domain dimension 3!")

        rho = self.root.density(U)
        stess_tensor = self.root.deviatoric_stress_tensor(U, Q)
        tyx, tyy = stess_tensor[1, 0], stess_tensor[1, 1]
        ux, uy = U.u

        B = ngs.CF((
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            -tyx*ux/rho - tyy*uy/rho, tyx/rho, tyy/rho, 0
        ), dims=(4, 4))

        return B

    def get_conservative_diffusive_jacobian(
            self, U: flowfields, Q: flowfields, unit_vector: ngs.CF) -> ngs.CF:
        if self.mesh.dim == 3:
            raise NotImplementedError("StrainHeat method is not implemented for domain dimension 3!")

        unit_vector = bla.as_vector(unit_vector)

        A = self.get_conservative_diffusive_jacobian_x(U, Q)
        B = self.get_conservative_diffusive_jacobian_y(U, Q)
        return A * unit_vector[0] + B * unit_vector[1]

    def get_mixed_diffusive_jacobian_x(self, U: flowfields):

        if self.mesh.dim == 3:
            raise NotImplementedError("StrainHeat method is not implemented for domain dimension 3!")

        Re = self.root.scaling.reference_reynolds_number
        Pr = self.root.prandtl_number
        mu = self.root.viscosity(U)

        ux, uy = U.u

        A = mu/Re * ngs.CF((
            0, 0, 0, 0, 0,
            2, 0, 0, 0, 0,
            0, 2, 0, 0, 0,
            2*ux, 2*uy, 0, 1/Pr, 0
        ), dims=(4, 5))

        return A

    def get_mixed_diffusive_jacobian_y(self, U: flowfields):

        if self.mesh.dim == 3:
            raise NotImplementedError("StrainHeat method is not implemented for domain dimension 3!")

        Re = self.root.scaling.reference_reynolds_number
        Pr = self.root.prandtl_number
        mu = self.root.viscosity(U)

        ux, uy = U.u

        B = mu/Re * ngs.CF((
            0, 0, 0, 0, 0,
            0, 2, 0, 0, 0,
            0, 0, 2, 0, 0,
            0, 2*ux, 2*uy, 0, 1/Pr
        ), dims=(4, 5))

        return B

    def get_mixed_diffusive_jacobian(self, U: flowfields, unit_vector: ngs.CF) -> ngs.CF:

        if self.mesh.dim == 3:
            raise NotImplementedError("StrainHeat method is not implemented for domain dimension 3!")

        unit_vector = bla.as_vector(unit_vector)
        A = self.get_mixed_diffusive_jacobian_x(U)
        B = self.get_mixed_diffusive_jacobian_y(U)
        return A * unit_vector[0] + B * unit_vector[1]


class Gradient(MixedMethod):

    name: str = "gradient"

    def add_mixed_finite_element_spaces(self, fes: dict[str, ngs.FESpace]) -> None:

        if self.root.dynamic_viscosity.is_inviscid:
            raise TypeError(f"Inviscid configuration does not require mixed method!")

        dim = self.mesh.dim + 2
        order = self.fem.order

        Q = ngs.VectorL2(self.mesh, order=order)
        Q = self.root.dcs.reduce_psponge_layers_order_elementwise(Q)

        fes['Q'] = Q**dim

    def add_mixed_form(self, blf: Integrals, lf: Integrals) -> None:

        Q, P = self.TnT['Q']
        U, _ = self.fem.method.TnT['U']
        Uhat, _ = self.fem.method.TnT['Uhat']

        blf['Q']['mixed'] = ngs.InnerProduct(Q, P) * ngs.dx
        blf['Q']['mixed'] += ngs.InnerProduct(U, ngs.div(P)) * ngs.dx
        blf['Q']['mixed'] -= ngs.InnerProduct(Uhat, P*self.mesh.normal) * ngs.dx(element_boundary=True)

    def get_mixed_fields(self, Q: ngs.CoefficientFunction):

        if isinstance(Q, ngs.GridFunction):
            Q = Q.components

        Q_ = flowfields()
        Q_.grad_rho = Q[0]
        Q_.grad_rho_u = Q[slice(1, self.mesh.dim + 1)]
        Q_.grad_rho_E = Q[self.mesh.dim + 1]

        if isinstance(Q, ngs.comp.ProxyFunction):
            Q_.Q = Q

        return Q_


class ConservativeMethod(Configuration, is_interface=True):

    root: CompressibleFlowSolver

    @property
    def TnT(self) -> dict[str, tuple[ngs.comp.ProxyFunction, ...]]:
        return self.root.fem.TnT

    @property
    def gfu(self) -> dict[str, ngs.GridFunction]:
        return self.root.fem.gfus

    def get_time_scheme_spaces(self) -> dict[str, ngs.FESpace]:
        return {'U': self.root.fem.spaces['U']}

    def get_conservative_fields(self, U: ngs.CoefficientFunction, with_gradients: bool = False) -> flowfields:

        if isinstance(U, ngs.GridFunction):
            U = U.components

        U_ = flowfields()
        U_.rho = U[0]
        U_.rho_u = U[slice(1, self.mesh.dim + 1)]
        U_.rho_E = U[self.mesh.dim + 1]

        U_.u = self.root.velocity(U_)
        U_.rho_Ek = self.root.kinetic_energy(U_)
        U_.rho_Ei = self.root.inner_energy(U_)
        U_.p = self.root.pressure(U_)
        U_.T = self.root.temperature(U_)
        U_.c = self.root.speed_of_sound(U_)

        if isinstance(U, ngs.comp.ProxyFunction):
            U_.U = U

        if isinstance(U, ngs.GridFunction) and with_gradients:
            dU = ngs.grad(U)

            U_.grad_rho = dU[0, :]
            U_.grad_rho_u = dU[slice(1, self.mesh.dim + 1), :]
            U_.grad_rho_E = dU[self.mesh.dim + 1, :]

            U_.grad_u = self.root.velocity_gradient(U_, U_)
            U_.grad_rho_Ek = self.root.kinetic_energy_gradient(U_, U_)
            U_.grad_rho_Ei = self.root.inner_energy_gradient(U_, U_)
            U_.grad_p = self.root.pressure_gradient(U_, U_)
            U_.grad_T = self.root.temperature_gradient(U_, U_)

        return U_

    def set_initial_conditions(self, U: ngs.CF = None):

        if U is None:
            U = self.mesh.MaterialCF({dom: ngs.CF(
                (self.root.density(dc.fields),
                 self.root.momentum(dc.fields),
                 self.root.energy(dc.fields))) for dom, dc in self.root.dcs.to_pattern(Initial).items()})

        self.gfu['U'].Set(U)

    def get_domain_boundary_mask(self) -> ngs.GridFunction:
        """ 
        Returns a Gridfunction that is 0 on the domain boundaries and 1 on the domain interior.
        """

        fes = ngs.FacetFESpace(self.mesh, order=0)
        mask = ngs.GridFunction(fes, name="mask")
        mask.vec[:] = 0

        bnd_dofs = fes.GetDofs(self.mesh.Boundaries(self.root.bcs.get_domain_boundaries(True)))
        mask.vec[~bnd_dofs] = 1

        return mask


class DG(ConservativeMethod):

    name: str = "dg"

    @dream_configuration
    def scheme(self) -> ExplicitEuler | SSPRK3 | CRK4:
        return self._scheme

    @scheme.setter
    def scheme(self, scheme: TimeSchemes) -> None:

        if not isinstance(self.root.time, TransientRoutine):
            raise TypeError("DG method only supports transient time routines!")

        OPTIONS = [ExplicitEuler, SSPRK3, CRK4]
        self._scheme = self._get_configuration_option(scheme, OPTIONS, TimeSchemes)

    def add_finite_element_spaces(self, fes: dict[str, ngs.FESpace]) -> None:

        order = self.root.fem.order
        dim = self.mesh.dim + 2

        U = ngs.L2(self.mesh, order=order, dgjumps=True)

        fes['U'] = U**dim

    def add_convection_form(self,
                            blf: dict[str, ngs.comp.SumOfIntegrals],
                            lf:  dict[str, ngs.comp.SumOfIntegrals]):

        # Extract the bonus integration order, if specified.
        bonus = self.root.optimizations.bonus_int_order

        # Obtain the relevant test and trial functions. Notice, the solution "U"
        # is assumed to be an unknown in the bilinear form, despite being explicit
        # in time. This works, because we invoke the "Apply" function when solving.
        U, V = self.TnT['U']

        # Get a mask that is nonzero (unity) for only the internal faces.
        mask = self.get_domain_boundary_mask()

        # Current/owned solution.
        Ui = self.get_conservative_fields(U)

        # Neighboring solution.
        Uj = self.get_conservative_fields(U.Other())

        # Compute the flux of the solution on the volume elements.
        F = self.root.get_convective_flux(Ui)

        # Compute the flux on the surface of an element, in the normal direction.
        Fn = self.root.riemann_solver.get_convective_numerical_flux_dg(Ui, Uj, self.mesh.normal)

        # Assemble the explicit bilinear form, keeping in mind this is placed on the RHS.
        blf['U']['convection'] = bla.inner(F, ngs.grad(V)) * ngs.dx(bonus_intorder=bonus.vol)
        blf['U']['convection'] += -mask*bla.inner(Fn, V) * ngs.dx(element_boundary=True, bonus_intorder=bonus.bnd)

    def add_boundary_conditions(self,
                                blf: dict[str, ngs.comp.SumOfIntegrals],
                                lf:  dict[str, ngs.comp.SumOfIntegrals]):

        bnds = self.root.bcs.to_pattern()

        for bnd, bc in bnds.items():

            logger.debug(f"Adding boundary condition {bc} on boundary {bnd}.")

            # NOTE: for now, we only implement periodic conditions.
            if isinstance(bc, Periodic):
                continue

            else:
                raise TypeError(f"Boundary condition {bc} not implemented in {self}!")

    def add_domain_conditions(self,
                              blf: dict[str, ngs.comp.SumOfIntegrals],
                              lf:  dict[str, ngs.comp.SumOfIntegrals]):

        doms = self.root.dcs.to_pattern()

        for dom, dc in doms.items():

            logger.debug(f"Adding domain condition {dc} on domain {dom}.")

            if isinstance(dc, Initial):
                continue

            else:
                raise TypeError(f"Domain condition {dc} not implemented in {self}!")



class HDG(ConservativeMethod):

    name: str = "hdg"

    @dream_configuration
    def scheme(self) -> ImplicitEuler | BDF2 | IMEXRK_ARS443 | SDIRK22 | SDIRK33 | SDIRK54 | DIRK34_LDD | DIRK43_WSO2:
        """ Time scheme for the HDG method depending on the choosen time routine.

            :getter: Returns the time scheme
            :setter: Sets the time scheme
        """
        return self._scheme

    @scheme.setter
    def scheme(self, scheme: TimeSchemes) -> None:
        if isinstance(self.root.time, TransientRoutine):
            OPTIONS = [ImplicitEuler, BDF2, IMEXRK_ARS443, SDIRK22, SDIRK33, SDIRK54, DIRK34_LDD, DIRK43_WSO2]
        elif isinstance(self.root.time, PseudoTimeSteppingRoutine):
            OPTIONS = [ImplicitEuler, BDF2]
        else:
            raise TypeError("HDG method only supports transient or pseudo time stepping routines!")
        self._scheme = self._get_configuration_option(scheme, OPTIONS, TimeSchemes)

    @property
    def mixed_method(self) -> MixedMethod:
        return self.root.fem.mixed_method

    def add_finite_element_spaces(self, fes: dict[str, ngs.FESpace]) -> None:

        order = self.root.fem.order
        dim = self.mesh.dim + 2

        U = ngs.L2(self.mesh, order=order)
        Uhat = ngs.FacetFESpace(self.mesh, order=order)

        psponge_layers = self.root.dcs.to_pattern(PSpongeLayer)
        if psponge_layers:
            U = self.root.dcs.reduce_psponge_layers_order_elementwise(U, psponge_layers)
            Uhat = self.root.dcs.reduce_psponge_layers_order_facetwise(Uhat, psponge_layers)

        if self.root.bcs.has_condition(Periodic):
            Uhat = ngs.Periodic(Uhat)

        fes['U'] = U**dim
        fes['Uhat'] = Uhat**dim

    def add_convection_form(self, blf: Integrals, lf: Integrals):

        bonus = self.root.optimizations.bonus_int_order

        mask = self.get_domain_boundary_mask()

        U, V = self.TnT['U']
        Uhat, Vhat = self.TnT['Uhat']

        U = self.get_conservative_fields(U)
        Uhat = self.get_conservative_fields(Uhat)

        F = self.root.get_convective_flux(U)
        Fn = self.get_convective_numerical_flux(U, Uhat, self.mesh.normal)

        blf['U']['convection'] = -bla.inner(F, ngs.grad(V)) * ngs.dx(bonus_intorder=bonus.vol)
        blf['U']['convection'] += bla.inner(Fn, V) * ngs.dx(element_boundary=True, bonus_intorder=bonus.bnd)
        blf['Uhat']['convection'] = -mask * bla.inner(Fn,
                                                      Vhat) * ngs.dx(element_boundary=True, bonus_intorder=bonus.bnd)

    def add_diffusion_form(self, blf: Integrals, lf: Integrals):

        bonus = self.root.optimizations.bonus_int_order

        mask = self.get_domain_boundary_mask()

        U, V = self.TnT['U']
        Uhat, Vhat = self.TnT['Uhat']
        Q, _ = self.mixed_method.TnT['Q']

        U = self.get_conservative_fields(U)
        Uhat = self.get_conservative_fields(Uhat)
        Q = self.root.fem.mixed_method.get_mixed_fields(Q)

        G = self.root.get_diffusive_flux(U, Q)
        Gn = self.get_diffusive_numerical_flux(U, Uhat, Q, self.mesh.normal)

        blf['U']['diffusion'] = ngs.InnerProduct(G, ngs.grad(V)) * ngs.dx(bonus_intorder=bonus.vol)
        blf['U']['diffusion'] -= ngs.InnerProduct(Gn, V) * ngs.dx(element_boundary=True, bonus_intorder=bonus.bnd)
        blf['Uhat']['diffusion'] = mask * ngs.InnerProduct(Gn,
                                                           Vhat) * ngs.dx(element_boundary=True, bonus_intorder=bonus.bnd)

    def add_boundary_conditions(self, blf: Integrals, lf: Integrals):

        bnds = self.root.bcs.to_pattern()

        for bnd, bc in bnds.items():

            logger.debug(f"Adding boundary condition {bc} on boundary {bnd}.")

            if isinstance(bc, FarField):
                self.add_farfield_formulation(blf, lf, bc, bnd)

            elif isinstance(bc, CBC):
                self.add_cbc_formulation(blf, lf, bc, bnd)

            elif isinstance(bc, Outflow):
                self.add_outflow_formulation(blf, lf, bc, bnd)

            elif isinstance(bc, (InviscidWall, Symmetry)):
                self.add_inviscid_wall_formulation(blf, lf, bc, bnd)

            elif isinstance(bc, IsothermalWall):
                self.add_isothermal_wall_formulation(blf, lf, bc, bnd)

            elif isinstance(bc, Periodic):
                continue

            elif isinstance(bc, AdiabaticWall):
                self.add_adiabatic_wall_formulation(blf, lf, bc, bnd)

            else:
                raise TypeError(f"Boundary condition {bc} not implemented in {self}!")

    def add_domain_conditions(self, blf: Integrals, lf: Integrals):

        doms = self.root.dcs.to_pattern()

        for dom, dc in doms.items():

            logger.debug(f"Adding domain condition {dc} on domain {dom}.")

            if isinstance(dc, SpongeLayer):
                self.add_sponge_layer_formulation(blf, lf, dc, dom)

            elif isinstance(dc, PSpongeLayer):
                self.add_psponge_layer_formulation(blf, lf, dc, dom)

            elif isinstance(dc, Initial):
                continue

            else:
                raise TypeError(f"Domain condition {dc} not implemented in {self}!")

    def add_farfield_formulation(self, blf: Integrals, lf: Integrals, bc: FarField, bnd: str):

        bonus = self.root.optimizations.bonus_int_order
        dS = ngs.ds(skeleton=True, definedon=self.mesh.Boundaries(bnd), bonus_intorder=bonus.bnd)

        U, _ = self.TnT['U']
        Uhat, Vhat = self.TnT['Uhat']
        Uhat = self.get_conservative_fields(Uhat)

        U_infty = ngs.CF(
            (self.root.density(bc.fields),
             self.root.momentum(bc.fields),
             self.root.energy(bc.fields)))

        if bc.use_identity_jacobian:
            Q_in = self.root.get_conservative_convective_identity(Uhat, self.mesh.normal, 'incoming')
            Q_out = self.root.get_conservative_convective_identity(Uhat, self.mesh.normal, 'outgoing')
            Gamma_infty = ngs.InnerProduct(Uhat.U - Q_out * U - Q_in * U_infty, Vhat)
        else:
            An_in = self.root.get_conservative_convective_jacobian(Uhat, self.mesh.normal, 'incoming')
            An_out = self.root.get_conservative_convective_jacobian(Uhat, self.mesh.normal, 'outgoing')
            Gamma_infty = ngs.InnerProduct(An_out * (Uhat.U - U) - An_in * (Uhat.U - U_infty), Vhat)

        blf['Uhat'][f"{bc.name}_{bnd}"] = Gamma_infty * dS

    def add_outflow_formulation(self, blf: Integrals, lf: Integrals, bc: Outflow, bnd: str):

        bonus = self.root.optimizations.bonus_int_order
        dS = ngs.ds(skeleton=True, definedon=self.mesh.Boundaries(bnd), bonus_intorder=bonus.bnd)

        U, _ = self.TnT['U']
        Uhat, Vhat = self.TnT['Uhat']

        U = self.get_conservative_fields(U)
        U_bc = flowfields(rho=U.rho, rho_u=U.rho_u, rho_Ek=U.rho_Ek, p=bc.fields.p)
        U_bc = ngs.CF((self.root.density(U_bc), self.root.momentum(U_bc), self.root.energy(U_bc)))

        Gamma_out = ngs.InnerProduct(Uhat - U_bc, Vhat)
        blf['Uhat'][f"{bc.name}_{bnd}"] = Gamma_out * dS

    def add_cbc_formulation(self, blf: Integrals, lf: Integrals, bc: CBC, bnd: str):

        bonus = self.root.optimizations.bonus_int_order
        label = f"{bc.name}_{bnd}"
        dS = ngs.ds(skeleton=True, definedon=self.mesh.Boundaries(bnd), bonus_intorder=bonus.bnd)
        scheme = self.root.fem.scheme

        U, _ = self.TnT['U']
        Uhat, Vhat = self.TnT['Uhat']

        U = self.get_conservative_fields(U)
        Uhat = self.get_conservative_fields(Uhat)

        if bc.target == "farfield":
            U_bc = ngs.CF(
                (self.root.density(bc.fields),
                 self.root.momentum(bc.fields),
                 self.root.energy(bc.fields)))

        elif bc.target == "outflow":
            U_bc = flowfields(rho=U.rho, rho_u=U.rho_u, rho_Ek=U.rho_Ek, p=bc.fields.p)
            U_bc = ngs.CF((self.root.density(U_bc), self.root.momentum(U_bc), self.root.energy(U_bc)))

        elif bc.target == "mass_inflow":
            U_bc = flowfields(rho=bc.fields.rho, rho_u=bc.fields.rho_u, rho_Ek=bc.fields.rho_Ek, p=U.p)
            U_bc = ngs.CF((self.root.density(U_bc), self.root.momentum(U_bc), self.root.energy(U_bc)))

        elif bc.target == "temperature_inflow":
            rho_ = self.root.isentropic_density(U, bc.fields)
            U_bc = flowfields(rho=rho_, u=bc.fields.u, T=U.T)
            U_bc.Ek = self.root.specific_kinetic_energy(U_bc)
            U_bc = ngs.CF((self.root.density(U_bc), self.root.momentum(U_bc), self.root.energy(U_bc)))

        D = bc.get_relaxation_matrix(self.mesh.dim,
                                     dt=self.root.time.timer.step, c=self.root.speed_of_sound(Uhat),
                                     M=self.root.mach_number)
        D = self.root.transform_characteristic_to_conservative(D, Uhat, self.mesh.normal)

        beta = bc.tangential_relaxation
        Qin = self.root.get_conservative_convective_identity(Uhat, self.mesh.normal, "incoming")
        Qout = self.root.get_conservative_convective_identity(Uhat, self.mesh.normal, "outgoing")
        B = self.root.get_conservative_convective_jacobian(Uhat, self.mesh.tangential)

        dt = scheme.get_time_step(True)
        Uhat_n = scheme.get_current_level('Uhat', True)

        blf['Uhat'][label] = (Uhat.U - Qout * U.U - Qin * Uhat_n) * Vhat * dS
        blf['Uhat'][label] -= dt * Qin * D * (U_bc - Uhat.U) * Vhat * dS
        blf['Uhat'][label] += dt * beta * Qin * B * (ngs.grad(Uhat.U) * self.mesh.tangential) * Vhat * dS

        if bc.is_viscous_fluxes:
            blf['Uhat'][label] -= dt * Qin * self.mixed_method.get_cbc_viscous_terms(bc) * Vhat * dS

    def add_inviscid_wall_formulation(self, blf: Integrals, lf: Integrals, bc: InviscidWall, bnd: str):

        n = self.mesh.normal
        dS = ngs.ds(skeleton=True, definedon=self.mesh.Boundaries(bnd))

        U, _ = self.TnT['U']
        Uhat, Vhat = self.TnT['Uhat']

        U = self.get_conservative_fields(U)

        rho = self.root.density(U)
        rho_u = self.root.momentum(U)
        rho_E = self.root.energy(U)
        U_bc = ngs.CF((rho, rho_u - ngs.InnerProduct(rho_u, n)*n, rho_E))

        Gamma_inv = ngs.InnerProduct(Uhat - U_bc, Vhat)
        blf['Uhat'][f"{bc.name}_{bnd}"] = Gamma_inv * dS

    def add_isothermal_wall_formulation(self, blf: Integrals, lf: Integrals, bc: IsothermalWall, bnd: str):

        bonus = self.root.optimizations.bonus_int_order
        dS = ngs.ds(skeleton=True, definedon=self.mesh.Boundaries(bnd), bonus_intorder=bonus.bnd)

        U, _ = self.TnT['U']
        Uhat, Vhat = self.TnT['Uhat']

        U = self.get_conservative_fields(U)
        U_bc = flowfields(rho=U.rho, rho_u=tuple(0 for _ in range(self.mesh.dim)), rho_Ek=0, T=bc.fields.T)
        U_bc = ngs.CF((self.root.density(U_bc), self.root.momentum(U_bc), self.root.inner_energy(U_bc)))

        Gamma_iso = ngs.InnerProduct(Uhat - U_bc, Vhat)
        blf['Uhat'][f"{bc.name}_{bnd}"] = Gamma_iso * dS

    def add_adiabatic_wall_formulation(self, blf: Integrals, lf: Integrals, bc: AdiabaticWall, bnd: str):

        if not isinstance(self.mixed_method, StrainHeat):
            raise NotImplementedError(f"Adiabatic wall not implemented for {self.mixed_method}")

        bonus = self.root.optimizations.bonus_int_order
        dS = ngs.ds(skeleton=True, definedon=self.mesh.Boundaries(bnd), bonus_intorder=bonus.bnd)

        n = self.mesh.normal

        U, _ = self.TnT['U']
        Uhat, Vhat = self.TnT['Uhat']
        Q, _ = self.mixed_method.TnT['Q']

        U = self.get_conservative_fields(U)
        Uhat = self.get_conservative_fields(Uhat)
        Q = self.mixed_method.get_mixed_fields(Q)

        tau = self.mixed_method.get_diffusive_stabilisation_matrix(U)
        T_grad = self.root.temperature_gradient(U, Q)

        U_bc = ngs.CF((Uhat.rho - U.rho, Uhat.rho_u, tau * (Uhat.rho_E - U.rho_E + T_grad * n)))

        Gamma_ad = ngs.InnerProduct(Uhat.U - U_bc, Vhat)
        blf['Uhat'][f"{bc.name}_{bnd}"] = Gamma_ad * dS

    def add_sponge_layer_formulation(self, blf: Integrals, lf: Integrals, dc: SpongeLayer, dom: str):

        dX = ngs.dx(definedon=self.mesh.Materials(dom), bonus_intorder=dc.order)

        U, V = self.TnT['U']
        U_target = ngs.CF(
            (self.root.density(dc.target_state),
             self.root.momentum(dc.target_state),
             self.root.energy(dc.target_state)))

        blf['Uhat'][f"{dc.name}_{dom}"] = dc.function * (U - U_target) * V * dX

    def add_psponge_layer_formulation(self, blf: Integrals, lf: Integrals, dc: PSpongeLayer, dom: str):

        dX = ngs.dx(definedon=self.mesh.Materials(dom), bonus_intorder=dc.order)

        U, V = self.TnT['U']

        if dc.is_equal_order:

            U_target = ngs.CF(
                (self.root.density(dc.target_state),
                 self.root.momentum(dc.target_state),
                 self.root.energy(dc.target_state)))

            Delta_U = U - U_target

        else:

            low_order_space = ngs.L2(self.mesh, order=dc.low_order)
            U_low = ngs.CF(tuple(ngs.Interpolate(proxy, low_order_space) for proxy in U))
            Delta_U = U - U_low

        blf['Uhat'][f"{dc.name}_{dom}"] = dc.function * Delta_U * V * dX

    def get_convective_numerical_flux(self, U: flowfields, Uhat: flowfields, unit_vector: bla.VECTOR):
        r"""
        Convective numerical flux

        .. math::

            \widehat{\vec{F}\vec{n}}  := \vec{F}(\hat{\vec{U}}) \vec{n} + \mat{\tau}_c(\hat{\vec{U}}) (\vec{U} - \hat{\vec{U}}) \qquad [E22a]

        .. [1] See equation :math:`[E22a]` in :cite:`vila-perezHybridisableDiscontinuousGalerkin2021`
        .. [2] See :class:`dream.compressible.riemann_solver` for more details on the definition of :math:`\mat{\tau}_c`
        """
        unit_vector = bla.as_vector(unit_vector)

        tau_c = self.root.riemann_solver.get_convective_stabilisation_matrix_hdg(Uhat, unit_vector)

        return self.root.get_convective_flux(Uhat) * unit_vector + tau_c * (U.U - Uhat.U)

    def get_diffusive_numerical_flux(
            self, U: flowfields, Uhat: flowfields, Q: flowfields, unit_vector: bla.VECTOR):
        r"""
        Diffusive numerical flux

        .. math::

            \widehat{\vec{G}\vec{n}}  := \vec{G}(\hat{\vec{U}}, \vec{Q}) \vec{n} + \mat{\tau}_d (\vec{U} - \hat{\vec{U}}).

        .. [1] See equation :math:`[E22b]` in :cite:`vila-perezHybridisableDiscontinuousGalerkin2021`
        .. [2] See :class:`MixedMethod` for more details on the definition of :math:`\mat{\tau}_d`
        """
        unit_vector = bla.as_vector(unit_vector)

        tau_d = self.root.fem.mixed_method.get_diffusive_stabilisation_matrix(Uhat)

        return self.root.get_diffusive_flux(Uhat, Q)*unit_vector - tau_d * (U.U - Uhat.U)

    def get_time_scheme_spaces(self) -> dict[str, ngs.FESpace]:
        U = super().get_time_scheme_spaces()
        if self.root.bcs.has_condition(CBC):
            U['Uhat'] = self.root.fem.spaces['Uhat']
        return U

    def set_initial_conditions(self, U: ngs.CF = None):

        if U is None:
            U = self.mesh.MaterialCF({dom: ngs.CF(
                (self.root.density(dc.fields),
                 self.root.momentum(dc.fields),
                 self.root.energy(dc.fields))) for dom, dc in self.root.dcs.to_pattern(Initial).items()})

        super().set_initial_conditions(U)

        gfu = self.gfu['Uhat']
        fes = self.gfu['Uhat'].space
        u, v = fes.TnT()

        blf = ngs.BilinearForm(fes)
        blf += u * v * ngs.dx(element_boundary=True)

        f = ngs.LinearForm(fes)
        f += U * v * ngs.dx(element_boundary=True)

        with ngs.TaskManager():
            blf.Assemble()
            f.Assemble()
            gfu.vec.data = blf.mat.Inverse(freedofs=fes.FreeDofs(), inverse="sparsecholesky") * f.vec


class DG_HDG(ConservativeMethod):

    name: str = "dg_hdg"

    @dream_configuration
    def scheme(self) -> IMEXRK_ARS443:
        return self._scheme

    @scheme.setter
    def scheme(self, scheme: TimeSchemes) -> None:

        if not isinstance(self.root.time, TransientRoutine):
            raise TypeError("DG method only supports transient time routines!")

        OPTIONS = [IMEXRK_ARS443]
        self._scheme = self._get_configuration_option(scheme, OPTIONS, TimeSchemes)

    @property
    def mixed_method(self) -> MixedMethod:
        return self.root.fem.mixed_method

    def add_finite_element_spaces(self, fes: dict[str, ngs.FESpace]) -> None:

        order = self.root.fem.order
        dim = self.mesh.dim + 2

        U = ngs.L2(self.mesh, order=order, dgjumps=True)
        Uhat = ngs.FacetFESpace(self.mesh, order=order)

        if self.root.bcs.has_condition(Periodic):
            Uhat = ngs.Periodic(Uhat)

        fes['U'] = U**dim
        fes['Uhat'] = Uhat**dim

    # In this (IMEX-)specialized class, the inviscid terms are handled via a standard DG.
    def add_convection_form(self,
                            blf: dict[str, ngs.comp.SumOfIntegrals],
                            lf:  dict[str, ngs.comp.SumOfIntegrals]):

        # Extract the bonus integration order, if specified.
        bonus = self.root.optimizations.bonus_int_order

        # Obtain the relevant test and trial functions. Notice, the solution "U"
        # is assumed to be an unknown in the bilinear form, despite being explicit
        # in time. This works, because we invoke the "Apply" function when solving.
        U, V = self.TnT['U']

        # Get a mask that is nonzero (unity) for only the internal faces.
        mask = self.get_domain_boundary_mask()

        # Current/owned solution.
        Ui = self.get_conservative_fields(U)

        # Neighboring solution.
        Uj = self.get_conservative_fields(U.Other())

        # Compute the flux of the solution on the volume elements.
        F = self.root.get_convective_flux(Ui)

        # Compute the flux on the surface of an element, in the normal direction.
        Fn = self.root.riemann_solver.get_convective_numerical_flux_dg(Ui, Uj, self.mesh.normal)

        # Assemble the explicit bilinear form, keeping in mind this is also placed on the LHS.
        blf['U']['convection'] = -bla.inner(F, ngs.grad(V)) * ngs.dx(bonus_intorder=bonus.vol)
        blf['U']['convection'] += mask*bla.inner(Fn, V) * ngs.dx(element_boundary=True, bonus_intorder=bonus.bnd)

    # In this (IMEX-)specialized class, the elliptic terms are handled via an HDG.

    def add_diffusion_form(self, blf: Integrals, lf: Integrals):

        bonus = self.root.optimizations.bonus_int_order

        mask = self.get_domain_boundary_mask()

        U, V = self.TnT['U']
        Uhat, Vhat = self.TnT['Uhat']
        Q, _ = self.mixed_method.TnT['Q']

        U = self.get_conservative_fields(U)
        Uhat = self.get_conservative_fields(Uhat)
        Q = self.root.fem.mixed_method.get_mixed_fields(Q)

        G = self.root.get_diffusive_flux(U, Q)
        Gn = self.get_diffusive_numerical_flux(U, Uhat, Q, self.mesh.normal)

        blf['U']['diffusion'] = ngs.InnerProduct(G, ngs.grad(V)) * ngs.dx(bonus_intorder=bonus.vol)
        blf['U']['diffusion'] -= ngs.InnerProduct(Gn, V) * ngs.dx(element_boundary=True, bonus_intorder=bonus.bnd)
        blf['Uhat']['diffusion'] = mask * ngs.InnerProduct(Gn, Vhat) * ngs.dx(element_boundary=True, bonus_intorder=bonus.bnd)

        # NOTE, to obtain a well-posed formulation, we require a value for rho_hat, since we need it on the facets.
        # To this end, we estimate its value as the average of the density on the surface (w.r.t. neighboring elements).
        # Recall, we solve for a Uhat implicitly, but use it explicitly in the next time step -- also note, rho is
        # solved for explicitly, as it's governed by a pure hyperbolic equation (continuity eq).
        rho = self.root.density(U)
        rhoHat = self.root.density(Uhat)
        rho_avg = rho - rhoHat
        eq = ngs.CF((rho_avg, 0, 0, 0))

        blf['Uhat']['test'] = mask * ngs.InnerProduct(eq,
                                                      Vhat) * ngs.dx(element_boundary=True, bonus_intorder=bonus.bnd)

    def get_diffusive_numerical_flux(
            self, U: flowfields, Uhat: flowfields, Q: flowfields, unit_vector: bla.VECTOR):
        r"""
        Diffusive numerical flux

        .. math::

            \widehat{\vec{G}\vec{n}}  := \vec{G}(\hat{\vec{U}}, \vec{Q}) \vec{n} + \mat{\tau}_d (\vec{U} - \hat{\vec{U}}).

        .. [1] See equation :math:`[E22b]` in :cite:`vila-perezHybridisableDiscontinuousGalerkin2021`
        .. [2] See :class:`MixedMethod` for more details on the definition of :math:`\mat{\tau}_d`
        """
        unit_vector = bla.as_vector(unit_vector)

        tau_d = self.root.fem.mixed_method.get_diffusive_stabilisation_matrix(Uhat)

        return self.root.get_diffusive_flux(Uhat, Q)*unit_vector - tau_d * (U.U - Uhat.U)

    def get_time_scheme_spaces(self) -> dict[str, ngs.FESpace]:
        U = super().get_time_scheme_spaces()
        if self.root.bcs.has_condition(CBC):
            U['Uhat'] = self.root.fem.spaces['Uhat']
        return U

    # NOTE, for now we restrict ourselves to periodic BCs.

    def add_boundary_conditions(self, blf: Integrals, lf: Integrals):

        bnds = self.root.bcs.to_pattern()

        for bnd, bc in bnds.items():

            logger.debug(f"Adding boundary condition {bc} on boundary {bnd}.")

            if isinstance(bc, Periodic):
                continue
            else:
                raise TypeError(f"Boundary condition {bc} not implemented in {self}!")

    def add_domain_conditions(self,
                              blf: dict[str, ngs.comp.SumOfIntegrals],
                              lf:  dict[str, ngs.comp.SumOfIntegrals]):

        doms = self.root.dcs.to_pattern()

        for dom, dc in doms.items():

            logger.debug(f"Adding domain condition {dc} on domain {dom}.")

            if isinstance(dc, Initial):
                continue

            else:
                raise TypeError(f"Domain condition {dc} not implemented in {self}!")

    def set_initial_conditions(self, U: ngs.CF = None):

        if U is None:
            U = self.mesh.MaterialCF({dom: ngs.CF(
                (self.root.density(dc.fields),
                 self.root.momentum(dc.fields),
                 self.root.energy(dc.fields))) for dom, dc in self.root.dcs.to_pattern(Initial).items()})

        super().set_initial_conditions(U)

        gfu = self.gfu['Uhat']
        fes = self.gfu['Uhat'].space
        u, v = fes.TnT()

        blf = ngs.BilinearForm(fes)
        blf += u * v * ngs.dx(element_boundary=True)

        f = ngs.LinearForm(fes)
        f += U * v * ngs.dx(element_boundary=True)

        with ngs.TaskManager():
            blf.Assemble()
            f.Assemble()
            gfu.vec.data = blf.mat.Inverse(freedofs=fes.FreeDofs(), inverse="sparsecholesky") * f.vec


class ConservativeFiniteElementMethod(CompressibleFiniteElementMethod):

    root: CompressibleFlowSolver
    name: str = "conservative"

    def __init__(self, mesh, root=None, **default):
        DEFAULT = {
            'method': "hdg",
            'mixed_method': "inactive",
        }
        DEFAULT.update(default)
        super().__init__(mesh, root, **DEFAULT)

    @dream_configuration
    def method(self) -> HDG | DG | DG_HDG:
        """
        The method to be used for the compressible flow solver.
        """
        return self._method

    @method.setter
    def method(self, value: str | ConservativeMethod):
        OPTIONS = [HDG, DG, DG_HDG]
        self._method = self._get_configuration_option(value, OPTIONS, ConservativeMethod)

    @dream_configuration
    def mixed_method(self) -> Inactive | StrainHeat | Gradient:
        """
        The mixed method to be used for the compressible flow solver.
        """
        return self._mixed_method

    @mixed_method.setter
    def mixed_method(self, value: str | MixedMethod):
        OPTIONS = [Inactive, StrainHeat, Gradient]
        self._mixed_method = self._get_configuration_option(value, OPTIONS, MixedMethod)

    @property
    def scheme(self) -> TimeSchemes:
        return self.method.scheme

    @scheme.setter
    def scheme(self, scheme: str | TimeSchemes):
        self.method.scheme = scheme

    def add_finite_element_spaces(self, fes: dict[str, ngs.FESpace]):
        self.method.add_finite_element_spaces(fes)
        self.mixed_method.add_mixed_finite_element_spaces(fes)

    def add_symbolic_spatial_forms(self, blf: Integrals, lf: Integrals):

        self.method.add_convection_form(blf, lf)

        if not self.root.dynamic_viscosity.is_inviscid:
            self.method.add_diffusion_form(blf, lf)

        self.mixed_method.add_mixed_form(blf, lf)

        self.method.add_boundary_conditions(blf, lf)
        self.method.add_domain_conditions(blf, lf)

    def get_solution_fields(self) -> flowfields:
        U = self.method.get_conservative_fields(self.method.gfu['U'], with_gradients=True)
        if not isinstance(self.mixed_method, Inactive):
            U.update(self.mixed_method.get_mixed_fields(self.mixed_method.gfu['Q']))
        return U

    def initialize_time_scheme_gridfunctions(self):
        spaces = self.method.get_time_scheme_spaces()
        super().initialize_time_scheme_gridfunctions(*spaces)

    def set_initial_conditions(self) -> None:
        U = self.mesh.MaterialCF({dom: ngs.CF(
            (self.root.density(dc.fields),
             self.root.momentum(dc.fields),
             self.root.energy(dc.fields))) for dom, dc in self.root.dcs.to_pattern(Initial).items()})

        self.method.set_initial_conditions(U)

        super().set_initial_conditions()
