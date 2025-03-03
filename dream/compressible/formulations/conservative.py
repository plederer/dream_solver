from __future__ import annotations
import logging
import ngsolve as ngs
import typing
import dream.bla as bla

from dream.config import InterfaceConfiguration, interface
from dream.mesh import SpongeLayer, PSpongeLayer, Periodic, Initial
from dream.compressible.config import (flowfields,
                                       CompressibleFiniteElement,
                                       FarField,
                                       Outflow,
                                       InviscidWall,
                                       Symmetry,
                                       IsothermalWall,
                                       AdiabaticWall,
                                       CBC)

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from ..solver import CompressibleFlowSolver


# --- Conservative --- #

class MixedMethod(InterfaceConfiguration, is_interface=True):

    cfg: CompressibleFlowSolver

    @property
    def fem(self) -> ConservativeFiniteElementMethod:
        return self.cfg.fem

    @property
    def TnT(self) -> dict[str, tuple[ngs.comp.ProxyFunction, ...]]:
        return self.cfg.TnT

    @property
    def gfu(self) -> dict[str, ngs.GridFunction]:
        return self.cfg.gfus

    def add_mixed_finite_element_spaces(self, fes: dict[str, ngs.FESpace]) -> None:
        raise NotImplementedError("Mixed method must implement get_mixed_finite_element_spaces method!")

    def add_mixed_form(self,
                       blf: dict[str, ngs.comp.SumOfIntegrals],
                       lf: dict[str, ngs.comp.SumOfIntegrals]) -> None:
        pass

    def get_cbc_viscous_terms(self, bc: CBC) -> ngs.CF:
        return ngs.CF(tuple(0 for _ in range(self.mesh.dim + 2)))

    def get_diffusive_stabilisation_matrix(self, U: flowfields) -> bla.MATRIX:
        Re = self.cfg.reference_reynolds_number
        Pr = self.cfg.prandtl_number
        mu = self.cfg.viscosity(U)

        tau_d = [0] + [1 for _ in range(self.mesh.dim)] + [1/Pr]
        return bla.diagonal(tau_d) * mu / Re


class Inactive(MixedMethod):

    name: str = "inactive"

    def add_mixed_finite_element_spaces(self, fes: dict[str, ngs.FESpace]) -> None:

        if not self.cfg.dynamic_viscosity.is_inviscid:
            raise TypeError(f"Viscous configuration requires mixed method!")


class StrainHeat(MixedMethod):

    name: str = "strain_heat"

    def add_mixed_finite_element_spaces(self, fes: dict[str, ngs.FESpace]) -> None:

        if self.cfg.dynamic_viscosity.is_inviscid:
            raise TypeError(f"Inviscid configuration does not require mixed method!")

        dim = 4*self.mesh.dim - 3
        order = self.fem.order

        Q = ngs.L2(self.mesh, order=order)
        Q = self.cfg.dcs.reduce_psponge_layers_order_elementwise(Q)

        fes['Q'] = Q**dim

    def add_mixed_form(self,
                       blf: dict[str, ngs.comp.SumOfIntegrals],
                       lf: dict[str, ngs.comp.SumOfIntegrals]) -> None:

        if self.mesh.dim == 3:
            raise NotImplementedError("StrainHeat method is not implemented for domain dimension 3!")

        bonus = self.cfg.optimizations.bonus_int_order

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
        blf['mixed'] = ngs.InnerProduct(Q.eps, P.eps) * ngs.dx
        blf['mixed'] += ngs.InnerProduct(U.u, div_dev_zeta) * ngs.dx(bonus_intorder=bonus.vol)
        blf['mixed'] -= ngs.InnerProduct(Uhat.u, dev_zeta*self.mesh.normal) * \
            ngs.dx(element_boundary=True, bonus_intorder=bonus.bnd)

        div_xi = gradient_P[3, 0] + gradient_P[4, 1]
        blf['mixed'] += ngs.InnerProduct(Q.grad_T, P.grad_T) * ngs.dx
        blf['mixed'] += ngs.InnerProduct(U.T, div_xi) * ngs.dx(bonus_intorder=bonus.vol)
        blf['mixed'] -= ngs.InnerProduct(Uhat.T*self.mesh.normal,
                                         P.grad_T) * ngs.dx(element_boundary=True, bonus_intorder=bonus.bnd)

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

        rho = self.cfg.density(U)
        stess_tensor = self.cfg.deviatoric_stress_tensor(U, Q)
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

        rho = self.cfg.density(U)
        stess_tensor = self.cfg.deviatoric_stress_tensor(U, Q)
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

        Re = self.cfg.reference_reynolds_number
        Pr = self.cfg.prandtl_number
        mu = self.cfg.viscosity(U)

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

        Re = self.cfg.reference_reynolds_number
        Pr = self.cfg.prandtl_number
        mu = self.cfg.viscosity(U)

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

        if self.cfg.dynamic_viscosity.is_inviscid:
            raise TypeError(f"Inviscid configuration does not require mixed method!")

        dim = self.mesh.dim + 2
        order = self.fem.order

        Q = ngs.VectorL2(self.mesh, order=order)
        Q = self.cfg.dcs.reduce_psponge_layers_order_elementwise(Q)

        fes['Q'] = Q**dim

    def add_mixed_form(self,
                       blf: dict[str, ngs.comp.SumOfIntegrals],
                       lf: dict[str, ngs.comp.SumOfIntegrals]) -> None:

        Q, P = self.TnT['Q']
        U, _ = self.fem.method.TnT['U']
        Uhat, _ = self.fem.method.TnT['Uhat']

        blf['mixed'] = ngs.InnerProduct(Q, P) * ngs.dx
        blf['mixed'] += ngs.InnerProduct(U, ngs.div(P)) * ngs.dx
        blf['mixed'] -= ngs.InnerProduct(Uhat, P*self.mesh.normal) * ngs.dx(element_boundary=True)

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


class ConservativeMethod(InterfaceConfiguration, is_interface=True):

    cfg: CompressibleFlowSolver

    @property
    def TnT(self) -> dict[str, tuple[ngs.comp.ProxyFunction, ...]]:
        return self.cfg.TnT

    @property
    def gfu(self) -> dict[str, ngs.GridFunction]:
        return self.cfg.gfus

    def add_symbolic_temporal_forms(self, blf: dict[str, ngs.comp.SumOfIntegrals],
                                    lf: dict[str, ngs.comp.SumOfIntegrals]):

        self.cfg.time.scheme.add_symbolic_temporal_forms('U', blf, lf)

    def get_temporal_integrators(self):
        return {'U': ngs.dx}

    def get_conservative_fields(self, U: ngs.CoefficientFunction) -> flowfields:

        if isinstance(U, ngs.GridFunction):
            U = U.components

        U_ = flowfields()
        U_.rho = U[0]
        U_.rho_u = U[slice(1, self.mesh.dim + 1)]
        U_.rho_E = U[self.mesh.dim + 1]

        U_.u = self.cfg.velocity(U_)
        U_.rho_Ek = self.cfg.kinetic_energy(U_)
        U_.rho_Ei = self.cfg.inner_energy(U_)
        U_.p = self.cfg.pressure(U_)
        U_.T = self.cfg.temperature(U_)
        U_.c = self.cfg.speed_of_sound(U_)

        if isinstance(U, ngs.comp.ProxyFunction):
            U_.U = U

        return U_

    def get_conservative_gradient_fields(self, U: ngs.CoefficientFunction) -> flowfields:

        U_ = self.get_conservative_fields(U)

        if isinstance(U, ngs.GridFunction):
            dU = ngs.grad(U)

            U_.grad_rho = dU[0, :]
            U_.grad_rho_u = dU[slice(1, self.mesh.dim + 1), :]
            U_.grad_rho_E = dU[self.mesh.dim + 1, :]

            U_.grad_u = self.cfg.velocity_gradient(U_, U_)
            U_.grad_rho_Ek = self.cfg.kinetic_energy_gradient(U_, U_)
            U_.grad_rho_Ei = self.cfg.inner_energy_gradient(U_, U_)
            U_.grad_p = self.cfg.pressure_gradient(U_, U_)
            U_.grad_T = self.cfg.temperature_gradient(U_, U_)

        return U_

    def set_initial_conditions(self, U: ngs.CF = None):

        if U is None:
            U = self.mesh.MaterialCF({dom: ngs.CF(
                (self.cfg.density(dc.fields),
                 self.cfg.momentum(dc.fields),
                 self.cfg.energy(dc.fields))) for dom, dc in self.cfg.dcs.to_pattern(Initial).items()})

        self.gfu['U'].Set(U)


class HDG(ConservativeMethod):

    name: str = "hdg"

    @property
    def mixed_method(self) -> MixedMethod:
        return self.cfg.fem.mixed_method

    def add_finite_element_spaces(self, fes: dict[str, ngs.FESpace]) -> None:

        order = self.cfg.fem.order
        dim = self.mesh.dim + 2

        U = ngs.L2(self.mesh, order=order)
        Uhat = ngs.FacetFESpace(self.mesh, order=order)

        psponge_layers = self.cfg.dcs.to_pattern(PSpongeLayer)
        if psponge_layers:
            U = self.cfg.dcs.reduce_psponge_layers_order_elementwise(U, psponge_layers)
            Uhat = self.cfg.dcs.reduce_psponge_layers_order_facetwise(Uhat, psponge_layers)

        if self.cfg.bcs.has_condition(Periodic):
            Uhat = ngs.Periodic(Uhat)

        fes['U'] = U**dim
        fes['Uhat'] = Uhat**dim

    def add_convection_form(
            self, blf: dict[str, ngs.comp.SumOfIntegrals],
            lf: dict[str, ngs.comp.SumOfIntegrals]):

        bonus = self.cfg.optimizations.bonus_int_order

        mask = self.get_domain_boundary_mask()

        U, V = self.TnT['U']
        Uhat, Vhat = self.TnT['Uhat']

        U = self.get_conservative_fields(U)
        Uhat = self.get_conservative_fields(Uhat)

        F = self.cfg.get_convective_flux(U)
        Fn = self.get_convective_numerical_flux(U, Uhat, self.mesh.normal)

        blf['convection'] = -bla.inner(F, ngs.grad(V)) * ngs.dx(bonus_intorder=bonus.vol)
        blf['convection'] += bla.inner(Fn, V) * ngs.dx(element_boundary=True, bonus_intorder=bonus.bnd)
        blf['convection'] += -mask * bla.inner(Fn, Vhat) * ngs.dx(element_boundary=True, bonus_intorder=bonus.bnd)

    def add_diffusion_form(
            self, blf: dict[str, ngs.comp.SumOfIntegrals],
            lf: dict[str, ngs.comp.SumOfIntegrals]):

        bonus = self.cfg.optimizations.bonus_int_order

        mask = self.get_domain_boundary_mask()

        U, V = self.TnT['U']
        Uhat, Vhat = self.TnT['Uhat']
        Q, _ = self.mixed_method.TnT['Q']

        U = self.get_conservative_fields(U)
        Uhat = self.get_conservative_fields(Uhat)
        Q = self.cfg.fem.mixed_method.get_mixed_fields(Q)

        G = self.cfg.get_diffusive_flux(U, Q)
        Gn = self.get_diffusive_numerical_flux(U, Uhat, Q, self.mesh.normal)

        blf['diffusion'] = ngs.InnerProduct(G, ngs.grad(V)) * ngs.dx(bonus_intorder=bonus.vol)
        blf['diffusion'] -= ngs.InnerProduct(Gn, V) * ngs.dx(element_boundary=True, bonus_intorder=bonus.bnd)
        blf['diffusion'] += mask * ngs.InnerProduct(Gn, Vhat) * ngs.dx(element_boundary=True, bonus_intorder=bonus.bnd)

    def add_boundary_conditions(self,
                                blf: dict[str, ngs.comp.SumOfIntegrals],
                                lf: dict[str, ngs.comp.SumOfIntegrals]):

        bnds = self.cfg.bcs.to_pattern()

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

    def add_domain_conditions(self,
                              blf: dict[str, ngs.comp.SumOfIntegrals],
                              lf: dict[str, ngs.comp.SumOfIntegrals]):

        doms = self.cfg.dcs.to_pattern()

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

    def add_farfield_formulation(
            self, blf: dict[str, ngs.comp.SumOfIntegrals],
            lf: dict[str, ngs.comp.SumOfIntegrals],
            bc: FarField, bnd: str):

        bonus = self.cfg.optimizations.bonus_int_order
        dS = ngs.ds(skeleton=True, definedon=self.mesh.Boundaries(bnd), bonus_intorder=bonus.bnd)

        U, _ = self.TnT['U']
        Uhat, Vhat = self.TnT['Uhat']
        Uhat = self.get_conservative_fields(Uhat)

        U_infty = ngs.CF(
            (self.cfg.density(bc.fields),
             self.cfg.momentum(bc.fields),
             self.cfg.energy(bc.fields)))

        if bc.identity_jacobian:
            Q_in = self.cfg.get_conservative_convective_identity(Uhat, self.mesh.normal, 'incoming')
            Q_out = self.cfg.get_conservative_convective_identity(Uhat, self.mesh.normal, 'outgoing')
            Gamma_infty = ngs.InnerProduct(Uhat.U - Q_out * U - Q_in * U_infty, Vhat)
        else:
            An_in = self.cfg.get_conservative_convective_jacobian(Uhat, self.mesh.normal, 'incoming')
            An_out = self.cfg.get_conservative_convective_jacobian(Uhat, self.mesh.normal, 'outgoing')
            Gamma_infty = ngs.InnerProduct(An_out * (Uhat.U - U) - An_in * (Uhat.U - U_infty), Vhat)

        blf[f"{bc.name}_{bnd}"] = Gamma_infty * dS

    def add_outflow_formulation(
            self, blf: dict[str, ngs.comp.SumOfIntegrals],
            lf: dict[str, ngs.comp.SumOfIntegrals],
            bc: Outflow, bnd: str):

        bonus = self.cfg.optimizations.bonus_int_order
        dS = ngs.ds(skeleton=True, definedon=self.mesh.Boundaries(bnd), bonus_intorder=bonus.bnd)

        U, _ = self.TnT['U']
        Uhat, Vhat = self.TnT['Uhat']

        U = self.get_conservative_fields(U)
        U_bc = flowfields(rho=U.rho, rho_u=U.rho_u, rho_Ek=U.rho_Ek, p=bc.fields.p)
        U_bc = ngs.CF((self.cfg.density(U_bc), self.cfg.momentum(U_bc), self.cfg.energy(U_bc)))

        Gamma_out = ngs.InnerProduct(Uhat - U_bc, Vhat)
        blf[f"{bc.name}_{bnd}"] = Gamma_out * dS

    def add_cbc_formulation(self,
                            blf: dict[str, ngs.comp.SumOfIntegrals],
                            lf: dict[str, ngs.comp.SumOfIntegrals],
                            bc: CBC, bnd: str):

        bonus = self.cfg.optimizations.bonus_int_order
        label = f"{bc.name}_{bnd}"
        dS = ngs.ds(skeleton=True, definedon=self.mesh.Boundaries(bnd), bonus_intorder=bonus.bnd)
        scheme = self.cfg.time.scheme

        U, _ = self.TnT['U']
        Uhat, Vhat = self.TnT['Uhat']

        U = self.get_conservative_fields(U)
        Uhat = self.get_conservative_fields(Uhat)

        if bc.target == "farfield":
            U_bc = ngs.CF(
                (self.cfg.density(bc.fields),
                 self.cfg.momentum(bc.fields),
                 self.cfg.energy(bc.fields)))

        elif bc.target == "outflow":
            U_bc = flowfields(rho=U.rho, rho_u=U.rho_u, rho_Ek=U.rho_Ek, p=bc.fields.p)
            U_bc = ngs.CF((self.cfg.density(U_bc), self.cfg.momentum(U_bc), self.cfg.energy(U_bc)))

        elif bc.target == "mass_inflow":
            U_bc = flowfields(rho=bc.fields.rho, rho_u=bc.fields.rho_u, rho_Ek=bc.fields.rho_Ek, p=U.p)
            U_bc = ngs.CF((self.cfg.density(U_bc), self.cfg.momentum(U_bc), self.cfg.energy(U_bc)))

        elif bc.target == "temperature_inflow":
            rho_ = self.cfg.isentropic_density(U, bc.fields)
            U_bc = flowfields(rho=rho_, u=bc.fields.u, T=U.T)
            U_bc.Ek = self.cfg.specific_kinetic_energy(U_bc)
            U_bc = ngs.CF((self.cfg.density(U_bc), self.cfg.momentum(U_bc), self.cfg.energy(U_bc)))

        D = bc.get_relaxation_matrix(
            dt=self.cfg.time.timer.step, c=self.cfg.speed_of_sound(Uhat),
            M=self.cfg.mach_number)
        D = self.cfg.transform_characteristic_to_conservative(D, Uhat, self.mesh.normal)

        beta = bc.tangential_relaxation
        Qin = self.cfg.get_conservative_convective_identity(Uhat, self.mesh.normal, "incoming")
        Qout = self.cfg.get_conservative_convective_identity(Uhat, self.mesh.normal, "outgoing")
        B = self.cfg.get_conservative_convective_jacobian(Uhat, self.mesh.tangential)

        dt = scheme.get_time_step(True)
        Uhat_n = scheme.get_current_level('Uhat', True)

        blf[label] = (Uhat.U - Qout * U.U - Qin * Uhat_n) * Vhat * dS
        blf[label] -= dt * Qin * D * (U_bc - Uhat.U) * Vhat * dS
        blf[label] += dt * beta * Qin * B * (ngs.grad(Uhat.U) * self.mesh.tangential) * Vhat * dS

        if bc.is_viscous_fluxes:
            blf[label] -= dt * Qin * self.mixed_method.get_cbc_viscous_terms(bc) * Vhat * dS

    def add_inviscid_wall_formulation(
            self, blf: dict[str, ngs.comp.SumOfIntegrals],
            lf: dict[str, ngs.comp.SumOfIntegrals],
            bc: InviscidWall, bnd: str):

        n = self.mesh.normal
        dS = ngs.ds(skeleton=True, definedon=self.mesh.Boundaries(bnd))

        U, _ = self.TnT['U']
        Uhat, Vhat = self.TnT['Uhat']

        U = self.get_conservative_fields(U)

        rho = self.cfg.density(U)
        rho_u = self.cfg.momentum(U)
        rho_E = self.cfg.energy(U)
        U_bc = ngs.CF((rho, rho_u - ngs.InnerProduct(rho_u, n)*n, rho_E))

        Gamma_inv = ngs.InnerProduct(Uhat - U_bc, Vhat)
        blf[f"{bc.name}_{bnd}"] = Gamma_inv * dS

    def add_isothermal_wall_formulation(
            self, blf: dict[str, ngs.comp.SumOfIntegrals],
            lf: dict[str, ngs.comp.SumOfIntegrals],
            bc: IsothermalWall, bnd: str):

        bonus = self.cfg.optimizations.bonus_int_order
        dS = ngs.ds(skeleton=True, definedon=self.mesh.Boundaries(bnd), bonus_intorder=bonus.bnd)

        U, _ = self.TnT['U']
        Uhat, Vhat = self.TnT['Uhat']

        U = self.get_conservative_fields(U)
        U_bc = flowfields(rho=U.rho, rho_u=tuple(0 for _ in range(self.mesh.dim)), rho_Ek=0, T=bc.fields.T)
        U_bc = ngs.CF((self.cfg.density(U_bc), self.cfg.momentum(U_bc), self.cfg.inner_energy(U_bc)))

        Gamma_iso = ngs.InnerProduct(Uhat - U_bc, Vhat)
        blf[f"{bc.name}_{bnd}"] = Gamma_iso * dS

    def add_adiabatic_wall_formulation(
            self, blf: dict[str, ngs.comp.SumOfIntegrals],
            lf: dict[str, ngs.comp.SumOfIntegrals],
            bc: AdiabaticWall, bnd: str):

        if not isinstance(self.mixed_method, StrainHeat):
            raise NotImplementedError(f"Adiabatic wall not implemented for {self.mixed_method}")

        bonus = self.cfg.optimizations.bonus_int_order
        dS = ngs.ds(skeleton=True, definedon=self.mesh.Boundaries(bnd), bonus_intorder=bonus.bnd)

        n = self.mesh.normal

        U, _ = self.TnT['U']
        Uhat, Vhat = self.TnT['Uhat']
        Q, _ = self.mixed_method.TnT['Q']

        U = self.get_conservative_fields(U)
        Uhat = self.get_conservative_fields(Uhat)
        Q = self.mixed_method.get_mixed_fields(Q)

        tau = self.mixed_method.get_diffusive_stabilisation_matrix(U)
        T_grad = self.cfg.temperature_gradient(U, Q)

        U_bc = ngs.CF((Uhat.rho - U.rho, Uhat.rho_u, tau * (Uhat.rho_E - U.rho_E + T_grad * n)))

        Gamma_ad = ngs.InnerProduct(Uhat.U - U_bc, Vhat)
        blf[f"{bc.name}_{bnd}"] = Gamma_ad * dS

    def add_sponge_layer_formulation(
            self, blf: dict[str, ngs.comp.SumOfIntegrals],
            lf: dict[str, ngs.comp.SumOfIntegrals],
            dc: SpongeLayer, dom: str):

        dX = ngs.dx(definedon=self.mesh.Materials(dom), bonus_intorder=dc.order)

        U, V = self.TnT['U']
        U_target = ngs.CF(
            (self.cfg.density(dc.target_state),
             self.cfg.momentum(dc.target_state),
             self.cfg.energy(dc.target_state)))

        blf[f"{dc.name}_{dom}"] = dc.function * (U - U_target) * V * dX

    def add_psponge_layer_formulation(
            self, blf: dict[str, ngs.comp.SumOfIntegrals],
            lf: dict[str, ngs.comp.SumOfIntegrals],
            dc: PSpongeLayer, dom: str):

        dX = ngs.dx(definedon=self.mesh.Materials(dom), bonus_intorder=dc.order)

        U, V = self.TnT['U']

        if dc.is_equal_order:

            U_target = ngs.CF(
                (self.cfg.density(dc.target_state),
                 self.cfg.momentum(dc.target_state),
                 self.cfg.energy(dc.target_state)))

            Delta_U = U - U_target

        else:

            low_order_space = ngs.L2(self.mesh, order=dc.low_order)
            U_low = ngs.CF(tuple(ngs.Interpolate(proxy, low_order_space) for proxy in U))
            Delta_U = U - U_low

        blf[f"{dc.name}_{dom}"] = dc.function * Delta_U * V * dX

    def get_convective_numerical_flux(self, U: flowfields, Uhat: flowfields, unit_vector: bla.VECTOR):
        """
        Convective numerical flux

        Equation 22a, page 11

        Literature:
        [1] - Vila-Pérez, J., Giacomini, M., Sevilla, R. et al.
              Hybridisable Discontinuous Galerkin Formulation of Compressible Flows.
              Arch Computat Methods Eng 28, 753–784 (2021).
              https://doi.org/10.1007/s11831-020-09508-z
        """
        unit_vector = bla.as_vector(unit_vector)

        tau = self.cfg.riemann_solver.get_convective_stabilisation_matrix(Uhat, unit_vector)

        return self.cfg.get_convective_flux(Uhat) * unit_vector + tau * (U.U - Uhat.U)

    def get_diffusive_numerical_flux(
            self, U: flowfields, Uhat: flowfields, Q: flowfields, unit_vector: bla.VECTOR):
        """
        Diffusive numerical flux

        Equation 22b, page 11

        Literature:
        [1] - Vila-Pérez, J., Giacomini, M., Sevilla, R. et al.
              Hybridisable Discontinuous Galerkin Formulation of Compressible Flows.
              Arch Computat Methods Eng 28, 753–784 (2021).
              https://doi.org/10.1007/s11831-020-09508-z
        """
        unit_vector = bla.as_vector(unit_vector)

        tau_d = self.cfg.fem.mixed_method.get_diffusive_stabilisation_matrix(Uhat)

        return self.cfg.get_diffusive_flux(Uhat, Q)*unit_vector - tau_d * (U.U - Uhat.U)

    def get_temporal_integrators(self):
        U = super().get_temporal_integrators()
        if self.cfg.bcs.has_condition(CBC):
            U['Uhat'] = ngs.ds(skeleton=True)
        return U

    def get_domain_boundary_mask(self) -> ngs.GridFunction:
        """ 
        Returns a Gridfunction that is 0 on the domain boundaries and 1 on the domain interior.
        """

        fes = ngs.FacetFESpace(self.mesh, order=0)
        mask = ngs.GridFunction(fes, name="mask")
        mask.vec[:] = 0

        bnd_dofs = fes.GetDofs(self.mesh.Boundaries(self.cfg.bcs.get_domain_boundaries(True)))
        mask.vec[~bnd_dofs] = 1

        return mask

    def set_initial_conditions(self, U: ngs.CF = None):

        if U is None:
            U = self.mesh.MaterialCF({dom: ngs.CF(
                (self.cfg.density(dc.fields),
                 self.cfg.momentum(dc.fields),
                 self.cfg.energy(dc.fields))) for dom, dc in self.cfg.dcs.to_pattern(Initial).items()})

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

            gfu.vec.data = blf.mat.Inverse(inverse="sparsecholesky") * f.vec


class ConservativeFiniteElementMethod(CompressibleFiniteElement):

    cfg: CompressibleFlowSolver
    name: str = "conservative"

    @interface(default=HDG)
    def method(self, method):
        return method

    @interface(default=Inactive)
    def mixed_method(self, mixed_method):
        return mixed_method

    def add_finite_element_spaces(self, fes: dict[str, ngs.FESpace]):
        self.method.add_finite_element_spaces(fes)
        self.mixed_method.add_mixed_finite_element_spaces(fes)

    def add_symbolic_spatial_forms(self,
                                   blf: dict[str, ngs.comp.SumOfIntegrals],
                                   lf: dict[str, ngs.comp.SumOfIntegrals]):

        self.method.add_convection_form(blf, lf)

        if not self.cfg.dynamic_viscosity.is_inviscid:
            self.method.add_diffusion_form(blf, lf)

        self.mixed_method.add_mixed_form(blf, lf)

        self.method.add_boundary_conditions(blf, lf)
        self.method.add_domain_conditions(blf, lf)

    def add_symbolic_temporal_forms(self,
                                    blf: dict[str, ngs.comp.SumOfIntegrals],
                                    lf: dict[str, ngs.comp.SumOfIntegrals]):
        self.method.add_symbolic_temporal_forms(blf, lf)

    def get_temporal_integrators(self):
        return self.method.get_temporal_integrators()

    def get_fields(self, quantities: dict[str, bool]) -> flowfields:

        U = self.method.get_conservative_gradient_fields(self.method.gfu['U'])
        if not isinstance(self.mixed_method, Inactive):
            U.update(self.mixed_method.get_mixed_fields(self.mixed_method.gfu['Q']))

        defaults = {'rho': True, 'u': True, 'p': True, 'T': True}
        defaults.update(quantities)

        fields = flowfields()
        for symbol, value in defaults.items():
            if not value:
                continue

            name = symbol
            if symbol in U.symbols:
                name = U.symbols[symbol]

            if name in U:
                fields[name] = U[name]

                if symbol in quantities:
                    quantities.pop(symbol)
                elif name in quantities:
                    quantities.pop(name)

        return fields

    def set_initial_conditions(self) -> None:
        super().set_initial_conditions()

        U = self.mesh.MaterialCF({dom: ngs.CF(
            (self.cfg.density(dc.fields),
             self.cfg.momentum(dc.fields),
             self.cfg.energy(dc.fields))) for dom, dc in self.cfg.dcs.to_pattern(Initial).items()})

        self.method.set_initial_conditions(U)

    method: HDG
    mixed_method: Inactive | StrainHeat | Gradient
