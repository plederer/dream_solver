from __future__ import annotations
import numpy as np
from ngsolve import *

from .interface import TestAndTrialFunction, MixedMethods, _Formulation
from ..region import BoundaryConditions as bcs
from ..region import DomainConditions as dcs

from typing import Optional, NamedTuple


class Indices(NamedTuple):
    PRESSURE: int
    VELOCITY: slice
    TEMPERATURE: int


class PrimitiveFormulation(_Formulation):

    _indices: Indices

    def add_time_bilinearform(self, blf):

        bonus_order_vol = self.cfg.bonus_int_order_vol
        compile_flag = self.cfg.compile_flag

        U, V = self.TnT.PRIMAL

        time_levels_gfu = self._gfus.get_component(0)
        time_levels_gfu['n+1'] = U
        Gamma = self.DME_from_PVT(U)

        var_form = InnerProduct(Gamma * self.time_scheme.apply(time_levels_gfu), V) * dx(bonus_intorder=bonus_order_vol)

        blf += (var_form).Compile(compile_flag)

    def add_convective_bilinearform(self, blf):

        compile_flag = self.cfg.compile_flag
        bonus_order_vol = self.cfg.bonus_int_order_vol
        bonus_order_bnd = self.cfg.bonus_int_order_bnd

        U, V = self.TnT.PRIMAL
        Uhat, Vhat = self.TnT.PRIMAL_FACET

        # Subtract boundary regions
        mask_fes = FacetFESpace(self.mesh, order=0)
        mask = GridFunction(mask_fes)
        mask.vec[:] = 0
        mask.vec[~mask_fes.GetDofs(self.dmesh.boundary(self.dmesh.bcs.pattern))] = 1

        var_form = -InnerProduct(self.convective_flux(U), grad(V)) * dx(bonus_intorder=bonus_order_vol)
        var_form += InnerProduct(self.convective_numerical_flux(U, Uhat, self.normal),
                                 V) * dx(element_boundary=True, bonus_intorder=bonus_order_bnd)
        var_form += mask * InnerProduct(self.convective_numerical_flux(U, Uhat, self.normal),
                                        Vhat) * dx(element_boundary=True, bonus_intorder=bonus_order_bnd)

        blf += var_form.Compile(compile_flag)

    def add_diffusive_bilinearform(self, blf) -> None:
        raise NotImplementedError()

    def add_initial_linearform(self, lf):

        mixed_method = self.cfg.mixed_method
        bonus_vol = self.cfg.bonus_int_order_vol
        bonus_bnd = self.cfg.bonus_int_order_bnd

        for domain, dc in self.dmesh.dcs.initial_conditions.items():

            dim = self.mesh.dim

            _, V = self.TnT.PRIMAL
            _, Vhat = self.TnT.PRIMAL_FACET
            _, P = self.TnT.MIXED

            state = self.calc.determine_missing(dc.state)
            U_f = CF((state.pressure, state.velocity, state.temperature))

            cf = U_f * V * dx(definedon=domain, bonus_intorder=bonus_vol)
            cf += U_f * Vhat * dx(element_boundary=True, definedon=domain, bonus_intorder=bonus_bnd)

            if mixed_method is not MixedMethods.NONE:
                raise NotImplementedError()
            lf += cf

    def add_perturbation_linearform(self, lf):
        bonus_vol = self.cfg.bonus_int_order_vol
        bonus_bnd = self.cfg.bonus_int_order_bnd

        for domain, dc in self.dmesh.dcs.perturbations.items():

            _, V = self.TnT.PRIMAL
            _, Vhat = self.TnT.PRIMAL_FACET

            state = self.calc.determine_missing(dc.state)
            U_f = CF((state.pressure, state.velocity, state.temperature))

            cf = U_f * V * dx(definedon=domain, bonus_intorder=bonus_vol)
            cf += U_f * Vhat * dx(element_boundary=True, definedon=domain, bonus_intorder=bonus_bnd)

            lf += cf

    def add_forcing_linearform(self, lf):
        bonus_vol = self.cfg.bonus_int_order_vol
        bonus_bnd = self.cfg.bonus_int_order_bnd

        for domain, dc in self.dmesh.dcs.force.items():

            _, V = self.TnT.PRIMAL
            _, Vhat = self.TnT.PRIMAL_FACET

            if dc.state is None:
                continue

            state = self.calc.determine_missing(dc.state)
            U_f = CF((state.pressure, state.velocity, state.temperature))

            cf = U_f * V * dx(definedon=domain, bonus_intorder=bonus_vol)
            cf += U_f * Vhat * dx(element_boundary=True, definedon=domain, bonus_intorder=bonus_bnd)

            lf += cf

    def _add_farfield_bilinearform(self, blf,  boundary: Region, bc: bcs.FarField):

        compile_flag = self.cfg.compile_flag
        bonus_order_bnd = self.cfg.bonus_int_order_bnd

        state = self.calc.determine_missing(bc.state)
        farfield = CF((state.pressure, state.velocity, state.temperature))

        U, _ = self.TnT.PRIMAL
        Uhat, Vhat = self.TnT.PRIMAL_FACET

        u_out = self.characteristic_velocities(Uhat, self.normal, type="out", as_matrix=True)
        u_in = self.characteristic_velocities(Uhat, self.normal, type="in", as_matrix=True)

        An_out = self.PVT_from_CHAR_matrix(u_out, Uhat, self.normal)
        An_in = self.PVT_from_CHAR_matrix(u_in, Uhat, self.normal)

        cf = An_out * (U - Uhat)
        cf += An_in * (farfield - Uhat)
        cf = cf * Vhat * ds(skeleton=True, definedon=boundary, bonus_intorder=bonus_order_bnd)

        blf += cf.Compile(compile_flag)

    def _add_inviscid_wall_bilinearform(self, blf, boundary: Region, bc: bcs.InviscidWall):

        bonus_order_bnd = self.cfg.bonus_int_order_bnd
        compile_flag = self.cfg.compile_flag

        U, _ = self.TnT.PRIMAL
        Uhat, Vhat = self.TnT.PRIMAL_FACET

        n = self.normal
        p = self.pressure(U)
        u = self.velocity(U)
        T = self.temperature(U)

        U_wall = CF((p, u - InnerProduct(u, n)*n, T))

        cf = InnerProduct(U_wall - Uhat, Vhat)
        cf = cf * ds(skeleton=True, definedon=boundary, bonus_intorder=bonus_order_bnd)

        blf += cf.Compile(compile_flag)

    def _add_outflow_bilinearform(self, blf, boundary: Region, bc: bcs.Outflow):

        bonus_order_bnd = self.cfg.bonus_int_order_bnd
        compile_flag = self.cfg.compile_flag

        U, _ = self.TnT.PRIMAL
        Uhat, Vhat = self.TnT.PRIMAL_FACET

        u = self.velocity(U)
        T = self.temperature(U)

        outflow = CF((bc.state.pressure, u, T))

        cf = InnerProduct(outflow - Uhat, Vhat)
        cf = cf * ds(skeleton=True, definedon=boundary, bonus_intorder=bonus_order_bnd)

        blf += cf.Compile(compile_flag)

    def convective_numerical_flux(self, U, Uhat, unit_vector: CF):
        """
        Convective numerical flux

        Equation 34, page 16

        Literature:
        [1] - Vila-Pérez, J., Giacomini, M., Sevilla, R. et al.
              Hybridisable Discontinuous Galerkin Formulation of Compressible Flows.
              Arch Computat Methods Eng 28, 753–784 (2021).
              https://doi.org/10.1007/s11831-020-09508-z
        """
        G = self.DME_from_PVT(Uhat)
        tau_c = self.convective_stabilisation_matrix(Uhat, unit_vector)
        return self.convective_flux(Uhat)*unit_vector + G * (tau_c * (U - Uhat))

    def pressure(self, U: Optional[CF] = None):
        U = self._if_none_replace_with_gfu(U)
        return U[self._indices.PRESSURE]

    def velocity(self, U: Optional[CF] = None):
        U = self._if_none_replace_with_gfu(U)
        return U[self._indices.VELOCITY]

    def temperature(self, U: Optional[CF] = None):
        U = self._if_none_replace_with_gfu(U)
        return U[self._indices.TEMPERATURE]

    def density(self, U: Optional[CF] = None):
        gamma = self.cfg.heat_capacity_ratio
        p = self.pressure(U)
        T = self.temperature(U)
        return gamma/(gamma - 1) * p/T

    def momentum(self, U: Optional[CF] = None):
        rho = self.density(U)
        u = self.velocity(U)
        return rho * u

    def energy(self, U: Optional[CF] = None):
        rho = self.density(U)
        E = self.specific_energy(U)
        return rho * E

    def specific_energy(self, U: Optional[CF] = None):
        Ei = self.specific_inner_energy(U)
        Ek = self.specific_kinetic_energy(U)
        return Ei + Ek

    def kinetic_energy(self, U: Optional[CF] = None):
        rho = self.density(U)
        Ek = self.specific_kinetic_energy(U)
        return rho*Ek

    def specific_kinetic_energy(self, U: Optional[CF] = None):
        u = self.velocity(U)
        return InnerProduct(u, u)/2

    def inner_energy(self, U: Optional[CF] = None):
        gamma = self.cfg.heat_capacity_ratio
        p = self.pressure(U)
        return p/(gamma - 1)

    def specific_inner_energy(self, U: Optional[CF] = None):
        gamma = self.cfg.heat_capacity_ratio
        T = self.temperature(U)
        return T/gamma

    def enthalpy(self, U: Optional[CF] = None):
        return self.energy(U) + self.pressure(U)

    def specific_enthalpy(self, U: Optional[CF] = None):
        gamma = self.cfg.heat_capacity_ratio
        return self.specific_energy(U) + (gamma - 1)/gamma * self.temperature(U)

    def speed_of_sound(self, U: Optional[CF] = None):
        gamma = self.cfg.heat_capacity_ratio
        T = self.temperature(U)
        return sqrt((gamma - 1) * T)

    def density_gradient(self, U: Optional[CF] = None, Q: Optional[CF] = None):
        gamma = self.cfg.heat_capacity_ratio

        p = self.pressure(U)
        gradient_p = self.pressure_gradient(U, Q)
        T = self.temperature(U)
        gradient_T = self.temperature_gradient(U, Q)

        gradient_rho = gamma/(gamma - 1) * (gradient_p/T - p/T**2 * gradient_T)
        return gradient_rho

    def velocity_gradient(self, U: Optional[CF] = None, Q: Optional[CF] = None):
        mixed_method = self.cfg.mixed_method

        if mixed_method is MixedMethods.GRADIENT:
            Q = self._if_none_replace_with_gfu(Q, component=2)
            gradient_u = Q[self._indices.VELOCITY, :]
        else:
            U = self._if_none_replace_with_gfu(U)
            gradient_U = grad(U)
            gradient_u = gradient_U[self._indices.VELOCITY, :]

        return gradient_u

    def momentum_gradient(self, U: Optional[CF] = None, Q: Optional[CF] = None):
        rho = self.density(U)
        gradient_rho = self.density_gradient(U, Q)

        u = self.velocity(U)
        gradient_u = self.velocity_gradient(U, Q)

        return rho * gradient_u + OuterProduct(u, gradient_rho)

    def pressure_gradient(self, U: Optional[CF] = None, Q: Optional[CF] = None):
        mixed_method = self.cfg.mixed_method

        if mixed_method is MixedMethods.GRADIENT:
            Q = self._if_none_replace_with_gfu(Q, component=2)
            gradient_p = Q[self._indices.PRESSURE, :]
        else:
            U = self._if_none_replace_with_gfu(U)
            gradient_U = grad(U)
            gradient_p = gradient_U[self._indices.PRESSURE, :]

        return gradient_p

    def temperature_gradient(self, U: Optional[CF] = None, Q: Optional[CF] = None):
        mixed_method = self.cfg.mixed_method

        if mixed_method is MixedMethods.GRADIENT:
            Q = self._if_none_replace_with_gfu(Q, component=2)
            gradient_T = Q[self._indices.TEMPERATURE, :]
        else:
            U = self._if_none_replace_with_gfu(U)
            gradient_U = grad(U)
            gradient_T = gradient_U[self._indices.TEMPERATURE, :]

        return gradient_T

    def inner_energy_gradient(self, U: Optional[CF] = None, Q: Optional[CF] = None):
        gamma = self.cfg.heat_capacity_ratio
        gradient_p = self.pressure_gradient(U, Q)
        return gradient_p / (gamma - 1)

    def specific_inner_energy_gradient(self, U: Optional[CF] = None, Q: Optional[CF] = None):
        gamma = self.cfg.heat_capacity_ratio
        gradient_T = self.temperature_gradient(U, Q)
        return gradient_T / gamma

    def kinetic_energy_gradient(self, U: Optional[CF] = True, Q: Optional[CF] = None):
        rho = self.density(U)
        gradient_rho = self.density_gradient(U, Q)

        Ek = self.specific_kinetic_energy(U)
        gradient_Ek = self.specific_kinetic_energy_gradient(U, Q)

        return rho * gradient_Ek + Ek * gradient_rho

    def specific_kinetic_energy_gradient(self, U: Optional[CF] = None, Q: Optional[CF] = None):
        u = self.velocity(U)
        gradient_u = self.velocity_gradient(U, Q)
        return gradient_u.trans * u

    def energy_gradient(self, U: Optional[CF] = None, Q: Optional[CF] = None):
        gradient_rho_Ei = self.inner_energy_gradient(U, Q)
        gradient_rho_Ek = self.kinetic_energy_gradient(U, Q)
        return gradient_rho_Ei + gradient_rho_Ek

    def specific_energy_gradient(self, U: Optional[CF] = None, Q: Optional[CF] = None):
        gradient_Ei = self.specific_inner_energy_gradient(U, Q)
        gradient_Ek = self.specific_kinetic_energy_gradient(U, Q)
        return gradient_Ei + gradient_Ek

    def enthalpy_gradient(self, U: Optional[CF] = None, Q: Optional[CF] = None):
        gradient_p = self.temperature_gradient(U, Q)
        return self.energy_gradient(U, Q) + gradient_p

    def specific_enthalpy_gradient(self, U: Optional[CF] = None, Q: Optional[CF] = None):
        gamma = self.cfg.heat_capacity_ratio
        gradient_T = self.temperature_gradient(U, Q)
        gradient_E = self.specific_energy_gradient(U, Q)
        return gradient_E + (gamma - 1)/gamma * gradient_T


class PrimitiveFormulation2D(PrimitiveFormulation):

    _indices = Indices(PRESSURE=0, VELOCITY=slice(1, 3), TEMPERATURE=3)

    def _initialize_FE_space(self) -> ProductSpace:

        order = self.cfg.order
        mixed_method = self.cfg.mixed_method

        p_sponge_layers = self.dmesh.dcs.psponge_layers
        if self.dmesh.highest_order_psponge > order:
            raise ValueError("Polynomial sponge order higher than polynomial discretization order")

        order_policy = ORDER_POLICY.CONSTANT
        if p_sponge_layers:
            order_policy = ORDER_POLICY.VARIABLE

        V = L2(self.mesh, order=order, order_policy=order_policy)
        VHAT = FacetFESpace(self.mesh, order=order)
        Q = VectorL2(self.mesh, order=order, order_policy=order_policy)

        if p_sponge_layers:

            vhat_dofs = BitArray(VHAT.ndof)
            vhat_dofs.Clear()

            for domain, bc in p_sponge_layers.items():
                domain = self.dmesh.domain(domain)

                for el in domain.Elements():
                    V.SetOrder(NodeId(ELEMENT, el.nr), bc.high_order)

                domain_dofs = VHAT.GetDofs(domain)
                for i in range(bc.high_order + 1, order + 1, 1):
                    domain_dofs[i::order + 1] = 0

                vhat_dofs |= domain_dofs

            p_dofs = ~VHAT.GetDofs(self.dmesh.domain(self.dmesh.pattern(p_sponge_layers.keys())))
            p_dofs |= vhat_dofs

            for idx in np.flatnonzero(~p_dofs):
                VHAT.SetCouplingType(idx, COUPLING_TYPE.UNUSED_DOF)

            VHAT = Compress(VHAT)
            V.UpdateDofTables()

        if self.dmesh.is_periodic:
            VHAT = Periodic(VHAT)

        space = V**4 * VHAT**4

        if mixed_method is MixedMethods.NONE:
            pass
        elif mixed_method is MixedMethods.GRADIENT:
            space *= Q**4
        else:
            raise NotImplementedError(f"Mixed method {mixed_method} not implemented for {self}!")

        return space

    def _initialize_TnT(self) -> TestAndTrialFunction:
        return TestAndTrialFunction(*zip(*self.fes.TnT()))

    # def _add_nonreflecting_outflow_bilinearform(self, blf, boundary: str, bc: co.NRBC_Outflow):

    #     bonus_order_bnd = self.cfg.bonus_int_order_bnd
    #     compile_flag = self.cfg.compile_flag
    #     mu = self.cfg.dynamic_viscosity

    #     n = self.normal
    #     t = self.tangential
    #     region = self.mesh.Boundaries(boundary)

    #     U, _ = self.TnT.PRIMAL
    #     Uhat, Vhat = self.TnT.PRIMAL_FACET
    #     Q, _ = self.TnT.MIXED

    #     time_levels_gfu = self._gfus.get_component(1)
    #     time_levels_gfu['n+1'] = Uhat

    #     cf = InnerProduct(self.time_scheme.apply(time_levels_gfu), Vhat)

    #     amplitude_out = self.characteristic_amplitudes(U, Q, Uhat, self.normal, type="out")

    #     rho = self.density(Uhat)
    #     c = self.speed_of_sound(Uhat)
    #     ut = InnerProduct(self.velocity(Uhat), t)
    #     un = InnerProduct(self.velocity(Uhat), n)
    #     Mn = IfPos(un, un, -un)/c
    #     M = self.cfg.Mach_number

    #     P = self.PVT_from_CHAR(Uhat, self.normal)
    #     Pinv = self.CHAR_from_PVT(Uhat, self.normal)

    #     if bc.type is bc.TYPE.PERFECT:
    #         amplitude_in = CF((0, 0, 0, 0))

    #     elif bc.type is bc.TYPE.PARTIALLY:
    #         ref_length = bc.reference_length

    #         amp = bc.sigma * c * (1 - Mn**2)/ref_length * (self.pressure(Uhat) - bc.pressure)
    #         amplitude_in = CF((amp, 0, 0, 0))

    #     if bc.tang_conv_flux:

    #         gradient_p_t = InnerProduct(self.pressure_gradient(U, Q), t)
    #         gradient_u_t = self.velocity_gradient(U, Q) * t

    #         beta_l = Mn
    #         beta_t = Mn

    #         amp_t = (1 - beta_l) * ut * (gradient_p_t - c*rho*InnerProduct(gradient_u_t, n))
    #         amp_t += (1 - beta_t) * c**2 * rho * InnerProduct(gradient_u_t, t)

    #         cf += (self.DME_convective_jacobian(Uhat, t) * (grad(U) * t)) * Vhat
    #         amplitude_in -= CF((amp_t, 0, 0, 0))

    #     if not mu.is_inviscid:

    #         if bc.tang_visc_flux:

    #             cons_diff_jac_tang = self.conservative_diffusive_jacobian(Uhat, Q, t)
    #             cf -= (cons_diff_jac_tang * (grad(U) * t)) * Vhat
    #             amplitude_in += Pinv * (cons_diff_jac_tang * (grad(U) * t))

    #             mixe_diff_jac_tang = self.mixed_diffusive_jacobian(Uhat, t)
    #             cf -= (mixe_diff_jac_tang * (grad(Q) * t)) * Vhat
    #             amplitude_in += Pinv * (mixe_diff_jac_tang * (grad(Q) * t))

    #         if bc.norm_visc_flux:

    #             cons_diff_jac_norm = self.conservative_diffusive_jacobian(Uhat, Q, n)
    #             cf -= cons_diff_jac_norm * (grad(U) * n) * Vhat
    #             amplitude_in += Pinv * (cons_diff_jac_norm * (grad(U) * n))

    #             # test = grad(Q)
    #             # test = CF((test[0, :], 0, 0,  0, 0, 0, 0, 0,0), dims=(5, 2))

    #             # mixe_diff_jac_norm = self.mixed_diffusive_jacobian(Uhat, n)
    #             # cf -= (mixe_diff_jac_norm * (grad(Q) * n)) * Vhat
    #             # amplitude_in += Pinv * ((mixe_diff_jac_norm) * (grad(Q) * n))

    #     amplitudes = amplitude_out + self.identity_matrix_incoming(Uhat, self.normal) * amplitude_in
    #     cf += (P * amplitudes) * Vhat
    #     cf = cf * ds(skeleton=True, definedon=region, bonus_intorder=bonus_order_bnd)
    #     blf += cf.Compile(compile_flag)
