from __future__ import annotations

from ngsolve import *

from .interface import TestAndTrialFunction, MixedMethods, RiemannSolver
from .primitive import PrimitiveFormulation, Indices
from .. import conditions as co


class PrimitiveFormulation2D(PrimitiveFormulation):

    _indices = Indices(PRESSURE=0, VELOCITY=slice(1, 3), TEMPERATURE=3)

    def _initialize_FE_space(self) -> ProductSpace:

        order = self.cfg.order
        mixed_method = self.cfg.mixed_method
        periodic = self.cfg.periodic

        V = L2(self.mesh, order=order)
        VHAT = FacetFESpace(self.mesh, order=order)
        Q = VectorL2(self.mesh, order=order)

        if periodic:
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

    def _add_linearform(self, lf, domain, value: co._Domain):
        bonus_order_vol = self.cfg.bonus_int_order_vol
        bonus_order_bnd = self.cfg.bonus_int_order_bnd

        region = self.mesh.Materials(domain)
        _, V = self.TnT.PRIMAL
        _, Vhat = self.TnT.PRIMAL_FACET

        initial_U = CF((value.pressure, value.velocity, value.temperature))
        cf = initial_U * V * dx(definedon=region, bonus_intorder=bonus_order_vol)
        cf += initial_U * Vhat * dx(element_boundary=True, definedon=region, bonus_intorder=bonus_order_bnd)

        lf += cf

    def _add_farfield_bilinearform(self, blf,  boundary: str, bc: co.FarField):

        bonus_order_bnd = self.cfg.bonus_int_order_bnd
        compile_flag = self.cfg.compile_flag

        region = self.mesh.Boundaries(boundary)

        farfield = CF((bc.pressure, bc.velocity, bc.temperature))

        U, _ = self.TnT.PRIMAL
        Uhat, Vhat = self.TnT.PRIMAL_FACET

        u_out = self.characteristic_velocities(Uhat, self.normal, type="out", as_matrix=True)
        u_in = self.characteristic_velocities(Uhat, self.normal, type="in", as_matrix=True)

        An_out = self.PVT_from_CHAR_matrix(u_out, Uhat, self.normal)
        An_in = self.PVT_from_CHAR_matrix(u_in, Uhat, self.normal)

        cf = An_out * (U - Uhat)
        cf += An_in * (farfield - Uhat)
        cf = cf * Vhat * ds(skeleton=True, definedon=region, bonus_intorder=bonus_order_bnd)

        blf += cf.Compile(compile_flag)

    def _add_inviscid_wall_bilinearform(self, blf, boundary: str, bc: co.InviscidWall):

        bonus_order_bnd = self.cfg.bonus_int_order_bnd
        compile_flag = self.cfg.compile_flag

        region = self.mesh.Boundaries(boundary)

        U, _ = self.TnT.PRIMAL
        Uhat, Vhat = self.TnT.PRIMAL_FACET

        n = self.normal
        p = self.pressure(U)
        u = self.velocity(U)
        T = self.temperature(U)

        U_wall = CF((p, u - InnerProduct(u, n)*n, T))

        cf = InnerProduct(U_wall - Uhat, Vhat)
        cf = cf * ds(skeleton=True, definedon=region, bonus_intorder=bonus_order_bnd)

        blf += cf.Compile(compile_flag)

    def _add_outflow_bilinearform(self, blf, boundary: str, bc: co.Outflow):

        bonus_order_bnd = self.cfg.bonus_int_order_bnd
        compile_flag = self.cfg.compile_flag

        region = self.mesh.Boundaries(boundary)

        U, _ = self.TnT.PRIMAL
        Uhat, Vhat = self.TnT.PRIMAL_FACET

        u = self.velocity(U)
        T = self.temperature(U)

        outflow = CF((bc.pressure, u, T))

        cf = InnerProduct(outflow - Uhat, Vhat)
        cf = cf * ds(skeleton=True, definedon=region, bonus_intorder=bonus_order_bnd)

        blf += cf.Compile(compile_flag)

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