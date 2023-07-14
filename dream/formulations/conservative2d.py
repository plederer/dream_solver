from __future__ import annotations
from ngsolve import *

from .interface import MixedMethods, TensorIndices, TestAndTrialFunction
from .conservative import ConservativeFormulation, Indices
from .. import conditions as co


class ConservativeFormulation2D(ConservativeFormulation):

    _indices = Indices(DENSITY=0,
                       MOMENTUM=slice(1, 3),
                       ENERGY=3,
                       STRAIN=TensorIndices(XX=0, XY=1, YX=1, YY=2),
                       TEMPERATURE_GRADIENT=slice(3, 5))

    def _initialize_FE_space(self) -> ProductSpace:

        order = self.cfg.order
        mixed_method = self.cfg.mixed_method
        periodic = self.cfg.periodic

        V = L2(self.mesh, order=order)
        VHAT = FacetFESpace(self.mesh, order=order)
        # VHAT = H1(self.mesh, order=order, orderinner=0)
        Q = VectorL2(self.mesh, order=order)

        if periodic:
            VHAT = Periodic(VHAT)

        space = V**4 * VHAT**4

        if mixed_method is MixedMethods.NONE:
            pass
        elif mixed_method is MixedMethods.STRAIN_HEAT:
            space *= V**5
        elif mixed_method is MixedMethods.GRADIENT:
            space *= Q**4
        else:
            raise NotImplementedError(f"Mixed method {mixed_method} not implemented for {self}!")

        return space

    def _initialize_TnT(self) -> TestAndTrialFunction:
        return TestAndTrialFunction(*zip(*self.fes.TnT()))

    def _add_mixed_bilinearform(self, blf):

        bonus_vol = self.cfg.bonus_int_order_vol
        bonus_bnd = self.cfg.bonus_int_order_bnd
        mixed_method = self.cfg.mixed_method
        compile_flag = self.cfg.compile_flag
        n = self.normal

        U, V = self.TnT.PRIMAL
        Uhat, _ = self.TnT.PRIMAL_FACET
        Q, P = self.TnT.MIXED

        if mixed_method is MixedMethods.STRAIN_HEAT:

            gradient_P = grad(P)

            # Deviatoric Strain tensor
            u = self.velocity(U)
            uhat = self.velocity(Uhat)

            eps = self.deviatoric_strain_tensor(U, Q)
            zeta = self.deviatoric_strain_tensor(V, P)

            dev_zeta = 2 * zeta - 2/3 * (zeta[0, 0] + zeta[1, 1]) * Id(2)
            div_dev_zeta = 2 * CF((gradient_P[0, 0] + gradient_P[1, 1], gradient_P[1, 0] + gradient_P[2, 1]))
            div_dev_zeta -= 2/3 * CF((gradient_P[0, 0] + gradient_P[2, 0], gradient_P[0, 1] + gradient_P[2, 1]))

            var_form = InnerProduct(eps, zeta) * dx(bonus_intorder=bonus_vol)
            var_form += InnerProduct(u, div_dev_zeta) * dx(bonus_intorder=bonus_vol)
            var_form -= InnerProduct(uhat, dev_zeta*n) * dx(element_boundary=True, bonus_intorder=bonus_bnd)

            # Temperature gradient

            phi = self.temperature_gradient(U, Q)
            xi = self.temperature_gradient(V, P)
            T = self.temperature(U)
            That = self.temperature(Uhat)

            div_xi = gradient_P[3, 0] + gradient_P[4, 1]

            var_form += InnerProduct(phi, xi) * dx(bonus_intorder=bonus_vol)
            var_form += InnerProduct(T, div_xi) * dx(bonus_intorder=bonus_vol)
            var_form -= InnerProduct(That*n, xi) * dx(element_boundary=True, bonus_intorder=bonus_bnd)

        elif mixed_method is MixedMethods.GRADIENT:

            var_form = InnerProduct(Q, P) * dx(bonus_intorder=bonus_vol)
            var_form += InnerProduct(U, div(P)) * dx(bonus_intorder=bonus_vol)
            var_form -= InnerProduct(Uhat, P*n) * dx(element_boundary=True, bonus_intorder=bonus_bnd)

        else:
            raise NotImplementedError(f"Mixed Bilinearform: {mixed_method}")

        blf += var_form.Compile(compile_flag)

    def _add_linearform(self, lf, domain, value: co._Domain):

        mixed_method = self.cfg.mixed_method
        bonus_order_vol = self.cfg.bonus_int_order_vol
        bonus_order_bnd = self.cfg.bonus_int_order_bnd
        dim = self.mesh.dim

        region = self.mesh.Materials(domain)
        _, V = self.TnT.PRIMAL
        _, Vhat = self.TnT.PRIMAL_FACET
        _, P = self.TnT.MIXED

        U_f = CF((value.density, value.momentum, value.energy))
        cf = U_f * V * dx(definedon=region, bonus_intorder=bonus_order_vol)
        cf += U_f * Vhat * dx(element_boundary=True, definedon=region, bonus_intorder=bonus_order_bnd)

        if not isinstance(value, co.Perturbation):

            if mixed_method is MixedMethods.GRADIENT:
                Q_f = CF(tuple(U_f.Diff(dir) for dir in (x, y)), dims=(dim, dim+2)).trans
                cf += InnerProduct(Q_f, P) * dx(definedon=region, bonus_intorder=bonus_order_vol)

            elif mixed_method is MixedMethods.STRAIN_HEAT:
                velocity_gradient = CF(tuple(value.velocity.Diff(dir) for dir in (x, y)), dims=(dim, dim)).trans

                strain = velocity_gradient + velocity_gradient.trans
                strain -= 2/3 * (velocity_gradient[0, 0] + velocity_gradient[1, 1]) * Id(dim)

                gradient_T = CF(tuple(value.temperature.Diff(dir) for dir in (x, y)))

                Q_f = CF((strain[0, 0], strain[0, 1], strain[1, 1], gradient_T[0], gradient_T[1]))

                cf += InnerProduct(Q_f, P) * dx(definedon=region, bonus_intorder=bonus_order_vol)

        lf += cf

    def _add_sponge_bilinearform(self, blf, domain: str, dc: co.SpongeLayer):

        compile_flag = self.cfg.compile_flag

        region = self.mesh.Materials(domain)

        reference_values = CF((dc.density, dc.momentum, dc.energy))

        U, V = self.TnT.PRIMAL
        cf = dc.weight_function * (U - reference_values)
        cf = cf * V * dx(definedon=region, bonus_intorder=dc.weight_function_order)

        blf += cf.Compile(compile_flag)

    def _add_dirichlet_bilinearform(self, blf, boundary: str, bc: co.Dirichlet):

        bonus_order_bnd = self.cfg.bonus_int_order_bnd
        compile_flag = self.cfg.compile_flag

        region = self.mesh.Boundaries(boundary)

        Uhat, Vhat = self.TnT.PRIMAL_FACET

        dirichlet = CF((bc.density, bc.momentum, bc.energy))

        cf = (dirichlet-Uhat)
        cf = cf * Vhat * ds(skeleton=True, definedon=region, bonus_intorder=bonus_order_bnd)
        blf += cf.Compile(compile_flag)

    def _add_farfield_bilinearform(self, blf,  boundary: str, bc: co.FarField):

        bonus_order_bnd = self.cfg.bonus_int_order_bnd
        compile_flag = self.cfg.compile_flag

        region = self.mesh.Boundaries(boundary)

        farfield = CF((bc.density, bc.momentum, bc.energy))

        U, _ = self.TnT.PRIMAL
        Uhat, Vhat = self.TnT.PRIMAL_FACET

        u_out = self.characteristic_velocities(Uhat, self.normal, type="out", as_matrix=True)
        u_in = self.characteristic_velocities(Uhat, self.normal, type="in", as_matrix=True)

        An_out = self.DME_from_CHAR_matrix(u_out, Uhat, self.normal)
        An_in = self.DME_from_CHAR_matrix(u_in, Uhat, self.normal)

        cf = An_out * (U - Uhat)
        cf -= An_in * (farfield - Uhat)
        cf = cf * Vhat * ds(skeleton=True, definedon=region, bonus_intorder=bonus_order_bnd)

        blf += cf.Compile(compile_flag)

    def _add_outflow_bilinearform(self, blf, boundary: str, bc: co.Outflow):

        bonus_order_bnd = self.cfg.bonus_int_order_bnd
        compile_flag = self.cfg.compile_flag
        gamma = self.cfg.heat_capacity_ratio

        region = self.mesh.Boundaries(boundary)

        U, _ = self.TnT.PRIMAL
        Uhat, Vhat = self.TnT.PRIMAL_FACET

        rho = self.density(U)
        rho_u = self.momentum(U)

        energy = bc.pressure/(gamma - 1) + 1/(2*rho) * InnerProduct(rho_u, rho_u)
        outflow = CF((rho, rho_u, energy))

        cf = InnerProduct(outflow - Uhat, Vhat)
        cf = cf * ds(skeleton=True, definedon=region, bonus_intorder=bonus_order_bnd)

        blf += cf.Compile(compile_flag)

    def _add_nonreflecting_inflow_bilinearform(self, blf, boundary: str, bc: co.NRBC_Inflow):
        raise NotImplementedError()
        # bonus_order_bnd = self.cfg.bonus_int_order_bnd
        # compile_flag = self.cfg.compile_flag
        # mu = self.cfg.dynamic_viscosity
        # M = self.cfg.Mach_number

        # n = self.normal
        # t = self.tangential
        # region = self.mesh.Boundaries(boundary)

        # U, _ = self.TnT.PRIMAL
        # Uhat, Vhat = self.TnT.PRIMAL_FACET
        # Q, _ = self.TnT.MIXED

        # time_levels_gfu = self._gfus.get_component(1)
        # time_levels_gfu['n+1'] = Uhat

        # cf = InnerProduct(self.time_scheme.apply(time_levels_gfu), Vhat)

        # amplitude_out = self.characteristic_amplitudes_outgoing(U, Q, Uhat, self.normal)

        # rho = self.density(Uhat)
        # c = self.speed_of_sound(Uhat)
        # ut = InnerProduct(self.velocity(Uhat), t)
        # un = InnerProduct(self.velocity(Uhat), n)
        # Mn = IfPos(un, un, -un)/c

        # if bc.type is bc.TYPE.PERFECT:
        #     amplitude_in = CF((0, 0, 0, 0))

        # elif bc.type is bc.TYPE.PARTIALLY:
        #     ref_length = bc.reference_length

        #     amp = bc.sigma * c * (1 - Mn**2)/ref_length * (self.pressure(Uhat) - bc.pressure)
        #     amplitude_in = CF((amp, 0, 0, 0))

        # if bc.tang_conv_flux:

        #     tang_conv_flux = self.conservative_convective_jacobian(Uhat, t) * (grad(U) * t)

        #     cf += tang_conv_flux * Vhat
        #     amplitude_in -= CF((tang_conv_flux, tang_conv_flux, tang_conv_flux, 0))

        # if not mu.is_inviscid:

        #     if bc.tang_visc_flux:

        #         cons_diff_jac_tang = self.conservative_diffusive_jacobian(Uhat, Q, t) * (grad(U) * t)
        #         mixe_diff_jac_tang = self.mixed_diffusive_jacobian(Uhat, t) * (grad(Q) * t)

        #         cf -= cons_diff_jac_tang * Vhat
        #         cf -= mixe_diff_jac_tang * Vhat

        #         amplitude_in += self.P_inverse_matrix(Uhat, self.normal) * cons_diff_jac_tang
        #         amplitude_in += self.P_inverse_matrix(Uhat, self.normal) * mixe_diff_jac_tang

        #     if bc.norm_visc_flux:
        #         test = grad(Q)
        #         test = CF((test[0, :], 0, 0, test[2, :], 0, 0, test[4, :]), dims=(5, 2))
        #         mixe_diff_jac_norm_test = self.mixed_diffusive_jacobian(Uhat, n) * (test * n)

        #         cons_diff_jac_norm = self.conservative_diffusive_jacobian(Uhat, Q, n) * (grad(U) * n)
        #         mixe_diff_jac_norm = self.mixed_diffusive_jacobian(Uhat, n) * (grad(Q) * n)

        #         cf -= cons_diff_jac_norm * Vhat
        #         # cf -= mixe_diff_jac_norm * Vhat

        #         amplitude_in += self.P_inverse_matrix(Uhat, self.normal) * cons_diff_jac_norm
        #         # amplitude_in += self.P_inverse_matrix(Uhat, self.normal) * mixe_diff_jac_norm

        # amplitudes = amplitude_out + self.identity_matrix_incoming(Uhat, self.normal) * amplitude_in
        # cf += (self.P_matrix(Uhat, self.normal) * amplitudes) * Vhat
        # cf = cf * ds(skeleton=True, definedon=region, bonus_intorder=bonus_order_bnd)
        # blf += cf.Compile(compile_flag)

    def _add_nonreflecting_outflow_bilinearform(self, blf, boundary: str, bc: co.NRBC_Outflow):

        bonus_order_bnd = self.cfg.bonus_int_order_bnd
        compile_flag = self.cfg.compile_flag
        mu = self.cfg.dynamic_viscosity

        n = self.normal
        t = self.tangential
        region = self.mesh.Boundaries(boundary)

        U, _ = self.TnT.PRIMAL
        Uhat, Vhat = self.TnT.PRIMAL_FACET
        Q, _ = self.TnT.MIXED

        time_levels_gfu = self._gfus.get_component(1)
        time_levels_gfu['n+1'] = Uhat

        cf = InnerProduct(self.time_scheme.apply(time_levels_gfu), Vhat)

        amplitude_out = self.characteristic_amplitudes(U, Q, Uhat, self.normal, type="out")

        rho = self.density(Uhat)
        c = self.speed_of_sound(Uhat)
        ut = InnerProduct(self.velocity(Uhat), t)
        un = InnerProduct(self.velocity(Uhat), n)
        Mn = IfPos(un, un, -un)/c
        M = self.cfg.Mach_number

        P = self.DME_from_CHAR(Uhat, self.normal)
        Pinv = self.CHAR_from_DME(Uhat, self.normal)

        if bc.type is bc.TYPE.PERFECT:
            amplitude_in = CF((0, 0, 0, 0))

        elif bc.type is bc.TYPE.PARTIALLY:
            ref_length = bc.reference_length

            amp = bc.sigma * c * (1 - Mn**2)/ref_length * (self.pressure(Uhat) - bc.pressure)
            amplitude_in = CF((amp, 0, 0, 0))

        if bc.tang_conv_flux:

            gradient_p_t = InnerProduct(self.pressure_gradient(U, Q), t)
            gradient_u_t = self.velocity_gradient(U, Q) * t

            beta_l = Mn
            beta_t = Mn

            amp_t = (1 - beta_l) * ut * (gradient_p_t - c*rho*InnerProduct(gradient_u_t, n))
            amp_t += (1 - beta_t) * c**2 * rho * InnerProduct(gradient_u_t, t)

            cf += (self.DME_convective_jacobian(Uhat, t) * (grad(U) * t)) * Vhat
            amplitude_in -= CF((amp_t, 0, 0, 0))

        if not mu.is_inviscid:

            if bc.tang_visc_flux:

                cons_diff_jac_tang = self.conservative_diffusive_jacobian(Uhat, Q, t)
                cf -= (cons_diff_jac_tang * (grad(U) * t)) * Vhat
                amplitude_in += Pinv * (cons_diff_jac_tang * (grad(U) * t))

                mixe_diff_jac_tang = self.mixed_diffusive_jacobian(Uhat, t)
                cf -= (mixe_diff_jac_tang * (grad(Q) * t)) * Vhat
                amplitude_in += Pinv * (mixe_diff_jac_tang * (grad(Q) * t))

            if bc.norm_visc_flux:

                cons_diff_jac_norm = self.conservative_diffusive_jacobian(Uhat, Q, n)
                cf -= cons_diff_jac_norm * (grad(U) * n) * Vhat
                amplitude_in += Pinv * (cons_diff_jac_norm * (grad(U) * n))

                # test = grad(Q)
                # test = CF((test[0, :], 0, 0,  0, 0, 0, 0, 0,0), dims=(5, 2))

                # mixe_diff_jac_norm = self.mixed_diffusive_jacobian(Uhat, n)
                # cf -= (mixe_diff_jac_norm * (grad(Q) * n)) * Vhat
                # amplitude_in += Pinv * ((mixe_diff_jac_norm) * (grad(Q) * n))

        amplitudes = amplitude_out + self.identity_matrix_incoming(Uhat, self.normal) * amplitude_in
        cf += (P * amplitudes) * Vhat
        cf = cf * ds(skeleton=True, definedon=region, bonus_intorder=bonus_order_bnd)
        blf += cf.Compile(compile_flag)

    def _add_inviscid_wall_bilinearform(self, blf, boundary: str, bc: co.InviscidWall):

        bonus_order_bnd = self.cfg.bonus_int_order_bnd
        compile_flag = self.cfg.compile_flag

        region = self.mesh.Boundaries(boundary)

        U, _ = self.TnT.PRIMAL
        Uhat, Vhat = self.TnT.PRIMAL_FACET

        n = self.normal

        rho = self.density(U)
        rho_u = self.momentum(U)
        rho_E = self.energy(U)
        U_wall = CF((rho, rho_u - InnerProduct(rho_u, n)*n, rho_E))

        cf = InnerProduct(U_wall-Uhat, Vhat)
        cf = cf * ds(skeleton=True, definedon=region, bonus_intorder=bonus_order_bnd)

        blf += cf.Compile(compile_flag)

    def _add_isothermal_wall_bilinearform(self, blf, boundary: str, bc: co.IsothermalWall):

        bonus_order_bnd = self.cfg.bonus_int_order_bnd
        compile_flag = self.cfg.compile_flag
        gamma = self.cfg.heat_capacity_ratio

        region = self.mesh.Boundaries(boundary)

        U, _ = self.TnT.PRIMAL
        Uhat, Vhat = self.TnT.PRIMAL_FACET

        rho = self.density(U)
        rho_E = rho * bc.temperature / gamma

        cf = InnerProduct(CF((rho, 0, 0, rho_E)) - Uhat, Vhat)
        cf = cf * ds(skeleton=True, definedon=region, bonus_intorder=bonus_order_bnd)

        blf += cf.Compile(compile_flag)

    def _add_adiabatic_wall_bilinearform(self, blf, boundary: str, bc: co.AdiabaticWall):

        mixed_method = self.cfg.mixed_method
        if mixed_method is MixedMethods.NONE:
            raise NotImplementedError(f"Adiabatic wall not implemented for {mixed_method}")

        bonus_order_bnd = self.cfg.bonus_int_order_bnd
        compile_flag = self.cfg.compile_flag

        Re = self.cfg.Reynolds_number
        Pr = self.cfg.Prandtl_number
        n = self.normal

        region = self.mesh.Boundaries(boundary)

        U, _ = self.TnT.PRIMAL
        Uhat, Vhat = self.TnT.PRIMAL_FACET
        Q, _ = self.TnT.MIXED

        tau_dE = self.diffusive_stabilisation_matrix(Uhat)[self.mesh.dim+1, self.mesh.dim+1]

        diff_rho = self.density(U) - self.density(Uhat)
        diff_rho_u = -self.momentum(Uhat)
        diff_rho_E = 1/(Re * Pr) * self.temperature_gradient(U, Q)*n - tau_dE * (self.energy(U) - self.energy(Uhat))

        cf = InnerProduct(CF((diff_rho, diff_rho_u, diff_rho_E)), Vhat)
        cf = cf * ds(skeleton=True, definedon=region, bonus_intorder=bonus_order_bnd)
        blf += cf.Compile(compile_flag)

    def conservative_diffusive_jacobian_x(self, U, Q):

        rho = self.density(U)
        stess_tensor = self.deviatoric_stress_tensor(U, Q)
        txx, txy = stess_tensor[0, 0], stess_tensor[0, 1]
        ux, uy = self.velocity(U)

        A = CF((
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            -txx*ux/rho - txy*uy/rho, txx/rho, txy/rho, 0
        ), dims=(4, 4))

        return A

    def conservative_diffusive_jacobian_y(self, U, Q):

        rho = self.density(U)
        stress_tensor = self.deviatoric_stress_tensor(U, Q)
        tyx, tyy = stress_tensor[1, 0], stress_tensor[1, 1]
        ux, uy = self.velocity(U)

        B = CF((
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            -tyx*ux/rho - tyy*uy/rho, tyx/rho, tyy/rho, 0
        ), dims=(4, 4))

        return B

    def conservative_diffusive_jacobian(self, U, Q, unit_vector: CF) -> CF:
        A = self.conservative_diffusive_jacobian_x(U, Q)
        B = self.conservative_diffusive_jacobian_y(U, Q)
        return A * unit_vector[0] + B * unit_vector[1]

    def mixed_diffusive_jacobian_x(self, U):

        Re = self.cfg.Reynolds_number
        Pr = self.cfg.Prandtl_number
        mu = self.dynamic_viscosity(U)

        ux, uy = self.velocity(U)

        A = CF((
            0, 0, 0, 0, 0,
            1, 0, 0, 0, 0,
            0, 1, 0, 0, 0,
            ux, uy, 0, 1/Pr, 0
        ), dims=(4, 5)) * mu/Re

        return A

    def mixed_diffusive_jacobian_y(self, U):

        Re = self.cfg.Reynolds_number
        Pr = self.cfg.Prandtl_number
        mu = self.dynamic_viscosity(U)

        ux, uy = self.velocity(U)

        B = CF((
            0, 0, 0, 0, 0,
            0, 1, 0, 0, 0,
            0, 0, 1, 0, 0,
            0, ux, uy, 0, 1/Pr
        ), dims=(4, 5)) * mu/Re

        return B

    def mixed_diffusive_jacobian(self, U, unit_vector: CF) -> CF:
        A = self.mixed_diffusive_jacobian_x(U)
        B = self.mixed_diffusive_jacobian_y(U)
        return A * unit_vector[0] + B * unit_vector[1]
