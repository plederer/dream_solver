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

        An_out = self.conservative_convective_jacobian_outgoing(Uhat, self.normal)
        An_in = self.conservative_convective_jacobian_incoming(Uhat, self.normal)

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

        amplitude_out = self.characteristic_amplitudes_outgoing(U, Q, Uhat, self.normal)

        rho = self.density(Uhat)
        c = self.speed_of_sound(Uhat)
        ut = InnerProduct(self.velocity(Uhat), t)
        un = InnerProduct(self.velocity(Uhat), n)
        Mn = IfPos(un, un, -un)/c

        if bc.type is bc.TYPE.PERFECT:
            amplitude_in = CF((0, 0, 0, 0))

        elif bc.type is bc.TYPE.PARTIALLY:
            ref_length = bc.reference_length

            amp = bc.sigma * c * (1 - Mn**2)/ref_length * (self.pressure(Uhat) - bc.pressure)
            amplitude_in = CF((amp, 0, 0, 0))

        if bc.tang_conv_flux:

            cf += self.conservative_convective_jacobian(Uhat, t) * (grad(U) * t) * Vhat
            gradient_p_t = InnerProduct(self.pressure_gradient(U, Q), t)
            gradient_u_t = self.velocity_gradient(U, Q) * t

            beta_l = 1
            beta_t = Mn

            amp_t = (1 - beta_l) * ut * (gradient_p_t - c*rho*InnerProduct(gradient_u_t, n))
            amp_t += (1 - beta_t) * c**2 * rho * InnerProduct(gradient_u_t, t)

            amplitude_in -= CF((amp_t, 0, 0, 0))

        if not mu.is_inviscid:

            if bc.tang_visc_flux:

                cons_diff_jac_tang = self.conservative_diffusive_jacobian(Uhat, Q, t) * (grad(U) * t)
                mixe_diff_jac_tang = self.mixed_diffusive_jacobian(Uhat, t) * (grad(Q) * t)

                cf -= cons_diff_jac_tang * Vhat
                cf -= mixe_diff_jac_tang * Vhat

                amplitude_in += self.P_inverse_matrix(Uhat, self.normal) * cons_diff_jac_tang
                amplitude_in += self.P_inverse_matrix(Uhat, self.normal) * mixe_diff_jac_tang

            if bc.norm_visc_flux:
                # test = grad(Q)
                # test = CF((test[0, :], 0, 0, test[2, :], 0, 0, test[4, :]), dims=(5, 2))
                # mixe_diff_jac_norm_test = self.mixed_diffusive_jacobian(Uhat, n) * (test * n)

                cons_diff_jac_norm = self.conservative_diffusive_jacobian(Uhat, Q, n) * (grad(U) * n)
                mixe_diff_jac_norm = self.mixed_diffusive_jacobian(Uhat, n) * (grad(Q) * n)

                cf -= cons_diff_jac_norm * Vhat
                # cf -= mixe_diff_jac_norm * Vhat

                amplitude_in += self.P_inverse_matrix(Uhat, self.normal) * cons_diff_jac_norm
                # amplitude_in += self.P_inverse_matrix(Uhat, self.normal) * mixe_diff_jac_norm

        amplitudes = amplitude_out + self.identity_matrix_incoming(Uhat, self.normal) * amplitude_in
        cf += (self.P_matrix(Uhat, self.normal) * amplitudes) * Vhat
        cf = cf * ds(skeleton=True, definedon=region, bonus_intorder=bonus_order_bnd)
        blf += cf.Compile(compile_flag)

    def _add_inviscid_wall_bilinearform(self, blf, boundary: str, bc: co.InviscidWall):

        bonus_order_bnd = self.cfg.bonus_int_order_bnd
        compile_flag = self.cfg.compile_flag

        region = self.mesh.Boundaries(boundary)

        U, _ = self.TnT.PRIMAL
        Uhat, Vhat = self.TnT.PRIMAL_FACET

        cf = InnerProduct(self.reflect(U)-Uhat, Vhat)
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

        tau_dE = self.diffusive_stabilisation_term(Uhat)[self.mesh.dim+1, self.mesh.dim+1]

        diff_rho = self.density(U) - self.density(Uhat)
        diff_rho_u = -self.momentum(Uhat)
        diff_rho_E = 1/(Re * Pr) * self.temperature_gradient(U, Q)*n - tau_dE * (self.energy(U) - self.energy(Uhat))

        cf = InnerProduct(CF((diff_rho, diff_rho_u, diff_rho_E)), Vhat)
        cf = cf * ds(skeleton=True, definedon=region, bonus_intorder=bonus_order_bnd)
        blf += cf.Compile(compile_flag)

    def M_matrix(self, U) -> CF:
        """
        The M matrix transforms primitive variables to conservative variables

        Equation E16.2.10, page 149

        Literature:
        [1] - C. Hirsch,
              Numerical Computation of Internal and External Flows Volume 2:
              Computational Methods for Inviscid and Viscous Flows,
              Vrije Universiteit Brussel, Brussels, Belgium
              ISBN: 978-0-471-92452-4
        """

        gamma = self.cfg.heat_capacity_ratio

        rho = self.density(U)
        u = self.velocity(U)
        ux, uy = u[0], u[1]

        M = CF((1, 0, 0, 0,
                ux, rho, 0, 0,
                uy, 0, rho, 0,
                0.5*InnerProduct(u, u), rho*ux, rho*uy, 1/(gamma - 1)), dims=(4, 4))

        return M

    def M_inverse_matrix(self, U) -> CF:
        """
        The M inverse matrix transforms conservative variables to primitive variables

        Equation E16.2.11, page 149

        Literature:
        [1] - C. Hirsch,
              Numerical Computation of Internal and External Flows Volume 2:
              Computational Methods for Inviscid and Viscous Flows,
              Vrije Universiteit Brussel, Brussels, Belgium
              ISBN: 978-0-471-92452-4
        """

        gamma = self.cfg.heat_capacity_ratio

        rho = self.density(U)
        u = self.velocity(U)
        ux, uy = u[0], u[1]

        Minv = CF((1, 0, 0, 0,
                   -ux/rho, 1/rho, 0, 0,
                   -uy/rho, 0, 1/rho, 0,
                   (gamma - 1)/2 * InnerProduct(u, u), -(gamma - 1) * ux,
                   -(gamma - 1) * uy, gamma - 1), dims=(4, 4))

        return Minv

    def L_matrix(self, U, unit_vector: CF) -> CF:
        """
        The L matrix transforms characteristic variables to primitive variables

        Equation E16.5.2, page 183

        Literature:
        [1] - C. Hirsch,
              Numerical Computation of Internal and External Flows Volume 2:
              Computational Methods for Inviscid and Viscous Flows,
              Vrije Universiteit Brussel, Brussels, Belgium
              ISBN: 978-0-471-92452-4
        """
        rho = self.density(U)
        c = self.speed_of_sound(U)

        d0, d1 = unit_vector[0], unit_vector[1]

        L = CF((0.5/c**2, 1/c**2, 0, 0.5/c**2,
                -d0/(2*c*rho), 0, d1, d0/(2*c*rho),
                -d1/(2*c*rho), 0, -d0, d1/(2*c*rho),
                0.5, 0, 0, 0.5), dims=(4, 4))

        return L

    def L_inverse_matrix(self, U, unit_vector: CF) -> CF:
        """
        The L inverse matrix transforms primitive variables to charactersitic variables

        Equation E16.5.1, page 182

        Literature:
        [1] - C. Hirsch,
              Numerical Computation of Internal and External Flows Volume 2:
              Computational Methods for Inviscid and Viscous Flows,
              Vrije Universiteit Brussel, Brussels, Belgium
              ISBN: 978-0-471-92452-4
        """
        rho = self.density(U)
        c = self.speed_of_sound(U)

        d0, d1 = unit_vector[0], unit_vector[1]

        Linv = CF((0, -rho*c*d0, -rho*c*d1, 1,
                   c**2, 0, 0, -1,
                   0, d1, -d0, 0,
                   0, rho*c*d0, rho*c*d1, 1), dims=(4, 4))

        return Linv

    def P_matrix(self, U, unit_vector: CF) -> CF:
        """
        The P matrix transforms characteristic variables to conservative variables

        Equation E16.5.3, page 183

        Literature:
        [1] - C. Hirsch,
              Numerical Computation of Internal and External Flows Volume 2:
              Computational Methods for Inviscid and Viscous Flows,
              Vrije Universiteit Brussel, Brussels, Belgium
              ISBN: 978-0-471-92452-4
        """
        return self.M_matrix(U) * self.L_matrix(U, unit_vector)

    def P_inverse_matrix(self, U, unit_vector: CF) -> CF:
        """
        The P inverse matrix transforms conservative variables to characteristic variables

        Equation E16.5.4, page 183

        Literature:
        [1] - C. Hirsch,
              Numerical Computation of Internal and External Flows Volume 2:
              Computational Methods for Inviscid and Viscous Flows,
              Vrije Universiteit Brussel, Brussels, Belgium
              ISBN: 978-0-471-92452-4
        """
        return self.L_inverse_matrix(U, unit_vector) * self.M_inverse_matrix(U)

    def primitive_convective_jacobian_x(self, U) -> CF:

        rho = self.density(U)
        c = self.speed_of_sound(U)
        u = self.velocity(U)[0]

        A = CF((
            u, rho, 0, 0,
            0, u, 0, 1/rho,
            0, 0, u, 0,
            0, rho*c**2, 0, u),
            dims=(4, 4))

        return A

    def primitive_convective_jacobian_y(self, U) -> CF:
        rho = self.density(U)
        c = self.speed_of_sound(U)
        v = self.velocity(U)[1]

        B = CF((v, 0, rho, 0,
                0, v, 0, 0,
                0, 0, v, 1/rho,
                0, 0, rho*c**2, v),
               dims=(4, 4))

        return B

    def primitive_convective_jacobian(self, U, unit_vector: CF) -> CF:
        A = self.primitive_convective_jacobian_x(U)
        B = self.primitive_convective_jacobian_y(U)
        return A * unit_vector[0] + B * unit_vector[1]

    def conservative_convective_jacobian_x(self, U) -> CF:
        '''
        First Jacobian of the convective Euler Fluxes F_c = (f_c, g_c) for conservative variables U
        A = \partial f_c / \partial U
        input: u = (rho, rho * u, rho * E)
        See also Page 144 in C. Hirsch, Numerical Computation of Internal and External Flows: Vol.2 
        '''

        gamma = self.cfg.heat_capacity_ratio
        velocity = self.velocity(U)
        ux, uy = velocity[0], velocity[1]
        u = InnerProduct(velocity, velocity)
        E = self.specific_energy(U)

        A = CF((
            0, 1, 0, 0,
            (gamma - 3)/2 * ux**2 + (gamma - 1)/2 * uy**2, (3 - gamma) * ux, -(gamma - 1) * uy, gamma - 1,
            -ux*uy, uy, ux, 0,
            -gamma*ux*E + (gamma - 1)*ux*u, gamma*E - (gamma - 1)/2 * (uy**2 + 3*ux**2), -(gamma - 1)*ux*uy, gamma*ux),
            dims=(4, 4))

        return A

    def conservative_convective_jacobian_y(self, U) -> CF:
        '''
        Second Jacobian of the convective Euler Fluxes F_c = (f_c, g_c)for conservative variables U
        B = \partial g_c / \partial U
        input: u = (rho, rho * u, rho * E)
        See also Page 144 in C. Hirsch, Numerical Computation of Internal and External Flows: Vol.2 
        '''
        gamma = self.cfg.heat_capacity_ratio
        velocity = self.velocity(U)
        ux, uy = velocity[0], velocity[1]
        u = InnerProduct(velocity, velocity)
        E = self.specific_energy(U)

        B = CF(
            (0, 0, 1, 0, -ux * uy, uy, ux, 0, (gamma - 3) / 2 * uy ** 2 + (gamma - 1) / 2 * ux ** 2, -(gamma - 1) * ux,
             (3 - gamma) * uy, gamma - 1, -gamma * uy * E + (gamma - 1) * uy * u, -(gamma - 1) * ux * uy, gamma * E -
             (gamma - 1) / 2 * (ux ** 2 + 3 * uy ** 2),
             gamma * uy),
            dims=(4, 4))

        return B

    def conservative_convective_jacobian(self, U, unit_vector: CF) -> CF:
        A = self.primitive_convective_jacobian(U, unit_vector)
        return self.primitive_to_conservative(A, U)

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

    def primitive_to_conservative(self, matrix, U) -> CF:
        return self.M_matrix(U) * matrix * self.M_inverse_matrix(U)

    def characteristic_to_conservative(self, matrix, U, unit_vector) -> CF:
        return self.P_matrix(U, unit_vector) * matrix * self.P_inverse_matrix(U, unit_vector)

    def conservative_to_primitive(self, matrix, U) -> CF:
        return self.M_inverse_matrix(U) * matrix * self.M_matrix(U)

    def conservative_to_characteristic(self, matrix, U, unit_vector) -> CF:
        return self.P_inverse_matrix(U, unit_vector) * matrix * self.P_matrix(U, unit_vector)

    def conservative_convective_jacobian_outgoing(self, U, unit_vector: CF) -> CF:
        Lambda = self.characteristic_velocities_outgoing(U, unit_vector)
        return self.characteristic_to_conservative(Lambda, U, unit_vector)

    def conservative_convective_jacobian_incoming(self, U, unit_vector: CF) -> CF:
        Lambda = self.characteristic_velocities_incoming(U, unit_vector)
        return self.characteristic_to_conservative(Lambda, U, unit_vector)

    def characteristic_variables(self, U, Q, Uhat, unit_vector: CF) -> CF:
        """
        The charachteristic amplitudes are defined as

            Amplitudes = Lambda * L_inverse * dV/dn,

        where Lambda is the eigenvalue matrix, L_inverse is the mapping from
        primitive variables to charachteristic variables and dV/dn is the
        derivative normal to the boundary.
        """
        rho = self.density(Uhat)
        c = self.speed_of_sound(Uhat)

        gradient_rho_dir = InnerProduct(self.density_gradient(U, Q), unit_vector)
        gradient_p_dir = InnerProduct(self.pressure_gradient(U, Q), unit_vector)
        gradient_u_dir = self.velocity_gradient(U, Q) * unit_vector

        variables = CF((
            gradient_p_dir - InnerProduct(gradient_u_dir, unit_vector) * c * rho,
            gradient_rho_dir * c**2 - gradient_p_dir,
            gradient_u_dir[0] * unit_vector[1] - gradient_u_dir[1] * unit_vector[0],
            gradient_p_dir + InnerProduct(gradient_u_dir, unit_vector) * c * rho
        ))

        return variables

    def characteristic_velocities(self, U, unit_vector: CF, absolute_value: bool = False) -> CF:
        """
        The Lambda matrix contains the eigenvalues of the Jacobian matrices

        Equation E16.5.21, page 180

        Literature:
        [1] - C. Hirsch,
              Numerical Computation of Internal and External Flows Volume 2:
              Computational Methods for Inviscid and Viscous Flows,
              Vrije Universiteit Brussel, Brussels, Belgium
              ISBN: 978-0-471-92452-4
        """
        c = self.speed_of_sound(U)
        u_dir = InnerProduct(self.velocity(U), unit_vector)

        lam_m_c = u_dir - c
        lam = u_dir
        lam_p_c = u_dir + c

        if absolute_value:
            lam_m_c = IfPos(lam_m_c, lam_m_c, -lam_m_c)
            lam = IfPos(lam, lam, -lam)
            lam_p_c = IfPos(lam_p_c, lam_p_c, -lam_p_c)

        Lambda = CF((lam_m_c, 0, 0, 0,
                     0, lam, 0, 0,
                     0, 0, lam, 0,
                     0, 0, 0, lam_p_c), dims=(4, 4))

        return Lambda

    def identity_matrix_outgoing(self, U, unit_vector: CF) -> CF:
        c = self.speed_of_sound(U)
        u_dir = InnerProduct(self.velocity(U), unit_vector)

        lam_m_c = IfPos(u_dir - c, 1, 0)
        lam = IfPos(u_dir, 1, 0)
        lam_p_c = IfPos(u_dir + c, 1, 0)

        identity = CF((lam_m_c, 0, 0, 0,
                       0, lam, 0, 0,
                       0, 0, lam, 0,
                       0, 0, 0, lam_p_c), dims=(4, 4))

        return identity

    def identity_matrix_incoming(self, U, unit_vector: CF) -> CF:
        c = self.speed_of_sound(U)
        u_dir = InnerProduct(self.velocity(U), unit_vector)

        lam_m_c = IfPos(u_dir - c, 0, 1)
        lam = IfPos(u_dir, 0, 1)
        lam_p_c = IfPos(u_dir + c, 0, 1)

        identity = CF((lam_m_c, 0, 0, 0,
                       0, lam, 0, 0,
                       0, 0, lam, 0,
                       0, 0, 0, lam_p_c), dims=(4, 4))

        return identity

    def characteristic_velocities_outgoing(self, U, unit_vector: CF, absolute_value: bool = False) -> CF:
        I_out = self.identity_matrix_outgoing(U, unit_vector)
        Lambda = self.characteristic_velocities(U, unit_vector, absolute_value)
        return I_out * Lambda

    def characteristic_velocities_incoming(self, U, unit_vector: CF, absolute_value: bool = False) -> CF:
        I_in = self.identity_matrix_incoming(U, unit_vector)
        Lambda = self.characteristic_velocities(U, unit_vector, absolute_value)
        return I_in * Lambda

    def characteristic_amplitudes_outgoing(self, U, Q, Uhat, unit_vector: CF) -> CF:
        Lambda_out = self.characteristic_velocities_outgoing(Uhat, unit_vector)
        W = self.characteristic_variables(U, Q, Uhat, unit_vector)
        return Lambda_out * W

    def characteristic_amplitudes_incoming(self, U, Q, Uhat, unit_vector: CF) -> CF:
        Lambda_in = self.characteristic_velocities_incoming(Uhat, unit_vector)
        W = self.characteristic_variables(U, Q, Uhat, unit_vector)
        return Lambda_in * W
