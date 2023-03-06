from __future__ import annotations
from ngsolve import *

from .interface import MixedMethods, Indices, VectorCoordinates, TensorCoordinates, RiemannSolver, TestAndTrialFunction
from .conservative import ConservativeFormulation
from .. import conditions as co
from .. import viscosity as visc


class ConservativeFormulation2D(ConservativeFormulation):

    _indices = Indices(DENSITY=0,
                       MOMENTUM=VectorCoordinates(X=1, Y=2),
                       ENERGY=3,
                       STRAIN=TensorCoordinates(XX=0, XY=1, YX=1, YY=2),
                       TEMPERATURE_GRADIENT=VectorCoordinates(X=3, Y=4))

    def _initialize_FE_space(self) -> ProductSpace:

        order = self.solver_configuration.order
        mixed_method = self.solver_configuration.mixed_method
        periodic = self.solver_configuration.periodic

        V = L2(self.mesh, order=order)
        VHAT = FacetFESpace(self.mesh, order=order)
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

    def add_time_bilinearform(self, blf, old_components):

        bonus_order_vol = self.solver_configuration.bonus_int_order_vol
        compile_flag = self.solver_configuration.compile_flag

        U, V = self.TnT.PRIMAL

        old_components = tuple(gfu.components[0] for gfu in old_components)
        var_form = InnerProduct(self.time_scheme(U, *old_components), V) * dx(bonus_intorder=bonus_order_vol)

        blf += (var_form).Compile(compile_flag)

    def add_convective_bilinearform(self, blf):

        compile_flag = self.solver_configuration.compile_flag
        bonus_order_vol = self.solver_configuration.bonus_int_order_vol
        bonus_order_bnd = self.solver_configuration.bonus_int_order_bnd

        U, V = self.TnT.PRIMAL
        Uhat, Vhat = self.TnT.PRIMAL_FACET

        # Subtract boundary regions
        mask_fes = FacetFESpace(self.mesh, order=0)
        mask = GridFunction(mask_fes)
        mask.vec[~mask_fes.GetDofs(self.mesh.Boundaries(self.bcs.pattern))] = 1

        var_form = -InnerProduct(self.convective_flux(U), grad(V)) * dx(bonus_intorder=bonus_order_vol)
        var_form += InnerProduct(self.convective_numerical_flux(U, Uhat),
                                 V) * dx(element_boundary=True, bonus_intorder=bonus_order_bnd)
        var_form += mask * InnerProduct(self.convective_numerical_flux(U, Uhat),
                                        Vhat) * dx(element_boundary=True, bonus_intorder=bonus_order_bnd)

        blf += var_form.Compile(compile_flag)

    def add_diffusive_bilinearform(self, blf):

        mixed_method = self.solver_configuration.mixed_method
        compile_flag = self.solver_configuration.compile_flag
        bonus_order_vol = self.solver_configuration.bonus_int_order_vol
        bonus_order_bnd = self.solver_configuration.bonus_int_order_bnd

        U, V = self.TnT.PRIMAL
        Uhat, Vhat = self.TnT.PRIMAL_FACET
        Q, _ = self.TnT.MIXED

        # Subtract boundary regions
        mask_fes = FacetFESpace(self.mesh, order=0)
        mask = GridFunction(mask_fes)
        mask.vec[~mask_fes.GetDofs(self.mesh.Boundaries(self.bcs.pattern))] = 1

        var_form = InnerProduct(self.diffusive_flux(U, Q), grad(V)) * dx(bonus_intorder=bonus_order_vol)
        var_form -= InnerProduct(self.diffusive_numerical_flux(U, Uhat, Q),
                                 V) * dx(element_boundary=True, bonus_intorder=bonus_order_bnd)
        var_form -= mask * InnerProduct(self.diffusive_numerical_flux(U, Uhat, Q),
                                        Vhat) * dx(element_boundary=True, bonus_intorder=bonus_order_bnd)

        blf += var_form.Compile(compile_flag)

        if mixed_method is not MixedMethods.NONE:
            self._add_mixed_bilinearform(blf)

    def _add_mixed_bilinearform(self, blf):

        bonus_vol = self.solver_configuration.bonus_int_order_vol
        bonus_bnd = self.solver_configuration.bonus_int_order_bnd
        mixed_method = self.solver_configuration.mixed_method
        compile_flag = self.solver_configuration.compile_flag
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

    def _add_initial_linearform(self, lf, domain, value: co.Initial):

        mixed_method = self.solver_configuration.mixed_method
        bonus_order_vol = self.solver_configuration.bonus_int_order_vol
        bonus_order_bnd = self.solver_configuration.bonus_int_order_bnd
        compile_flag = self.solver_configuration.compile_flag

        region = self.mesh.Materials(domain)
        _, V = self.TnT.PRIMAL
        _, Vhat = self.TnT.PRIMAL_FACET
        _, P = self.TnT.MIXED

        initial_U = CF((value.density, value.momentum, value.energy))
        cf = initial_U * V * dx(definedon=region, bonus_intorder=bonus_order_vol)
        cf += initial_U * Vhat * dx(element_boundary=True, definedon=region, bonus_intorder=bonus_order_bnd)

        if mixed_method is MixedMethods.GRADIENT:
            initial_Q = CF(tuple(initial_U.Diff(dir) for dir in (x, y)), dims=(2, 4)).trans
            cf += InnerProduct(initial_Q, P) * dx(definedon=region, bonus_intorder=bonus_order_vol)

        elif mixed_method is MixedMethods.STRAIN_HEAT:
            velocity = self.velocity(initial_U)
            velocity_gradient = CF(tuple(velocity.Diff(dir) for dir in (x, y)), dims=(2, 2)).trans

            strain = velocity_gradient + velocity_gradient.trans
            strain -= 2/3 * (velocity_gradient[0, 0] + velocity_gradient[1, 1]) * Id(2)

            temperature = self.temperature(initial_U)
            temperature_gradient = CF(tuple(temperature.Diff(dir) for dir in (x, y)))

            initial_Q = CF(
                (strain[0, 0],
                 strain[0, 1],
                 strain[1, 1],
                 temperature_gradient[0],
                 temperature_gradient[1]))

            cf += InnerProduct(initial_Q, P) * dx(definedon=region, bonus_intorder=bonus_order_vol)

        lf += cf.Compile(compile_flag)

    def _add_sponge_bilinearform(self, blf, domain: str, dc: co.SpongeLayer):

        compile_flag = self.solver_configuration.compile_flag

        region = self.mesh.Materials(domain)

        reference_values = CF((dc.density, dc.momentum, dc.energy))

        U, V = self.TnT.PRIMAL
        cf = dc.weight_function * (U - reference_values)
        cf = cf * V * dx(definedon=region, bonus_intorder=dc.weight_function_order)

        blf += cf.Compile(compile_flag)

    def _add_dirichlet_bilinearform(self, blf, boundary: str, bc: co.Dirichlet):

        bonus_order_bnd = self.solver_configuration.bonus_int_order_bnd
        compile_flag = self.solver_configuration.compile_flag

        region = self.mesh.Boundaries(boundary)

        Uhat, Vhat = self.TnT.PRIMAL_FACET

        dirichlet = CF((bc.density, bc.momentum, bc.energy))

        cf = (dirichlet-Uhat)
        cf = cf * Vhat * ds(skeleton=True, definedon=region, bonus_intorder=bonus_order_bnd)
        blf += cf.Compile(compile_flag)

    def _add_farfield_bilinearform(self, blf,  boundary: str, bc: co.FarField):

        bonus_order_bnd = self.solver_configuration.bonus_int_order_bnd
        compile_flag = self.solver_configuration.compile_flag

        region = self.mesh.Boundaries(boundary)

        farfield = CF((bc.density, bc.momentum, bc.energy))

        U, _ = self.TnT.PRIMAL
        Uhat, Vhat = self.TnT.PRIMAL_FACET

        cf = self.An_matrix_outgoing(Uhat) * (U - Uhat)
        cf -= self.An_matrix_incoming(Uhat) * (farfield - Uhat)
        cf = cf * Vhat * ds(skeleton=True, definedon=region, bonus_intorder=bonus_order_bnd)

        blf += cf.Compile(compile_flag)

    def _add_outflow_bilinearform(self, blf, boundary: str, bc: co.Outflow):

        bonus_order_bnd = self.solver_configuration.bonus_int_order_bnd
        compile_flag = self.solver_configuration.compile_flag
        gamma = self.solver_configuration.heat_capacity_ratio

        region = self.mesh.Boundaries(boundary)

        U, _ = self.TnT.PRIMAL
        Uhat, Vhat = self.TnT.PRIMAL_FACET

        rho = self.density(U)
        rho_m = self.momentum(U)

        energy = bc.pressure/(gamma - 1) + 1/(2*rho) * InnerProduct(rho_m, rho_m)
        outflow = CF((rho, rho_m, energy))

        cf = InnerProduct(outflow - Uhat, Vhat)
        cf = cf * ds(skeleton=True, definedon=region, bonus_intorder=bonus_order_bnd)

        blf += cf.Compile(compile_flag)

    def _add_nonreflecting_inflow_bilinearform(self, blf, boundary: str, bc: co.NRBC_Inflow, old_components):

        bonus_order_bnd = self.solver_configuration.bonus_int_order_bnd
        compile_flag = self.solver_configuration.compile_flag
        viscosity = self.solver_configuration.dynamic_viscosity

        n = self.normal
        t = self.tangential
        region = self.mesh.Boundaries(boundary)

        old_components = tuple(gfu.components[1] for gfu in old_components)

        U, _ = self.TnT.PRIMAL
        Uhat, Vhat = self.TnT.PRIMAL_FACET

        cf = InnerProduct(self.time_scheme(Uhat, *old_components), Vhat)

        amplitude_out = self.characteristic_amplitudes_outgoing(U, Q, Uhat)
        I_in = self.identity_matrix_incoming(Uhat)

        if bc.type is bc.TYPE.PERFECT:
            amplitude_in = CF((0, 0, 0, 0))

        elif bc.type is bc.TYPE.PARTIALLY:
            diff_u = self.velocity(Uhat) - bc.velocity
            rho = self.density(Uhat)
            c = self.speed_of_sound(Uhat)
            M = self.solver_configuration.Mach_number
            entropy_amp = bc.sigma * c/bc.reference_length * (self.pressure(Uhat) - bc.pressure)
            contact_amp = bc.sigma * c/bc.reference_length * (diff_u[1] * n[0] - diff_u[0] * n[1])
            acoustic_amp = bc.sigma * rho * c**2 * (1-M**2)/bc.reference_length * InnerProduct(diff_u, n)
            amplitude_in = I_in * CF((entropy_amp, contact_amp, 0, acoustic_amp))

        P_inverse_inc = I_in * self.P_inverse_matrix(Uhat)
        if bc.tang_conv_flux:
            conv_flux_gradient = self.convective_flux_gradient(U, Q)
            conv_flux_gradient_tang = sum([conv_flux_gradient[:, i, :] * t[i] for i in range(self.mesh.dim)]) * t
            cf += conv_flux_gradient_tang * Vhat
            amplitude_in -= P_inverse_inc * conv_flux_gradient_tang

        if viscosity is not visc.DynamicViscosity.INVISCID:

            Q, _ = self.TnT.MIXED

            diff_flux_gradient = self.diffusive_flux_gradient(U, Q)

            if bc.tang_visc_flux:
                visc_flux_gradient_tang = fem.Einsum('ijk,k->ij', diff_flux_gradient, t) * t
                cf -= visc_flux_gradient_tang * Vhat
                amplitude_in += P_inverse_inc * visc_flux_gradient_tang

            if bc.norm_visc_flux:
                visc_flux_gradient_norm = fem.Einsum('ijk,k->ij', diff_flux_gradient, n) * n
                cf -= visc_flux_gradient_norm * Vhat
                amplitude_in += P_inverse_inc * visc_flux_gradient_norm

        amplitudes = amplitude_out + amplitude_in
        cf += self.P_matrix(Uhat) * amplitudes * Vhat

        cf = cf * ds(skeleton=True, definedon=region, bonus_intorder=bonus_order_bnd)
        blf += cf.Compile(compile_flag)

    def _add_nonreflecting_outflow_bilinearform(self, blf, boundary: str, bc: co.NRBC_Outflow, old_components):

        bonus_order_bnd = self.solver_configuration.bonus_int_order_bnd
        compile_flag = self.solver_configuration.compile_flag
        viscosity = self.solver_configuration.dynamic_viscosity

        n = self.normal
        t = self.tangential
        region = self.mesh.Boundaries(boundary)

        old_components = tuple(gfu.components[1] for gfu in old_components)

        U, _ = self.TnT.PRIMAL
        Uhat, Vhat = self.TnT.PRIMAL_FACET

        cf = InnerProduct(self.time_scheme(Uhat, *old_components), Vhat)

        amplitude_out = self.characteristic_amplitudes_outgoing(U, Q, Uhat)
        I_in = self.identity_matrix_incoming(Uhat)

        if bc.type is bc.TYPE.PERFECT:
            amplitude_in = CF((0, 0, 0, 0))

        elif bc.type is bc.TYPE.PARTIALLY:
            c = self.speed_of_sound(Uhat)
            M = self.solver_configuration.Mach_number
            ref_length = bc.reference_length
            rho = self.density(Uhat)

            amp = bc.sigma * c * (1 - M**2)/ref_length * (self.pressure(Uhat) - bc.pressure)
            amplitude_in = I_in * CF((0, 0, 0, amp))

        elif bc.type is bc.TYPE.PIROZZOLI:
            raise ValueError(f"Implementation details needed {bc.type}")
            rho = self.density(Uhat)
            dt = self.solver_configuration.time_step
            lam = self.characteristic_velocities(Uhat, False)[3, 3]
            M = self.mach_number(Uhat)

            L3 = (1 - M)/(lam * dt * rho) * (self.pressure(Uhat) - bc.pressure)
            amplitude_out = CF((amplitude_out[0], amplitude_out[1], amplitude_out[2], L3))

        P_inverse_inc = I_in * self.P_inverse_matrix(Uhat)
        if bc.tang_conv_flux:
            conv_flux_gradient = self.convective_flux_gradient(U, Q)
            conv_flux_gradient_tang = sum([conv_flux_gradient[:, i, :] * t[i] for i in range(2)]) * t
            cf += conv_flux_gradient_tang * Vhat
            amplitude_in -= P_inverse_inc * conv_flux_gradient_tang

        if viscosity is not visc.DynamicViscosity.INVISCID:

            Q, _ = self.TnT.MIXED

            diff_flux_gradient = self.diffusive_flux_gradient(U, Q)
            diff_flux_gradient_normal = self.diffusive_flux_gradient_normal(U, Q)

            if bc.tang_visc_flux:
                visc_flux_gradient_tang = fem.Einsum('ijk,k->ij', diff_flux_gradient, t) * t
                cf -= visc_flux_gradient_tang * Vhat
                amplitude_in += P_inverse_inc * visc_flux_gradient_tang

            if bc.norm_visc_flux:
                visc_flux_gradient_norm = fem.Einsum('ijk,k->ij', diff_flux_gradient, n) * n
                cf -= visc_flux_gradient_norm * Vhat
                visc_flux_gradient_norm = fem.Einsum('ijk,k->ij', diff_flux_gradient_normal, n) * n
                amplitude_in += P_inverse_inc * visc_flux_gradient_norm

        amplitudes = amplitude_out + amplitude_in
        cf += self.P_matrix(Uhat) * amplitudes * Vhat

        cf = cf * ds(skeleton=True, definedon=region, bonus_intorder=bonus_order_bnd)
        blf += cf.Compile(compile_flag)

    def _add_inviscid_wall_bilinearform(self, blf, boundary: str, bc: co.InviscidWall):

        bonus_order_bnd = self.solver_configuration.bonus_int_order_bnd
        compile_flag = self.solver_configuration.compile_flag

        region = self.mesh.Boundaries(boundary)

        U, _ = self.TnT.PRIMAL
        Uhat, Vhat = self.TnT.PRIMAL_FACET

        cf = InnerProduct(self.reflect(U)-Uhat, Vhat)
        cf = cf * ds(skeleton=True, definedon=region, bonus_intorder=bonus_order_bnd)

        blf += cf.Compile(compile_flag)

    def _add_isothermal_wall_bilinearform(self, blf, boundary: str, bc: co.IsothermalWall):

        bonus_order_bnd = self.solver_configuration.bonus_int_order_bnd
        compile_flag = self.solver_configuration.compile_flag
        gamma = self.solver_configuration.heat_capacity_ratio

        region = self.mesh.Boundaries(boundary)

        U, _ = self.TnT.PRIMAL
        Uhat, Vhat = self.TnT.PRIMAL_FACET

        rho = self.density(U)
        rho_E = rho * bc.temperature / gamma

        cf = InnerProduct(CF((rho, 0, 0, rho_E)) - Uhat, Vhat)
        cf = cf * ds(skeleton=True, definedon=region, bonus_intorder=bonus_order_bnd)

        blf += cf.Compile(compile_flag)

    def _add_adiabatic_wall_bilinearform(self, blf, boundary: str, bc: co.AdiabaticWall):

        mixed_method = self.solver_configuration.mixed_method
        if mixed_method is MixedMethods.NONE:
            raise NotImplementedError(f"Adiabatic wall not implemented for {mixed_method}")

        bonus_order_bnd = self.solver_configuration.bonus_int_order_bnd
        compile_flag = self.solver_configuration.compile_flag

        Re = self.solver_configuration.Reynolds_number
        Pr = self.solver_configuration.Prandtl_number
        n = self.normal

        region = self.mesh.Boundaries(boundary)

        U, _ = self.TnT.PRIMAL
        Uhat, Vhat = self.TnT.PRIMAL_FACET
        Q, _ = self.TnT.MIXED

        tau_dE = self.diffusive_stabilisation_term(Uhat, Q)[self.mesh.dim+1, self.mesh.dim+1]

        diff_rho = self.density(U) - self.density(Uhat)
        diff_rho_u = -self.momentum(Uhat)
        diff_rho_E = 1/(Re * Pr) * self.temperature_gradient(U, Q)*n - tau_dE * (self.energy(U) - self.energy(Uhat))

        cf = InnerProduct(CF((diff_rho, diff_rho_u, diff_rho_E)), Vhat)
        cf = cf * ds(skeleton=True, definedon=region, bonus_intorder=bonus_order_bnd)
        blf += cf.Compile(compile_flag)

    def convective_numerical_flux(self, U, Uhat):
        """
        Convective numerical flux

        Equation 34, page 16

        Literature:
        [1] - Vila-Pérez, J., Giacomini, M., Sevilla, R. et al.
              Hybridisable Discontinuous Galerkin Formulation of Compressible Flows.
              Arch Computat Methods Eng 28, 753–784 (2021).
              https://doi.org/10.1007/s11831-020-09508-z
        """
        riemann_solver = self.solver_configuration.riemann_solver
        n = self.normal

        if riemann_solver is RiemannSolver.LAX_FRIEDRICH:
            un = InnerProduct(self.velocity(Uhat), n)
            un_abs = IfPos(un, un, -un)
            c = self.speed_of_sound(Uhat)
            lambda_max = un_abs + c

            stabilisation_matrix = lambda_max * Id(self.mesh.dim + 2)

        elif riemann_solver is RiemannSolver.ROE:
            stabilisation_matrix = self.An_matrix(Uhat, absolute_value=True)

        elif riemann_solver is RiemannSolver.HLL:
            un = InnerProduct(self.velocity(Uhat), n)
            c = self.speed_of_sound(Uhat)
            splus = IfPos(un + c, un + c, 0)

            stabilisation_matrix = splus * Id(self.mesh.dim + 2)

        elif riemann_solver is RiemannSolver.HLLEM:
            un = InnerProduct(self.velocity(Uhat), n)
            un_abs = IfPos(un, un, -un)
            c = self.speed_of_sound(Uhat)

            theta_0 = 1e-6
            theta = un_abs/(un_abs + c)
            IfPos(theta - theta_0, theta, theta_0)
            Theta = CF((theta, 0, 0, 0,
                        0, theta, 0, 0,
                        0, 0, 1, 0,
                        0, 0, 0, 1), dims=(4, 4))

            Theta = self.P_matrix(Uhat) * Theta * self.P_inverse_matrix(Uhat)
            splus = IfPos(un + c, un + c, 0)

            stabilisation_matrix = splus * Theta

        return self.convective_flux(Uhat)*n + stabilisation_matrix * (U-Uhat)

    def diffusive_numerical_flux(self, U, Uhat, Q):

        n = self.normal
        tau_d = self.diffusive_stabilisation_term(Uhat, Q)

        return self.diffusive_flux(Uhat, Q)*n - tau_d * (U-Uhat)

    def diffusive_stabilisation_term(self, Uhat, Q):

        Re = self.solver_configuration.Reynolds_number
        Pr = self.solver_configuration.Prandtl_number
        mu = self.mu.get(Uhat, Q)

        tau_d = CF((0, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1/Pr), dims=(4, 4)) * mu / Re

        return tau_d

    def M_matrix(self, U):
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

        gamma = self.solver_configuration.heat_capacity_ratio

        rho = self.density(U)
        u = self.velocity(U)
        ux, uy = u[0], u[1]

        M = CF((1, 0, 0, 0,
                ux, rho, 0, 0,
                uy, 0, rho, 0,
                0.5*InnerProduct(u, u), rho*ux, rho*uy, 1/(gamma - 1)), dims=(4, 4))

        return M

    def M_inverse_matrix(self, U):
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

        gamma = self.solver_configuration.heat_capacity_ratio

        rho = self.density(U)
        u = self.velocity(U)
        ux, uy = u[0], u[1]

        Minv = CF((1, 0, 0, 0,
                   -ux/rho, 1/rho, 0, 0,
                   -uy/rho, 0, 1/rho, 0,
                   (gamma - 1)/2 * InnerProduct(u, u), -(gamma - 1) * ux,
                   -(gamma - 1) * uy, gamma - 1), dims=(4, 4))

        return Minv

    def L_matrix(self, U):
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

        n = self.normal
        nx, ny = n[0], n[1]

        L = CF((1/c**2, 0, 0.5/c**2, 0.5/c**2,
                0, -ny, nx/(2*c*rho), -nx/(2*c*rho),
                0, nx, ny/(2*c*rho), -ny/(2*c*rho),
                0, 0, 0.5, 0.5), dims=(4, 4))

        return L

    def L_inverse_matrix(self, U):
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

        n = self.normal
        nx, ny = n[0], n[1]

        Linv = CF((c**2, 0, 0, -1,
                   0, -ny, nx, 0,
                   0, rho*c*nx, rho*c*ny, 1,
                   0, -rho*c*nx, -rho*c*ny, 1), dims=(4, 4))

        return Linv

    def P_matrix(self, U):
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
        return self.M_matrix(U) * self.L_matrix(U)

    def P_inverse_matrix(self, U):
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
        return self.L_inverse_matrix(U) * self.M_inverse_matrix(U)

    def An_matrix(self, U, absolute_value: bool = False):
        return self.P_matrix(U) * self.characteristic_velocities(U, absolute_value) * self.P_inverse_matrix(U)

    def An_matrix_outgoing(self, U):
        return self.P_matrix(U) * self.characteristic_velocities_outgoing(U) * self.P_inverse_matrix(U)

    def An_matrix_incoming(self, U):
        return self.P_matrix(U) * self.characteristic_velocities_incoming(U) * self.P_inverse_matrix(U)

    def characteristic_variables(self, U, Q, Uhat):
        """
        The charachteristic amplitudes are defined as

            Amplitudes = Lambda * L_inverse * dV/dn,

        where Lambda is the eigenvalue matrix, L_inverse is the mapping from
        primitive variables to charachteristic variables and dV/dn is the
        derivative normal to the boundary.
        """
        rho = self.density(Uhat)
        c = self.speed_of_sound(Uhat)
        n = self.normal

        gradient_rho_normal = InnerProduct(self.density_gradient(U, Q), n)
        gradient_p_normal = InnerProduct(self.pressure_gradient(U, Q), n)
        gradient_u_normal = self.velocity_gradient(U, Q) * n

        variables = CF((
            gradient_rho_normal * c**2 - gradient_p_normal,
            gradient_u_normal[1] * n[0] - gradient_u_normal[0] * n[1],
            gradient_p_normal + InnerProduct(gradient_u_normal, n) * (c * rho),
            gradient_p_normal - InnerProduct(gradient_u_normal, n) * (c * rho)
        ))

        return variables

    def characteristic_velocities(self, U, absolute_value: bool = False) -> CF:
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
        vn = InnerProduct(self.velocity(U), self.normal)

        lam = vn
        lam_p_c = vn + c
        lam_m_c = vn - c

        if absolute_value:
            lam = IfPos(lam, lam, -lam)
            lam_p_c = IfPos(lam_p_c, lam_p_c, -lam_p_c)
            lam_m_c = IfPos(lam_m_c, lam_m_c, -lam_m_c)

        Lambda = CF((lam, 0, 0, 0,
                     0, lam, 0, 0,
                     0, 0, lam_p_c, 0,
                     0, 0, 0, lam_m_c), dims=(4, 4))

        return Lambda

    def characteristic_velocities_incoming(self, U, absolute_value=False):
        I_in = self.identity_matrix_incoming(U)
        return I_in * self.characteristic_velocities(U, absolute_value)

    def characteristic_velocities_outgoing(self, U, absolute_value=False):
        I_out = self.identity_matrix_outgoing(U)
        return I_out * self.characteristic_velocities(U, absolute_value)

    def identity_matrix_outgoing(self, U):
        c = self.speed_of_sound(U)
        vn = InnerProduct(self.velocity(U), self.normal)

        lam = IfPos(vn, 1, 0)
        lam_p_c = IfPos(vn + c, 1, 0)
        lam_m_c = IfPos(vn - c, 1, 0)

        identity = CF((lam, 0, 0, 0,
                       0, lam, 0, 0,
                       0, 0, lam_p_c, 0,
                       0, 0, 0, lam_m_c), dims=(4, 4))

        return identity

    def identity_matrix_incoming(self, U):
        c = self.speed_of_sound(U)
        vn = InnerProduct(self.velocity(U), self.normal)

        lam = IfPos(vn, 0, 1)
        lam_p_c = IfPos(vn + c, 0, 1)
        lam_m_c = IfPos(vn - c, 0, 1)

        identity = CF((lam, 0, 0, 0,
                       0, lam, 0, 0,
                       0, 0, lam_p_c, 0,
                       0, 0, 0, lam_m_c), dims=(4, 4))

        return identity

    def characteristic_amplitudes_outgoing(self, U, Q, Uhat):
        return self.characteristic_velocities_outgoing(Uhat) * self.characteristic_variables(U, Q, Uhat)

    def characteristic_amplitudes_incoming(self, U, Q, Uhat):
        return self.characteristic_velocities_incoming(Uhat) * self.characteristic_variables(U, Q, Uhat)
