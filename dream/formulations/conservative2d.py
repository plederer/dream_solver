from __future__ import annotations
from ngsolve import *

from .base import MixedMethods, Indices, VectorCoordinates, TensorCoordinates
from .conservative import ConservativeFormulation
import dream.boundary_conditions as bc
import dream.viscosity as visc


class ConservativeFormulation2D(ConservativeFormulation):

    _indices = Indices(DENSITY=0,
                       MOMENTUM=VectorCoordinates(X=1, Y=2),
                       ENERGY=3,
                       STRAIN=TensorCoordinates(XX=0, XY=1, YX=1, YY=2),
                       TEMPERATURE_GRADIENT=VectorCoordinates(X=3, Y=4))

    def get_FESpace(self) -> ProductSpace:

        order = self.solver_configuration.order
        mixed_method = self.solver_configuration.mixed_method

        V = L2(self.mesh, order=order)
        VHAT = FacetFESpace(self.mesh, order=order)
        Q = VectorL2(self.mesh, order=order)

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

    def get_TnT(self):

        mixed_method = self.solver_configuration.mixed_method

        trial = self.fes.TrialFunction()
        test = self.fes.TestFunction()

        if mixed_method is MixedMethods.NONE:
            trial += [None]
            test += [None]

        return tuple(trial), tuple(test)

    def add_time_bilinearform(self, blf, old_components):

        bonus_order_vol = self.solver_configuration.bonus_int_order_vol
        compile_flag = self.solver_configuration.compile_flag

        (U, _, _), (V, _, _) = self.TnT

        old_components = tuple(gfu.components[0] for gfu in old_components)
        var_form = InnerProduct(self.time_scheme(U, *old_components), V) * dx(bonus_intorder=bonus_order_vol)

        blf += (var_form).Compile(compile_flag)

    def add_mixed_bilinearform(self, blf):

        bonus_vol = self.solver_configuration.bonus_int_order_vol
        bonus_bnd = self.solver_configuration.bonus_int_order_bnd
        mixed_variant = self.solver_configuration.mixed_method
        compile_flag = self.solver_configuration.compile_flag

        (U, Uhat, Q), (V, _, P) = self.TnT
        n = self.normal

        if mixed_variant is MixedMethods.STRAIN_HEAT:

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
            var_form -= InnerProduct(That, xi*n) * dx(element_boundary=True, bonus_intorder=bonus_bnd)

        elif mixed_variant is MixedMethods.GRADIENT:

            var_form = InnerProduct(Q, P) * dx(bonus_intorder=bonus_vol)
            var_form += InnerProduct(U, div(P)) * dx(bonus_intorder=bonus_vol)
            var_form -= InnerProduct(Uhat, P*n) * dx(element_boundary=True, bonus_intorder=bonus_bnd)

        blf += var_form.Compile(compile_flag)

    def add_convective_bilinearform(self, blf):

        compile_flag = self.solver_configuration.compile_flag
        bonus_order_vol = self.solver_configuration.bonus_int_order_vol
        bonus_order_bnd = self.solver_configuration.bonus_int_order_bnd

        (U, Uhat, _), (V, Vhat, _) = self.TnT

        var_form = -InnerProduct(self.convective_flux(U), grad(V)) * dx(bonus_intorder=bonus_order_vol)
        var_form += InnerProduct(self.convective_numerical_flux(U, Uhat),
                                 V) * dx(element_boundary=True, bonus_intorder=bonus_order_bnd)
        var_form += InnerProduct(self.convective_numerical_flux(U, Uhat),
                                 Vhat) * dx(element_boundary=True, bonus_intorder=bonus_order_bnd)

        # Subtract boundary regions
        regions = self.mesh.Boundaries("|".join(self.bcs.boundaries.keys()))
        var_form -= InnerProduct(self.convective_numerical_flux(U, Uhat),
                                 Vhat) * ds(skeleton=True, definedon=regions, bonus_intorder=bonus_order_bnd)

        blf += var_form.Compile(compile_flag)

    def add_diffusive_bilinearform(self, blf):

        compile_flag = self.solver_configuration.compile_flag
        bonus_order_vol = self.solver_configuration.bonus_int_order_vol
        bonus_order_bnd = self.solver_configuration.bonus_int_order_bnd

        (U, Uhat, Q), (V, Vhat, _) = self.TnT

        var_form = InnerProduct(self.diffusive_flux(U, Q), grad(V)) * dx(bonus_intorder=bonus_order_vol)
        var_form -= InnerProduct(self.diffusive_numerical_flux(U, Uhat, Q),
                                 V) * dx(element_boundary=True, bonus_intorder=bonus_order_bnd)
        var_form -= InnerProduct(self.diffusive_numerical_flux(U, Uhat, Q),
                                 Vhat) * dx(element_boundary=True, bonus_intorder=bonus_order_bnd)

        # Subtract boundary regions
        regions = self.mesh.Boundaries("|".join(self.bcs.boundaries.keys()))
        var_form += InnerProduct(self.diffusive_numerical_flux(U, Uhat, Q),
                                 Vhat) * ds(skeleton=True, definedon=regions, bonus_intorder=bonus_order_bnd)

        blf += var_form.Compile(compile_flag)

    def _add_initial_linearform(self, lf, domain, value: bc.Initial):

        mixed_method = self.solver_configuration.mixed_method
        bonus_order_vol = self.solver_configuration.bonus_int_order_vol
        bonus_order_bnd = self.solver_configuration.bonus_int_order_bnd
        compile_flag = self.solver_configuration.compile_flag

        region = self.mesh.Materials(domain)
        (_, _, _), (V, Vhat, P) = self.TnT

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

    def _add_dirichlet_bilinearform(self, blf, boundary: str, bc: bc.Dirichlet):

        bonus_order_bnd = self.solver_configuration.bonus_int_order_bnd
        compile_flag = self.solver_configuration.compile_flag

        region = self.mesh.Boundaries(boundary)

        (_, Uhat, _), (_, Vhat, _) = self.TnT
        dirichlet = CF((bc.density, bc.momentum, bc.energy))

        cf = (dirichlet-Uhat)
        cf = cf * Vhat * ds(skeleton=True, definedon=region, bonus_intorder=bonus_order_bnd)
        blf += cf.Compile(compile_flag)

    def _add_farfield_bilinearform(self, blf,  boundary: str, bc: bc.FarField):

        bonus_order_bnd = self.solver_configuration.bonus_int_order_bnd
        compile_flag = self.solver_configuration.compile_flag

        region = self.mesh.Boundaries(boundary)

        farfield = CF((bc.density, bc.momentum, bc.energy))

        (U, Uhat, _), (_, Vhat, _) = self.TnT

        cf = self.positive_A_matrix(Uhat) * (U-Uhat)
        cf += self.negative_A_matrix(Uhat) * (farfield - Uhat)
        cf = cf * Vhat * ds(skeleton=True, definedon=region, bonus_intorder=bonus_order_bnd)

        blf += cf.Compile(compile_flag)

    def _add_outflow_bilinearform(self, blf, boundary: str, bc: bc.Outflow):

        bonus_order_bnd = self.solver_configuration.bonus_int_order_bnd
        compile_flag = self.solver_configuration.compile_flag
        gamma = self.solver_configuration.heat_capacity_ratio

        region = self.mesh.Boundaries(boundary)

        (U, Uhat, _), (_, Vhat, _) = self.TnT
        rho = self.density(U)
        rho_m = self.momentum(U)

        energy = bc.pressure/(gamma - 1) + 1/(2*rho) * InnerProduct(rho_m, rho_m)
        outflow = CF((rho, rho_m, energy))

        cf = InnerProduct(outflow - Uhat, Vhat)
        cf = cf * ds(skeleton=True, definedon=region, bonus_intorder=bonus_order_bnd)

        blf += cf.Compile(compile_flag)

    def _add_nonreflecting_outflow_bilinearform(self, blf, boundary: str, bc: bc.NonReflectingOutflow, old_components):

        bonus_order_bnd = self.solver_configuration.bonus_int_order_bnd
        compile_flag = self.solver_configuration.compile_flag
        viscosity = self.solver_configuration.dynamic_viscosity

        n = self.normal
        t = self.tangential

        region = self.mesh.Boundaries(boundary)

        (U, Uhat, Q), (_, Vhat, _) = self.TnT

        L = self.charachteristic_amplitudes(U, Q, Uhat)

        if bc.type is bc.TYPE.PERFECT:
            L = CF((L[0], L[1], L[2], 0))

        elif bc.type is bc.TYPE.POINSOT:
            c = self.speed_of_sound(Uhat)
            M = self.mach_number(Uhat)
            ref_length = bc.reference_length

            L3 = bc.sigma * c * (1 - M)/ref_length * (self.pressure(Uhat) - bc.pressure)
            L = CF((L[0], L[1], L[2], L3))

        elif bc.type is bc.TYPE.PIROZZOLI:
            c = self.speed_of_sound(Uhat)
            M = self.mach_number(Uhat)
            ref_length = bc.reference_length

            L3 = bc.sigma * c * (1 - M)/ref_length * (self.pressure(Uhat) - bc.pressure)
            L = CF((L[0], L[1], L[2], L3))

        D = (self.P_matrix(Uhat) * L)

        old_components = tuple(gfu.components[1] for gfu in old_components)

        cf = InnerProduct(self.time_scheme(Uhat, *old_components), Vhat)
        cf += D * Vhat

        if bc.tang_conv_flux:
            cf += self.convective_flux_gradient(U, Q, t) * t * Vhat

        if viscosity is not visc.DynamicViscosity.INVISCID:

            if bc.tang_visc_flux:
                cf -= self.diffusive_flux_gradient(U, Q, t) * t * Vhat

            if bc.norm_visc_flux:
                cf -= self.diffusive_flux_gradient_test(U, Q, n) * n * Vhat

        cf = cf * ds(skeleton=True, definedon=region, bonus_intorder=bonus_order_bnd)
        blf += cf.Compile(compile_flag)

    def _add_inviscid_wall_bilinearform(self, blf, boundary: str, bc: bc.InviscidWall):

        bonus_order_bnd = self.solver_configuration.bonus_int_order_bnd
        compile_flag = self.solver_configuration.compile_flag

        region = self.mesh.Boundaries(boundary)

        (U, Uhat, _), (_, Vhat, _) = self.TnT
        cf = InnerProduct(self.reflect(U)-Uhat, Vhat)
        cf = cf * ds(skeleton=True, definedon=region, bonus_intorder=bonus_order_bnd)

        blf += cf.Compile(compile_flag)

    def _add_isothermal_wall_bilinearform(self, blf, boundary: str, bc: bc.IsothermalWall):

        bonus_order_bnd = self.solver_configuration.bonus_int_order_bnd
        compile_flag = self.solver_configuration.compile_flag
        gamma = self.solver_configuration.heat_capacity_ratio

        region = self.mesh.Boundaries(boundary)

        (U, Uhat, _), (_, Vhat, _) = self.TnT

        rho = self.density(U)
        rho_E = rho * bc.temperature / gamma

        cf = InnerProduct(CF((rho, 0, 0, rho_E)) - Uhat, Vhat)
        cf = cf * ds(skeleton=True, definedon=region, bonus_intorder=bonus_order_bnd)

        blf += cf.Compile(compile_flag)

    def _add_adiabatic_wall_bilinearform(self, blf, boundary: str, bc: bc.IsothermalWall):

        mixed_method = self.solver_configuration.mixed_method
        if mixed_method is MixedMethods.NONE:
            raise NotImplementedError(f"Adiabatic wall not implemented for {mixed_method}")

        bonus_order_bnd = self.solver_configuration.bonus_int_order_bnd
        compile_flag = self.solver_configuration.compile_flag

        Re = self.solver_configuration.Reynold_number
        Pr = self.solver_configuration.Prandtl_number
        n = self.normal
        tau_dE = self.diffusive_stabilisation_term()[self.mesh.dim+2, self.mesh.dim+2]

        region = self.mesh.Boundaries(boundary)

        (U, Uhat, _), (Q, Vhat, _) = self.TnT

        diff_rho = self.density(U) - self.density(Uhat)
        diff_rho_u = -self.momentum(Uhat)
        diff_rho_E = 1/(Re * Pr) * self.temperature_gradient(U, Q)*n - tau_dE * (self.energy(U) - self.energy(Uhat))

        cf = InnerProduct(CF((diff_rho, diff_rho_u, diff_rho_E)), Vhat)
        cf = cf * ds(skeleton=True, definedon=region, bonus_intorder=bonus_order_bnd)
        blf += cf.Compile(compile_flag)

    def convective_flux(self, U):
        """
        Convective flux F

        Equation 2, page 5

        Literature:
        [1] - Vila-Pérez, J., Giacomini, M., Sevilla, R. et al.
              Hybridisable Discontinuous Galerkin Formulation of Compressible Flows.
              Arch Computat Methods Eng 28, 753–784 (2021).
              https://doi.org/10.1007/s11831-020-09508-z
        """

        rho = self.density(U)
        rho_u = self.momentum(U)
        rho_E = self.energy(U)
        u = self.velocity(U)
        p = self.pressure(U)

        flux = tuple([rho_u, OuterProduct(rho_u, rho_u)/rho + p * Id(2), (rho_E + p) * u])

        return CF(flux, dims=(4, 2))

    def convective_numerical_flux(self, U, Uhat):
        """
        Lax-Friedrichs numerical flux

        Equation 34, page 16

        Literature:
        [1] - Vila-Pérez, J., Giacomini, M., Sevilla, R. et al.
              Hybridisable Discontinuous Galerkin Formulation of Compressible Flows.
              Arch Computat Methods Eng 28, 753–784 (2021).
              https://doi.org/10.1007/s11831-020-09508-z
        """
        n = self.normal

        An = self.P_matrix(Uhat) * self.Lambda_matrix(Uhat, True) * self.P_inverse_matrix(Uhat)
        return self.convective_flux(Uhat)*n + An * (U-Uhat)

    def diffusive_flux(self, U, Q):
        """
        Diffusive flux G

        Equation 2, page 5

        Literature:
        [1] - Vila-Pérez, J., Giacomini, M., Sevilla, R. et al.
              Hybridisable Discontinuous Galerkin Formulation of Compressible Flows.
              Arch Computat Methods Eng 28, 753–784 (2021).
              https://doi.org/10.1007/s11831-020-09508-z
        """

        u = self.velocity(U)
        tau = self.deviatoric_stress_tensor(U, Q)
        heat_flux = self.heat_flux(U, Q)

        flux = CF((0, 0, tau, tau*u + heat_flux), dims=(4, 2))

        return flux

    def diffusive_numerical_flux(self, U, Uhat, Q):

        n = self.normal
        tau_d = self.diffusive_stabilisation_term()

        return self.diffusive_flux(Uhat, Q)*n - tau_d * (U-Uhat)

    def diffusive_stabilisation_term(self):

        Re = self.solver_configuration.Reynold_number
        Pr = self.solver_configuration.Prandtl_number

        tau_d = CF((0, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1/Pr), dims=(4, 4)) / Re

        return tau_d

    def convective_flux_f(self, U):
        flux = self.convective_flux(U)
        return CF(tuple(flux[i, 0] for i in range(4)))

    def convective_flux_g(self, U):
        flux = self.convective_flux(U)
        return CF(tuple(flux[i, 1] for i in range(4)))

    def convective_flux_f_gradient(self, U, Q):

        rho = self.density(U)
        rho_u = self.momentum(U)
        rho_H = self.enthalpy(U)

        gradient_rho = self.density_gradient(U, Q)
        gradient_rho_u = self.momentum_gradient(U, Q)
        gradient_rho_H = self.enthalpy_gradient(U, Q)
        gradient_p = self.pressure_gradient(U, Q)

        rho_ux = rho_u[0]
        rho_uy = rho_u[1]
        gradient_rho_ux = gradient_rho_u[0, ...]
        gradient_rho_uy = gradient_rho_u[1, ...]

        gradf = CF(
            (gradient_rho_ux,
             2 * rho_ux * gradient_rho_ux / rho + gradient_p - rho_ux**2 * gradient_rho / rho**2,
             (rho_uy * gradient_rho_ux + rho_ux * gradient_rho_uy) / rho - rho_ux * rho_uy * gradient_rho / rho**2,
             (rho_H * gradient_rho_ux + rho_ux * gradient_rho_H) / rho - rho_ux * rho_H * gradient_rho / rho**2),
            dims=(4, 2))

        return gradf

    def convective_flux_g_gradient(self, U, Q):

        rho = self.density(U)
        rho_u = self.momentum(U)
        rho_H = self.enthalpy(U)

        gradient_rho = self.density_gradient(U, Q)
        gradient_rho_u = self.momentum_gradient(U, Q)
        gradient_rho_H = self.enthalpy_gradient(U, Q)
        gradient_p = self.pressure_gradient(U, Q)

        rho_ux = rho_u[0]
        rho_uy = rho_u[1]
        gradient_rho_ux = gradient_rho_u[0, ...]
        gradient_rho_uy = gradient_rho_u[1, ...]

        gradg = CF(
            (gradient_rho_uy,
             (rho_uy * gradient_rho_ux + rho_ux * gradient_rho_uy) / rho - rho_ux * rho_uy * gradient_rho / rho**2,
             2 * rho_uy * gradient_rho_uy / rho + gradient_p - rho_uy**2 * gradient_rho / rho**2,
             (rho_H * gradient_rho_uy + rho_uy * gradient_rho_H) / rho - rho_uy * rho_H * gradient_rho / rho**2),
            dims=(4, 2))

        return gradg

    def convective_flux_gradient(self, U, Q, vec):
        return self.convective_flux_f_gradient(U, Q) * vec[0] + self.convective_flux_g_gradient(U, Q) * vec[1]

    def diffusive_flux_f(self, U, Q):
        flux = self.diffusive_flux(U, Q)
        return CF(tuple(flux[i, 0] for i in range(4)))

    def diffusive_flux_g(self, U, Q):
        flux = self.diffusive_flux(U, Q)
        return CF(tuple(flux[i, 1] for i in range(4)))

    def diffusive_flux_f_gradient(self, U, Q):

        u = self.velocity(U)
        stress = self.deviatoric_stress_tensor(U, Q)

        gradient_u = self.velocity_gradient(U, Q)
        gradient_heat_flux = self.heat_flux_gradient(U, Q)
        gradient_stress = self.deviatoric_stress_tensor_gradient(U, Q)

        ux, uy = u[0], u[1]
        tauxx, tauyx = stress[0, 0], stress[1, 0]

        grad_tauxx = gradient_stress[0, 0, ...]
        grad_tauyx = gradient_stress[1, 0, ...]
        grad_phix = gradient_heat_flux[0, ...]
        grad_ux = gradient_u[0, ...]
        grad_uy = gradient_u[1, ...]

        grad_flux_f = CF((
            0, 0,
            grad_tauxx,
            grad_tauyx,
            grad_tauxx * ux + tauxx * grad_ux + grad_tauyx * uy + tauyx * grad_uy + grad_phix
        ), dims=(4, 2))

        return grad_flux_f

    def diffusive_flux_g_gradient(self, U, Q):

        u = self.velocity(U)
        stress = self.deviatoric_stress_tensor(U, Q)

        gradient_u = self.velocity_gradient(U, Q)
        gradient_heat_flux = self.heat_flux_gradient(U, Q)
        gradient_stress = self.deviatoric_stress_tensor_gradient(U, Q)

        ux, uy = u[0], u[1]
        tauxy, tauyy = stress[0, 1], stress[1, 1]

        grad_tauxy = gradient_stress[0, 1, ...]
        grad_tauyy = gradient_stress[1, 1, ...]
        grad_phiy = gradient_heat_flux[1, ...]
        grad_ux = gradient_u[0, ...]
        grad_uy = gradient_u[1, ...]

        grad_flux_g = CF((
            0, 0,
            grad_tauxy,
            grad_tauyy,
            grad_tauxy * ux + tauxy * grad_ux + grad_tauyy * uy + tauyy * grad_uy + grad_phiy
        ), dims=(4, 2))

        return grad_flux_g

    def diffusive_flux_f_gradient_test(self, U, Q):

        u = self.velocity(U)
        stress = self.deviatoric_stress_tensor(U, Q)

        gradient_u = self.velocity_gradient(U, Q)
        gradient_stress = self.deviatoric_stress_tensor_gradient(U, Q)

        ux, uy = u[0], u[1]
        tauxx, tauyx = stress[0, 0], stress[1, 0]

        grad_tauxx = gradient_stress[0, 0, ...]
        grad_tauyx = gradient_stress[1, 0, ...]
        grad_ux = gradient_u[0, ...]
        grad_uy = gradient_u[1, ...]

        grad_flux_f = CF((
            0, 0,
            grad_tauxx,
            0, 0,
            grad_tauxx * ux + tauxx * grad_ux + tauyx * grad_uy
        ), dims=(4, 2))

        return grad_flux_f

    def diffusive_flux_g_gradient_test(self, U, Q):

        u = self.velocity(U)
        stress = self.deviatoric_stress_tensor(U, Q)

        gradient_u = self.velocity_gradient(U, Q)
        gradient_stress = self.deviatoric_stress_tensor_gradient(U, Q)

        ux, uy = u[0], u[1]
        tauxy, tauyy = stress[0, 1], stress[1, 1]

        grad_tauxy = gradient_stress[0, 1, ...]
        grad_tauyy = gradient_stress[1, 1, ...]
        grad_ux = gradient_u[0, ...]
        grad_uy = gradient_u[1, ...]

        grad_flux_g = CF((
            0, 0,
            0, 0,
            grad_tauyy,
            grad_tauyy * uy + tauyy * grad_uy + tauxy * grad_ux
        ), dims=(4, 2))

        return grad_flux_g

    def diffusive_flux_gradient(self, U, Q, vec):
        return self.diffusive_flux_f_gradient(U, Q) * vec[0] + self.diffusive_flux_g_gradient(U, Q) * vec[1]

    def diffusive_flux_gradient_test(self, U, Q, vec):
        return self.diffusive_flux_f_gradient_test(U, Q) * vec[0] + self.diffusive_flux_g_gradient_test(U, Q) * vec[1]

    def A_jacobian(self, U):
        """
        First Jacobian of the convective Euler Fluxes F = (f, g) for conservative variables U.

            A = \partial f / \partial U

        Equation E16.2.5, page 144

        Literature:
        [1] - C. Hirsch,
              Numerical Computation of Internal and External Flows Volume 2:
              Computational Methods for Inviscid and Viscous Flows,
              Vrije Universiteit Brussel, Brussels, Belgium
              ISBN: 978-0-471-92452-4
        """

        gamma = self.solver_configuration.heat_capacity_ratio

        u = self.velocity(U)
        rho_E = self.energy(U)

        ux, uy = u[0], u[1]

        a11 = 0
        a12 = 1
        a13 = 0
        a14 = 0
        a21 = (gamma - 3) / 2 * ux**2 + (gamma - 1) / 2 * uy**2
        a22 = (3 - gamma) * ux
        a23 = -(gamma - 1) * uy
        a24 = gamma - 1
        a31 = -ux * uy
        a32 = uy
        a33 = ux
        a34 = 0
        a41 = -gamma * ux * rho_E + (gamma - 1) * ux * (ux**2 + uy**2)
        a42 = gamma * rho_E - (gamma - 1) / 2 * (uy**2 + 3 * ux**2)
        a43 = -(gamma - 1) * ux * uy
        a44 = gamma * ux

        return CoefficientFunction(
            (a11, a12, a13, a14,
             a21, a22, a23, a24,
             a31, a32, a33, a34,
             a41, a42, a43, a44),
            dims=(4, 4))

    def B_jacobian(self, U):
        """
        Second Jacobian of the convective Euler Fluxes F = (f, g) for conservative variables U.

            B = \partial g / \partial U

        Equation E16.2.6, page 145

        Literature:
        [1] - C. Hirsch,
              Numerical Computation of Internal and External Flows Volume 2:
              Computational Methods for Inviscid and Viscous Flows,
              Vrije Universiteit Brussel, Brussels, Belgium
              ISBN: 978-0-471-92452-4
        """
        gamma = self.solver_configuration.heat_capacity_ratio

        u = self.velocity(U)
        rho_E = self.energy(U)

        ux, uy = u[0], u[1]

        b11 = 0
        b12 = 0
        b13 = 1
        b14 = 0
        b21 = -ux * uy
        b22 = uy
        b23 = ux
        b24 = 0
        b31 = (gamma - 3) / 2 * uy**2 + (gamma - 1) / 2 * ux**2
        b32 = -(gamma - 1) * ux
        b33 = (3 - gamma) * uy
        b34 = gamma - 1
        b41 = -gamma * uy * rho_E + (gamma - 1) * uy * (ux**2 + uy**2)
        b42 = -(gamma - 1) * ux * uy
        b43 = gamma * rho_E - (gamma - 1) / 2 * (ux**2 + 3 * uy**2)
        b44 = gamma * uy

        return CoefficientFunction(
            (b11, b12, b13, b14,
             b21, b22, b23, b24,
             b31, b32, b33, b34,
             b41, b42, b43, b44),
            dims=(4, 4))

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

        rho = self.density(U)
        u = self.velocity(U)
        c = self.speed_of_sound(U)
        H = self.specific_enthalpy(U)
        n = self.normal

        ux, uy = u[0], u[1]
        nx, ny = n[0], n[1]

        p11 = 1
        p12 = 0
        p13 = rho / (2 * c)
        p14 = rho / (2 * c)
        p21 = ux
        p22 = rho * ny
        p23 = rho / (2 * c) * (ux + c * nx)
        p24 = rho / (2 * c) * (ux - c * nx)
        p31 = uy
        p32 = -rho * nx
        p33 = rho / (2 * c) * (uy + c * ny)
        p34 = rho / (2 * c) * (uy - c * ny)
        p41 = InnerProduct(u, u) / 2
        p42 = rho * (ux * ny - uy * nx)
        p43 = rho / (2 * c) * (H + c * InnerProduct(u, n))
        p44 = rho / (2 * c) * (H - c * InnerProduct(u, n))

        P = CF((p11, p12, p13, p14,
                p21, p22, p23, p24,
                p31, p32, p33, p34,
                p41, p42, p43, p44),
               dims=(4, 4))

        return P

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
        gamma = self.solver_configuration.heat_capacity_ratio

        rho = self.density(U)
        u = self.velocity(U)
        c = self.speed_of_sound(U)
        M = self.mach_number(U)
        n = self.normal

        ux, uy = u[0], u[1]
        nx, ny = n[0], n[1]

        p11 = 1 - (gamma - 1) / 2 * M**2
        p12 = (gamma - 1) * ux / c**2
        p13 = (gamma - 1) * uy / c**2
        p14 = -(gamma - 1) / c**2
        p21 = 1/rho * (uy * nx - ux * ny)
        p22 = ny / rho
        p23 = -nx / rho
        p24 = 0
        p31 = c/rho * ((gamma - 1)/2 * M**2 - InnerProduct(u, n)/c)
        p32 = 1/rho * (nx - (gamma - 1) * ux / c)
        p33 = 1/rho * (ny - (gamma - 1) * uy / c)
        p34 = (gamma - 1) / (rho * c)
        p41 = c/rho * ((gamma - 1)/2 * M**2 + InnerProduct(u, n)/c)
        p42 = -1/rho * (nx + (gamma - 1) * ux / c)
        p43 = -1/rho * (ny + (gamma - 1) * uy / c)
        p44 = (gamma - 1) / (rho * c)

        Pinv = CF((p11, p12, p13, p14,
                   p21, p22, p23, p24,
                   p31, p32, p33, p34,
                   p41, p42, p43, p44),
                  dims=(4, 4))

        return Pinv

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

        L = CF((1, 0, rho/(2*c), rho/(2*c),
                0,  ny, nx/2, -nx/2,
                0, -nx, ny/2, -ny/2,
                0, 0, rho*c/2, rho*c/2), dims=(4, 4))

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

        Linv = CF((1, 0, 0, -1/c**2,
                   0, ny, -nx, 0,
                   0, nx, ny, 1/(rho*c),
                   0, -nx, -ny, 1/(rho*c)), dims=(4, 4))

        return Linv

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

    def Lambda_matrix(self, U, absolute_value=False):
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

        vel = self.velocity(U)
        c = self.speed_of_sound(U)
        n = self.normal

        vn = InnerProduct(vel, n)
        vn_p_c = vn + c
        vn_m_c = vn - c

        if absolute_value:
            vn = IfPos(vn, vn, -vn)
            vn_p_c = IfPos(vn_p_c, vn_p_c, -vn_p_c)
            vn_m_c = IfPos(vn_m_c, vn_m_c, -vn_m_c)

        Lambda = CF((vn, 0, 0, 0,
                     0, vn, 0, 0,
                     0, 0, vn_p_c, 0,
                     0, 0, 0, vn_m_c), dims=(4, 4))

        return Lambda

    def charachteristic_amplitudes(self, U, Q, Uhat):
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

        amplitudes = CF((
            gradient_rho_normal - gradient_p_normal/c**2,
            gradient_u_normal[0] * n[1] - gradient_u_normal[1] * n[0],
            gradient_p_normal / (c * rho) + InnerProduct(gradient_u_normal, n),
            gradient_p_normal / (c * rho) - InnerProduct(gradient_u_normal, n)
        ))

        return self.Lambda_matrix(Uhat) * amplitudes

    def positive_A_matrix(self, U):
        positive_lambda = self.Lambda_matrix(U, False) + self.Lambda_matrix(U, True)
        return 0.5 * (self.P_matrix(U) * positive_lambda * self.P_inverse_matrix(U))

    def negative_A_matrix(self, U):
        negative_lambda = self.Lambda_matrix(U, False) - self.Lambda_matrix(U, True)
        return 0.5 * (self.P_matrix(U) * negative_lambda * self.P_inverse_matrix(U))
