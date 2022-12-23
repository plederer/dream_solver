from .base import ConservativeFormulation, Indices, Coordinates
from ngsolve import *

from configuration import MixedMethods, DynamicViscosity
import boundary_conditions as bc


class ConservativeFormulation2D(ConservativeFormulation):

    _indices = Indices(DENSITY=0, MOMENTUM=Coordinates(1, 2), ENERGY=3, TEMPERATURE_GRADIENT=Coordinates(3, 4))

    def get_FESpace(self) -> ProductSpace:

        order = self.solver_configuration.order
        mixed = self.solver_configuration.mixed_method

        V = L2(self.mesh, order=order)
        VHAT = FacetFESpace(self.mesh, order=order)
        Q = VectorL2(self.mesh, order=order)

        space = V**4 * VHAT**4

        if mixed is MixedMethods.STRAIN_HEAT:
            space *= V**5
        elif mixed is MixedMethods.GRADIENT:
            space *= Q**4

        return space

    def get_TnT(self):

        trial = self.fes.TrialFunction()
        test = self.fes.TestFunction()

        trial = trial + [None for i in range(len(trial) - 3)]
        test = test + [None for i in range(len(trial) - 3)]

        return tuple(trial), tuple(test)

    def set_time_bilinearform(self, blf, old_components):

        bonus_vol = self.solver_configuration.bonus_int_order_vol

        (U, _, _), (V, _, _) = self.TnT

        blf += InnerProduct(self.time_scheme(U, *old_components), V) * dx(bonus_intorder=bonus_vol)

    def set_mixed_bilinearform(self, blf):

        bonus_vol = self.solver_configuration.bonus_int_order_vol
        bonus_bnd = self.solver_configuration.bonus_int_order_bnd
        mixed_variant = self.solver_configuration.mixed_method

        (U, Uhat, Q), (_, _, P) = self.TnT
        n = self.normal

        if mixed_variant is MixedMethods.STRAIN_HEAT:

            vel = self.velocity(U)
            vel_hat = self.velocity(Uhat)
            T = self.temperature(U)
            T_hat = self.temperature(Uhat)

            gradient_P = CF(grad(P), dims=(5, 2))

            eps = CF((Q[0], Q[1], Q[1], Q[2]), dims=(2, 2))
            zeta = CF((P[0], P[1], P[1], P[2]), dims=(2, 2))

            dev_zeta = 2 * zeta - 2/3 * (zeta[0, 0] + zeta[1, 1]) * Id(2)
            div_dev_zeta = 2 * CF((gradient_P[0, 0] + gradient_P[1, 1], gradient_P[1, 0] + gradient_P[2, 1]))
            div_dev_zeta -= 2/3 * CF((gradient_P[0, 0] + gradient_P[2, 0], gradient_P[0, 1] + gradient_P[2, 1]))

            blf += (InnerProduct(eps, zeta) + InnerProduct(vel, div_dev_zeta)) * dx(bonus_intorder=bonus_vol)
            blf += -InnerProduct(vel_hat, dev_zeta*n) * dx(element_boundary=True, bonus_intorder=bonus_bnd)

            phi = CF((Q[3], Q[4]))
            xi = CF((P[3], P[4]))

            div_xi = gradient_P[3, 0] + gradient_P[4, 1]

            blf += (InnerProduct(phi, xi) + InnerProduct(T, div_xi)) * dx(bonus_intorder=bonus_vol)
            blf += -InnerProduct(T_hat, xi*n) * dx(element_boundary=True, bonus_intorder=bonus_bnd)

        elif mixed_variant is MixedMethods.GRADIENT:

            blf += (InnerProduct(Q, P) + InnerProduct(U, div(P))) * dx(bonus_intorder=bonus_vol)
            blf += -InnerProduct(Uhat, P*n) * dx(element_boundary=True, bonus_intorder=bonus_bnd)

    def set_convective_bilinearform(self, blf):

        bonus_vol = self.solver_configuration.bonus_int_order_vol
        bonus_bnd = self.solver_configuration.bonus_int_order_bnd

        (U, Uhat, _), (V, Vhat, _) = self.TnT

        blf += -InnerProduct(self.convective_flux(U), grad(V)) * dx(bonus_intorder=bonus_vol)
        blf += InnerProduct(self.convective_numerical_flux(U, Uhat),
                            V) * dx(element_boundary=True, bonus_intorder=bonus_bnd)
        blf += InnerProduct(self.convective_numerical_flux(U, Uhat),
                            Vhat) * dx(element_boundary=True, bonus_intorder=bonus_bnd)

        # Subtract boundary regions
        regions = self.mesh.Boundaries("|".join(self.bcs.keys()))
        blf += -InnerProduct(self.convective_numerical_flux(U, Uhat), Vhat) * ds(skeleton=True,
                                                                                 definedon=regions, bonus_intorder=bonus_bnd)

    def set_diffusive_bilinearform(self, blf):

        bonus_vol = self.solver_configuration.bonus_int_order_vol
        bonus_bnd = self.solver_configuration.bonus_int_order_bnd

        (U, Uhat, Q), (V, Vhat, _) = self.TnT
        blf += InnerProduct(self.diffusive_flux(U, Q), grad(V)) * dx(bonus_intorder=bonus_vol)
        blf += -InnerProduct(self.diffusive_numerical_flux(U, Uhat, Q),
                             V) * dx(element_boundary=True, bonus_intorder=bonus_bnd)
        blf += -InnerProduct(self.diffusive_numerical_flux(U, Uhat, Q),
                             Vhat) * dx(element_boundary=True, bonus_intorder=bonus_bnd)

        # Subtract boundary regions
        regions = self.mesh.Boundaries("|".join(self.bcs.keys()))
        blf += InnerProduct(self.diffusive_numerical_flux(U, Uhat, Q),
                            Vhat) * ds(skeleton=True, definedon=regions, bonus_intorder=bonus_bnd)

    def _set_dirichlet(self, blf, boundary: str, bc: bc.Dirichlet):

        bonus_order_bnd = self.solver_configuration.bonus_int_order_bnd
        compile_flag = self.solver_configuration.compile_flag

        region = self.mesh.Boundaries(boundary)

        (_, Uhat, _), (_, Vhat, _) = self.TnT
        dirichlet = CF((bc.density, bc.velocity, bc.energy))
        bc = ((dirichlet-Uhat) * Vhat).Compile(compile_flag)

        blf += bc * ds(skeleton=True, definedon=region, bonus_intorder=bonus_order_bnd)

    def _set_farfield(self, blf,  boundary: str, bc: bc.FarField):

        bonus_order_bnd = self.solver_configuration.bonus_int_order_bnd
        compile_flag = self.solver_configuration.compile_flag

        region = self.mesh.Boundaries(boundary)

        n = self.normal
        farfield = CF((bc.density, bc.velocity, bc.energy))

        (U, Uhat, _), (_, Vhat, _) = self.TnT
        Bhat = self.Aplus(Uhat, n) * (U-Uhat)
        Bhat += self.Aminus(Uhat, n) * (farfield - Uhat)
        bc = (Bhat * Vhat).Compile(compile_flag)

        blf += bc * ds(skeleton=True, definedon=region, bonus_intorder=bonus_order_bnd)

    def _set_outflow(self, blf, boundary: str, bc: bc.Outflow):

        bonus_order_bnd = self.solver_configuration.bonus_int_order_bnd
        compile_flag = self.solver_configuration.compile_flag
        gamma = self.solver_configuration.heat_capacity_ratio

        region = self.mesh.Boundaries(boundary)

        (U, Uhat, _), (_, Vhat, _) = self.TnT
        rho = self.density(U)
        rho_m = self.momentum(U)

        energy = bc.pressure/(gamma - 1) + 1/(2*rho) * InnerProduct(rho_m, rho_m)
        outflow = CF((rho, rho_m, energy))
        bc = (InnerProduct(outflow - Uhat, Vhat)).Compile(compile_flag)

        blf += bc * ds(skeleton=True, definedon=region, bonus_intorder=bonus_order_bnd)

    def _set_nonreflecting_outflow(self, blf, boundary: str, bc: bc.NonReflectingOutflow):

        bonus_order_bnd = self.solver_configuration.bonus_int_order_bnd
        compile_flag = self.solver_configuration.compile_flag
        gamma = self.solver_configuration.heat_capacity_ratio

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

        D = self.P_matrix(Uhat) * L

        blf += self.time_scheme.

        Ft = self.FU.tangential_flux_gradient(u, q, t)
        self.a += 3/2*1/self.FU.dt * InnerProduct(uhat - 4/3*self.gfu_old.components[1] + 1/3 * self.gfu_old_2.components[1], vhat) * ds(
            skeleton=True, definedon=self.mesh.Boundaries(self.bnd_data["NRBC"][0]), bonus_intorder=10)
        self.a += (D * vhat) * ds(skeleton=True,
                                  definedon=self.mesh.Boundaries(self.bnd_data["NRBC"][0]), bonus_intorder=10)
        self.a += ((Ft * t) * vhat) * ds(skeleton=True,
                                         definedon=self.mesh.Boundaries(self.bnd_data["NRBC"][0]), bonus_intorder=10)

    def _set_inviscid_wall(self, blf, boundary: str, bc: bc.InviscidWall):

        bonus_order_bnd = self.solver_configuration.bonus_int_order_bnd
        compile_flag = self.solver_configuration.compile_flag

        region = self.mesh.Boundaries(boundary)

        (U, Uhat, _), (_, Vhat, _) = self.TnT
        bc = (InnerProduct(self.reflect(U)-Uhat, Vhat)).Compile(compile_flag)

        blf += bc * ds(skeleton=True, definedon=region, bonus_intorder=bonus_order_bnd)

    def _set_isothermal_wall(self, blf, boundary: str, bc: bc.IsothermalWall):

        bonus_order_bnd = self.solver_configuration.bonus_int_order_bnd
        compile_flag = self.solver_configuration.compile_flag
        gamma = self.solver_configuration.heat_capacity_ratio

        region = self.mesh.Boundaries(boundary)

        (U, Uhat, _), (_, Vhat, _) = self.TnT

        rho = self.density(U)
        rho_E = rho * bc.temperature / gamma
        bc = InnerProduct(CF((rho, 0, 0, rho_E)) - Uhat, Vhat).Compile(compile_flag)

        blf += bc * ds(skeleton=True, definedon=region, bonus_intorder=bonus_order_bnd)

    def _set_adiabatic_wall(self, blf, boundary: str, bc: bc.IsothermalWall):

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
        bc = InnerProduct(CF((diff_rho, diff_rho_u, diff_rho_E)), Vhat).Compile(compile_flag)

        blf += bc * ds(skeleton=True, definedon=region, bonus_intorder=bonus_order_bnd)

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
        compile = self.solver_configuration.compile_flag

        rho = self.density(U)
        rho_m = self.momentum(U)
        rho_E = self.energy(U)
        p = self.pressure(U)

        flux = tuple([rho_m, 1/rho * rho_m*rho_m.trans + p*Id(2), 1/rho * (rho_E+p) * rho_m])

        return CoefficientFunction(flux, dims=(4, 2)).Compile(compile)

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
        compile = self.solver_configuration.compile_flag

        Re = self.solver_configuration.Reynold_number
        Pr = self.solver_configuration.Prandtl_number

        vel = self.velocity(U)

        mixed_method = self.solver_configuration.mixed_method
        viscosity = self.solver_configuration.dynamic_viscosity

        if mixed_method is MixedMethods.STRAIN_HEAT and viscosity is DynamicViscosity.CONSTANT:

            tau = 1/Re * CF((Q[0], Q[1], Q[1], Q[2]), dims=(2, 2))
            grad_T = CF((Q[3], Q[4]))

        elif mixed_method is MixedMethods.GRADIENT and viscosity is DynamicViscosity.CONSTANT:
            grad_vel = self.velocity_gradient(U, Q)
            tau = 1/Re * (2 * (grad_vel+grad_vel.trans) - 2/3 * (grad_vel[0, 0] + grad_vel[1, 1]) * Id(2))
            grad_T = self.temperature_gradient(U, Q)

        else:
            raise NotImplementedError()

        # CoefficientFunction((tau[0,0] * vel[0] + tau[0,1] * vel[1],tau[1,0] * vel[0] + tau[1,1] * vel[1]))
        tau_vel = tau * vel
        k = 1 / (Re.Get() * Pr.Get())

        flux = CF((0, 0,
                   tau[0, 0], tau[0, 1],
                   tau[1, 0], tau[1, 1],
                   tau_vel[0] + k*grad_T[0], tau_vel[1] + k*grad_T[1]), dims=(4, 2)).Compile(compile)

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

    def f_convective_flux(self, u):

        rho = self.rho(u)
        rho_u = self.momentum(u)
        rho_E = u[3]
        p = self.p(u)

        m = rho_u[0]
        n = rho_u[1]

        return CF((m, m**2/rho + p, m*n/rho, m/rho*(rho_E + p)))

    def f_gradient_convective_flux(self, u, q):

        gradU = grad(u)

        p = self.p(u)
        rho = self.rho(u)
        rho_u = self.momentum(u)
        rho_E = u[3]

        m = rho_u[0]
        n = rho_u[1]

        Dp = self.gradp(u, q)
        Drho = CF((gradU[0, 0], gradU[0, 1]))
        Dm = CF((gradU[1, 0], gradU[1, 1]))
        Dn = CF((gradU[2, 0], gradU[2, 1]))
        Drho_E = CF((gradU[3, 0], gradU[3, 1]))

        gradf = CF((
            Dm,
            2*m*Dm/rho - Drho*m**2/rho**2 + Dp,
            (Dm * n + Dn * m)/rho - Drho*m*n/rho**2,
            (Dm*(rho_E + p) + (Drho_E + Dp)*m)/rho - Drho*m*(rho_E + p)/rho**2
        ), dims=(4, 2))

        return gradf

    def g_gradient_convective_flux(self, u, q):

        gradU = grad(u)

        p = self.p(u)
        rho = self.rho(u)
        rho_u = self.momentum(u)
        rho_E = u[3]

        m = rho_u[0]
        n = rho_u[1]

        Dp = self.gradp(u, q)
        Drho = CF((gradU[0, 0], gradU[0, 1]))
        Dm = CF((gradU[1, 0], gradU[1, 1]))
        Dn = CF((gradU[2, 0], gradU[2, 1]))
        Drho_E = CF((gradU[3, 0], gradU[3, 1]))

        gradg = CF((
            Dn,
            (Dm * n + Dn * m)/rho - Drho*m*n/rho**2,
            2*n*Dn/rho - Drho*n**2/rho**2 + Dp,
            (Dn*(rho_E + p) + (Drho_E + Dp)*n)/rho - Drho*n*(rho_E + p)/rho**2
        ), dims=(4, 2))

        return gradg

    def g_convective_flux(self, u):

        rho = self.rho(u)
        rho_u = self.momentum(u)
        rho_E = u[3]
        p = self.p(u)

        m = rho_u[0]
        n = rho_u[1]

        return CF((n, m*n/rho, n**2/rho + p,  n/rho*(rho_E + p)))

    def tangential_flux_gradient(self, u, q, t):
        return self.f_gradient_convective_flux(u, q) * t[0] + self.g_gradient_convective_flux(u, q) * t[1]

    def A_jacobian(self, u):
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

        vel = self.vel(u)
        E = self.E(u)

        vel_1 = vel[0]
        vel_2 = vel[1]

        a11 = 0
        a12 = 1
        a13 = 0
        a14 = 0
        a21 = (self.gamma - 3) / 2 * vel_1**2 + (self.gamma - 1) / 2 * vel_2**2
        a22 = (3 - self.gamma) * vel_1
        a23 = -(self.gamma - 1) * vel_2
        a24 = self.gamma - 1
        a31 = -vel_1 * vel_2
        a32 = vel_2
        a33 = vel_1
        a34 = 0
        a41 = -self.gamma * vel_1 * E + (self.gamma - 1) * vel_1 * (vel_1**2 + vel_2**2)
        a42 = self.gamma * E - (self.gamma - 1) / 2 * (vel_2**2 + 3 * vel_1**2)
        a43 = -(self.gamma - 1) * vel_1 * vel_2
        a44 = self.gamma * vel_1

        return CoefficientFunction(
            (a11, a12, a13, a14,
             a21, a22, a23, a24,
             a31, a32, a33, a34,
             a41, a42, a43, a44),
            dims=(4, 4)).Compile()

    def B_jacobian(self, u):
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

        vel = self.vel(u)
        E = self.E(u)

        vel_1 = vel[0]
        vel_2 = vel[1]

        b11 = 0
        b12 = 0
        b13 = 1
        b14 = 0
        b21 = -vel_1 * vel_2
        b22 = vel_2
        b23 = vel_1
        b24 = 0
        b31 = (self.gamma - 3) / 2 * vel_2**2 + (self.gamma - 1) / 2 * vel_1**2
        b32 = -(self.gamma - 1) * vel_1
        b33 = (3 - self.gamma) * vel_2
        b34 = self.gamma - 1
        b41 = -self.gamma * vel_2 * E + (self.gamma - 1) * vel_2 * (vel_1**2 + vel_2**2)
        b42 = -(self.gamma - 1) * vel_1 * vel_2
        b43 = self.gamma * E - (self.gamma - 1) / 2 * (vel_1**2 + 3 * vel_2**2)
        b44 = self.gamma * vel_2

        return CoefficientFunction(
            (b11, b12, b13, b14,
             b21, b22, b23, b24,
             b31, b32, b33, b34,
             b41, b42, b43, b44),
            dims=(4, 4)).Compile()

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
        compile = self.solver_configuration.compile_flag

        rho = self.density(U)
        vel = self.velocity(U)
        c = self.speed_of_sound(U)
        H = self.enthalpy(U)

        u = vel[0]
        v = vel[1]

        n = self.normal
        nx = n[0]
        ny = n[1]

        p11 = 1
        p12 = 0
        p13 = rho / (2 * c)
        p14 = rho / (2 * c)
        p21 = u
        p22 = rho * ny
        p23 = rho / (2 * c) * (u + c * nx)
        p24 = rho / (2 * c) * (u - c * nx)
        p31 = v
        p32 = -rho * nx
        p33 = rho / (2 * c) * (v + c * ny)
        p34 = rho / (2 * c) * (v - c * ny)
        p41 = InnerProduct(vel, vel) / 2
        p42 = rho * (u * ny - v * nx)
        p43 = rho / (2 * c) * (H + c * InnerProduct(vel, n))
        p44 = rho / (2 * c) * (H - c * InnerProduct(vel, n))

        P = CF((p11, p12, p13, p14,
                p21, p22, p23, p24,
                p31, p32, p33, p34,
                p41, p42, p43, p44),
               dims=(4, 4)).Compile(compile)

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
        compile = self.solver_configuration.compile_flag
        gamma = self.solver_configuration.heat_capacity_ratio

        rho = self.density(U)
        vel = self.velocity(U)
        c = self.speed_of_sound(U)
        M = self.mach_number(U)

        u = vel[0]
        v = vel[1]

        n = self.normal
        nx = n[0]
        ny = n[1]

        p11 = 1 - (gamma - 1) / 2 * M**2
        p12 = (gamma - 1) * u / c**2
        p13 = (gamma - 1) * v / c**2
        p14 = -(gamma - 1) / c**2
        p21 = 1/rho * (v * nx - u * ny)
        p22 = ny / rho
        p23 = -nx / rho
        p24 = 0
        p31 = c/rho * ((gamma - 1)/2 * M**2 - InnerProduct(vel, n)/c)
        p32 = 1/rho * (nx - (gamma - 1) * u / c)
        p33 = 1/rho * (ny - (gamma - 1) * v / c)
        p34 = (gamma - 1) / (rho * c)
        p41 = c/rho * ((gamma - 1)/2 * M**2 + InnerProduct(vel, n)/c)
        p42 = -1/rho * (nx + (gamma - 1) * u / c)
        p43 = -1/rho * (ny + (gamma - 1) * v / c)
        p44 = (gamma - 1) / (rho * c)

        Pinv = CF((p11, p12, p13, p14,
                   p21, p22, p23, p24,
                   p31, p32, p33, p34,
                   p41, p42, p43, p44),
                  dims=(4, 4)).Compile(compile)

        return Pinv

    def L_matrix(self, u, k):
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

        rho = self.rho(u)
        c = self.c(u)
        kx, ky = k[0], k[1]

        L = CF((1, 0, rho/(2*c), rho/(2*c),
                0,  ky, kx/2, -kx/2,
                0, -kx, ky/2, -ky/2,
                0, 0, rho*c/2, rho*c/2), dims=(4, 4)).Compile()

        return L

    def L_inverse_matrix(self, u, k):
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

        rho = self.rho(u)
        c = self.c(u)
        kx, ky = k[0], k[1]

        Linv = CF((1, 0, 0, -1/c**2,
                   0, ky, -kx, 0,
                   0, kx, ky, 1/(rho*c),
                   0, -kx, -ky, 1/(rho*c)), dims=(4, 4)).Compile()

        return Linv

    def M_matrix(self, u, k):
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

        rho = self.rho(u)
        vel = self.vel(u)
        velx, vely = vel[0], vel[1]

        M = CF((1, 0, 0, 0,
                velx,  rho, 0, 0,
                vely, 0, rho, 0,
                0.5*InnerProduct(vel, vel), rho*velx, rho*vely, 1/(self.gamma - 1)), dims=(4, 4)).Compile()

        return M

    def M_inverse_matrix(self, u, k):
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

        rho = self.rho(u)
        vel = self.vel(u)
        velx, vely = vel[0], vel[1]

        Minv = CF((1, 0, 0, 0,
                   -velx/rho, 1/rho, 0, 0,
                   -vely/rho, 0, 1/rho, 0,
                   (self.gamma - 1)/2 * InnerProduct(vel, vel), -(self.gamma - 1) * velx,
                   -(self.gamma - 1) * vely, self.gamma - 1), dims=(4, 4)).Compile()

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
        compile = self.solver_configuration.compile_flag

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
                     0, 0, 0, vn_m_c), dims=(4, 4)).Compile(compile)

        return Lambda

    def charachteristic_amplitudes(self, u, q, k, uhat):
        """
        The charachteristic amplitudes are defined as

            Amplitudes = Lambda * L_inverse * dV/dn,

        where Lambda is the eigenvalue matrix, L_inverse is the mapping from
        primitive variables to charachteristic variables and dV/dn is the
        derivative normal to the boundary.
        """
        rho = self.rho(uhat)
        c = self.c(uhat)

        gradient_rho_normal = InnerProduct(self.gradrho(u, q), k)
        gradient_p_normal = InnerProduct(self.gradp(u, q), k)
        gradient_vel_normal = self.gradvel(u, q) * k

        amplitudes = CF((
            gradient_rho_normal - gradient_p_normal/c**2,
            gradient_vel_normal[0] * k[1] - gradient_vel_normal[1] * k[0],
            gradient_p_normal / (c * rho) + InnerProduct(gradient_vel_normal, k),
            gradient_p_normal / (c * rho) - InnerProduct(gradient_vel_normal, k)
        ))

        return self.Lambda_matrix(uhat, k) * amplitudes

    def Aplus(self, u, k):
        positive_lambda = self.Lambda_matrix(u, k, False) + self.Lambda_matrix(u, k, True)
        return 0.5 * (self.P_matrix(u, k) * positive_lambda * self.P_inverse_matrix(u, k))

    def Aminus(self, u, k):
        negative_lambda = self.Lambda_matrix(u, k, False) - self.Lambda_matrix(u, k, True)
        return 0.5 * (self.P_matrix(u, k) * negative_lambda * self.P_inverse_matrix(u, k))
