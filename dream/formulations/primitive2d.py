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
        riemann_solver = self.cfg.riemann_solver
        un = InnerProduct(self.velocity(Uhat), unit_vector)
        un_abs = IfPos(un, un, -un)
        c = self.speed_of_sound(Uhat)
        G = self.G_matrix(U)

        if riemann_solver is RiemannSolver.LAX_FRIEDRICH:
            lambda_max = un_abs + c
            stabilisation_matrix = lambda_max * Id(self.mesh.dim + 2)

        elif riemann_solver is RiemannSolver.ROE:
            stabilisation_matrix = self.primitive_convective_jacobian(Uhat, unit_vector, True)

        elif riemann_solver is RiemannSolver.HLL:
            splus = IfPos(un + c, un + c, 0)
            stabilisation_matrix = splus * Id(self.mesh.dim + 2)

        elif riemann_solver is RiemannSolver.HLLEM:
            theta_0 = 1e-8
            theta = un_abs/(un_abs + c)
            IfPos(theta - theta_0, theta, theta_0)
            Theta = CF((1, 0, 0, 0,
                        0, theta, 0, 0,
                        0, 0, theta, 0,
                        0, 0, 0, 1), dims=(4, 4))

            Theta = self.characteristic_to_primitive(Theta, Uhat, unit_vector)
            splus = IfPos(un + c, un + c, 0)

            stabilisation_matrix = splus * Theta

        return self.convective_flux(Uhat)*unit_vector + G * (stabilisation_matrix * (U-Uhat))

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

        An_out = self.primitive_convective_jacobian_outgoing(Uhat, self.normal)
        An_in = self.primitive_convective_jacobian_incoming(Uhat, self.normal)

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

        cf = InnerProduct(self.reflect(U)-Uhat, Vhat)
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

    def reflect(self, U):

        n = self.normal

        p = self.pressure(U)
        u = self.velocity(U)
        T = self.temperature(U)

        return CF((p, u - InnerProduct(u, n)*n, T))

    def G_matrix(self, U):

        gamma = self.cfg.heat_capacity_ratio
        R = (gamma - 1)/gamma

        p = self.pressure(U)
        T = self.temperature(U)
        u, v = self.velocity(U)
        E = self.specific_energy(U)
        Ek = self.specific_kinetic_energy(U)

        M = CF((
            1, 0, 0, -p/T,
            u, p, 0, -p*u/T,
            v, 0, p, -p*v/T,
            E, p*u, p*v, -p*Ek/T
        ), dims=(4, 4))

        M *= 1/(R * T)

        return M

    def G_inverse_matrix(self, U):

        gamma = self.cfg.heat_capacity_ratio
        R = (gamma - 1)/gamma

        rho = self.density(U)
        T = self.temperature(U)
        u, v = self.velocity(U)
        Ek = self.specific_kinetic_energy(U)

        Minv = CF((
            R*Ek*rho, -R*rho*u, -R*rho*v, R*rho,
            -u/gamma, 1/gamma, 0, 0,
            -v/gamma, 0, 1/gamma, 0,
            -T/gamma + Ek, -u, -v, 1
        ), dims=(4, 4))

        Minv *= gamma/rho

        return Minv

    def T_matrix(self, U):
        """ From low mach primitive to compressible primitive """
        gamma = self.cfg.heat_capacity_ratio
        R = (gamma - 1)/gamma

        p = self.pressure(U)
        T = self.temperature(U)

        T_mat = CF((
            1/(R*T), 0, 0, -p/(R * T**2),
            0, 1, 0, 0,
            0, 0, 1, 0,
            1, 0, 0, 0
        ), dims=(4, 4))

        return T_mat

    def T_inverse_matrix(self, U):
        """ From compressible primitive to low mach primitive """
        gamma = self.cfg.heat_capacity_ratio
        R = (gamma - 1)/gamma

        p = self.pressure(U)
        rho = self.density(U)

        Tinv = CF((
            0, 0, 0, 1,
            0, 1, 0, 0,
            0, 0, 1, 0,
            -p/(R*rho**2), 0, 0, 1/(R * rho),
        ), dims=(4, 4))

        return Tinv

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

    def characteristic_to_primitive(self, matrix, U, unit_vector) -> CF:
        Tinv = self.T_inverse_matrix(U)
        L = self.L_matrix(U, unit_vector)
        T = self.T_matrix(U)
        Linv = self.L_inverse_matrix(U, unit_vector)
        return (Tinv * L) * matrix * (Linv * T)

    def primitive_convective_jacobian_outgoing(self, U, unit_vector: CF) -> CF:
        Lambda = self.characteristic_velocities_outgoing(U, unit_vector)
        return self.characteristic_to_primitive(Lambda, U, unit_vector)

    def primitive_convective_jacobian_incoming(self, U, unit_vector: CF) -> CF:
        Lambda = self.characteristic_velocities_incoming(U, unit_vector)
        return self.characteristic_to_primitive(Lambda, U, unit_vector)

    def primitive_convective_jacobian(self, U, unit_vector: CF, absolute_value: bool = False) -> CF:
        Lambda = self.characteristic_velocities(U, unit_vector, absolute_value)
        return self.characteristic_to_primitive(Lambda, U, unit_vector)

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
