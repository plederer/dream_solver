from ngsolve import *

import math
from math import pi, atan2


class FlowUtils():
    def __init__(self, ff_data):

        # stationary solution without pseudo time stepping
        if "dt" not in ff_data.keys():
            ff_data["dt"] = 1e10

        if "Du" not in ff_data.keys():
            ff_data["Du"] = True
        self.Du = ff_data["Du"]

        for k in ["Re", "Pr", "mu"]:
            if k not in ff_data.keys():
                print("Setting standard value: {} = 1".format(k))
                ff_data[k] = 1
            setattr(self, k, Parameter(ff_data[k]))

        for k in ["Minf", "gamma"]:
            if k not in ff_data.keys():
                print("Setting standard value: {} = 1".format(k))
                ff_data[k] = 1
            setattr(self, k, ff_data[k])

        # time step for pseudo time stepping
        self.dt = Parameter(ff_data["dt"])

    def GetData(self):
        ff_data = {}
        ff_data["dt"] = self.dt.Get()
        ff_data["Du"] = self.Du
        ff_data["Re"] = self.Re.Get()
        ff_data["Pr"] = self.Pr.Get()
        ff_data["mu"] = self.mu.Get()
        ff_data["Minf"] = self.Minf
        ff_data["gamma"] = self.gamma

        return ff_data

    def rho(self, u):
        return u[0]

    def gradrho(self, u, q=None):

        if not self.Du and q is not None:
            gradrho = CF((q[0], q[1]))
        else:
            gradrho = grad(u)
            gradrho = CF((gradrho[0, 0], gradrho[0, 1]))

        return gradrho

    def momentum(self, u):
        return CF((u[1], u[2]))

    def E(self, u):
        return u[3]/self.rho(u)

    def vel(self, u):
        return self.momentum(u)/self.rho(u)

    def p(self, u):
        momentum = self.momentum(u)
        return (self.gamma-1) * (u[3] - InnerProduct(momentum, momentum)/(2 * self.rho(u)))

    def T(self, u):
        return self.gamma/(self.gamma - 1) * self.p(u)/self.rho(u)

    def c(self, u):
        return sqrt(self.gamma * self.p(u)/self.rho(u))

    def H(self, u):
        return self.E(u) + self.p(u)/self.rho(u)

    def M(self, u):
        velocity = self.vel(u)
        return sqrt(InnerProduct(velocity, velocity)) / self.c(u)

    def gradvel(self, u, q=None):

        rho = self.rho(u)

        if not self.Du and q is not None:
            gradu = 1/rho * (q[1] - q[0] * u[1]/rho)
            gradv = 1/rho * (q[2] - q[0] * u[2]/rho)
        else:
            gradients = grad(u)
            momentum_gradient_x = CF(tuple(gradients[1, i] for i in range(2)))
            momentum_gradient_y = CF(tuple(gradients[2, i] for i in range(2)))
            gradu = 1/rho * (momentum_gradient_x - self.gradrho(u, q) * u[1]/rho)
            gradv = 1/rho * (momentum_gradient_y - self.gradrho(u, q) * u[2]/rho)
        return CF((gradu, gradv), dims=(2, 2))

    def gradp(self, u, q=None):

        momentum = self.momentum(u)

        if q is None:
            rho = self.rho(u)
            gradients = grad(u)
            momentum_gradient_x = CF(tuple(gradients[1, i] for i in range(2)))
            momentum_gradient_y = CF(tuple(gradients[2, i] for i in range(2)))
            energy_gradient = CF(tuple(gradients[3, i] for i in range(2)))
            gradp = (self.gamma - 1) * (energy_gradient - (u[1] * momentum_gradient_x + u[2] * momentum_gradient_y)/rho + (
                self.gradrho(u, q) * InnerProduct(momentum, momentum)) / (2 * rho**2))
        elif not self.Du:
            raise NotImplementedError()
        else:
            gradp = (self.gamma - 1)/self.gamma * (self.T(u) * grad(self.rho(u)) + self.rho(u) * self.gradT(u, q))

        return gradp

    def gradE(self, u, q):
        # E_x = 1/rho * [(rho*E)_x - rho_x*E]
        E_x = 1/self.rho(u) * (q[3, 0] - q[0, 0] * u[3]/self.rho(u))
        E_y = 1/self.rho(u) * (q[3, 1] - q[0, 1] * u[3]/self.rho(u))
        return CF((E_x, E_y))

    def gradT(self, u, q=None):
        # T = (self.gamma-1) * self.gamma * Minf**2(E - 1/2 (u**2 + v**2)
        # self.gamma * Minf**2 comes due to the non-dimensional NVS T = self.gamma Minf**2*p/rho
        # temp flux
        rho = self.rho(u)
        if q is None:
            gradT = (self.gamma - 1)/self.gamma * (self.gradp(u, q)/rho - self.gradrho(u, q)*self.p(u)/rho**2)
        elif not self.Du:
            vel = self.vel(u)
            grad_vel = self.gradvel(u, q)
            u_x = grad_vel[0, 0]
            v_x = grad_vel[1, 0]
            u_y = grad_vel[0, 1]
            v_y = grad_vel[1, 1]

            E_x = self.gradE(u, q)[0]
            E_y = self.gradE(u, q)[1]

            # T_x = (self.gamma-1)*(self.gamma * self.Minf2) * (E_x - (u_x * vel[0] + v_x * vel[1]))
            # T_y = (self.gamma-1)*(self.gamma * self.Minf2) * (E_y - (u_y * vel[0] + v_y * vel[1]))
            T_x = (self.gamma-1)/self.R * (E_x - (u_x * vel[0] + v_x * vel[1]))
            T_y = (self.gamma-1)/self.R * (E_y - (u_y * vel[0] + v_y * vel[1]))
            gradT = CF((T_x, T_y))
        else:
            gradT = CF((q[3], q[4]))
        return gradT

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

    def P_matrix(self, u, k):
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
        rho = self.rho(u)
        vel = self.vel(u)
        c = self.c(u)
        H = self.H(u)

        vel_x = vel[0]
        vel_y = vel[1]

        # should be the same as:
        # H = (vel_1**2 + vel_2**2)/2 + c**2/(self.gamma-1)

        # k is assumed to be normalized!
        k_x = k[0]  # / sqrt(k*k)
        k_y = k[1]  # / sqrt(k*k)

        p11 = 1
        p12 = 0
        p13 = rho / (2 * c)
        p14 = rho / (2 * c)
        p21 = vel_x
        p22 = rho * k_y
        p23 = rho / (2 * c) * (vel_x + c * k_x)
        p24 = rho / (2 * c) * (vel_x - c * k_x)
        p31 = vel_y
        p32 = -rho * k_x
        p33 = rho / (2 * c) * (vel_y + c * k_y)
        p34 = rho / (2 * c) * (vel_y - c * k_y)
        p41 = (vel_x ** 2 + vel_y ** 2) / 2
        p42 = rho * (vel_x * k_y - vel_y * k_x)
        p43 = rho / (2 * c) * (H + c * InnerProduct(vel, k))
        p44 = rho / (2 * c) * (H - c * InnerProduct(vel, k))

        P = CF((p11, p12, p13, p14,
                p21, p22, p23, p24,
                p31, p32, p33, p34,
                p41, p42, p43, p44),
               dims=(4, 4)).Compile()

        return P

    def P_inverse_matrix(self, u, k):
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

        rho = self.rho(u)
        vel = self.vel(u)
        c = self.c(u)
        M = self.M(u)

        vel_x = vel[0]
        vel_y = vel[1]

        k_x = k[0]  # / sqrt(k*k)
        k_y = k[1]  # / sqrt(k*k)

        p11 = 1 - (self.gamma - 1) / 2 * M**2
        p12 = (self.gamma - 1) * vel_x / c**2
        p13 = (self.gamma - 1) * vel_y / c**2
        p14 = -(self.gamma - 1) / c**2
        p21 = 1/rho * (vel_y * k_x - vel_x * k_y)
        p22 = k_y / rho
        p23 = -k_x / rho
        p24 = 0
        p31 = c/rho * ((self.gamma - 1)/2 * M**2 - InnerProduct(vel, k)/c)
        p32 = 1/rho * (k_x - (self.gamma - 1) * vel_x / c)
        p33 = 1/rho * (k_y - (self.gamma - 1) * vel_y / c)
        p34 = (self.gamma - 1) / (rho * c)
        p41 = c/rho * ((self.gamma - 1)/2 * M**2 + InnerProduct(vel, k)/c)
        p42 = -1/rho * (k_x + (self.gamma - 1) * vel_x / c)
        p43 = -1/rho * (k_y + (self.gamma - 1) * vel_y / c)
        p44 = (self.gamma - 1) / (rho * c)

        Pinv = CF((p11, p12, p13, p14,
                   p21, p22, p23, p24,
                   p31, p32, p33, p34,
                   p41, p42, p43, p44),
                  dims=(4, 4)).Compile()

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

    def Lambda_matrix(self, u, k, absolute_value=False):
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
        vel = self.vel(u)
        c = self.c(u)

        vn = InnerProduct(vel, k)
        vn_p_c = vn + c
        vn_m_c = vn - c

        if absolute_value:
            vn = IfPos(vn, vn, -vn)
            vn_p_c = IfPos(vn_p_c, vn_p_c, -vn_p_c)
            vn_m_c = IfPos(vn_m_c, vn_m_c, -vn_m_c)

        Lambda = CF((vn, 0, 0, 0,
                     0, vn, 0, 0,
                     0, 0, vn_p_c, 0,
                     0, 0, 0, vn_m_c), dims=(4, 4)).Compile()

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

    def convective_flux(self, u):
        """
        Convective flux F

        Equation 2, page 5

        Literature:
        [1] - Vila-Pérez, J., Giacomini, M., Sevilla, R. et al.
              Hybridisable Discontinuous Galerkin Formulation of Compressible Flows.
              Arch Computat Methods Eng 28, 753–784 (2021).
              https://doi.org/10.1007/s11831-020-09508-z
        """
        rho = self.rho(u)
        m = CoefficientFunction((u[1], u[2]), dims=(2, 1))
        p = self.p(u)  # (self.gamma-1) * (u[3] - 0.5*InnerProduct(m,m)/u[0])
        return CoefficientFunction(tuple([m, 1/rho * m*m.trans + p*Id(2), 1/rho * (u[3]+p) * m]),
                                   dims=(4, 2))

    def numerical_convective_flux(self, uhatold, u, uhat, n):
        """
        Lax-Friedrichs numerical flux

        Equation 34, page 16

        Literature:
        [1] - Vila-Pérez, J., Giacomini, M., Sevilla, R. et al.
              Hybridisable Discontinuous Galerkin Formulation of Compressible Flows.
              Arch Computat Methods Eng 28, 753–784 (2021).
              https://doi.org/10.1007/s11831-020-09508-z
        """

        # Roe
        # An = self.P_matrix(uhat, n) * self.Lambda_matrix(uhat, n, True) * self.P_inverse_matrix(uhat, n)
        # return self.convective_flux(uhat)*n + An * (u-uhat)

        # LF
        vel = self.vel(uhat)
        c = self.c(uhat)

        vn = InnerProduct(vel, n)
        
        lam_max = IfPos(vn, vn, -vn) + c
        return self.convective_flux(uhat)*n + lam_max * (u-uhat)

    def diffusive_flux(self, u, q):
        """
        Diffusive flux G

        Equation 2, page 5

        Literature:
        [1] - Vila-Pérez, J., Giacomini, M., Sevilla, R. et al.
              Hybridisable Discontinuous Galerkin Formulation of Compressible Flows.
              Arch Computat Methods Eng 28, 753–784 (2021).
              https://doi.org/10.1007/s11831-020-09508-z
        """

        if not self.Du:
            grad_vel = self.gradvel(u, q)
            tau = self.mu.Get()/self.Re.Get() * (2 * (grad_vel+grad_vel.trans) -
                                                 2/3 * (grad_vel[0, 0] + grad_vel[1, 1]) * Id(2))
            grad_T = self.gradT(u, q)
        else:
            tau = self.mu.Get()/self.Re.Get() * \
                CF((q[0], q[1], q[1], q[2]), dims=(2, 2))
            grad_T = CF((q[3], q[4]))

        # CoefficientFunction((tau[0,0] * vel[0] + tau[0,1] * vel[1],tau[1,0] * vel[0] + tau[1,1] * vel[1]))
        tau_vel = tau * self.vel(u)

        # k = 1/((self.gamma-1) * self.Minf2 * self.Re*self.Pr)
        k = self.mu.Get() / (self.Re.Get() * self.Pr.Get())

        return CoefficientFunction((0, 0,
                                    tau[0, 0], tau[0, 1],
                                    tau[1, 0], tau[1, 1],
                                    tau_vel[0] + k*grad_T[0], tau_vel[1] + k*grad_T[1]), dims=(4, 2))

    def numerical_diffusive_flux(self, u, uhat, q, n):
        C = CoefficientFunction(
            (0, 0, 0, 0,
             0, 1/self.Re.Get(), 0, 0,
             0, 0, 1/self.Re.Get(), 0,
             0, 0, 0, self.mu.Get()/(self.Re.Get() * self.Pr.Get())), dims=(4, 4))

        if self.Du:
            return self.diffusive_flux(uhat, q)*n - C * (u-uhat)
        else:
            return self.diffusive_flux(uhat, q)*n

    def reflect(self, u, n):
        m = CoefficientFunction(tuple([u[i] for i in range(1, 3)]), dims=(2, 1))
        mn = InnerProduct(m, n)
        m -= mn*n
        # P = Id(2) - OuterProduct(n,n)
        return CoefficientFunction(tuple([self.rho(u), m, u[3]]), dims=(4, 1))
