from __future__ import annotations
from ngsolve import *
from typing import Optional, NamedTuple

from .interface import Formulation, MixedMethods, TensorIndices, RiemannSolver


class Indices(NamedTuple):
    DENSITY: int
    MOMENTUM: slice
    ENERGY: int
    TEMPERATURE_GRADIENT: slice
    STRAIN: TensorIndices


class ConservativeFormulation(Formulation):

    _indices: Indices

    def add_time_bilinearform(self, blf):

        bonus_order_vol = self.cfg.bonus_int_order_vol
        compile_flag = self.cfg.compile_flag

        U, V = self.TnT.PRIMAL

        time_levels_gfu = self._gfus.get_component(0)
        time_levels_gfu['n+1'] = U

        var_form = InnerProduct(self.time_scheme.apply(time_levels_gfu), V) * dx(bonus_intorder=bonus_order_vol)

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
        mask.vec[~mask_fes.GetDofs(self.mesh.Boundaries(self.bcs.pattern))] = 1

        var_form = -InnerProduct(self.convective_flux(U), grad(V)) * dx(bonus_intorder=bonus_order_vol)
        var_form += InnerProduct(self.convective_numerical_flux(U, Uhat, self.normal),
                                 V) * dx(element_boundary=True, bonus_intorder=bonus_order_bnd)
        var_form += mask * InnerProduct(self.convective_numerical_flux(U, Uhat, self.normal),
                                        Vhat) * dx(element_boundary=True, bonus_intorder=bonus_order_bnd)

        blf += var_form.Compile(compile_flag)

    def add_diffusive_bilinearform(self, blf):

        mixed_method = self.cfg.mixed_method
        compile_flag = self.cfg.compile_flag
        bonus_order_vol = self.cfg.bonus_int_order_vol
        bonus_order_bnd = self.cfg.bonus_int_order_bnd

        U, V = self.TnT.PRIMAL
        Uhat, Vhat = self.TnT.PRIMAL_FACET
        Q, _ = self.TnT.MIXED

        # Subtract boundary regions
        mask_fes = FacetFESpace(self.mesh, order=0)
        mask = GridFunction(mask_fes)
        mask.vec[:] = 0
        mask.vec[~mask_fes.GetDofs(self.mesh.Boundaries(self.bcs.pattern))] = 1

        var_form = InnerProduct(self.diffusive_flux(U, Q), grad(V)) * dx(bonus_intorder=bonus_order_vol)
        var_form -= InnerProduct(self.diffusive_numerical_flux(U, Uhat, Q, self.normal),
                                 V) * dx(element_boundary=True, bonus_intorder=bonus_order_bnd)
        var_form -= mask * InnerProduct(self.diffusive_numerical_flux(U, Uhat, Q, self.normal),
                                        Vhat) * dx(element_boundary=True, bonus_intorder=bonus_order_bnd)

        blf += var_form.Compile(compile_flag)

        if mixed_method is not MixedMethods.NONE:
            self._add_mixed_bilinearform(blf)

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
        gamma = self.cfg.heat_capacity_ratio
        Ma = self.cfg.Mach_number
        Mn = un_abs/c

        # rho = self.density(Uhat)
        # rho_jump = self.density(U) - rho

        # u = self.velocity(Uhat)
        # u_jump = self.velocity(U) - u

        # p = self.pressure(Uhat)
        # p_jump = self.pressure(U) - p

        # Ekin = self.specific_kinetic_energy(Uhat)
        # Ekin_jump = self.specific_kinetic_energy(U) - Ekin

        # Qu = CF((0, u_jump, Ekin_jump)) * rho
        # Qrho = CF((rho_jump, rho_jump * self.velocity(Uhat), rho_jump * self.specific_kinetic_energy(Uhat) + p_jump/(gamma - 1)))
        # Q = Qrho + self.mach_number(Uhat) * Qu

        if riemann_solver is RiemannSolver.LAX_FRIEDRICH:
            lambda_max = un_abs + c
            stabilisation_matrix = lambda_max * Id(self.mesh.dim + 2)

        elif riemann_solver is RiemannSolver.ROE:
            Lambda_abs = self.characteristic_velocities(Uhat, unit_vector, absolute_value=True)
            stabilisation_matrix = self.characteristic_to_conservative(Lambda_abs, Uhat, unit_vector)

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

            Theta = self.characteristic_to_conservative(Theta, Uhat, unit_vector)
            splus = IfPos(un + c, un + c, 0)

            stabilisation_matrix = splus * Theta

        return self.convective_flux(Uhat)*unit_vector + stabilisation_matrix * (U - Uhat)

    def diffusive_numerical_flux(self, U, Uhat, Q, unit_vector: CF):
        tau_d = self.diffusive_stabilisation_term(Uhat)
        return self.diffusive_flux(Uhat, Q)*unit_vector - tau_d * (U-Uhat)

    def diffusive_stabilisation_term(self, Uhat):

        Re = self.cfg.Reynolds_number
        Pr = self.cfg.Prandtl_number
        mu = self.dynamic_viscosity(Uhat)

        tau_d = CF((0, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1/Pr), dims=(4, 4)) * mu / Re

        return tau_d

    def density(self, U: Optional[CF] = None):
        U = self._if_none_replace_with_gfu(U)
        return U[self._indices.DENSITY]

    def velocity(self, U: Optional[CF] = None):
        rho = self.density(U)
        rho_u = self.momentum(U)
        return rho_u/rho

    def momentum(self, U: Optional[CF] = None):
        U = self._if_none_replace_with_gfu(U)
        return U[self._indices.MOMENTUM]

    def pressure(self, U: Optional[CF] = None):
        rho_Ei = self.inner_energy(U)
        gamma = self.cfg.heat_capacity_ratio
        return (gamma-1) * rho_Ei

    def temperature(self, U: Optional[CF] = None):
        rho = self.density(U)
        rho_Ei = self.inner_energy(U)
        gamma = self.cfg.heat_capacity_ratio
        return gamma/rho * rho_Ei

    def energy(self, U: Optional[CF] = None):
        U = self._if_none_replace_with_gfu(U)
        return U[self._indices.ENERGY]

    def specific_energy(self, U: Optional[CF] = None):
        rho = self.density(U)
        rho_E = self.energy(U)
        return rho_E/rho

    def kinetic_energy(self, U: Optional[CF] = None):
        rho = self.density(U)
        rho_u = self.momentum(U)
        return InnerProduct(rho_u, rho_u)/(2 * rho)

    def specific_kinetic_energy(self, U: Optional[CF] = None):
        rho = self.density(U)
        rho_Ek = self.kinetic_energy(U)
        return rho_Ek/rho

    def enthalpy(self, U: Optional[CF] = None):
        p = self.pressure(U)
        rho_E = self.energy(U)
        return rho_E + p

    def specific_enthalpy(self, U: Optional[CF] = None):
        rho_H = self.enthalpy(U)
        rho = self.density(U)
        return rho_H/rho

    def inner_energy(self, U: Optional[CF] = None):
        rho_Ek = self.kinetic_energy(U)
        rho_E = self.energy(U)
        return rho_E - rho_Ek

    def specific_inner_energy(self, U: Optional[CF] = None):
        rho = self.density(U)
        rho_Ei = self.inner_energy(U)
        return rho_Ei/rho

    def speed_of_sound(self, U: Optional[CF] = None):
        p = self.pressure(U)
        rho = self.density(U)
        gamma = self.cfg.heat_capacity_ratio
        return sqrt(gamma * p/rho)

    def density_gradient(self, U: Optional[CF] = None, Q: Optional[CF] = None):

        mixed_method = self.cfg.mixed_method

        if mixed_method is MixedMethods.GRADIENT:
            Q = self._if_none_replace_with_gfu(Q, component=2)
            gradient_rho = Q[self._indices.DENSITY, :]
        else:
            U = self._if_none_replace_with_gfu(U)
            gradient_U = grad(U)
            gradient_rho = gradient_U[self._indices.DENSITY, :]

        return gradient_rho

    def velocity_gradient(self,  U: Optional[CF] = None, Q: Optional[CF] = None):

        rho = self.density(U)
        rho_u = self.momentum(U)

        gradient_rho = self.density_gradient(U, Q)
        gradient_rho_u = self.momentum_gradient(U, Q)

        gradient_u = gradient_rho_u/rho - OuterProduct(rho_u, gradient_rho)/rho**2

        return gradient_u

    def momentum_gradient(self, U: Optional[CF] = None, Q: Optional[CF] = None):

        mixed_method = self.cfg.mixed_method

        if mixed_method is MixedMethods.GRADIENT:
            Q = self._if_none_replace_with_gfu(Q, component=2)
            gradient_rho_u = Q[self._indices.MOMENTUM, :]
        else:
            U = self._if_none_replace_with_gfu(U)
            gradient_U = grad(U)
            gradient_rho_u = gradient_U[self._indices.MOMENTUM, :]

        return gradient_rho_u

    def pressure_gradient(self, U: Optional[CF] = None, Q: Optional[CF] = None):
        mixed_method = self.cfg.mixed_method
        gamma = self.cfg.heat_capacity_ratio

        rho = self.density(U)
        gradient_rho = self.density_gradient(U, Q)

        if mixed_method is MixedMethods.STRAIN_HEAT:
            T = self.temperature(U)
            gradient_T = self.temperature_gradient(U, Q)

            gradient_p = (gamma - 1)/gamma * (T * gradient_rho + rho * gradient_T)

        else:
            gradient_rho_Ei = self.inner_energy_gradient(U, Q)
            gradient_p = (gamma - 1) * gradient_rho_Ei
        return gradient_p

    def temperature_gradient(self, U: Optional[CF] = None, Q: Optional[CF] = None):

        mixed_method = self.cfg.mixed_method
        gamma = self.cfg.heat_capacity_ratio

        if mixed_method is MixedMethods.STRAIN_HEAT:
            Q = self._if_none_replace_with_gfu(Q, 2)
            gradient_T = Q[self._indices.TEMPERATURE_GRADIENT]
        else:
            gradient_Ei = self.specific_inner_energy_gradient(U, Q)
            gradient_T = gamma * gradient_Ei

        return gradient_T

    def inner_energy_gradient(self, U: Optional[CF] = None, Q: Optional[CF] = None):
        gradient_rho_E = self.energy_gradient(U, Q)
        gradient_rho_Ek = self.kinetic_energy_gradient(U, Q)
        return gradient_rho_E - gradient_rho_Ek

    def specific_inner_energy_gradient(self, U: Optional[CF] = None, Q: Optional[CF] = None):
        gradient_E = self.specific_energy_gradient(U, Q)
        gradient_Ek = self.specific_kinetic_energy_gradient(U, Q)
        return gradient_E - gradient_Ek

    def kinetic_energy_gradient(self, U: Optional[CF] = True, Q: Optional[CF] = None):
        rho = self.density(U)
        gradient_rho = self.density_gradient(U, Q)

        rho_u = self.momentum(U)
        gradient_rho_m = self.momentum_gradient(U, Q)
        return gradient_rho_m.trans*rho_u/rho - InnerProduct(rho_u, rho_u)*gradient_rho/(2*rho**2)

    def specific_kinetic_energy_gradient(self, U: Optional[CF] = None, Q: Optional[CF] = None):
        rho = self.density(U)
        gradient_rho = self.density_gradient(U, Q)

        rho_u = self.momentum(U)
        gradient_rho_m = self.momentum_gradient(U, Q)
        return gradient_rho_m.trans*rho_u/rho**2 - InnerProduct(rho_u, rho_u)*gradient_rho/rho**3

    def energy_gradient(self, U: Optional[CF] = None, Q: Optional[CF] = None):

        mixed_method = self.cfg.mixed_method

        if mixed_method is MixedMethods.GRADIENT:
            Q = self._if_none_replace_with_gfu(Q, component=2)
            gradient_rho_E = Q[self._indices.ENERGY, :]
        else:
            U = self._if_none_replace_with_gfu(U)
            gradient_U = grad(U)
            gradient_rho_E = gradient_U[self._indices.ENERGY, :]

        return gradient_rho_E

    def specific_energy_gradient(self, U: Optional[CF] = None, Q: Optional[CF] = None):
        rho = self.density(U)
        gradient_rho = self.density_gradient(U, Q)

        rho_E = self.energy(U)
        gradient_rho_E = self.energy_gradient(U, Q)

        return gradient_rho_E/rho - gradient_rho*rho_E/rho**2

    def enthalpy_gradient(self, U: Optional[CF] = None, Q: Optional[CF] = None):
        gradient_rho_E = self.energy_gradient(U, Q)
        gradient_p = self.pressure_gradient(U, Q)
        return gradient_rho_E + gradient_p

    def specific_enthalpy_gradient(self, U: Optional[CF] = None, Q: Optional[CF] = None):
        gamma = self.cfg.heat_capacity_ratio

        gradient_E = self.specific_energy_gradient(U, Q)
        gradient_T = self.temperature_gradient(U, Q)

        return gradient_E + (gamma - 1)/gamma * gradient_T

    def vorticity(self, U: Optional[CF] = None, Q: Optional[CF] = None):

        gradient_u = self.velocity_gradient(U, Q)
        rate_of_rotation = gradient_u - gradient_u.trans

        if self.mesh.dim == 2:
            return rate_of_rotation[1, 0]
        elif self.mesh.dim == 3:
            return CF(tuple(rate_of_rotation[2, 1], rate_of_rotation[0, 2], rate_of_rotation[1, 0]))

    def heat_flux_gradient(self, U: Optional[CF] = None, Q: Optional[CF] = None):

        Re = self.cfg.Reynolds_number
        Pr = self.cfg.Prandtl_number
        mixed_method = self.cfg.mixed_method
        dim = self.mesh.dim

        if mixed_method is MixedMethods.STRAIN_HEAT:
            Q = self._if_none_replace_with_gfu(Q, 2)
            gradient_Q = grad(Q)
            phi = self.temperature_gradient(U, Q)
            gradient_phi = CF(tuple(gradient_Q[index, :]
                              for index in self._indices.TEMPERATURE_GRADIENT), dims=(dim, dim))

            mu = self.dynamic_viscosity(U)
            gradient_mu = self.dynamic_viscosity_gradient(U, Q)

            gradient_heat = -1/(Re * Pr) * (OuterProduct(phi, gradient_mu) + mu * gradient_phi)
        else:
            raise NotImplementedError(f"Deviatoric strain tensor: {mixed_method}")

        return gradient_heat

    def deviatoric_strain_tensor(self, U: Optional[CF] = None, Q: Optional[CF] = None):

        mixed_method = self.cfg.mixed_method
        dim = self.mesh.dim

        if mixed_method is MixedMethods.STRAIN_HEAT:
            Q = self._if_none_replace_with_gfu(Q, component=2)
            strain = CF(tuple(Q[index] for index in self._indices.STRAIN), dims=(dim, dim))

        elif mixed_method is MixedMethods.GRADIENT:
            gradient_u = self.velocity_gradient(U, Q)
            trace_gradient_u = sum(CF(tuple(gradient_u[i, i] for i in range(dim))))
            strain = gradient_u + gradient_u.trans - 2/3 * trace_gradient_u * Id(dim)

        else:
            raise NotImplementedError(f"Deviatoric strain tensor: {mixed_method}")

        return strain

    def deviatoric_strain_tensor_gradient(self, U: Optional[CF] = None, Q: Optional[CF] = None):

        mixed_method = self.cfg.mixed_method
        dim = self.mesh.dim

        if mixed_method is MixedMethods.STRAIN_HEAT:
            Q = self._if_none_replace_with_gfu(Q, 2)
            gradient_Q = grad(Q)
            strain_gradient = CF(tuple(gradient_Q[index, :] for index in self._indices.STRAIN), dims=(dim, dim, dim))

        else:
            raise NotImplementedError(f"Deviatoric strain tensor: {mixed_method}")

        return strain_gradient

    def deviatoric_stress_tensor_gradient(self, U: Optional[CF] = None, Q: Optional[CF] = None):

        Re = self.cfg.Reynolds_number
        dim = self.mesh.dim

        strain = self.deviatoric_strain_tensor(U, Q)
        gradient_strain = self.deviatoric_strain_tensor_gradient(U, Q)

        mu = self.dynamic_viscosity(U)
        gradient_mu = self.dynamic_viscosity_gradient(U, Q)

        gradient_stress = OuterProduct(strain, gradient_mu).Reshape((dim, dim, dim)) + mu * gradient_strain
        gradient_stress /= Re

        return gradient_stress

    def convective_flux_gradient(self, U, Q):

        dim = self.mesh.dim
        shape = (dim, dim, dim)

        rho = self.density(U)
        rho_u = self.momentum(U)
        rho_H = self.enthalpy(U)

        gradient_rho = self.density_gradient(U, Q)
        gradient_rho_u = self.momentum_gradient(U, Q)
        gradient_rho_H = self.enthalpy_gradient(U, Q)
        gradient_p = self.pressure_gradient(U, Q)

        continuity_gradient = gradient_rho_u
        momentum_gradient = OuterProduct(rho_u, gradient_rho_u).Reshape(shape)/rho
        momentum_gradient += momentum_gradient.TensorTranspose((1, 0, 2))
        momentum_gradient += OuterProduct(Id(dim), gradient_p).Reshape(shape)
        momentum_gradient -= OuterProduct(OuterProduct(rho_u, rho_u), gradient_rho).Reshape(shape) / rho**2
        energy_gradient = gradient_rho_u * rho_H/rho
        energy_gradient += OuterProduct(rho_u, gradient_rho_H)/rho
        energy_gradient -= OuterProduct(rho_u, gradient_rho) * rho_H / rho**2

        convective_flux = CF(
            (continuity_gradient,
             momentum_gradient,
             energy_gradient), dims=(dim + 2, dim, dim))

        return convective_flux

    def diffusive_flux_gradient(self, U, Q):

        dim = self.mesh.dim

        u = self.velocity(U)
        stress = self.deviatoric_stress_tensor(U, Q)

        gradient_u = self.velocity_gradient(U, Q)
        gradient_heat_flux = self.heat_flux_gradient(U, Q)
        gradient_stress = self.deviatoric_stress_tensor_gradient(U, Q)

        continuity = CF(tuple(0 for i in range(dim * dim)), dims=(dim, dim))
        momentum = gradient_stress
        energy = stress * gradient_u + fem.Einsum('ijk,j->ik', gradient_stress, u) - gradient_heat_flux

        flux_gradient = CF((continuity, momentum, energy), dims=(dim+2, dim, dim))

        return flux_gradient

    def reflect(self, U):

        n = self.normal

        rho = self.density(U)
        rho_u = self.momentum(U)
        rho_E = self.energy(U)

        return CF((rho, rho_u - InnerProduct(rho_u, n)*n, rho_E))
