from __future__ import annotations
from .base import Formulation, MixedMethods, GridFunctionComponents
from ngsolve import *


class ConservativeFormulation(Formulation):

    def get_gridfunction_components(self, gfu: GridFunction):
        mixed_method = self.solver_configuration.mixed_method

        if mixed_method is not MixedMethods.NONE:
            comp = GridFunctionComponents(
                PRIMAL=gfu.components[0],
                PRIMAL_FACET=gfu.components[1],
                MIXED=gfu.components[2])
        else:
            comp = GridFunctionComponents(
                PRIMAL=gfu.components[0],
                PRIMAL_FACET=gfu.components[1])

        return comp

    def density(self, U):
        return U[self._indices.DENSITY]

    def momentum(self, U):
        return CF(tuple(U[index] for index in self._indices.MOMENTUM))

    def energy(self, U):
        return U[self._indices.ENERGY]

    def specific_energy(self, U):

        rho = self.density(U)
        rho_E = self.energy(U)

        return rho_E/rho

    def specific_inner_energy(self, U):

        rho = self.density(U)
        rho_u = self.momentum(U)
        rho_E = self.energy(U)

        return (rho_E - InnerProduct(rho_u, rho_u)/(2 * rho))/rho

    def pressure(self, U):

        rho = self.density(U)
        rho_u = self.momentum(U)
        rho_E = self.energy(U)
        gamma = self.solver_configuration.heat_capacity_ratio

        return (gamma-1) * (rho_E - InnerProduct(rho_u, rho_u)/(2 * rho))

    def velocity(self, U):

        rho = self.density(U)
        rho_u = self.momentum(U)

        return rho_u/rho

    def temperature(self, U):

        rho = self.density(U)
        rho_u = self.momentum(U)
        rho_E = self.energy(U)
        gamma = self.solver_configuration.heat_capacity_ratio

        return gamma/rho * (rho_E - InnerProduct(rho_u, rho_u)/(2 * rho))

    def enthalpy(self, U):

        p = self.pressure(U)
        rho_E = self.energy(U)

        return rho_E + p

    def specific_enthalpy(self, U):

        rho_H = self.enthalpy(U)
        rho = self.density(U)

        return rho_H/rho

    def speed_of_sound(self, U):

        p = self.pressure(U)
        rho = self.density(U)
        gamma = self.solver_configuration.heat_capacity_ratio

        return sqrt(gamma * p/rho)

    def mach_number(self, U):

        u = self.velocity(U)
        c = self.speed_of_sound(U)

        return sqrt(InnerProduct(u, u)) / c

    def density_gradient(self, U, Q):

        mixed_method = self.solver_configuration.mixed_method

        if mixed_method is MixedMethods.GRADIENT:
            gradient_rho = Q[self._indices.DENSITY, ...]
        else:
            gradient_U = grad(U)
            gradient_rho = gradient_U[self._indices.DENSITY, ...]

        return gradient_rho

    def momentum_gradient(self, U, Q):

        mixed_method = self.solver_configuration.mixed_method
        dim = self.mesh.dim

        if mixed_method is MixedMethods.GRADIENT:
            gradient_rho_u = CF(tuple(Q[index, ...] for index in self._indices.MOMENTUM), dims=(dim, dim))
        else:
            gradient_U = grad(U)
            gradient_rho_u = CF(tuple(gradient_U[index, ...] for index in self._indices.MOMENTUM), dims=(dim, dim))

        return gradient_rho_u

    def energy_gradient(self, U, Q):

        mixed_method = self.solver_configuration.mixed_method

        if mixed_method is MixedMethods.GRADIENT:
            gradient_rho_E = Q[self._indices.ENERGY, ...]
        else:
            gradient_U = grad(U)
            gradient_rho_E = gradient_U[self._indices.ENERGY, ...]

        return gradient_rho_E

    def enthalpy_gradient(self, U, Q):
        gradient_rho_E = self.energy_gradient(U, Q)
        gradient_p = self.pressure_gradient(U, Q)
        return gradient_rho_E + gradient_p

    def pressure_gradient(self, U, Q):

        mixed_method = self.solver_configuration.mixed_method
        gamma = self.solver_configuration.heat_capacity_ratio

        rho = self.density(U)
        gradient_rho = self.density_gradient(U, Q)

        if mixed_method is MixedMethods.STRAIN_HEAT:
            T = self.temperature(U)
            gradient_T = self.temperature_gradient(U, Q)

            gradient_p = (gamma - 1)/gamma * (T * gradient_rho + rho * gradient_T)

        else:
            rho_u = self.momentum(U)

            gradient_rho_u = self.momentum_gradient(U, Q)
            gradient_rho_E = self.energy_gradient(U, Q)

            gradient_p = (gamma - 1) * (gradient_rho_E -
                                        gradient_rho_u.trans*rho_u/rho +
                                        InnerProduct(rho_u, rho_u)*gradient_rho/(2*rho**2))

        return gradient_p

    def temperature_gradient(self, U, Q):

        mixed_method = self.solver_configuration.mixed_method
        gamma = self.solver_configuration.heat_capacity_ratio

        if mixed_method is MixedMethods.STRAIN_HEAT:
            gradient_T = CF(tuple(Q[index] for index in self._indices.TEMPERATURE_GRADIENT))

        else:
            rho = self.density(U)
            rho_u = self.momentum(U)
            rho_E = self.energy(U)

            gradient_rho = self.density_gradient(U, Q)
            gradient_rho_m = self.momentum_gradient(U, Q)
            gradient_rho_E = self.energy_gradient(U, Q)

            gradient_T = gamma * (gradient_rho_E/rho -
                                  gradient_rho*rho_E/rho**2 -
                                  gradient_rho_m.trans*rho_u/rho**2 +
                                  InnerProduct(rho_u, rho_u)*gradient_rho/rho**3)

        return gradient_T

    def velocity_gradient(self, U, Q):

        rho = self.density(U)
        rho_u = self.momentum(U)

        gradient_rho = self.density_gradient(U, Q)
        gradient_rho_u = self.momentum_gradient(U, Q)

        gradient_u = gradient_rho_u/rho - OuterProduct(rho_u, gradient_rho)/rho**2

        return gradient_u

    def vorticity(self, U, Q):

        gradient_u = self.velocity_gradient(U, Q)
        rate_of_rotation = gradient_u - gradient_u.trans

        if self.mesh.dim == 2:
            return rate_of_rotation[1, 0]
        elif self.mesh.dim == 3:
            return CF(tuple(rate_of_rotation[2, 1], rate_of_rotation[0, 2], rate_of_rotation[1, 0]))

    def heat_flux(self, U, Q):

        Re = self.solver_configuration.Reynold_number
        Pr = self.solver_configuration.Prandtl_number

        gradient_T = self.temperature_gradient(U, Q)
        k = self.mu.get(U, Q) / (Re * Pr)

        return k * gradient_T

    def heat_flux_gradient(self, U, Q):

        Re = self.solver_configuration.Reynold_number
        Pr = self.solver_configuration.Prandtl_number
        mixed_method = self.solver_configuration.mixed_method
        dim = self.mesh.dim

        if mixed_method is MixedMethods.STRAIN_HEAT:
            gradient_Q = grad(Q)
            phi = self.temperature_gradient(U, Q)
            gradient_phi = CF(tuple(gradient_Q[index, ...]
                              for index in self._indices.TEMPERATURE_GRADIENT), dims=(dim, dim))

            mu = self.mu.get(U, Q)
            gradient_mu = self.mu.get_gradient(U, Q)

            gradient_heat = OuterProduct(phi, gradient_mu) + mu * gradient_phi
            gradient_heat /= Re * Pr

        else:
            raise NotImplementedError(f"Deviatoric strain tensor: {mixed_method}")

        return gradient_heat

    def deviatoric_strain_tensor(self, U, Q):

        mixed_method = self.solver_configuration.mixed_method
        dim = self.mesh.dim

        if mixed_method is MixedMethods.STRAIN_HEAT:
            strain = CF(tuple(Q[index] for index in self._indices.STRAIN), dims=(dim, dim))

        elif mixed_method is MixedMethods.GRADIENT:
            gradient_u = self.velocity_gradient(U, Q)
            trace_gradient_u = sum(CF(tuple(gradient_u[i, i] for i in range(dim))))
            strain = gradient_u + gradient_u.trans - 2/3 * trace_gradient_u * Id(dim)

        else:
            raise NotImplementedError(f"Deviatoric strain tensor: {mixed_method}")

        return strain

    def deviatoric_strain_tensor_gradient(self, U, Q):

        mixed_method = self.solver_configuration.mixed_method
        dim = self.mesh.dim

        if mixed_method is MixedMethods.STRAIN_HEAT:
            gradient_Q = grad(Q)
            strain_gradient = CF(tuple(gradient_Q[index, ...] for index in self._indices.STRAIN), dims=(dim, dim, dim))

        else:
            raise NotImplementedError(f"Deviatoric strain tensor: {mixed_method}")

        return strain_gradient

    def deviatoric_stress_tensor(self, U, Q):

        Re = self.solver_configuration.Reynold_number
        return self.mu.get(U, Q)/Re * self.deviatoric_strain_tensor(U, Q)

    def deviatoric_stress_tensor_gradient(self, U, Q):

        Re = self.solver_configuration.Reynold_number
        dim = self.mesh.dim

        strain = self.deviatoric_strain_tensor(U, Q)
        gradient_strain = self.deviatoric_strain_tensor_gradient(U, Q)

        mu = self.mu.get(U, Q)
        gradient_mu = self.mu.get_gradient(U, Q)

        gradient_stress = OuterProduct(strain, gradient_mu).Reshape((dim, dim, dim)) + mu * gradient_strain
        gradient_stress /= Re

        return gradient_stress

    def reflect(self, U):

        n = self.normal

        rho = self.density(U)
        rho_u = self.momentum(U)
        rho_E = self.energy(U)

        return CF((rho, rho_u - InnerProduct(rho_u, n)*n, rho_E))

    def add_mass_bilinearform(self, blf):

        mixed_method = self.solver_configuration.mixed_method

        (U, Uhat, Q), (V, Vhat, P) = self.TnT

        blf += U * V * dx
        blf += Uhat * Vhat * dx(element_boundary=True)

        if mixed_method is not MixedMethods.NONE:
            blf += InnerProduct(Q, P) * dx
