from __future__ import annotations
from ngsolve import *
from typing import Optional, NamedTuple

from .interface import Formulation, MixedMethods


class Indices(NamedTuple):
    PRESSURE: int
    VELOCITY: slice
    TEMPERATURE: int


class PrimitiveFormulation(Formulation):

    _indices: Indices

    def add_time_bilinearform(self, blf):

        bonus_order_vol = self.cfg.bonus_int_order_vol
        compile_flag = self.cfg.compile_flag

        U, V = self.TnT.PRIMAL

        time_levels_gfu = self._gfus.get_component(0)
        time_levels_gfu['n+1'] = U
        G = self.G_matrix(U)

        var_form = InnerProduct(G * self.time_scheme.apply(time_levels_gfu), V) * dx(bonus_intorder=bonus_order_vol)

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

    def add_diffusive_bilinearform(self, blf) -> None:
        raise NotImplementedError()

    def pressure(self, U: Optional[CF] = None):
        U = self._if_none_replace_with_gfu(U)
        return U[self._indices.PRESSURE]

    def velocity(self, U: Optional[CF] = None):
        U = self._if_none_replace_with_gfu(U)
        return U[self._indices.VELOCITY]

    def temperature(self, U: Optional[CF] = None):
        U = self._if_none_replace_with_gfu(U)
        return U[self._indices.TEMPERATURE]

    def density(self, U: Optional[CF] = None):
        gamma = self.cfg.heat_capacity_ratio
        p = self.pressure(U)
        T = self.temperature(U)
        return gamma/(gamma - 1) * p/T

    def momentum(self, U: Optional[CF] = None):
        rho = self.density(U)
        u = self.velocity(U)
        return rho * u

    def energy(self, U: Optional[CF] = None):
        rho = self.density(U)
        E = self.specific_energy(U)
        return rho * E

    def specific_energy(self, U: Optional[CF] = None):
        Ei = self.specific_inner_energy(U)
        Ek = self.specific_kinetic_energy(U)
        return Ei + Ek

    def kinetic_energy(self, U: Optional[CF] = None):
        rho = self.density(U)
        Ek = self.specific_kinetic_energy(U)
        return rho*Ek

    def specific_kinetic_energy(self, U: Optional[CF] = None):
        u = self.velocity(U)
        return InnerProduct(u, u)/2

    def inner_energy(self, U: Optional[CF] = None):
        gamma = self.cfg.heat_capacity_ratio
        p = self.pressure(U)
        return p/(gamma - 1)

    def specific_inner_energy(self, U: Optional[CF] = None):
        gamma = self.cfg.heat_capacity_ratio
        T = self.temperature(U)
        return T/gamma

    def enthalpy(self, U: Optional[CF] = None):
        return self.energy(U) + self.pressure(U)

    def specific_enthalpy(self, U: Optional[CF] = None):
        gamma = self.cfg.heat_capacity_ratio
        return self.specific_energy(U) + (gamma - 1)/gamma * self.temperature(U)

    def speed_of_sound(self, U: Optional[CF] = None):
        gamma = self.cfg.heat_capacity_ratio
        T = self.temperature(U)
        return sqrt((gamma - 1) * T)

    def density_gradient(self, U: Optional[CF] = None, Q: Optional[CF] = None):
        gamma = self.cfg.heat_capacity_ratio

        p = self.pressure(U)
        gradient_p = self.pressure_gradient(U, Q)
        T = self.temperature(U)
        gradient_T = self.temperature_gradient(U, Q)

        gradient_rho = gamma/(gamma - 1) * (gradient_p/T - p/T**2 * gradient_T)
        return gradient_rho

    def velocity_gradient(self, U: Optional[CF] = None, Q: Optional[CF] = None):
        mixed_method = self.cfg.mixed_method

        if mixed_method is MixedMethods.GRADIENT:
            Q = self._if_none_replace_with_gfu(Q, component=2)
            gradient_u = Q[self._indices.VELOCITY, :]
        else:
            U = self._if_none_replace_with_gfu(U)
            gradient_U = grad(U)
            gradient_u = gradient_U[self._indices.VELOCITY, :]

        return gradient_u

    def momentum_gradient(self, U: Optional[CF] = None, Q: Optional[CF] = None):
        rho = self.density(U)
        gradient_rho = self.density_gradient(U, Q)

        u = self.velocity(U)
        gradient_u = self.velocity_gradient(U, Q)

        return rho * gradient_u + OuterProduct(u, gradient_rho)

    def pressure_gradient(self, U: Optional[CF] = None, Q: Optional[CF] = None):
        mixed_method = self.cfg.mixed_method

        if mixed_method is MixedMethods.GRADIENT:
            Q = self._if_none_replace_with_gfu(Q, component=2)
            gradient_p = Q[self._indices.PRESSURE, :]
        else:
            U = self._if_none_replace_with_gfu(U)
            gradient_U = grad(U)
            gradient_p = gradient_U[self._indices.PRESSURE, :]

        return gradient_p

    def temperature_gradient(self, U: Optional[CF] = None, Q: Optional[CF] = None):
        mixed_method = self.cfg.mixed_method

        if mixed_method is MixedMethods.GRADIENT:
            Q = self._if_none_replace_with_gfu(Q, component=2)
            gradient_T = Q[self._indices.TEMPERATURE, :]
        else:
            U = self._if_none_replace_with_gfu(U)
            gradient_U = grad(U)
            gradient_T = gradient_U[self._indices.TEMPERATURE, :]

        return gradient_T

    def inner_energy_gradient(self, U: Optional[CF] = None, Q: Optional[CF] = None):
        gamma = self.cfg.heat_capacity_ratio
        gradient_p = self.pressure_gradient(U, Q)
        return gradient_p / (gamma - 1)

    def specific_inner_energy_gradient(self, U: Optional[CF] = None, Q: Optional[CF] = None):
        gamma = self.cfg.heat_capacity_ratio
        gradient_T = self.temperature_gradient(U, Q)
        return gradient_T / gamma

    def kinetic_energy_gradient(self, U: Optional[CF] = True, Q: Optional[CF] = None):
        rho = self.density(U)
        gradient_rho = self.density_gradient(U, Q)

        Ek = self.specific_kinetic_energy(U)
        gradient_Ek = self.specific_kinetic_energy_gradient(U, Q)

        return rho * gradient_Ek + Ek * gradient_rho

    def specific_kinetic_energy_gradient(self, U: Optional[CF] = None, Q: Optional[CF] = None):
        u = self.velocity(U)
        gradient_u = self.velocity_gradient(U, Q)
        return gradient_u.trans * u

    def energy_gradient(self, U: Optional[CF] = None, Q: Optional[CF] = None):
        gradient_rho_Ei = self.inner_energy_gradient(U, Q)
        gradient_rho_Ek = self.kinetic_energy_gradient(U, Q)
        return gradient_rho_Ei + gradient_rho_Ek

    def specific_energy_gradient(self, U: Optional[CF] = None, Q: Optional[CF] = None):
        gradient_Ei = self.specific_inner_energy_gradient(U, Q)
        gradient_Ek = self.specific_kinetic_energy_gradient(U, Q)
        return gradient_Ei + gradient_Ek

    def enthalpy_gradient(self, U: Optional[CF] = None, Q: Optional[CF] = None):
        gradient_p = self.temperature_gradient(U, Q)
        return self.energy_gradient(U, Q) + gradient_p

    def specific_enthalpy_gradient(self, U: Optional[CF] = None, Q: Optional[CF] = None):
        gamma = self.cfg.heat_capacity_ratio
        gradient_T = self.temperature_gradient(U, Q)
        gradient_E = self.specific_energy_gradient(U, Q)
        return gradient_E + (gamma - 1)/gamma * gradient_T
