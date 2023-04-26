from __future__ import annotations
from ngsolve import *
from typing import Optional, NamedTuple

from .interface import Formulation, MixedMethods, VectorIndices


class Indices(NamedTuple):
    PRESSURE: Optional[int] = None
    VELOCITY: Optional[VectorIndices] = None
    TEMPERATURE: Optional[int] = None


class PrimitiveFormulation(Formulation):

    _indices: Indices

    def pressure(self, U: Optional[CF] = None):
        U = self._if_none_replace_with_gfu(U)
        return U[self._indices.PRESSURE]

    def velocity(self, U: Optional[CF] = None):
        U = self._if_none_replace_with_gfu(U)
        return CF(tuple(U[index] for index in self._indices.VELOCITY))

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
        p = self.pressure(U)
        rho_E = self.energy(U)
        return rho_E + p

    def specific_enthalpy(self, U: Optional[CF] = None):
        rho_H = self.enthalpy(U)
        rho = self.density(U)
        return rho_H/rho

    def speed_of_sound(self, U: Optional[CF] = None):
        gamma = self.cfg.heat_capacity_ratio
        p = self.pressure(U)
        rho = self.density(U)
        return sqrt(gamma * p/rho)

    def mach_number(self, U: Optional[CF] = None):
        u = self.velocity(U)
        c = self.speed_of_sound(U)
        return sqrt(InnerProduct(u, u)) / c
