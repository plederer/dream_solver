from __future__ import annotations
from ngsolve import *
from typing import Optional, NamedTuple
from .interface import _Formulation, MixedMethods, TensorIndices, FiniteElementSpace
from ..region import BoundaryConditions as bcs
from ..region import DomainConditions as dcs


class Indices(NamedTuple):
    DENSITY: int
    MOMENTUM: slice
    ENERGY: int
    TEMPERATURE_GRADIENT: slice
    STRAIN: TensorIndices


class ConservativeFormulation(_Formulation):

    _indices: Indices

    def add_time_bilinearform(self, blf):

        compile_flag = self.cfg.compile_flag

        U, V = self.TnT.PRIMAL

        time_levels_gfu = self._gfus.get_component(0)
        time_levels_gfu['n+1'] = U

        var_form = InnerProduct(self.time_scheme.apply(time_levels_gfu), V) * dx

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
        mask.vec[~mask_fes.GetDofs(self.dmesh.boundary(self.dmesh.bcs.pattern))] = 1

        var_form = -InnerProduct(self.convective_flux(U), grad(V)) * dx(bonus_intorder=bonus_order_vol)
        var_form += InnerProduct(self.convective_numerical_flux(U, Uhat, self.normal),
                                 V) * dx(element_boundary=True, bonus_intorder=bonus_order_bnd)
        var_form += -mask * InnerProduct(self.convective_numerical_flux(U, Uhat, self.normal),
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
        mask.vec[~mask_fes.GetDofs(self.dmesh.boundary(self.dmesh.bcs.pattern))] = 1

        var_form = InnerProduct(self.diffusive_flux(U, Q), grad(V)) * dx(bonus_intorder=bonus_order_vol)
        var_form -= InnerProduct(self.diffusive_numerical_flux(U, Uhat, Q, self.normal),
                                 V) * dx(element_boundary=True, bonus_intorder=bonus_order_bnd)
        var_form += mask * InnerProduct(self.diffusive_numerical_flux(U, Uhat, Q, self.normal),
                                        Vhat) * dx(element_boundary=True, bonus_intorder=bonus_order_bnd)

        blf += var_form.Compile(compile_flag)

        if mixed_method is not MixedMethods.NONE:
            self._add_mixed_bilinearform(blf)

    def add_initial_linearform(self, lf):

        mixed_method = self.cfg.mixed_method
        bonus_vol = self.cfg.bonus_int_order_vol
        bonus_bnd = self.cfg.bonus_int_order_bnd

        for domain, dc in self.dmesh.dcs.initial_conditions.items():

            dim = self.mesh.dim

            _, V = self.TnT.PRIMAL
            _, Vhat = self.TnT.PRIMAL_FACET
            _, P = self.TnT.MIXED

            state = self.calc.determine_missing(dc.state)
            U_f = CF((state.density, state.momentum, state.energy))

            cf = U_f * V * dx(definedon=domain, bonus_intorder=bonus_vol)
            cf += U_f * Vhat * dx(element_boundary=True, definedon=domain, bonus_intorder=bonus_bnd)

            if mixed_method is MixedMethods.GRADIENT:
                Q_f = CF(tuple(U_f.Diff(dir) for dir in (x, y)), dims=(dim, dim+2)).trans
                cf += InnerProduct(Q_f, P) * dx(definedon=domain, bonus_intorder=bonus_vol)

            elif mixed_method is MixedMethods.STRAIN_HEAT:
                velocity_gradient = CF(tuple(CF(state.velocity).Diff(dir) for dir in (x, y)), dims=(dim, dim)).trans

                strain = velocity_gradient + velocity_gradient.trans
                strain -= 2/3 * (velocity_gradient[0, 0] + velocity_gradient[1, 1]) * Id(dim)

                gradient_T = CF(tuple(CF(state.temperature).Diff(dir) for dir in (x, y)))

                Q_f = CF((strain[0, 0], strain[0, 1], strain[1, 1], gradient_T[0], gradient_T[1]))

                cf += InnerProduct(Q_f, P) * dx(definedon=domain, bonus_intorder=bonus_vol)

            lf += cf

    def add_perturbation_linearform(self, lf):
        bonus_vol = self.cfg.bonus_int_order_vol
        bonus_bnd = self.cfg.bonus_int_order_bnd

        for domain, dc in self.dmesh.dcs.perturbations.items():

            _, V = self.TnT.PRIMAL
            _, Vhat = self.TnT.PRIMAL_FACET

            state = self.calc.determine_missing(dc.state)
            U_f = CF((state.density, state.momentum, state.energy))

            cf = U_f * V * dx(definedon=domain, bonus_intorder=bonus_vol)
            cf += U_f * Vhat * dx(element_boundary=True, definedon=domain, bonus_intorder=bonus_bnd)

            lf += cf

    def add_forcing_linearform(self, lf):
        bonus_vol = self.cfg.bonus_int_order_vol
        bonus_bnd = self.cfg.bonus_int_order_bnd

        for domain, dc in self.dmesh.dcs.force.items():

            _, V = self.TnT.PRIMAL
            _, Vhat = self.TnT.PRIMAL_FACET

            if dc.state is None:
                continue

            state = self.calc.determine_missing(dc.state)
            U_f = CF((state.density, state.momentum, state.energy))

            cf = U_f * V * dx(definedon=domain, bonus_intorder=bonus_vol)
            cf += U_f * Vhat * dx(element_boundary=True, definedon=domain, bonus_intorder=bonus_bnd)

            lf += cf

    def _add_dirichlet_bilinearform(self, blf, boundary: Region, bc: bcs.Dirichlet):

        compile_flag = self.cfg.compile_flag

        state = self.calc.determine_missing(bc.state)
        dirichlet = CF((state.density, state.momentum, state.energy))

        Uhat, Vhat = self.TnT.PRIMAL_FACET

        cf = (dirichlet-Uhat)
        cf = cf * Vhat * ds(skeleton=True, definedon=boundary)
        blf += cf.Compile(compile_flag)

    def _add_farfield_bilinearform(self, blf,  boundary: Region, bc: bcs.FarField):

        compile_flag = self.cfg.compile_flag
        bonus_order_bnd = self.cfg.bonus_int_order_bnd

        state = self.calc.determine_missing(bc.state)
        farfield = CF((state.density, state.momentum, state.energy))

        U, _ = self.TnT.PRIMAL
        Uhat, Vhat = self.TnT.PRIMAL_FACET

        riemann_solver = self.cfg.riemann_solver.LAX_FRIEDRICH

        An_out = self.convective_stabilisation_matrix(Uhat, self.normal, riemann_solver)
        An_in = self.convective_stabilisation_matrix(Uhat, -self.normal, riemann_solver)

        u_in = self.characteristic_velocities(Uhat, self.normal, type="in", as_matrix=True)
        u_out = self.characteristic_velocities(Uhat, self.normal, type="out", as_matrix=True)

        An_in = self.DME_from_CHAR_matrix(u_in, Uhat, self.normal)
        An_out = self.DME_from_CHAR_matrix(u_out, Uhat, self.normal)

        cf = An_out * (U - Uhat)
        cf += An_in * (farfield - Uhat)
        cf = cf * Vhat * ds(skeleton=True, definedon=boundary, bonus_intorder=bonus_order_bnd)

        blf += cf.Compile(compile_flag)

    def _add_outflow_bilinearform(self, blf, boundary: Region, bc: bcs.Outflow):

        bonus_order_bnd = self.cfg.bonus_int_order_bnd
        compile_flag = self.cfg.compile_flag
        gamma = self.cfg.heat_capacity_ratio

        U, _ = self.TnT.PRIMAL
        Uhat, Vhat = self.TnT.PRIMAL_FACET

        rho = self.density(U)
        rho_u = self.momentum(U)

        energy = bc.state.pressure/(gamma - 1) + 1/(2*rho) * InnerProduct(rho_u, rho_u)
        outflow = CF((rho, rho_u, energy))

        cf = InnerProduct(outflow - Uhat, Vhat)
        cf = cf * ds(skeleton=True, definedon=boundary, bonus_intorder=bonus_order_bnd)

        blf += cf.Compile(compile_flag)

    def _add_inviscid_wall_bilinearform(self, blf, boundary: Region, bc: bcs.InviscidWall):

        bonus_order_bnd = self.cfg.bonus_int_order_bnd
        compile_flag = self.cfg.compile_flag

        U, _ = self.TnT.PRIMAL
        Uhat, Vhat = self.TnT.PRIMAL_FACET
        n = self.normal

        rho = self.density(U)
        rho_u = self.momentum(U)
        rho_E = self.energy(U)
        U_wall = CF((rho, rho_u - InnerProduct(rho_u, n)*n, rho_E))

        cf = InnerProduct(U_wall-Uhat, Vhat)
        cf = cf * ds(skeleton=True, definedon=boundary, bonus_intorder=bonus_order_bnd)

        blf += cf.Compile(compile_flag)

    def _add_isothermal_wall_bilinearform(self, blf, boundary: Region, bc: bcs.IsothermalWall):

        compile_flag = self.cfg.compile_flag

        U, _ = self.TnT.PRIMAL
        Uhat, Vhat = self.TnT.PRIMAL_FACET

        rho = self.density(U)
        rho_u = tuple(0 for _ in range(self.mesh.dim))
        rho_E = self.calc.inner_energy_dT(rho, bc.state.temperature)

        cf = InnerProduct(CF((rho, rho_u, rho_E)) - Uhat, Vhat)
        cf = cf * ds(skeleton=True, definedon=boundary)

        blf += cf.Compile(compile_flag)

    def _add_adiabatic_wall_bilinearform(self, blf, boundary: Region, bc: bcs.AdiabaticWall):

        mixed_method = self.cfg.mixed_method
        if mixed_method is MixedMethods.NONE:
            raise NotImplementedError(f"Adiabatic wall not implemented for {mixed_method}")

        compile_flag = self.cfg.compile_flag
        n = self.normal

        U, _ = self.TnT.PRIMAL
        Uhat, Vhat = self.TnT.PRIMAL_FACET
        Q, _ = self.TnT.MIXED

        tau_dE = self.diffusive_stabilisation_matrix(Uhat)[self.mesh.dim+1, self.mesh.dim+1]

        diff_rho = self.density(U) - self.density(Uhat)
        diff_rho_u = -self.momentum(Uhat)
        diff_rho_E = tau_dE * (self.temperature_gradient(U, Q)*n - (self.energy(U) - self.energy(Uhat)))

        cf = InnerProduct(CF((diff_rho, diff_rho_u, diff_rho_E)), Vhat) * ds(skeleton=True, definedon=boundary)
        blf += cf.Compile(compile_flag)

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
        tau_c = self.convective_stabilisation_matrix(Uhat, unit_vector)
        return self.convective_flux(Uhat)*unit_vector + tau_c * (U - Uhat)

    def diffusive_numerical_flux(self, U, Uhat, Q, unit_vector: CF):
        tau_d = self.diffusive_stabilisation_matrix(Uhat)
        return self.diffusive_flux(Uhat, Q)*unit_vector - tau_d * (U-Uhat)

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

    def velocity_gradient(self,  U: Optional[CF] = None, Q: Optional[CF] = None, Uhat: Optional[CF] = None):
        if Uhat is None:
            Uhat = U

        rho = self.density(Uhat)
        rho_u = self.momentum(Uhat)

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

    def pressure_gradient(self, U: Optional[CF] = None, Q: Optional[CF] = None, Uhat: Optional[CF] = None):
        mixed_method = self.cfg.mixed_method
        gamma = self.cfg.heat_capacity_ratio

        rho = self.density(U)
        gradient_rho = self.density_gradient(U, Q)

        if mixed_method is MixedMethods.STRAIN_HEAT:
            T = self.temperature(U)
            gradient_T = self.temperature_gradient(U, Q)

            gradient_p = (gamma - 1)/gamma * (T * gradient_rho + rho * gradient_T)

        else:
            gradient_rho_Ei = self.inner_energy_gradient(U, Q, Uhat)
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

    def inner_energy_gradient(self, U: Optional[CF] = None, Q: Optional[CF] = None, Uhat: Optional[CF] = None):
        gradient_rho_E = self.energy_gradient(U, Q)
        gradient_rho_Ek = self.kinetic_energy_gradient(U, Q, Uhat)
        return gradient_rho_E - gradient_rho_Ek

    def specific_inner_energy_gradient(self, U: Optional[CF] = None, Q: Optional[CF] = None):
        gradient_E = self.specific_energy_gradient(U, Q)
        gradient_Ek = self.specific_kinetic_energy_gradient(U, Q)
        return gradient_E - gradient_Ek

    def kinetic_energy_gradient(self, U: Optional[CF] = None, Q: Optional[CF] = None, Uhat: Optional[CF] = None):
        if Uhat is None:
            Uhat = U

        rho = self.density(Uhat)
        rho_u = self.momentum(Uhat)

        gradient_rho = self.density_gradient(U, Q)
        gradient_rho_m = self.momentum_gradient(U, Q)

        return gradient_rho_m.trans*rho_u/rho - InnerProduct(rho_u, rho_u)*gradient_rho/(2*rho**2)

    def specific_kinetic_energy_gradient(self,
                                         U: Optional[CF] = None,
                                         Q: Optional[CF] = None,
                                         Uhat: Optional[CF] = None):
        if Uhat is None:
            Uhat = U

        rho = self.density(Uhat)
        rho_u = self.momentum(Uhat)

        gradient_rho = self.density_gradient(U, Q)
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

    def specific_energy_gradient(self, U: Optional[CF] = None, Q: Optional[CF] = None, Uhat: Optional[CF] = None):
        if Uhat is None:
            Uhat = U

        rho = self.density(Uhat)
        rho_E = self.energy(Uhat)

        gradient_rho = self.density_gradient(U, Q)
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


class ConservativeFormulation2D(ConservativeFormulation):

    _indices = Indices(DENSITY=0,
                       MOMENTUM=slice(1, 3),
                       ENERGY=3,
                       STRAIN=TensorIndices(XX=0, XY=1, YX=1, YY=2),
                       TEMPERATURE_GRADIENT=slice(3, 5))

    def _initialize_FE_space(self) -> FiniteElementSpace:

        order = self.cfg.order
        mixed_method = self.cfg.mixed_method

        p_sponge_layers = self.dmesh.dcs.psponge_layers
        if self.dmesh.highest_order_psponge > order:
            raise ValueError("Polynomial sponge order higher than polynomial discretization order")

        order_policy = ORDER_POLICY.CONSTANT
        if p_sponge_layers:
            order_policy = ORDER_POLICY.VARIABLE

        V = L2(self.mesh, order=order, order_policy=order_policy)
        if self.cfg.fem is self.cfg.fem.HDG:
            VHAT = FacetFESpace(self.mesh, order=order)
        elif self.cfg.fem is self.cfg.fem.EDG:
            VHAT = H1(self.mesh, order=order, orderinner=0)
        Q = VectorL2(self.mesh, order=order, order_policy=order_policy)

        if p_sponge_layers:

            sponge_region = self.dmesh.domain(self.dmesh.pattern(p_sponge_layers.keys()))
            vhat_dofs = ~VHAT.GetDofs(sponge_region)

            for domain, bc in p_sponge_layers.items():
                domain = self.dmesh.domain(domain)

                for el in domain.Elements():
                    V.SetOrder(NodeId(ELEMENT, el.nr), bc.order.high)

                domain_dofs = VHAT.GetDofs(domain)
                for i in range(bc.order.high + 1, order + 1, 1):
                    domain_dofs[i::order + 1] = 0

                vhat_dofs |= domain_dofs

            VHAT = Compress(VHAT, vhat_dofs)
            V.UpdateDofTables()

        if self.dmesh.is_periodic:
            VHAT = Periodic(VHAT)

        spaces = FiniteElementSpace.Settings(V**4, VHAT**4)

        if mixed_method is MixedMethods.NONE:
            pass
        elif mixed_method is MixedMethods.STRAIN_HEAT:
            spaces.MIXED = V**5
        elif mixed_method is MixedMethods.GRADIENT:
            spaces.MIXED = Q**4
        else:
            raise NotImplementedError(f"Mixed method {mixed_method} not implemented for {self}!")

        return FiniteElementSpace.from_settings(spaces)

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

            var_form = InnerProduct(eps, zeta) * dx()
            var_form += InnerProduct(u, div_dev_zeta) * dx(bonus_intorder=bonus_vol)
            var_form -= InnerProduct(uhat, dev_zeta*n) * dx(element_boundary=True, bonus_intorder=bonus_bnd)

            # Temperature gradient

            phi = self.temperature_gradient(U, Q)
            xi = self.temperature_gradient(V, P)
            T = self.temperature(U)
            That = self.temperature(Uhat)

            div_xi = gradient_P[3, 0] + gradient_P[4, 1]

            var_form += InnerProduct(phi, xi) * dx()
            var_form += InnerProduct(T, div_xi) * dx(bonus_intorder=bonus_vol)
            var_form -= InnerProduct(That*n, xi) * dx(element_boundary=True, bonus_intorder=bonus_bnd)

        elif mixed_method is MixedMethods.GRADIENT:

            var_form = InnerProduct(Q, P) * dx()
            var_form += InnerProduct(U, div(P)) * dx()
            var_form -= InnerProduct(Uhat, P*n) * dx(element_boundary=True)

        else:
            raise NotImplementedError(f"Mixed Bilinearform: {mixed_method}")

        blf += var_form.Compile(compile_flag)

    def _add_sponge_bilinearform(self, blf, domain: Region, dc: dcs.SpongeLayer, weight_function: GridFunction):
        compile_flag = self.cfg.compile_flag
        bonus_int_order = weight_function.space.globalorder

        state = self.calc.determine_missing(dc.state)
        ref = CF((state.density, state.momentum, state.energy))

        U, V = self.TnT.PRIMAL

        cf = weight_function * (U - ref)
        cf = cf * V * dx(definedon=domain, bonus_intorder=bonus_int_order)

        blf += cf.Compile(compile_flag)

    def _add_psponge_bilinearform(self, blf, domain: Region, dc: dcs.PSpongeLayer, weight_function: GridFunction):

        compile_flag = self.cfg.compile_flag
        bonus_int_order = weight_function.space.globalorder

        U, V = self.TnT.PRIMAL
        low_order_space = L2(self.mesh, order=dc.order.low)

        if dc.is_equal_order:
            state = self.calc.determine_missing(dc.state)
            ref = CF((state.density, state.momentum, state.energy))
            U_high = U - ref

        else:
            U_low = CF(tuple(Interpolate(proxy, low_order_space) for proxy in U))
            U_high = U - U_low

        cf = weight_function * U_high * V * dx(definedon=domain, bonus_intorder=bonus_int_order)
        blf += cf.Compile(compile_flag)

    def _add_nonreflecting_inflow_bilinearform(self, blf, boundary: Region, bc: bcs.Dirichlet):
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

    def _get_characteristic_amplitudes_incoming(self, bc: bcs.Outflow_NSCBC):

        M = self.cfg.Mach_number

        U, _ = self.TnT.PRIMAL
        Uhat, _ = self.TnT.PRIMAL_FACET
        Q, _ = self.TnT.MIXED

        rho = self.density(Uhat)
        c = self.speed_of_sound(Uhat)
        un = self.velocity(Uhat) * self.normal
        Mn = IfPos(un, un, -un)/c

        outflow_amp_in = bc.sigma * c * (1 - M**2)/bc.reference_length * (self.pressure(Uhat) - bc.state.pressure)

        if bc.tang_conv_flux:

            ut = InnerProduct(self.velocity(Uhat), self.tangential)

            gradient_p_t = InnerProduct(self.pressure_gradient(U, Q, Uhat), self.tangential)
            gradient_u_t = self.velocity_gradient(U, Q, Uhat) * self.tangential

            beta_l = M
            beta_t = M

            outflow_amp_in -= (1 - beta_l) * ut * (gradient_p_t - c*rho*InnerProduct(gradient_u_t, self.normal))
            outflow_amp_in -= (1 - beta_t) * c**2 * rho * InnerProduct(gradient_u_t, self.tangential)

        L = CF((outflow_amp_in, 0, 0, 0))

        return L

    def _add_nonreflecting_outflow_bilinearform(self, blf, boundary: Region, bc: bcs.Outflow_NSCBC):

        bonus_order_bnd = self.cfg.bonus_int_order_bnd
        compile_flag = self.cfg.compile_flag
        mixed_method = self.cfg.mixed_method

        U, _ = self.TnT.PRIMAL
        Uhat, Vhat = self.TnT.PRIMAL_FACET
        Q, _ = self.TnT.MIXED

        An_in = self.DME_convective_jacobian_incoming(Uhat, self.normal)
        An_out = self.DME_convective_jacobian_outgoing(Uhat, self.normal)

        LAM_in = self.characteristic_velocities(Uhat, self.normal, 'in', True)
        P_in = self.DME_from_CHAR(Uhat, self.normal) * LAM_in

        time_levels_gfu = self._gfus.get_component(1)
        time_levels_gfu['n+1'] = Uhat

        L = self._get_characteristic_amplitudes_incoming(bc)

        cf = InnerProduct(An_out*(U - Uhat), Vhat)
        cf += InnerProduct(An_in*self.time_scheme.apply(time_levels_gfu), Vhat)
        cf += An_in*(self.DME_convective_jacobian(Uhat, self.tangential) * (grad(U) * self.tangential)) * Vhat
        cf += InnerProduct(P_in * L, Vhat)

        if mixed_method is not MixedMethods.NONE:
            cf -= An_in*(self.conservative_diffusive_jacobian(Uhat, Q,
                         self.tangential)*(grad(U)*self.tangential)) * Vhat
            cf -= An_in*(self.conservative_diffusive_jacobian(Uhat, Q, self.normal) * (grad(U) * self.normal)) * Vhat

            cf -= An_in*(self.mixed_diffusive_jacobian(Uhat, self.normal) * (grad(Q) * self.normal)) * Vhat
            cf -= An_in*(self.mixed_diffusive_jacobian(Uhat, self.tangential) * (grad(Q) * self.tangential)) * Vhat

        cf = cf * ds(skeleton=True, definedon=boundary, bonus_intorder=bonus_order_bnd)
        blf += cf.Compile(compile_flag)

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
