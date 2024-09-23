from __future__ import annotations
import logging
import ngsolve as ngs
import typing

import dream.bla as bla

from dream.config import MultipleConfiguration, multiple
from dream.time_schemes import ImplicitEuler, BDF2

from dream.compressible.config import (
    CompressibleState, CompressibleStateGradient, CompressibleFormulation, DreamMesh, CharacteristicRelaxationInflow,
    CharacteristicRelaxationOutflow, FarField, Outflow, InviscidWall, Symmetry, IsothermalWall, BoundaryConditions,
    DomainConditions)


logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from dream.solver import SolverConfiguration


# --- Conservative --- #

class MixedMethod(MultipleConfiguration, is_interface=True):

    @property
    def Q(self) -> ngs.CF:
        return self.cfg.pde.formulation.TnT['Q'][0]

    @property
    def P(self) -> ngs.CF:
        return self.cfg.pde.formulation.TnT['Q'][1]

    def set_configuration_and_mesh(self, cfg: SolverConfiguration, dmesh: DreamMesh):
        self.cfg = cfg
        self.dmesh = dmesh

    def get_mixed_finite_element_spaces(self) -> dict[str, ngs.ProductSpace]:
        ...


class Inactive(MixedMethod):

    name: str = "inactive"

    def get_mixed_finite_element_spaces(self) -> dict:

        if not self.cfg.pde.dynamic_viscosity.is_inviscid:
            raise TypeError(f"Viscous configuration requires mixed method!")

        return {}


class StrainHeat(MixedMethod):

    name: str = "strain_heat"

    def get_mixed_finite_element_spaces(self) -> dict[str, ngs.ProductSpace]:

        if self.cfg.pde.dynamic_viscosity.is_inviscid:
            raise TypeError(f"Inviscid configuration does not require mixed method!")

        dim = 4*dim - 3
        order = self.cfg.fem.order
        mesh = self.dmesh.ngsmesh

        Q = ngs.L2(mesh, order=order)
        Q = self.dmesh._reduce_psponge_layers_order_elementwise(Q)

        return {'Q': Q**dim}

    def get_mixed_state(self, Q: ngs.CoefficientFunction):

        dim = self.dmesh.dim

        if isinstance(Q, ngs.GridFunction):
            Q = Q.components

        Q_ = CompressibleStateGradient()
        Q_.eps = bla.symmetric_matrix_from_vector(Q[:3*dim - 3])
        Q_.T = Q[3*dim - 3:]

        return Q_


class Gradient(MixedMethod):

    name: str = "gradient"

    def get_mixed_finite_element_spaces(self) -> dict[str, ngs.ProductSpace]:

        if self.cfg.pde.dynamic_viscosity.is_inviscid:
            raise TypeError(f"Inviscid configuration does not require mixed method!")

        dim = self.dmesh.dim + 2
        order = self.cfg.fem.order
        mesh = self.dmesh.ngsmesh

        Q = ngs.VectorL2(mesh, order=order)
        Q = self.dmesh._reduce_psponge_layers_order_elementwise(Q)

        return {'Q': Q**dim}

    def get_mixed_state(self, Q: ngs.CoefficientFunction):

        if isinstance(Q, ngs.GridFunction):
            Q = Q.components

        Q_ = CompressibleStateGradient()
        Q_.rho = Q[0]
        Q_.rho_u = Q[slice(1, self.dmesh.dim + 1)]
        Q_.rho_E = Q[self.dmesh.dim + 1]

        return Q_


class ConservativeMethod(MultipleConfiguration, is_interface=True):

    @property
    def U_TnT(self) -> tuple[ngs.comp.ProxyFunction, ...]:
        return self.cfg.pde.formulation.TnT['U']

    @property
    def U_gfu(self) -> ngs.GridFunction:
        return self.cfg.pde.formulation.gfus['U']

    @property
    def U_gfu_transient(self) -> dict[str, ngs.GridFunction]:
        return self.cfg.pde.formulation.gfus_transient['U']

    def set_configuration_and_mesh(self, cfg: SolverConfiguration, dmesh: DreamMesh):
        self.cfg = cfg
        self.dmesh = dmesh

    def get_transient_gridfunctions(self):
        return {'U': self.cfg.time.scheme.allocate_transient_gridfunctions(self.U_gfu)}

    def add_mass(self, blf: list[ngs.comp.SumOfIntegrals], lf: list[ngs.comp.SumOfIntegrals]):
        U, V = self.U_TnT
        blf += bla.inner(U, V) * ngs.dx

    def add_time_derivative(self, blf: list[ngs.comp.SumOfIntegrals], lf: list[ngs.comp.SumOfIntegrals]):

        U, V = self.U_TnT
        dt = self.cfg.time.timer.step

        U_dt = self.U_gfu_transient.copy()
        U_dt['n+1'] = U

        U_dt = self.cfg.time.scheme.get_discrete_time_derivative(self.U_gfu, dt)

        blf += bla.inner(U_dt, V) * ngs.dx

    def get_conservative_state(self, U: ngs.CoefficientFunction):

        if isinstance(U, ngs.GridFunction):
            U = U.components

        U_ = CompressibleState()
        U_.rho = U[0]
        U_.rho_u = U[slice(1, self.dmesh.dim + 1)]
        U_.rho_E = U[self.dmesh.dim + 1]

        U_.u = self.cfg.pde.equations.velocity(U_)
        U_.Ek = self.cfg.pde.equations.specific_kinetic_energy(U_)
        U_.Ei = self.cfg.pde.equations.specific_inner_energy(U_)
        U_.rho_Ek = self.cfg.pde.equations.kinetic_energy(U_)
        U_.rho_Ei = self.cfg.pde.equations.inner_energy(U_)

        U_.p = self.cfg.pde.equations.pressure(U_)
        U_.T = self.cfg.pde.equations.temperature(U_)

        U_.H = self.cfg.pde.equations.specific_enthalpy(U_)
        U_.rho_H = self.cfg.pde.equations.enthalpy(U_)

        return U_


class HDG(ConservativeMethod):

    name: str = "hdg"

    @property
    def Uhat_TnT(self) -> tuple[ngs.comp.ProxyFunction, ...]:
        return self.cfg.pde.formulation.TnT['Uhat']

    @property
    def Uhat_gfu(self) -> ngs.GridFunction:
        return self.cfg.pde.formulation.gfus['Uhat']

    @property
    def Uhat_gfu_transient(self) -> dict[str, ngs.GridFunction]:
        return self.cfg.pde.formulation.gfus_transient['Uhat']

    def get_finite_element_spaces(self):
        mesh = self.dmesh.ngsmesh
        dim = self.dmesh.dim + 2

        U = ngs.L2(mesh, self.cfg.fem.order)
        U = self.dmesh._reduce_psponge_layers_order_elementwise(U)

        Uhat = ngs.FacetFESpace(mesh, self.cfg.fem.order)
        Uhat = self.dmesh._reduce_psponge_layers_order_facetwise(Uhat)

        return {'U': U**dim, 'Uhat': Uhat**dim}

    def get_transient_gridfunctions(self):
        gfus = super().get_transient_gridfunctions()

        if self.dmesh.bcs.get(CharacteristicRelaxationInflow, CharacteristicRelaxationOutflow):
            gfus['Uhat'] = self.cfg.time.scheme.allocate_transient_gridfunctions(self.Uhat_gfu)

        return gfus

    def get_convective_numerical_flux(self, U, Uhat, unit_vector: bla.VECTOR):
        """
        Convective numerical flux

        Equation 34, page 16

        Literature:
        [1] - Vila-Pérez, J., Giacomini, M., Sevilla, R. et al.
              Hybridisable Discontinuous Galerkin Formulation of Compressible Flows.
              Arch Computat Methods Eng 28, 753–784 (2021).
              https://doi.org/10.1007/s11831-020-09508-z
        """
        unit_vector = bla.as_vector(unit_vector)
        eq = self.cfg.pde.equations

        Uhat_ = self.get_conservative_state(Uhat)
        tau = self.cfg.pde.riemann_solver.convective_stabilisation_matrix(Uhat_, unit_vector)

        return eq.convective_flux(Uhat_) * unit_vector + tau * (U - Uhat)

    def get_domain_boundary_mask(self):
        """ 
        Unsets the correct degrees of freedom on the domain boundaries
        """

        fes = ngs.FacetFESpace(self.dmesh.ngsmesh, order=0)
        mask = ngs.GridFunction(fes, name="mask")
        mask.vec[:] = 0
        mask.vec[~fes.GetDofs(self.dmesh.bcs.get_domain_boundaries(True))] = 1

        return mask

    def add_convection(self, blf: list[ngs.comp.SumOfIntegrals], lf: list[ngs.comp.SumOfIntegrals]):

        eq = self.cfg.pde.equations
        normal = self.dmesh.normal
        vol = self.cfg.fem.bonus_int_order.vol
        bnd = self.cfg.fem.bonus_int_order.bnd
        mask = self.get_domain_boundary_mask()

        U, V = self.U_TnT
        Uhat, Vhat = self.Uhat_TnT

        U_ = self.get_conservative_state(U)

        F = eq.convective_flux(U_)
        Fn = self.get_convective_numerical_flux(U, Uhat, normal)

        blf += -bla.inner(F, ngs.grad(V)) * ngs.dx(bonus_intorder=vol)
        blf += bla.inner(Fn, V) * ngs.dx(element_boundary=True, bonus_intorder=bnd)
        blf += -mask * bla.inner(Fn, Vhat) * ngs.dx(element_boundary=True, bonus_intorder=bnd)

    # def add_domain_conditions(self, blf: list[ngs.comp.SumOfIntegrals], lf: list[ngs.comp.SumOfIntegrals]):

    #     for dom, bc in self.dmesh.dcs.as_pattern().items():

    #         match bc:

    #             case


class ConservativeFormulation(CompressibleFormulation):

    name: str = "conservative"

    @multiple(default=HDG)
    def method(self, method):
        return method

    @multiple(default=Inactive)
    def mixed_method(self, mixed_method):
        return mixed_method

    def set_finite_element_spaces(self, spaces: dict[str, ngs.FESpace] | None = None) -> None:

        if spaces is None:
            spaces = {}

        self.method.set_configuration_and_mesh(self.cfg, self.dmesh)
        self.mixed_method.set_configuration_and_mesh(self.cfg, self.dmesh)

        spaces.update(self.method.get_finite_element_spaces())
        spaces.update(self.mixed_method.get_mixed_finite_element_spaces())

        super().set_finite_element_spaces(spaces)

    def set_transient_gridfunctions(self, gfus: dict[str, dict[str, ngs.GridFunction]] | None = None) -> None:

        if gfus is None:
            gfus = {}

        gfus.update(self.method.get_transient_gridfunctions())

        super().set_transient_gridfunctions(gfus)

    method: HDG
    mixed_method: Inactive
