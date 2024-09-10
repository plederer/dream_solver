from __future__ import annotations
import logging
import ngsolve as ngs
import typing

import dream.bla as bla

from dream.config import DescriptorConfiguration, descriptor_configuration
from dream.mesh import DreamMesh
from dream.formulation import Space, CompressibleFormulation
from dream.compressible.state import CompressibleState, CompressibleStateGradient


logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from dream.solver import SolverConfiguration


# --- Conservative --- #

class MixedMethod(DescriptorConfiguration, is_interface=True):

    def initialize(self, cfg: SolverConfiguration, dmesh: DreamMesh):
        self.cfg = cfg
        self.dmesh = dmesh

    def set_test_and_trial_function(self, TnT: dict[str, tuple[ngs.CF, ...]]):
        ...

    def get_mixed_spaces(self) -> dict[str, ngs.ProductSpace]:
        ...


class Inactive(MixedMethod):

    name: str = "inactive"

    def get_mixed_spaces(self) -> dict:

        if not self.cfg.pde.dynamic_viscosity.is_inviscid:
            raise TypeError(f"Viscous configuration requires mixed method!")

        return {}


class StrainHeat(MixedMethod):

    name: str = "strain_heat"

    def get_mixed_spaces(self) -> dict[str, ngs.ProductSpace]:

        if self.cfg.pde.dynamic_viscosity.is_inviscid:
            raise TypeError(f"Inviscid configuration does not require mixed method!")

        dim = 4*dim - 3
        order = self.cfg.fem.order
        mesh = self.dmesh.ngsmesh

        Q = ngs.L2(mesh, order=order)
        Q = self.dmesh._reduce_psponge_layers_order_elementwise(Q)

        return {'Q': Q**dim}

    def set_test_and_trial_function(self, TnT: dict[str, tuple[ngs.CF, ...]]):
        self.Q, self.P = TnT.pop('Q')

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

    def get_mixed_spaces(self) -> dict[str, ngs.ProductSpace]:

        if self.cfg.pde.dynamic_viscosity.is_inviscid:
            raise TypeError(f"Inviscid configuration does not require mixed method!")

        dim = self.dmesh.dim + 2
        order = self.cfg.fem.order
        mesh = self.dmesh.ngsmesh

        Q = ngs.VectorL2(mesh, order=order)
        Q = self.dmesh._reduce_psponge_layers_order_elementwise(Q)

        return {'Q': Q**dim}

    def set_test_and_trial_function(self, TnT: dict[str, tuple[ngs.CF, ...]]):
        self.Q, self.P = TnT.pop('Q')

    def get_mixed_state(self, Q: ngs.CoefficientFunction):

        if isinstance(Q, ngs.GridFunction):
            Q = Q.components

        Q_ = CompressibleStateGradient()
        Q_.rho = Q[0]
        Q_.rho_u = Q[slice(1, self.dmesh.dim + 1)]
        Q_.rho_E = Q[self.dmesh.dim + 1]

        return Q_


class Primal(Space):

    def get_state_from_variable(self, gfu: ngs.GridFunction = None) -> CompressibleState:
        if gfu is None:
            gfu = self.gfu

        state = CompressibleState()
        state.density = gfu[0]
        state.momentum = gfu[slice(1, self.dmesh.dim + 1)]
        state.energy = gfu[self.dmesh.dim + 1]

        return state

    def get_variable_from_state(self, state: State) -> ngs.CF:
        state = CompressibleState(**state)
        eq = self.cfg.pde.equations

        density = eq.density(state)
        momentum = eq.momentum(state)
        energy = eq.energy(state)
        return ngs.CF((density, momentum, energy))


class PrimalElement(Primal):

    def get_space(self) -> ngs.L2:
        dim = self.dmesh.dim + 2
        order = self.cfg.fem.order
        mesh = self.dmesh.ngsmesh

        V = ngs.L2(mesh, order=order)
        V = self.dmesh._reduce_psponge_layers_order_elementwise(V)

        return V**dim

    def set_configuration_flags(self):

        self.has_time_derivative = False
        if not self.cfg.simulation.is_stationary:
            self.has_time_derivative = True

    def get_transient_gridfunction(self) -> TransientGridfunction:
        if not self.cfg.simulation.is_stationary:
            return self.cfg.simulation.scheme.get_transient_gridfunction(self.gfu)


class PrimalFacet(Primal):

    @property
    def mask(self) -> ngs.GridFunction:
        """ Mask is a indicator Gridfunction, which vanishes on the domain boundaries.

            This is required to implement different boundary conditions on the the domain boundaries,
            while using a Riemann-Solver in the interior!
        """

        return getattr(self, "_mask", None)

    def get_space(self) -> ngs.FacetFESpace:
        dim = self.dmesh.dim + 2
        order = self.cfg.fem.order
        mesh = self.dmesh.ngsmesh

        VHAT = ngs.FacetFESpace(mesh, order=order)
        VHAT = self.dmesh._reduce_psponge_layers_order_facetwise(VHAT)

        if self.dmesh.is_periodic:
            VHAT = ngs.Periodic(VHAT)

        return VHAT**dim

    def get_transient_gridfunction(self) -> TransientGridfunction:
        if not self.cfg.simulation.is_stationary and self.dmesh.bcs.get(NSCBC):
            return self.cfg.simulation.scheme.get_transient_gridfunction(self.gfu)

    def add_mass_bilinearform(self, blf: ngs.BilinearForm, dx=ngs.dx, **dx_kwargs):
        return super().add_mass_bilinearform(blf, dx=ngs.dx, element_boundary=True, **dx_kwargs)

    def add_mass_linearform(self, state: CompressibleState, lf: ngs.LinearForm, dx=ngs.dx, **dx_kwargs):
        return super().add_mass_linearform(state, lf, dx=ngs.dx, element_boundary=True, **dx_kwargs)

    def set_mask(self):
        """ Unsets the correct degrees of freedom on the domain boundaries

        """

        mask = self.mask
        if mask is None:
            fes = ngs.FacetFESpace(self.dmesh.ngsmesh, order=0)
            mask = ngs.GridFunction(fes, name="mask")
            self._mask = mask

        mask.vec[:] = 0
        mask.vec[~fes.GetDofs(self.dmesh.bcs.get_domain_boundaries(True))] = 1


class ConservativeMethod(DescriptorConfiguration, is_interface=True):

    def initialize(self, cfg: SolverConfiguration, dmesh: DreamMesh):
        self.cfg = cfg
        self.dmesh = dmesh

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

    def get_spaces(self):
        ...


class HDG(ConservativeMethod):

    name: str = "hdg"

    def get_spaces(self):

        mesh = self.dmesh.ngsmesh
        dim = self.dmesh.dim + 2

        U = ngs.L2(mesh, self.cfg.fem.order)
        U = self.dmesh._reduce_psponge_layers_order_elementwise(U)

        Uhat = ngs.FacetFESpace(mesh, self.cfg.fem.order)
        Uhat = self.dmesh._reduce_psponge_layers_order_facetwise(Uhat)

        return {'U': U**dim, 'Uhat': Uhat**dim}

    def set_test_and_trial_function(self, TnT: dict[str, tuple[ngs.CF, ...]]):
        self.U, self.V = TnT.pop('U')
        self.Uhat, self.Vhat = TnT.pop('Uhat')

    def add_convection_bilinearform(self, blf: list[ngs.comp.SumOfIntegrals]):

        eq = self.cfg.pde.equations
        n = self.dmesh.normal
        bonus_vol = self.cfg.fem.bonus_int_order.vol
        bonus_bnd = self.cfg.fem.bonus_int_order.bnd

        mask = self.get_domain_boundary_mask()

        U = self.get_conservative_state(self.U)
        Uhat = self.get_conservative_state(self.Uhat)

        tau = self.cfg.pde.riemann_solver.convective_stabilisation_matrix(Uhat, n)

        F = eq.convective_flux(U)
        Fn = eq.convective_flux(Uhat) + tau * (self.U - self.Uhat)

        blf += -bla.inner(F, ngs.grad(self.V)) * ngs.dx(bonus_intorder=bonus_vol)
        blf += bla.inner(Fn, self.V) * ngs.dx(element_boundary=True, bonus_intorder=bonus_bnd)
        blf += -mask * bla.inner(Fn, self.Vhat) * ngs.dx(element_boundary=True, bonus_intorder=bonus_bnd)

    def get_domain_boundary_mask(self):
        """ 
        Unsets the correct degrees of freedom on the domain boundaries
        """

        fes = ngs.FacetFESpace(self.dmesh.ngsmesh, order=0)
        mask = ngs.GridFunction(fes, name="mask")
        mask.vec[:] = 0
        mask.vec[~fes.GetDofs(self.dmesh.bcs.get_domain_boundaries(True))] = 1

        return mask


class ConservativeFormulation(CompressibleFormulation):

    name: str = "conservative"

    @descriptor_configuration(default=HDG)
    def method(self, method):
        return method

    @descriptor_configuration(default=Inactive)
    def mixed_method(self, mixed_method):
        return mixed_method

    def initialize_finite_element_space(self, spaces: dict[str, ngs.FESpace] = None) -> None:

        if spaces is None:
            spaces = {}

        self.method.initialize(self.cfg, self.dmesh)
        self.mixed_method.initialize(self.cfg, self.dmesh)

        spaces.update(self.method.get_spaces(self.cfg, self.dmesh))
        spaces.update(self.mixed_method.get_mixed_spaces(self.cfg, self.dmesh))

        super().initialize_finite_element_space(spaces)

    def initialize_test_and_trial_function(self) -> None:
        super().initialize_test_and_trial_function()

        TnT = self.TnT.copy()

        self.method.set_test_and_trial_function(TnT)
        self.mixed_method.set_test_and_trial_function(TnT)

    def pre_assemble(self):

        self.U.get_trial_as_state()

        self.Uhat.set_mask()
        self.Uhat.get_trial_as_state()

    def get_system(self, blf: list[ngs.comp.SumOfIntegrals], lf: list[ngs.comp.SumOfIntegrals]):
        self.pre_assemble()

        if self.U.dt:
            self.add_time_derivative(blf, lf)

        self.add_convection(blf, lf)

    def add_time_derivative(self, blf: list[ngs.comp.SumOfIntegrals], lf: list[ngs.comp.SumOfIntegrals]):
        scheme = self.cfg.simulation.scheme
        U = self.U

        dt = U.dt.swap_level(U.trial)

        blf += bla.inner(scheme.scheme(dt), U.test) * ngs.dx

    def add_convection(self, blf: list[ngs.comp.SumOfIntegrals], lf: list[ngs.comp.SumOfIntegrals]):

        eq = self.cfg.pde.equations
        bonus_vol = self.cfg.fem.bonus_int_order.vol
        bonus_bnd = self.cfg.fem.bonus_int_order.bnd

        U, U_ = self.U, self.U.trial_state
        Uhat, Uhat_ = self.Uhat, self.Uhat.trial_state

        F = eq.convective_flux(U_)
        Fhat = self.convective_numerical_flux(self.normal)

        blf += -bla.inner(F, ngs.grad(U.test)) * ngs.dx(bonus_intorder=bonus_vol)
        blf += bla.inner(Fhat, U.test) * ngs.dx(element_boundary=True, bonus_intorder=bonus_bnd)
        blf += -Uhat.mask * bla.inner(Fhat, Uhat.test) * ngs.dx(element_boundary=True, bonus_intorder=bonus_bnd)

    def convective_numerical_flux(self, unit_vector: bla.VECTOR):
        eq = self.cfg.pde.equations
        U, Uhat = self.U, self.Uhat

        unit_vector = bla.as_vector(unit_vector)
        tau_c = self.cfg.pde.riemann_solver.convective_stabilisation_matrix(Uhat.trial_state, unit_vector)

        F = eq.convective_flux(Uhat.trial_state)

        return F * unit_vector + tau_c * (U.trial - Uhat.trial)

    method: HDG
    mixed_method: Inactive
