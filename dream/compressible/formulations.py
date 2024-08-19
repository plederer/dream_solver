from __future__ import annotations
import logging
import ngsolve as ngs

from dream.config import InterfaceConfiguration
from dream.mesh import DreamMesh
from dream.formulation import Space, Formulation

logger = logging.getLogger(__name__)

# ------- Formulations ------- #

class CompressibleFormulation(Formulation, is_interface=True):
    ...

# --- Conservative --- #

class MixedMethod(InterfaceConfiguration, is_interface=True):
    ...


class Inactive(MixedMethod):

    name: str = "inactive"

    def get_mixed_space(self, cfg: SolverConfiguration, dmesh: DreamMesh) -> None:
        if cfg.pde.dynamic_viscosity.is_inviscid:
            raise TypeError(f"Viscous configuration requires mixed method!")
        return None


class StrainHeat(MixedMethod):

    name: str = "strain_heat"

    def get_mixed_space(self, cfg: SolverConfiguration, dmesh: DreamMesh) -> StrainHeatSpace:
        if cfg.pde.dynamic_viscosity.is_inviscid:
            raise TypeError(f"Inviscid configuration does not require mixed method!")

        return StrainHeatSpace(cfg, dmesh)


class Gradient(MixedMethod):

    name: str = "gradient"

    def get_mixed_space(self, cfg: SolverConfiguration, dmesh: DreamMesh) -> GradientSpace:

        if cfg.pde.dynamic_viscosity.is_inviscid:
            raise TypeError(f"Inviscid configuration does not require mixed method!")

        return GradientSpace(cfg, dmesh)

class StrainHeatSpace(Space):

    def get_space(self) -> ngs.FESpace:
        dim = 4 * self.dmesh.dim - 3
        order = self.cfg.fem.order
        mesh = self.dmesh.ngsmesh

        Q = ngs.L2(mesh, order=order)
        Q = self.dmesh._reduce_psponge_layers_order_elementwise(Q)

        return Q**dim


class GradientSpace(Space):

    def get_space(self) -> ngs.FESpace:
        dim = 4 * self.dmesh.dim - 3
        order = self.cfg.fem.order
        mesh = self.dmesh.ngsmesh

        Q = ngs.L2(mesh, order=order)
        Q = self.dmesh._reduce_psponge_layers_order_elementwise(Q)

        return Q**dim


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


class StrainHeatSpace(Space):

    def get_space(self) -> ngs.FESpace:
        dim = 4 * self.dmesh.dim - 3
        order = self.cfg.fem.order
        mesh = self.dmesh.ngsmesh

        Q = ngs.L2(mesh, order=order)
        Q = self.dmesh._reduce_psponge_layers_order_elementwise(Q)

        return Q**dim


class GradientSpace(Space):

    def get_space(self) -> ngs.FESpace:
        dim = 4 * self.dmesh.dim - 3
        order = self.cfg.fem.order
        mesh = self.dmesh.ngsmesh

        Q = ngs.L2(mesh, order=order)
        Q = self.dmesh._reduce_psponge_layers_order_elementwise(Q)

        return Q**dim


class Conservative(CompressibleFormulation):

    label: str = "conservative"

    @property
    def U(self) -> PrimalElement:
        return self.spaces["U"]

    @property
    def Uhat(self) -> PrimalFacet:
        return self.spaces["Uhat"]

    @property
    def Q(self) -> StrainHeat | GradientSpace:
        return self.spaces["Q"]

    def get_space(self) -> form.Spaces:
        mixed_method = self.cfg.pde.mixed_method

        spaces = form.Spaces()
        spaces["U"] = PrimalElement(self.cfg, self.dmesh)
        spaces["Uhat"] = PrimalFacet(self.cfg, self.dmesh)
        spaces["Q"] = mixed_method.get_mixed_space(self.cfg, self.dmesh)

        return spaces

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


