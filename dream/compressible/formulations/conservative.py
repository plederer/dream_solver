from __future__ import annotations
import logging
import ngsolve as ngs
import typing

import dream.bla as bla

from dream.config import MultipleConfiguration, multiple

from dream.compressible.config import (
    CompressibleState, CompressibleStateGradient, CompressibleFiniteElement, FarField, Outflow, InviscidWall, Symmetry,
    IsothermalWall, AdiabaticWall)
from dream.mesh import SpongeLayer, PSpongeLayer, Periodic, Initial

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from dream.solver import SolverConfiguration


# --- Conservative --- #

class MixedMethod(MultipleConfiguration, is_interface=True):

    cfg: SolverConfiguration

    @property
    def mesh(self) -> ngs.Mesh:
        return self.cfg.mesh

    @property
    def equations(self):
        return self.cfg.pde.equations

    @property
    def Q_TnT(self) -> ngs.CF:
        return self.cfg.pde.TnT['Q']

    @property
    def Q_gfu(self) -> ngs.GridFunction:
        return self.cfg.pde.gfus['Q']

    def add_mixed_finite_element_spaces(self, fes: dict[str, ngs.FESpace]) -> None:
        raise NotImplementedError("Mixed method must implement get_mixed_finite_element_spaces method!")

    def add_mixed_formulation(self, blf: dict[str, ngs.comp.SumOfIntegrals],
                              lf: dict[str, ngs.comp.SumOfIntegrals]) -> None:
        pass

    def add_initial_conditions(self, blf: dict[str, ngs.comp.SumOfIntegrals], lf: dict[str, ngs.comp.SumOfIntegrals]):
        pass

    def add_transient_gridfunctions(self, gfus: dict[str, ngs.GridFunction]) -> None:
        # gfus['Q'] = self.cfg.time.scheme.allocate_transient_gridfunctions(self.Q_gfu)
        pass

    def get_diffusive_stabilisation_matrix(self, U: CompressibleState) -> bla.MATRIX:
        Re = self.cfg.pde.reference_reynolds_number
        Pr = self.cfg.pde.prandtl_number
        mu = self.cfg.pde.equations.viscosity(U)

        if self.mesh.dim == 2:

            tau_d = ngs.CF((0, 0, 0, 0,
                            0, 1, 0, 0,
                            0, 0, 1, 0,
                            0, 0, 0, 1/Pr), dims=(4, 4))

        elif self.mesh.dim == 3:

            tau_d = ngs.CF((0, 0, 0, 0, 0,
                            0, 1, 0, 0, 0,
                            0, 0, 1, 0, 0,
                            0, 0, 0, 1, 0,
                            0, 0, 0, 0, 1/Pr), dims=(5, 5))

        tau_d *= mu / Re

        return tau_d


class Inactive(MixedMethod):

    name: str = "inactive"

    def add_mixed_finite_element_spaces(self, fes: dict[str, ngs.FESpace]) -> None:

        if not self.cfg.pde.dynamic_viscosity.is_inviscid:
            raise TypeError(f"Viscous configuration requires mixed method!")


class StrainHeat(MixedMethod):

    name: str = "strain_heat"

    def add_mixed_finite_element_spaces(self, fes: dict[str, ngs.FESpace]) -> None:

        if self.cfg.pde.dynamic_viscosity.is_inviscid:
            raise TypeError(f"Inviscid configuration does not require mixed method!")

        dim = 4*dim - 3
        order = self.cfg.pde.fe.order

        Q = ngs.L2(self.mesh, order=order)
        Q = self.cfg.pde.dcs.reduce_psponge_layers_order_elementwise(Q)

        fes['Q'] = Q**dim

    def add_mixed_formulation(self, blf: dict[str, ngs.comp.SumOfIntegrals],
                              lf: dict[str, ngs.comp.SumOfIntegrals]) -> None:

        if self.mesh.dim == 3:
            raise NotImplementedError("StrainHeat method is not implemented for domain dimension 3!")

        bonus = self.cfg.optimizations.bonus_int_order

        U, _ = self.cfg.pde.fe.method.U_TnT
        Uhat, _ = self.cfg.pde.fe.method.Uhat_TnT
        Q, P = self.Q_TnT

        U = self.cfg.pde.fe.method.get_conservative_state(U)
        U.u = self.cfg.pde.equations.velocity(U)
        U.T = self.cfg.pde.equations.temperature(U)

        Uhat = self.cfg.pde.fe.method.get_conservative_state(Uhat)
        Uhat.u = self.cfg.pde.equations.velocity(Uhat)
        Uhat.T = self.cfg.pde.equations.temperature(Uhat)

        gradient_P = ngs.grad(P)
        Q = self.get_mixed_state(Q)
        P = self.get_mixed_state(P)

        dev_zeta = P.strain - bla.trace(P.strain) * ngs.Id(self.mesh.dim)/3
        div_dev_zeta = ngs.CF((gradient_P[0, 0] + gradient_P[1, 1], gradient_P[1, 0] + gradient_P[2, 1]))
        div_dev_zeta -= 1/3 * ngs.CF((gradient_P[0, 0] + gradient_P[2, 0], gradient_P[0, 1] + gradient_P[2, 1]))
        blf['mixed'] = ngs.InnerProduct(Q.strain, P.strain) * ngs.dx
        blf['mixed'] += ngs.InnerProduct(U.u, div_dev_zeta) * ngs.dx(bonus_intorder=bonus.vol)
        blf['mixed'] -= ngs.InnerProduct(Uhat.u,
                                         dev_zeta*self.mesh.normal) * ngs.dx(element_boundary=True, bonus_intorder=bonus.bnd)

        div_xi = gradient_P[3, 0] + gradient_P[4, 1]
        blf['mixed'] += ngs.InnerProduct(Q.grad_T, P.grad_T) * ngs.dx
        blf['mixed'] += ngs.InnerProduct(U.T, div_xi) * ngs.dx(bonus_intorder=bonus.vol)
        blf['mixed'] -= ngs.InnerProduct(Uhat.T*self.mesh.normal,
                                         P.grad_T) * ngs.dx(element_boundary=True, bonus_intorder=bonus.bnd)

    def add_initial_conditions(self, blf: dict[str, ngs.comp.SumOfIntegrals], lf: dict[str, ngs.comp.SumOfIntegrals]):

        Q, P = self.Q_TnT
        blf['Q'] = bla.inner(Q, P) * ngs.dx

        for dom, dc in self.cfg.pde.dcs.to_pattern(Initial).items():

            strain = self.equations.strain_rate_tensor(dc.state)
            Q_init = ngs.CF((strain[0, 0], strain[0, 1], strain[1, 1],
                            self.equations.temperature_gradient(dc.state, dc.state)))
            lf[f"{dc.name}_{dom}"] = Q_init * P * ngs.dx(definedon=self.mesh.Materials(dom))

    def get_mixed_state(self, Q: ngs.CoefficientFunction):

        dim = self.mesh.dim

        if isinstance(Q, ngs.GridFunction):
            Q = Q.components

        Q_ = CompressibleStateGradient()
        Q_.strain = bla.symmetric_matrix_from_vector(Q[:3*dim - 3])
        Q_.grad_T = Q[3*dim - 3:]

        if isinstance(Q, ngs.comp.ProxyFunction):
            Q_.Q = Q

        return Q_


class Gradient(MixedMethod):

    name: str = "gradient"

    def add_mixed_finite_element_spaces(self, fes: dict[str, ngs.FESpace]) -> None:

        if self.cfg.pde.dynamic_viscosity.is_inviscid:
            raise TypeError(f"Inviscid configuration does not require mixed method!")

        dim = self.mesh.dim + 2
        order = self.cfg.pde.fe.order

        Q = ngs.VectorL2(self.mesh, order=order)
        Q = self.cfg.pde.dcs.reduce_psponge_layers_order_elementwise(Q)

        fes['Q'] = Q**dim

    def add_mixed_formulation(self, blf: dict[str, ngs.comp.SumOfIntegrals],
                              lf: dict[str, ngs.comp.SumOfIntegrals]) -> None:

        Q, P = self.Q_TnT
        U, _ = self.cfg.pde.fe.method.U_TnT
        Uhat, _ = self.cfg.pde.fe.method.Uhat_TnT

        blf['mixed'] = ngs.InnerProduct(Q, P) * ngs.dx
        blf['mixed'] += ngs.InnerProduct(U, ngs.div(P)) * ngs.dx
        blf['mixed'] -= ngs.InnerProduct(Uhat, P*self.mesh.normal) * ngs.dx(element_boundary=True)

    def add_initial_conditions(self, blf: dict[str, ngs.comp.SumOfIntegrals], lf: dict[str, ngs.comp.SumOfIntegrals]):
        raise NotImplementedError("Initial Conditions not implemented!")

    def get_mixed_state(self, Q: ngs.CoefficientFunction):

        if isinstance(Q, ngs.GridFunction):
            Q = Q.components

        Q_ = CompressibleStateGradient()
        Q_.grad_rho = Q[0]
        Q_.grad_rho_u = Q[slice(1, self.mesh.dim + 1)]
        Q_.grad_rho_E = Q[self.mesh.dim + 1]

        if isinstance(Q, ngs.comp.ProxyFunction):
            Q_.Q = Q

        return Q_


class ConservativeMethod(MultipleConfiguration, is_interface=True):

    cfg: SolverConfiguration

    @property
    def mesh(self) -> ngs.Mesh:
        return self.cfg.mesh

    @property
    def equations(self):
        return self.cfg.pde.equations

    @property
    def U_TnT(self) -> tuple[ngs.comp.ProxyFunction, ...]:
        return self.cfg.pde.TnT['U']

    @property
    def U_gfu(self) -> ngs.GridFunction:
        return self.cfg.pde.gfus['U']

    @property
    def U_gfu_transient(self) -> dict[str, ngs.GridFunction]:
        return self.cfg.pde.gfus_transient['U']

    def add_transient_gridfunctions(self, gfus: dict[str, dict[str, ngs.GridFunction]]) -> None:
        gfus['U'] = self.cfg.time.scheme.allocate_transient_gridfunctions(self.U_gfu)

    def add_initial_conditions(self, blf: dict[str, ngs.comp.SumOfIntegrals], lf: dict[str, ngs.comp.SumOfIntegrals]):

        U, V = self.U_TnT
        blf['U'] = bla.inner(U, V) * ngs.dx

        for dom, dc in self.cfg.pde.dcs.to_pattern(Initial).items():

            U_init = ngs.CF(
                (self.equations.density(dc.state),
                 self.equations.momentum(dc.state),
                 self.equations.energy(dc.state)))

            lf[f"{dc.name}_{dom}"] = U_init * V * ngs.dx(definedon=self.mesh.Materials(dom))

    def add_time_derivative_formulation(
            self, blf: dict[str, ngs.comp.SumOfIntegrals],
            lf: dict[str, ngs.comp.SumOfIntegrals]):

        U, V = self.U_TnT
        dt = self.cfg.time.timer.step

        U_dt = self.U_gfu_transient.copy()
        U_dt['n+1'] = U

        U_dt = self.cfg.time.scheme.get_discrete_time_derivative(self.U_gfu, dt)

        blf['time'] = bla.inner(U_dt, V) * ngs.dx

    def get_conservative_state(self, U: ngs.CoefficientFunction) -> CompressibleState:

        # equations = self.cfg.pde.equations

        if isinstance(U, ngs.GridFunction):
            U = U.components

        U_ = CompressibleState()
        U_.rho = U[0]
        U_.rho_u = U[slice(1, self.mesh.dim + 1)]
        U_.rho_E = U[self.mesh.dim + 1]

        if isinstance(U, ngs.comp.ProxyFunction):
            U_.U = U

        return U_


class HDG(ConservativeMethod):

    name: str = "hdg"

    @property
    def mixed_method(self) -> MixedMethod:
        return self.cfg.pde.fe.mixed_method

    @property
    def Uhat_TnT(self) -> tuple[ngs.comp.ProxyFunction, ...]:
        return self.cfg.pde.TnT['Uhat']

    @property
    def Uhat_gfu(self) -> ngs.GridFunction:
        return self.cfg.pde.gfus['Uhat']

    @property
    def Uhat_gfu_transient(self) -> dict[str, ngs.GridFunction]:
        return self.cfg.pde.gfus_transient['Uhat']

    def add_finite_element_spaces(self, fes: dict[str, ngs.FESpace]) -> None:

        order = self.cfg.pde.fe.order

        dim = self.mesh.dim + 2

        U = ngs.L2(self.mesh, order)
        Uhat = ngs.FacetFESpace(self.mesh, order)

        psponge_layers = self.cfg.pde.dcs.to_pattern(PSpongeLayer)
        if psponge_layers:
            U = self.cfg.pde.dcs.reduce_psponge_layers_order_elementwise(U, psponge_layers)
            Uhat = self.cfg.pde.dcs.reduce_psponge_layers_order_facetwise(Uhat, psponge_layers)

        fes['U'] = U**dim
        fes['Uhat'] = Uhat**dim

        self.mixed_method.add_mixed_finite_element_spaces(fes)

    def add_transient_gridfunctions(self, gfus: dict[str, dict[str, ngs.GridFunction]]) -> None:
        super().add_transient_gridfunctions(gfus)
        gfus['Uhat'] = self.cfg.time.scheme.allocate_transient_gridfunctions(self.Uhat_gfu)

    def add_initial_conditions(self, blf: dict[str, ngs.comp.SumOfIntegrals], lf: dict[str, ngs.comp.SumOfIntegrals]):
        super().add_initial_conditions(blf, lf)

        Uhat, Vhat = self.Uhat_TnT
        blf['Uhat'] = bla.inner(Uhat, Vhat) * ngs.dx(element_boundary=True)

        for dom, dc in self.cfg.pde.dcs.to_pattern(Initial).items():

            U_init = ngs.CF(
                (self.equations.density(dc.state),
                 self.equations.momentum(dc.state),
                 self.equations.energy(dc.state)))

            lf[f"{dc.name}_{dom}"] = U_init * Vhat * ngs.dx(element_boundary=True, definedon=self.mesh.Materials(dom))

    def add_convection_formulation(
            self, blf: dict[str, ngs.comp.SumOfIntegrals],
            lf: dict[str, ngs.comp.SumOfIntegrals]):

        bonus = self.cfg.optimizations.bonus_int_order

        mask = self.get_domain_boundary_mask()

        U, V = self.U_TnT
        Uhat, Vhat = self.Uhat_TnT

        U = self.get_conservative_state(U)
        Uhat = self.get_conservative_state(Uhat)

        F = self.equations.convective_flux(U)
        Fn = self.get_convective_numerical_flux(U, Uhat, self.mesh.normal)

        blf['convection'] = -bla.inner(F, ngs.grad(V)) * ngs.dx(bonus_intorder=bonus.vol)
        blf['convection'] += bla.inner(Fn, V) * ngs.dx(element_boundary=True, bonus_intorder=bonus.bnd)
        blf['convection'] += -mask * bla.inner(Fn, Vhat) * ngs.dx(element_boundary=True, bonus_intorder=bonus.bnd)

    def add_diffusion_formulation(
            self, blf: dict[str, ngs.comp.SumOfIntegrals], lf: dict[str, ngs.comp.SumOfIntegrals]):

        bonus = self.cfg.optimizations.bonus_int_order

        mask = self.get_domain_boundary_mask()

        U, V = self.U_TnT
        Uhat, Vhat = self.Uhat_TnT
        Q, _ = self.mixed_method.Q_TnT

        U = self.get_conservative_state(U)
        Uhat = self.get_conservative_state(Uhat)
        Q = self.cfg.pde.fe.mixed_method.get_mixed_state(Q)

        G = self.equations.diffusive_flux(U, Q)
        Gn = self.get_diffusive_numerical_flux(U, Uhat, Q, self.mesh.normal)

        blf['diffusion'] = ngs.InnerProduct(G, ngs.grad(V)) * ngs.dx(bonus_intorder=bonus.vol)
        blf['diffusion'] -= ngs.InnerProduct(Gn, V) * ngs.dx(element_boundary=True, bonus_intorder=bonus.bnd)
        blf['diffusion'] += mask * ngs.InnerProduct(Gn, Vhat) * ngs.dx(element_boundary=True, bonus_intorder=bonus.bnd)

    def add_boundary_conditions(self, blf: dict[str, ngs.comp.SumOfIntegrals], lf: dict[str, ngs.comp.SumOfIntegrals]):

        bnds = self.cfg.pde.bcs.to_pattern()

        for bnd, bc in bnds.items():

            logger.debug(f"Adding boundary condition {bc} on boundary {bnd}.")

            if isinstance(bc, FarField):
                self.add_farfield_formulation(blf, lf, bc, bnd)

            elif isinstance(bc, Outflow):
                self.add_outflow_formulation(blf, lf, bc, bnd)

            elif isinstance(bc, (InviscidWall, Symmetry)):
                self.add_inviscid_wall_formulation(blf, lf, bc, bnd)

            elif isinstance(bc, IsothermalWall):
                self.add_isothermal_wall_formulation(blf, lf, bc, bnd)

            elif isinstance(bc, Periodic):
                continue

            elif isinstance(bc, AdiabaticWall):
                self.add_adiabatic_wall_formulation(blf, lf, bc, bnd)

            else:
                raise TypeError(f"Boundary condition {bc} not implemented in {self}!")

    def add_domain_conditions(self, blf: dict[str, ngs.comp.SumOfIntegrals], lf: dict[str, ngs.comp.SumOfIntegrals]):

        doms = self.cfg.pde.dcs.to_pattern()

        for dom, dc in doms.items():

            logger.debug(f"Adding domain condition {dc} on domain {dom}.")

            if isinstance(dc, SpongeLayer):
                self.add_sponge_layer_formulation(blf, lf, dc, dom)

            elif isinstance(dc, PSpongeLayer):
                self.add_psponge_layer_formulation(blf, lf, dc, dom)

            elif isinstance(dc, Initial):
                continue

            else:
                raise TypeError(f"Domain condition {dc} not implemented in {self}!")

    def add_farfield_formulation(
            self, blf: dict[str, ngs.comp.SumOfIntegrals],
            lf: dict[str, ngs.comp.SumOfIntegrals],
            bc: FarField, bnd: str):

        bonus = self.cfg.optimizations.bonus_int_order
        bnd_ = self.mesh.Boundaries(bnd)

        U, _ = self.U_TnT
        Uhat, Vhat = self.Uhat_TnT
        Uhat = self.get_conservative_state(Uhat)

        U_infty = ngs.CF(
            (self.equations.density(bc.state),
             self.equations.momentum(bc.state),
             self.equations.energy(bc.state)))

        An_in = self.equations.get_conservative_convective_jacobian(Uhat, self.mesh.normal, 'incoming')
        An_out = self.equations.get_conservative_convective_jacobian(Uhat, self.mesh.normal, 'outgoing')

        Gamma_infty = ngs.InnerProduct(An_out * (Uhat.U - U) - An_in * (Uhat.U - U_infty), Vhat)
        blf[f"{bc.name}_{bnd}"] = Gamma_infty * ngs.ds(skeleton=True, definedon=bnd_, bonus_intorder=bonus.bnd)

    def add_outflow_formulation(
            self, blf: dict[str, ngs.comp.SumOfIntegrals],
            lf: dict[str, ngs.comp.SumOfIntegrals],
            bc: Outflow, bnd: str):

        bonus = self.cfg.optimizations.bonus_int_order
        bnd_ = self.mesh.Boundaries(bnd)

        U, _ = self.U_TnT
        Uhat, Vhat = self.Uhat_TnT

        U = self.get_conservative_state(U)
        U.p = self.equations.pressure(bc.pressure)
        U_bc = ngs.CF((self.equations.density(U), self.equations.momentum(U), self.equations.energy(U)))

        Gamma_out = ngs.InnerProduct(Uhat - U_bc, Vhat)
        blf[f"{bc.name}_{bnd}"] = Gamma_out * ngs.ds(skeleton=True, definedon=bnd_, bonus_intorder=bonus.bnd)

    def add_inviscid_wall_formulation(
            self, blf: dict[str, ngs.comp.SumOfIntegrals],
            lf: dict[str, ngs.comp.SumOfIntegrals],
            bc: InviscidWall, bnd: str):

        n = self.mesh.normal
        bnd_ = self.mesh.Boundaries(bnd)

        U, _ = self.U_TnT
        Uhat, Vhat = self.Uhat_TnT

        U = self.get_conservative_state(U)

        rho = self.equations.density(U)
        rho_u = self.equations.momentum(U)
        rho_E = self.equations.energy(U)
        U_bc = ngs.CF((rho, rho_u - ngs.InnerProduct(rho_u, n)*n, rho_E))

        Gamma_inv = ngs.InnerProduct(Uhat - U_bc, Vhat)
        blf[f"{bc.name}_{bnd}"] = Gamma_inv * ngs.ds(skeleton=True, definedon=bnd_)

    def add_isothermal_wall_formulation(
            self, blf: dict[str, ngs.comp.SumOfIntegrals],
            lf: dict[str, ngs.comp.SumOfIntegrals],
            bc: IsothermalWall, bnd: str):

        bonus = self.cfg.optimizations.bonus_int_order
        bnd_ = self.mesh.Boundaries(bnd)

        U, _ = self.U_TnT
        Uhat, Vhat = self.Uhat_TnT

        U = self.get_conservative_state(U)
        U.T = bc.temperature

        U_bc = ngs.CF(
            (self.equations.density(U),
             tuple(0 for _ in range(self.mesh.dim)),
             self.equations.inner_energy(U)))

        Gamma_iso = ngs.InnerProduct(Uhat - U_bc, Vhat)
        blf[f"{bc.name}_{bnd}"] = Gamma_iso * ngs.ds(skeleton=True, definedon=bnd_, bonus_intorder=bonus.bnd)

    def add_adiabatic_wall_formulation(
            self, blf: dict[str, ngs.comp.SumOfIntegrals],
            lf: dict[str, ngs.comp.SumOfIntegrals],
            bc: AdiabaticWall, bnd: str):

        if not isinstance(self.mixed_method, StrainHeat):
            raise NotImplementedError(f"Adiabatic wall not implemented for {self.mixed_method}")

        bonus = self.cfg.optimizations.bonus_int_order
        bnd_ = self.mesh.Boundaries(bnd)
        n = self.mesh.normal

        U, _ = self.U_TnT
        Uhat, Vhat = self.Uhat_TnT
        Q, _ = self.mixed_method.Q_TnT

        U = self.get_conservative_state(U)
        Uhat = self.get_conservative_state(Uhat)
        Q = self.mixed_method.get_mixed_state(Q)

        tau = self.mixed_method.get_diffusive_stabilisation_matrix(U)
        T_grad = self.equations.temperature_gradient(U, Q)

        U_bc = ngs.CF((Uhat.rho - U.rho, Uhat.rho_u, tau * (Uhat.rho_E - U.rho_E + T_grad * n)))

        Gamma_ad = ngs.InnerProduct(Uhat.U - U_bc, Vhat)
        blf[f"{bc.name}_{bnd}"] = Gamma_ad * ngs.ds(skeleton=True, definedon=bnd_, bonus_intorder=bonus.bnd)

    def add_sponge_layer_formulation(
            self, blf: dict[str, ngs.comp.SumOfIntegrals],
            lf: dict[str, ngs.comp.SumOfIntegrals],
            dc: SpongeLayer, dom: str):

        dom_ = self.mesh.Materials(dom)

        U, V = self.U_TnT
        U_target = ngs.CF(
            (self.equations.density(dc.target_state),
             self.equations.momentum(dc.target_state),
             self.equations.energy(dc.target_state)))

        blf[f"{dc.name}_{dom}"] = dc.function * (U - U_target) * V * ngs.dx(definedon=dom_, bonus_intorder=dc.order)

    def add_psponge_layer_formulation(
            self, blf: dict[str, ngs.comp.SumOfIntegrals],
            lf: dict[str, ngs.comp.SumOfIntegrals],
            dc: PSpongeLayer, dom: str):

        dom_ = self.mesh.Materials(dom)

        U, V = self.U_TnT

        if dc.is_equal_order:

            U_target = ngs.CF(
                (self.equations.density(dc.target_state),
                 self.equations.momentum(dc.target_state),
                 self.equations.energy(dc.target_state)))

            Delta_U = U - U_target

        else:

            low_order_space = ngs.L2(self.mesh, order=dc.low_order)
            U_low = ngs.CF(tuple(ngs.Interpolate(proxy, low_order_space) for proxy in U))
            Delta_U = U - U_low

        blf[f"{dc.name}_{dom}"] = dc.function * Delta_U * V * ngs.dx(definedon=dom_, bonus_intorder=dc.order)

    def get_convective_numerical_flux(self, U: CompressibleState, Uhat: CompressibleState, unit_vector: bla.VECTOR):
        """
        Convective numerical flux

        Equation 22a, page 11

        Literature:
        [1] - Vila-Pérez, J., Giacomini, M., Sevilla, R. et al.
              Hybridisable Discontinuous Galerkin Formulation of Compressible Flows.
              Arch Computat Methods Eng 28, 753–784 (2021).
              https://doi.org/10.1007/s11831-020-09508-z
        """
        unit_vector = bla.as_vector(unit_vector)

        tau = self.cfg.pde.riemann_solver.get_convective_stabilisation_matrix(Uhat, unit_vector)

        return self.equations.convective_flux(Uhat) * unit_vector + tau * (U.U - Uhat.U)

    def get_diffusive_numerical_flux(
            self, U: CompressibleState, Uhat: CompressibleState, Q: CompressibleStateGradient, unit_vector: bla.VECTOR):
        """
        Diffusive numerical flux

        Equation 22b, page 11

        Literature:
        [1] - Vila-Pérez, J., Giacomini, M., Sevilla, R. et al.
              Hybridisable Discontinuous Galerkin Formulation of Compressible Flows.
              Arch Computat Methods Eng 28, 753–784 (2021).
              https://doi.org/10.1007/s11831-020-09508-z
        """
        unit_vector = bla.as_vector(unit_vector)

        tau_d = self.cfg.pde.fe.mixed_method.get_diffusive_stabilisation_matrix(Uhat)

        return self.equations.diffusive_flux(Uhat, Q)*unit_vector - tau_d * (U.U - Uhat.U)

    def get_domain_boundary_mask(self) -> ngs.GridFunction:
        """ 
        Returns a Gridfunction that is 0 on the domain boundaries and 1 on the domain interior.
        """

        fes = ngs.FacetFESpace(self.mesh, order=0)
        mask = ngs.GridFunction(fes, name="mask")
        mask.vec[:] = 0
        mask.vec[~fes.GetDofs(self.cfg.pde.bcs.get_domain_boundaries(True))] = 1

        return mask


class ConservativeFiniteElement(CompressibleFiniteElement):

    name: str = "conservative"

    @multiple(default=HDG)
    def method(self, method):
        return method

    @multiple(default=Inactive)
    def mixed_method(self, mixed_method):
        return mixed_method

    def get_finite_element_spaces(self, fes: dict[str, ngs.FESpace] | None = None) -> None:

        if fes is None:
            fes = {}

        self.method.add_finite_element_spaces(fes)
        self.mixed_method.add_mixed_finite_element_spaces(fes)

        super().get_finite_element_spaces(fes)

    def get_transient_gridfunctions(self, gfus: dict[str, dict[str, ngs.GridFunction]] | None = None) -> None:

        if gfus is None:
            gfus = {}

        self.method.add_transient_gridfunctions(gfus)
        self.mixed_method.add_transient_gridfunctions(gfus)

        super().get_transient_gridfunctions(gfus)

    def get_discrete_system(self, blf: dict[str, ngs.comp.SumOfIntegrals] = None,
                            lf: dict[str, ngs.comp.SumOfIntegrals] = None):

        if self.cfg.time.is_stationary:
            raise ValueError("Compressible flow requires transient solver!")

        if blf is None:
            blf = {}

        if lf is None:
            lf = {}

        self.method.add_time_derivative_formulation(blf, lf)
        self.method.add_convection_formulation(blf, lf)

        if not self.cfg.pde.dynamic_viscosity.is_inviscid:
            self.method.add_diffusion_formulation(blf, lf)

        self.mixed_method.add_mixed_formulation(blf, lf)

        self.method.add_boundary_conditions(blf, lf)
        self.method.add_domain_conditions(blf, lf)

        return blf, lf

    def set_initial_conditions(self) -> None:
        mass = {}
        rhs = {}

        self.method.add_initial_conditions(mass, rhs)
        self.mixed_method.add_initial_conditions(mass, rhs)

        blf = ngs.BilinearForm(self.cfg.pde.fes)
        for cf in mass.values():
            blf += cf

        lf = ngs.LinearForm(self.cfg.pde.fes)
        for cf in rhs.values():
            lf += cf

        with ngs.TaskManager():
            blf.Assemble()
            lf.Assemble()

            self.gfu.vec.data = blf.mat.Inverse(inverse="sparsecholesky") * lf.vec

    method: HDG
    mixed_method: Inactive | StrainHeat | Gradient
