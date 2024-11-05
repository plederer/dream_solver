from __future__ import annotations
import logging
import ngsolve as ngs
import typing

import dream.bla as bla

from dream.config import InterfaceConfiguration, interface

from dream.compressible.config import (flowstate, CompressibleFiniteElement, FarField, Outflow, InviscidWall, Symmetry,
                                       IsothermalWall, AdiabaticWall, CBC, GRCBC, NSCBC)
from dream.mesh import SpongeLayer, PSpongeLayer, Periodic, Initial

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from dream.solver import SolverConfiguration


# --- Conservative --- #

class MixedMethod(InterfaceConfiguration, is_interface=True):

    cfg: SolverConfiguration

    @property
    def mesh(self) -> ngs.Mesh:
        return self.cfg.mesh

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

    def get_cbc_viscous_terms(self, bc: CBC) -> ngs.CF:
        return ngs.CF(tuple(0 for _ in range(self.mesh.dim + 2)))

    def get_diffusive_stabilisation_matrix(self, U: flowstate) -> bla.MATRIX:
        Re = self.cfg.pde.reference_reynolds_number
        Pr = self.cfg.pde.prandtl_number
        mu = self.cfg.pde.viscosity(U)

        tau_d = [0] + [1 for _ in range(self.mesh.dim)] + [1/Pr]
        return bla.diagonal(tau_d) * mu / Re


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

        dim = 4*self.mesh.dim - 3
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
        Uhat = self.cfg.pde.fe.method.get_conservative_state(Uhat)

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

    def get_cbc_viscous_terms(self, bc: CBC):

        if self.mesh.dim == 3:
            raise NotImplementedError("StrainHeat method CBC is not implemented for domain dimension 3!")

        Q, _ = self.Q_TnT
        U, _ = self.cfg.pde.fe.method.U_TnT

        U = self.cfg.pde.fe.method.get_conservative_state(U)
        Q = self.get_mixed_state(Q)

        t = self.mesh.tangential
        n = self.mesh.normal

        R = ngs.CF((n, t), dims=(2, 2)).trans

        grad_Q = ngs.grad(Q.Q) * n
        grad_EPS = R.trans * ngs.CF((grad_Q[0], grad_Q[1], grad_Q[1], grad_Q[2]), dims=(2, 2)) * R
        grad_q = ngs.CF((grad_Q[3], grad_Q[4]))

        if bc.target == "outflow":
            grad_q = R.trans * ngs.CF((grad_Q[3], grad_Q[4]))
            grad_EPS = R * ngs.CF((grad_EPS[0], 0, 0, grad_EPS[2]), dims=(2, 2)) * R.trans
            grad_q = R * ngs.CF((0, grad_q[1]))
        else:
            grad_EPS = R * ngs.CF((0, grad_EPS[1], grad_EPS[1], grad_EPS[2]), dims=(2, 2)) * R.trans

        grad_Q = ngs.CF((grad_EPS[0], grad_EPS[1], grad_EPS[2], grad_q[0], grad_q[1]))

        S = self.get_conservative_diffusive_jacobian(U, Q, t) * (ngs.grad(U.U) * t)
        S += self.get_conservative_diffusive_jacobian(U, Q, n) * (ngs.grad(U.U) * n)
        S += self.get_mixed_diffusive_jacobian(U, t) * (ngs.grad(Q.Q) * t)
        S += self.get_mixed_diffusive_jacobian(U, n) * grad_Q

        return S

    def get_mixed_state(self, Q: ngs.CF):

        dim = self.mesh.dim

        if isinstance(Q, ngs.GridFunction):
            Q = Q.components

        Q_ = flowstate()
        Q_.strain = bla.symmetric_matrix_from_vector(Q[:3*dim - 3])
        Q_.grad_T = Q[3*dim - 3:]

        if isinstance(Q, ngs.comp.ProxyFunction):
            Q_.Q = Q

        return Q_

    def get_conservative_diffusive_jacobian_x(self, U: flowstate, Q: flowstate):

        if self.mesh.dim == 3:
            raise NotImplementedError("StrainHeat method is not implemented for domain dimension 3!")

        rho = self.cfg.pde.density(U)
        stess_tensor = self.cfg.pde.deviatoric_stress_tensor(U, Q)
        txx, txy = stess_tensor[0, 0], stess_tensor[0, 1]
        ux, uy = U.u

        A = ngs.CF((
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            -txx*ux/rho - txy*uy/rho, txx/rho, txy/rho, 0
        ), dims=(4, 4))

        return A

    def get_conservative_diffusive_jacobian_y(self, U: flowstate, Q: flowstate):

        if self.mesh.dim == 3:
            raise NotImplementedError("StrainHeat method is not implemented for domain dimension 3!")

        rho = self.cfg.pde.density(U)
        stess_tensor = self.cfg.pde.deviatoric_stress_tensor(U, Q)
        tyx, tyy = stess_tensor[1, 0], stess_tensor[1, 1]
        ux, uy = U.u

        B = ngs.CF((
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            -tyx*ux/rho - tyy*uy/rho, tyx/rho, tyy/rho, 0
        ), dims=(4, 4))

        return B

    def get_conservative_diffusive_jacobian(
            self, U: flowstate, Q: flowstate, unit_vector: ngs.CF) -> ngs.CF:
        if self.mesh.dim == 3:
            raise NotImplementedError("StrainHeat method is not implemented for domain dimension 3!")

        unit_vector = bla.as_vector(unit_vector)

        A = self.get_conservative_diffusive_jacobian_x(U, Q)
        B = self.get_conservative_diffusive_jacobian_y(U, Q)
        return A * unit_vector[0] + B * unit_vector[1]

    def get_mixed_diffusive_jacobian_x(self, U: flowstate):

        if self.mesh.dim == 3:
            raise NotImplementedError("StrainHeat method is not implemented for domain dimension 3!")

        Re = self.cfg.pde.reference_reynolds_number
        Pr = self.cfg.pde.prandtl_number
        mu = self.cfg.pde.viscosity(U)

        ux, uy = U.u

        A = mu/Re * ngs.CF((
            0, 0, 0, 0, 0,
            2, 0, 0, 0, 0,
            0, 2, 0, 0, 0,
            2*ux, 2*uy, 0, 1/Pr, 0
        ), dims=(4, 5))

        return A

    def get_mixed_diffusive_jacobian_y(self, U: flowstate):

        if self.mesh.dim == 3:
            raise NotImplementedError("StrainHeat method is not implemented for domain dimension 3!")

        Re = self.cfg.pde.reference_reynolds_number
        Pr = self.cfg.pde.prandtl_number
        mu = self.cfg.pde.viscosity(U)

        ux, uy = U.u

        B = mu/Re * ngs.CF((
            0, 0, 0, 0, 0,
            0, 2, 0, 0, 0,
            0, 0, 2, 0, 0,
            0, 2*ux, 2*uy, 0, 1/Pr
        ), dims=(4, 5))

        return B

    def get_mixed_diffusive_jacobian(self, U: flowstate, unit_vector: ngs.CF) -> ngs.CF:

        if self.mesh.dim == 3:
            raise NotImplementedError("StrainHeat method is not implemented for domain dimension 3!")

        unit_vector = bla.as_vector(unit_vector)
        A = self.get_mixed_diffusive_jacobian_x(U)
        B = self.get_mixed_diffusive_jacobian_y(U)
        return A * unit_vector[0] + B * unit_vector[1]


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

        Q_ = flowstate()
        Q_.grad_rho = Q[0]
        Q_.grad_rho_u = Q[slice(1, self.mesh.dim + 1)]
        Q_.grad_rho_E = Q[self.mesh.dim + 1]

        if isinstance(Q, ngs.comp.ProxyFunction):
            Q_.Q = Q

        return Q_


class ConservativeMethod(InterfaceConfiguration, is_interface=True):

    cfg: SolverConfiguration

    @property
    def mesh(self) -> ngs.Mesh:
        return self.cfg.mesh

    @property
    def U_TnT(self) -> tuple[ngs.comp.ProxyFunction, ...]:
        return self.cfg.pde.TnT['U']

    @property
    def U_gfu(self) -> ngs.GridFunction:
        return self.cfg.pde.gfus['U']

    @property
    def U_gfu_transient(self) -> dict[str, ngs.GridFunction]:
        return self.cfg.pde.transient_gfus['U']

    def add_transient_gridfunctions(self, gfus: dict[str, dict[str, ngs.GridFunction]]) -> None:
        gfus['U'] = self.cfg.time.scheme.get_transient_gridfunctions(self.U_gfu)

    def set_initial_conditions(self, U: ngs.CF = None):

        if U is None:
            U = self.mesh.MaterialCF({dom: ngs.CF(
                (self.cfg.pde.density(dc.state),
                 self.cfg.pde.momentum(dc.state),
                 self.cfg.pde.energy(dc.state))) for dom, dc in self.cfg.pde.dcs.to_pattern(Initial).items()})

        self.U_gfu.Set(U)

    def add_time_derivative_formulation(
            self, blf: dict[str, ngs.comp.SumOfIntegrals],
            lf: dict[str, ngs.comp.SumOfIntegrals]):

        U, V = self.U_TnT

        U_dt = self.U_gfu_transient.copy()
        U_dt['n+1'] = U

        U_dt = self.cfg.time.scheme.get_discrete_time_derivative(U_dt)

        blf['time'] = bla.inner(U_dt, V) * ngs.dx

    def get_conservative_state(self, U: ngs.CoefficientFunction) -> flowstate:

        if isinstance(U, ngs.GridFunction):
            U = U.components

        U_ = flowstate()
        U_.rho = U[0]
        U_.rho_u = U[slice(1, self.mesh.dim + 1)]
        U_.rho_E = U[self.mesh.dim + 1]

        U_.u = self.cfg.pde.velocity(U_)
        U_.rho_Ek = self.cfg.pde.kinetic_energy(U_)
        U_.rho_Ei = self.cfg.pde.inner_energy(U_)
        U_.p = self.cfg.pde.pressure(U_)
        U_.T = self.cfg.pde.temperature(U_)
        U_.c = self.cfg.pde.speed_of_sound(U_)

        if isinstance(U, ngs.comp.ProxyFunction):
            U_.U = U

        return U_

    def get_state(self, quantities: dict[str, bool]) -> flowstate:
        U = self.get_conservative_state(self.U_gfu)

        draw = flowstate()
        for symbol, name in U.symbols.items():
            if name in quantities and quantities[name]:
                quantities.pop(name)
                draw[symbol] = getattr(self.cfg.pde, name)(U)
            elif symbol in quantities and quantities[symbol]:
                quantities.pop(symbol)
                draw[symbol] = getattr(self.cfg.pde, name)(U)

        return draw


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
        return self.cfg.pde.transient_gfus['Uhat']

    def add_finite_element_spaces(self, fes: dict[str, ngs.FESpace]) -> None:

        order = self.cfg.pde.fe.order
        dim = self.mesh.dim + 2

        U = ngs.L2(self.mesh, order=order)
        Uhat = ngs.FacetFESpace(self.mesh, order=order)

        psponge_layers = self.cfg.pde.dcs.to_pattern(PSpongeLayer)
        if psponge_layers:
            U = self.cfg.pde.dcs.reduce_psponge_layers_order_elementwise(U, psponge_layers)
            Uhat = self.cfg.pde.dcs.reduce_psponge_layers_order_facetwise(Uhat, psponge_layers)

        if self.cfg.pde.bcs.has(Periodic):
            Uhat = ngs.Periodic(Uhat)

        fes['U'] = U**dim
        fes['Uhat'] = Uhat**dim

    def add_transient_gridfunctions(self, gfus: dict[str, dict[str, ngs.GridFunction]]) -> None:
        super().add_transient_gridfunctions(gfus)

        if self.cfg.pde.bcs.has(CBC):
            gfus['Uhat'] = self.cfg.time.scheme.get_transient_gridfunctions(self.Uhat_gfu)

    def set_initial_conditions(self, U: ngs.CF = None):

        if U is None:
            U = self.mesh.MaterialCF({dom: ngs.CF(
                (self.cfg.pde.density(dc.state),
                 self.cfg.pde.momentum(dc.state),
                 self.cfg.pde.energy(dc.state))) for dom, dc in self.cfg.pde.dcs.to_pattern(Initial).items()})

        super().set_initial_conditions(U)

        u, v = self.Uhat_gfu.space.TnT()

        blf = ngs.BilinearForm(self.Uhat_gfu.space)
        blf += u * v * ngs.dx(element_boundary=True)

        f = ngs.LinearForm(self.Uhat_gfu.space)
        f += U * v * ngs.dx(element_boundary=True)

        with ngs.TaskManager():
            blf.Assemble()
            f.Assemble()

            self.Uhat_gfu.vec.data = blf.mat.Inverse(inverse="sparsecholesky") * f.vec

    def add_convection_formulation(
            self, blf: dict[str, ngs.comp.SumOfIntegrals],
            lf: dict[str, ngs.comp.SumOfIntegrals]):

        bonus = self.cfg.optimizations.bonus_int_order

        mask = self.get_domain_boundary_mask()

        U, V = self.U_TnT
        Uhat, Vhat = self.Uhat_TnT

        U = self.get_conservative_state(U)
        Uhat = self.get_conservative_state(Uhat)

        F = self.cfg.pde.get_convective_flux(U)
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

        G = self.cfg.pde.get_diffusive_flux(U, Q)
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

            elif isinstance(bc, (GRCBC, NSCBC)):
                self.add_cbc_formulation(blf, lf, bc, bnd)

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
        dS = ngs.ds(skeleton=True, definedon=self.mesh.Boundaries(bnd), bonus_intorder=bonus.bnd)

        U, _ = self.U_TnT
        Uhat, Vhat = self.Uhat_TnT
        Uhat = self.get_conservative_state(Uhat)

        U_infty = ngs.CF(
            (self.cfg.pde.density(bc.state),
             self.cfg.pde.momentum(bc.state),
             self.cfg.pde.energy(bc.state)))

        if bc.identity_jacobian:
            Q_in = self.cfg.pde.get_conservative_convective_identity(Uhat, self.mesh.normal, 'incoming')
            Q_out = self.cfg.pde.get_conservative_convective_identity(Uhat, self.mesh.normal, 'outgoing')
            Gamma_infty = ngs.InnerProduct(Uhat.U - Q_out * U - Q_in * U_infty, Vhat)
        else:
            An_in = self.cfg.pde.get_conservative_convective_jacobian(Uhat, self.mesh.normal, 'incoming')
            An_out = self.cfg.pde.get_conservative_convective_jacobian(Uhat, self.mesh.normal, 'outgoing')
            Gamma_infty = ngs.InnerProduct(An_out * (Uhat.U - U) - An_in * (Uhat.U - U_infty), Vhat)

        blf[f"{bc.name}_{bnd}"] = Gamma_infty * dS

    def add_outflow_formulation(
            self, blf: dict[str, ngs.comp.SumOfIntegrals],
            lf: dict[str, ngs.comp.SumOfIntegrals],
            bc: Outflow, bnd: str):

        bonus = self.cfg.optimizations.bonus_int_order
        dS = ngs.ds(skeleton=True, definedon=self.mesh.Boundaries(bnd), bonus_intorder=bonus.bnd)

        U, _ = self.U_TnT
        Uhat, Vhat = self.Uhat_TnT

        U = self.get_conservative_state(U)
        U_bc = flowstate(rho=U.rho, rho_u=U.rho_u, rho_Ek=U.rho_Ek, p=bc.state.p)
        U_bc = ngs.CF((self.cfg.pde.density(U_bc), self.cfg.pde.momentum(U_bc), self.cfg.pde.energy(U_bc)))

        Gamma_out = ngs.InnerProduct(Uhat - U_bc, Vhat)
        blf[f"{bc.name}_{bnd}"] = Gamma_out * dS

    def add_cbc_formulation(self,
                            blf: dict[str, ngs.comp.SumOfIntegrals],
                            lf: dict[str, ngs.comp.SumOfIntegrals],
                            bc: CBC, bnd: str):

        bonus = self.cfg.optimizations.bonus_int_order
        label = f"{bc.name}_{bnd}"
        dS = ngs.ds(skeleton=True, definedon=self.mesh.Boundaries(bnd), bonus_intorder=bonus.bnd)

        U, _ = self.U_TnT
        Uhat, Vhat = self.Uhat_TnT

        U = self.get_conservative_state(U)
        Uhat = self.get_conservative_state(Uhat)

        if bc.target == "farfield":
            U_bc = ngs.CF(
                (self.cfg.pde.density(bc.state),
                 self.cfg.pde.momentum(bc.state),
                 self.cfg.pde.energy(bc.state)))

        elif bc.target == "outflow":
            U_bc = flowstate(rho=U.rho, rho_u=U.rho_u, rho_Ek=U.rho_Ek, p=bc.state.p)
            U_bc = ngs.CF((self.cfg.pde.density(U_bc), self.cfg.pde.momentum(U_bc), self.cfg.pde.energy(U_bc)))

        elif bc.target == "mass_inflow":
            U_bc = flowstate(rho=bc.state.rho, rho_u=bc.state.rho_u, rho_Ek=bc.state.rho_Ek, p=U.p)
            U_bc = ngs.CF((self.cfg.pde.density(U_bc), self.cfg.pde.momentum(U_bc), self.cfg.pde.energy(U_bc)))

        elif bc.target == "temperature_inflow":
            rho_ = self.cfg.pde.isentropic_density(U, bc.state)
            U_bc = flowstate(rho=rho_, u=bc.state.u, T=U.T)
            U_bc.Ek = self.cfg.pde.specific_kinetic_energy(U_bc)
            U_bc = ngs.CF((self.cfg.pde.density(U_bc), self.cfg.pde.momentum(U_bc), self.cfg.pde.energy(U_bc)))

        D = bc.get_relaxation_matrix(
            dt=self.cfg.time.timer.step, c=self.cfg.pde.speed_of_sound(Uhat),
            M=self.cfg.pde.mach_number)
        D = self.cfg.pde.transform_characteristic_to_conservative(D, Uhat, self.mesh.normal)

        beta = bc.tangential_relaxation
        Qin = self.cfg.pde.get_conservative_convective_identity(Uhat, self.mesh.normal, "incoming")
        Qout = self.cfg.pde.get_conservative_convective_identity(Uhat, self.mesh.normal, "outgoing")
        B = self.cfg.pde.get_conservative_convective_jacobian(Uhat, self.mesh.tangential)

        dt = self.cfg.time.scheme.get_normalized_time_step()
        Uhat_dt = self.Uhat_gfu_transient.copy()
        Uhat_dt['n+1'] = Uhat.U

        blf[label] = (Uhat.U - Qout * U.U - Qin * self.cfg.time.scheme.get_normalized_explicit_terms(Uhat_dt)) * Vhat * dS
        blf[label] -= dt * Qin * D * (U_bc - Uhat.U) * Vhat * dS
        blf[label] += dt * beta * Qin * B * (ngs.grad(Uhat.U) * self.mesh.tangential) * Vhat * dS

        if bc.is_viscous_fluxes:
            blf[label] -= dt * Qin * self.mixed_method.get_cbc_viscous_terms(bc) * Vhat * dS

    def add_inviscid_wall_formulation(
            self, blf: dict[str, ngs.comp.SumOfIntegrals],
            lf: dict[str, ngs.comp.SumOfIntegrals],
            bc: InviscidWall, bnd: str):

        n = self.mesh.normal
        dS = ngs.ds(skeleton=True, definedon=self.mesh.Boundaries(bnd))

        U, _ = self.U_TnT
        Uhat, Vhat = self.Uhat_TnT

        U = self.get_conservative_state(U)

        rho = self.cfg.pde.density(U)
        rho_u = self.cfg.pde.momentum(U)
        rho_E = self.cfg.pde.energy(U)
        U_bc = ngs.CF((rho, rho_u - ngs.InnerProduct(rho_u, n)*n, rho_E))

        Gamma_inv = ngs.InnerProduct(Uhat - U_bc, Vhat)
        blf[f"{bc.name}_{bnd}"] = Gamma_inv * dS

    def add_isothermal_wall_formulation(
            self, blf: dict[str, ngs.comp.SumOfIntegrals],
            lf: dict[str, ngs.comp.SumOfIntegrals],
            bc: IsothermalWall, bnd: str):

        bonus = self.cfg.optimizations.bonus_int_order
        dS = ngs.ds(skeleton=True, definedon=self.mesh.Boundaries(bnd), bonus_intorder=bonus.bnd)

        U, _ = self.U_TnT
        Uhat, Vhat = self.Uhat_TnT

        U = self.get_conservative_state(U)
        U_bc = flowstate(rho=U.rho, rho_u=tuple(0 for _ in range(self.mesh.dim)), rho_Ek=0, T=bc.state.T)
        U_bc = ngs.CF((self.cfg.pde.density(U_bc), self.cfg.pde.momentum(U_bc), self.cfg.pde.inner_energy(U_bc)))

        Gamma_iso = ngs.InnerProduct(Uhat - U_bc, Vhat)
        blf[f"{bc.name}_{bnd}"] = Gamma_iso * dS

    def add_adiabatic_wall_formulation(
            self, blf: dict[str, ngs.comp.SumOfIntegrals],
            lf: dict[str, ngs.comp.SumOfIntegrals],
            bc: AdiabaticWall, bnd: str):

        if not isinstance(self.mixed_method, StrainHeat):
            raise NotImplementedError(f"Adiabatic wall not implemented for {self.mixed_method}")

        bonus = self.cfg.optimizations.bonus_int_order
        dS = ngs.ds(skeleton=True, definedon=self.mesh.Boundaries(bnd), bonus_intorder=bonus.bnd)

        n = self.mesh.normal

        U, _ = self.U_TnT
        Uhat, Vhat = self.Uhat_TnT
        Q, _ = self.mixed_method.Q_TnT

        U = self.get_conservative_state(U)
        Uhat = self.get_conservative_state(Uhat)
        Q = self.mixed_method.get_mixed_state(Q)

        tau = self.mixed_method.get_diffusive_stabilisation_matrix(U)
        T_grad = self.cfg.pde.temperature_gradient(U, Q)

        U_bc = ngs.CF((Uhat.rho - U.rho, Uhat.rho_u, tau * (Uhat.rho_E - U.rho_E + T_grad * n)))

        Gamma_ad = ngs.InnerProduct(Uhat.U - U_bc, Vhat)
        blf[f"{bc.name}_{bnd}"] = Gamma_ad * dS

    def add_sponge_layer_formulation(
            self, blf: dict[str, ngs.comp.SumOfIntegrals],
            lf: dict[str, ngs.comp.SumOfIntegrals],
            dc: SpongeLayer, dom: str):

        dX = ngs.dx(definedon=self.mesh.Materials(dom), bonus_intorder=dc.order)

        U, V = self.U_TnT
        U_target = ngs.CF(
            (self.cfg.pde.density(dc.target_state),
             self.cfg.pde.momentum(dc.target_state),
             self.cfg.pde.energy(dc.target_state)))

        blf[f"{dc.name}_{dom}"] = dc.function * (U - U_target) * V * dX

    def add_psponge_layer_formulation(
            self, blf: dict[str, ngs.comp.SumOfIntegrals],
            lf: dict[str, ngs.comp.SumOfIntegrals],
            dc: PSpongeLayer, dom: str):

        dX = ngs.dx(definedon=self.mesh.Materials(dom), bonus_intorder=dc.order)

        U, V = self.U_TnT

        if dc.is_equal_order:

            U_target = ngs.CF(
                (self.cfg.pde.density(dc.target_state),
                 self.cfg.pde.momentum(dc.target_state),
                 self.cfg.pde.energy(dc.target_state)))

            Delta_U = U - U_target

        else:

            low_order_space = ngs.L2(self.mesh, order=dc.low_order)
            U_low = ngs.CF(tuple(ngs.Interpolate(proxy, low_order_space) for proxy in U))
            Delta_U = U - U_low

        blf[f"{dc.name}_{dom}"] = dc.function * Delta_U * V * dX

    def get_convective_numerical_flux(self, U: flowstate, Uhat: flowstate, unit_vector: bla.VECTOR):
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

        return self.cfg.pde.get_convective_flux(Uhat) * unit_vector + tau * (U.U - Uhat.U)

    def get_diffusive_numerical_flux(
            self, U: flowstate, Uhat: flowstate, Q: flowstate, unit_vector: bla.VECTOR):
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

        return self.cfg.pde.get_diffusive_flux(Uhat, Q)*unit_vector - tau_d * (U.U - Uhat.U)

    def get_domain_boundary_mask(self) -> ngs.GridFunction:
        """ 
        Returns a Gridfunction that is 0 on the domain boundaries and 1 on the domain interior.
        """

        fes = ngs.FacetFESpace(self.mesh, order=0)
        mask = ngs.GridFunction(fes, name="mask")
        mask.vec[:] = 0

        bnd_dofs = fes.GetDofs(self.mesh.Boundaries(self.cfg.pde.bcs.get_domain_boundaries(True)))
        mask.vec[~bnd_dofs] = 1

        return mask


class ConservativeFiniteElement(CompressibleFiniteElement):

    name: str = "conservative"

    @interface(default=HDG)
    def method(self, method):
        return method

    @interface(default=Inactive)
    def mixed_method(self, mixed_method):
        return mixed_method

    def add_finite_element_spaces(self, fes: dict[str, ngs.FESpace]):
        self.method.add_finite_element_spaces(fes)
        self.mixed_method.add_mixed_finite_element_spaces(fes)
        super().add_finite_element_spaces(fes)

    def add_transient_gridfunctions(self, gfus: dict[str, dict[str, ngs.GridFunction]]):
        self.method.add_transient_gridfunctions(gfus)
        self.mixed_method.add_transient_gridfunctions(gfus)
        super().add_transient_gridfunctions(gfus)

    def add_discrete_system(self, blf: dict[str, ngs.comp.SumOfIntegrals],
                            lf: dict[str, ngs.comp.SumOfIntegrals]):

        if self.cfg.time.is_stationary:
            raise ValueError("Compressible flow requires transient solver!")

        self.method.add_time_derivative_formulation(blf, lf)
        self.method.add_convection_formulation(blf, lf)

        if not self.cfg.pde.dynamic_viscosity.is_inviscid:
            self.method.add_diffusion_formulation(blf, lf)

        self.mixed_method.add_mixed_formulation(blf, lf)

        self.method.add_boundary_conditions(blf, lf)
        self.method.add_domain_conditions(blf, lf)
        super().add_discrete_system(blf, lf)

    def get_state(self, quantities: dict[str, bool]) -> flowstate:
        U = self.method.get_state(quantities)
        return U

    def set_initial_conditions(self) -> None:

        U = self.mesh.MaterialCF({dom: ngs.CF(
            (self.cfg.pde.density(dc.state),
             self.cfg.pde.momentum(dc.state),
             self.cfg.pde.energy(dc.state))) for dom, dc in self.cfg.pde.dcs.to_pattern(Initial).items()})

        self.method.set_initial_conditions(U)

    method: HDG
    mixed_method: Inactive | StrainHeat | Gradient
