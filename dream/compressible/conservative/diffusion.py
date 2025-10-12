""" Definitions of conservative hdg discretizations. """
from __future__ import annotations
import logging
import ngsolve as ngs
import typing
import dream.bla as bla

from dream.config import Configuration, dream_configuration, Integrals
from dream.mesh import SpongeLayer, PSpongeLayer, Periodic, Initial
from dream.compressible.config import (flowfields,
                                       ConservativeFiniteElementMethod,
                                       FarField,
                                       Outflow,
                                       InviscidWall,
                                       Symmetry,
                                       IsothermalWall,
                                       AdiabaticWall,
                                       InterfaceBC,
                                       Dirichlet,
                                       Force,
                                       CBC)

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from ..solver import CompressibleFlowSolver


class ViscousTreatment(Configuration, is_interface=True):

    root: CompressibleFlowSolver

    @property
    def fem(self) -> ConservativeFiniteElementMethod:
        return self.root.fem

    @property
    def TnT(self) -> dict[str, tuple[ngs.comp.ProxyFunction, ...]]:
        return self.root.fem.TnT

    @property
    def gfus(self) -> dict[str, ngs.GridFunction]:
        return self.root.fem.gfus

    def add_finite_element_spaces(self, fes: dict[str, ngs.FESpace]) -> None:
        pass

    def add_diffusion_form(self, blf: Integrals, lf: Integrals) -> None:
        raise NotImplementedError("A method for the viscous treatment of the diffusive forms must be implemented.")

    def add_adiabatic_wall_formulation(self, blf: Integrals, lf: Integrals, bc: AdiabaticWall, bnd: str):
        raise NotImplementedError("A method for the viscous treatment of the diffusive forms must be implemented.")

    def set_initial_conditions(self):
        pass

    def get_diffusive_numerical_flux(self, U: flowfields, Uhat: flowfields, Q: flowfields, unit_vector: ngs.CF):
        pass

    def get_solution_fields(self) -> flowfields:
        return flowfields()


class StrainHeat(ViscousTreatment):
    r""" Strain-tensor and temperature gradient mixed method for compressible flow.

    This mixed method is based on the strain-rate tensor 

    .. math::
        \mat{\varepsilon} = \frac{1}{2} \left( \grad{\vec{u}} + \grad{\vec{u}}^\T \right) - \frac{1}{3} \div{(\vec{u})} \mat{I}

    and the temperature gradient :math:`\phi = \grad{T}` as additional variables to the conservative variables. 
    It is used to solve the compressible Navier-Stokes equations with viscous effects.

    Find :math:`\left(\vec{U}_h,\hat{\vec{U}}_h, (\mat{\varepsilon}_h, \vec{\phi}_h) \right) \in U_h \times \hat{U}_h \times Q_h` such that

    .. math::

        \sum_{T  \in \mesh} \int_{T} \mat{\varepsilon}_h : \mat{\zeta}_h \, d\bm{x} + \int_{T} \vec{u}_h \cdot \div(\mat{\zeta}_h - \frac{1}{3}\tr(\mat{\zeta}_h)\I) \, d\bm{x}  - \int_{\partial T} \hat{\vec{u}}_h \cdot \left[\mat{\zeta}_h - \frac{1}{3}\tr(\mat{\zeta}_h)\mat{I} \right] \vec{n} \, d\bm{s} & = 0, \\
        \sum_{T  \in \mesh} \int_{T} \vec{\phi}_h \cdot \vec{\varphi}_h \, d\bm{x} + \int_{T} T_h \div(\vec{\varphi}_h) \, d\bm{x} - \int_{\partial T} \hat{T}_h \vec{\varphi}_h \cdot \vec{n}    \, d\bm{s}                                    & = 0,

    for all :math:`(\mat{\zeta}_h, \vec{\varphi}_h ) \in Q_h`. With the discrete space choosen as

    .. math::
        Q_h       & := \Xi_h \times \Theta_h, \\
        \Xi_h     & := L^2\left( (0, t_{end}] ; \mathbb{P}^k(\mesh, \mathbb{R}^{d \times d}_{\mathrm{sym}}) \right), \\
        \Theta_h  & := L^2\left( (0, t_{end}] ; \mathbb{P}^k(\mesh, \mathbb{R}^{d})                 \right).

    The discrete velocities :math:`\vec{u}_h := \vec{u}(\vec{U}_h)`, :math:`\hat{\vec{u}}_h := \vec{u}(\hat{\vec{U}}_h)`, 
    and the discrete temperatures :math:`\theta_h := \theta(\vec{U}_h)`, :math:`\hat{\theta}_h := \theta(\hat{\vec{U}}_h)` are functions
    of the conservative fields :math:`\vec{U}_h` and :math:`\hat{\vec{U}}_h`, respectively.

    :note: See :class:`HDG` for the definition of the conservative spaces :math:`U_h` and :math:`\hat{U}_h`.

    """

    name: str = "mixed_strain_temperature_gradient"

    def add_finite_element_spaces(self, fes: dict[str, ngs.FESpace]) -> None:

        if self.root.dynamic_viscosity.is_inviscid:
            raise TypeError(f"Inviscid configuration does not require mixed method!")

        dim = 4*self.mesh.dim - 3
        order = self.fem.order

        Q = ngs.L2(self.mesh, order=order)
        Q = self.root.dcs.reduce_psponge_layers_order_elementwise(Q)

        fes['Q'] = Q**dim

    def add_mixed_form(self, blf: Integrals, lf: Integrals) -> None:

        if self.mesh.dim == 3:
            raise NotImplementedError("StrainHeat method is not implemented for domain dimension 3!")

        bonus = self.fem.bonus_int_order['diffusion']

        U, _ = self.fem.TnT['U']
        Uhat, _ = self.fem.TnT['Uhat']
        Q, P = self.fem.TnT['Q']

        U = self.fem.get_conservative_fields(U)
        Uhat = self.fem.get_conservative_fields(Uhat)

        gradient_P = ngs.grad(P)
        Q = self.get_mixed_fields(Q)
        P = self.get_mixed_fields(P)

        dev_zeta = P.eps - bla.trace(P.eps) * ngs.Id(self.mesh.dim)/3
        div_dev_zeta = ngs.CF((gradient_P[0, 0] + gradient_P[1, 1], gradient_P[1, 0] + gradient_P[2, 1]))
        div_dev_zeta -= 1/3 * ngs.CF((gradient_P[0, 0] + gradient_P[2, 0], gradient_P[0, 1] + gradient_P[2, 1]))

        blf['Q']['mixed'] = ngs.InnerProduct(Q.eps, P.eps) * ngs.dx
        blf['Q']['mixed'] += ngs.InnerProduct(U.u, div_dev_zeta) * ngs.dx(bonus_intorder=bonus['vol'])
        blf['Q']['mixed'] -= ngs.InnerProduct(Uhat.u, dev_zeta*self.mesh.normal) * \
            ngs.dx(element_boundary=True, bonus_intorder=bonus['bnd'])

        div_xi = gradient_P[3, 0] + gradient_P[4, 1]
        blf['Q']['mixed'] += ngs.InnerProduct(Q.grad_T, P.grad_T) * ngs.dx
        blf['Q']['mixed'] += ngs.InnerProduct(U.T, div_xi) * ngs.dx(bonus_intorder=bonus['vol'])
        blf['Q']['mixed'] -= ngs.InnerProduct(Uhat.T*self.mesh.normal, P.grad_T) * \
            ngs.dx(element_boundary=True, bonus_intorder=bonus['bnd'])

    def add_diffusion_form(self, blf: Integrals, lf: Integrals):

        bonus = self.fem.bonus_int_order['diffusion']
        dX = ngs.dx(element_boundary=True, bonus_intorder=bonus['bnd'])

        mask = self.fem.get_domain_boundary_mask()

        U, V = self.TnT['U']
        Uhat, Vhat = self.TnT['Uhat']
        Q, _ = self.TnT['Q']

        U = self.fem.get_conservative_fields(U)
        Uhat = self.fem.get_conservative_fields(Uhat)
        Q = self.get_mixed_fields(Q)

        G = self.root.get_diffusive_flux(U, Q)
        Gn = self.get_diffusive_numerical_flux(U, Uhat, Q, self.mesh.normal)

        blf['U']['diffusion'] = ngs.InnerProduct(G, ngs.grad(V)) * ngs.dx(bonus_intorder=bonus['vol'])
        blf['U']['diffusion'] -= ngs.InnerProduct(Gn, V) * dX
        blf['Uhat']['diffusion'] = mask * ngs.InnerProduct(Gn, Vhat) * dX

        self.add_mixed_form(blf, lf)

    def add_cbc_formulation(self, blf: Integrals, lf: Integrals, bc: CBC, bnd: str):

        if self.mesh.dim == 3:
            raise NotImplementedError("StrainHeat method CBC is not implemented for domain dimension 3!")

        bonus = self.fem.bonus_int_order['diffusion']
        label = f"{bc.name}_{bnd}"
        dS = ngs.ds(skeleton=True, definedon=self.mesh.Boundaries(bnd), bonus_intorder=bonus['bnd'])

        Q, _ = self.TnT['Q']
        U, _ = self.fem.TnT['U']
        Uhat, Vhat = self.TnT['Uhat']

        U = self.fem.get_conservative_fields(U)
        Uhat = self.fem.get_conservative_fields(Uhat)
        Q = self.get_mixed_fields(Q)

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

        Qin = self.root.get_conservative_convective_identity(Uhat, self.mesh.normal, "incoming")
        dt = self.fem.scheme.get_time_step(True)

        S = self.get_conservative_diffusive_jacobian(U, Q, t) * (ngs.grad(U.U) * t)
        S += self.get_conservative_diffusive_jacobian(U, Q, n) * (ngs.grad(U.U) * n)
        S += self.get_mixed_diffusive_jacobian(U, t) * (ngs.grad(Q.Q) * t)
        S += self.get_mixed_diffusive_jacobian(U, n) * grad_Q

        blf['Uhat'][label] -= dt * Qin * S * Vhat * dS

    def add_adiabatic_wall_formulation(self, blf, lf, bc, bnd):

        bonus = self.fem.bonus_int_order['convection']
        dS = ngs.ds(skeleton=True, definedon=self.mesh.Boundaries(bnd), bonus_intorder=bonus['bnd'])

        n = self.mesh.normal

        U, _ = self.TnT['U']
        Uhat, Vhat = self.TnT['Uhat']
        Q, _ = self.TnT['Q']

        U = self.fem.get_conservative_fields(U)
        Uhat = self.fem.get_conservative_fields(Uhat)
        Q = self.get_mixed_fields(Q)

        tau = self.get_diffusive_stabilisation_matrix(Uhat)[self.mesh.dim+1, self.mesh.dim+1]
        q = self.root.heat_flux(Uhat, Q)

        U_bc = ngs.CF((Uhat.rho - U.rho, Uhat.rho_u, tau * (Uhat.rho_E - U.rho_E) - q * n))

        Gamma_ad = ngs.InnerProduct(U_bc, Vhat)
        blf['Uhat'][f"{bc.name}_{bnd}"] = Gamma_ad * dS

    def get_diffusive_numerical_flux(self, U: flowfields, Uhat: flowfields, Q: flowfields, unit_vector: ngs.CF):
        r"""
        Diffusive numerical flux

        .. math::

            \hat{\vec{G}}_h \vec{n}^\pm  := \vec{G}(\hat{\vec{U}_h}, \vec{Q}_h) \vec{n}^\pm + \mat{\tau}_d (\vec{U}_h - \hat{\vec{U}}_h).

        :note: See equation :math:`(E22b)` in :cite:`vila-perezHybridisableDiscontinuousGalerkin2021`.
        """
        unit_vector = bla.as_vector(unit_vector)

        tau_d = self.get_diffusive_stabilisation_matrix(Uhat)

        return self.root.get_diffusive_flux(Uhat, Q)*unit_vector - tau_d * (U.U - Uhat.U)

    def get_diffusive_stabilisation_matrix(self, U: flowfields) -> ngs.CF:
        Re = self.root.scaling.reference_reynolds_number
        Pr = self.root.prandtl_number
        mu = self.root.viscosity(U)

        tau_d = [0] + [1 for _ in range(self.mesh.dim)] + [1/Pr]
        return bla.diagonal(tau_d) * mu / Re

    def get_cbc_viscous_terms(self, bc: CBC):

        if self.mesh.dim == 3:
            raise NotImplementedError("StrainHeat method CBC is not implemented for domain dimension 3!")

        Q, _ = self.TnT['Q']
        U, _ = self.fem.TnT['U']

        U = self.fem.get_conservative_fields(U)
        Q = self.get_mixed_fields(Q)

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

    def get_mixed_fields(self, Q: ngs.CF):

        dim = self.mesh.dim

        if isinstance(Q, ngs.GridFunction):
            Q = Q.components

        Q_ = flowfields()
        Q_.eps = bla.symmetric_matrix_from_vector(Q[:3*dim - 3])
        Q_.grad_T = Q[3*dim - 3:]

        if isinstance(Q, ngs.comp.ProxyFunction):
            Q_.Q = Q

        return Q_

    def get_solution_fields(self):
        return self.get_mixed_fields(self.gfus['Q'])

    def get_conservative_diffusive_jacobian_x(self, U: flowfields, Q: flowfields):

        if self.mesh.dim == 3:
            raise NotImplementedError("StrainHeat method is not implemented for domain dimension 3!")

        rho = self.root.density(U)
        stess_tensor = self.root.deviatoric_stress_tensor(U, Q)
        txx, txy = stess_tensor[0, 0], stess_tensor[0, 1]
        ux, uy = U.u

        A = ngs.CF((
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            -txx*ux/rho - txy*uy/rho, txx/rho, txy/rho, 0
        ), dims=(4, 4))

        return A

    def get_conservative_diffusive_jacobian_y(self, U: flowfields, Q: flowfields):

        if self.mesh.dim == 3:
            raise NotImplementedError("StrainHeat method is not implemented for domain dimension 3!")

        rho = self.root.density(U)
        stess_tensor = self.root.deviatoric_stress_tensor(U, Q)
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
            self, U: flowfields, Q: flowfields, unit_vector: ngs.CF) -> ngs.CF:
        if self.mesh.dim == 3:
            raise NotImplementedError("StrainHeat method is not implemented for domain dimension 3!")

        unit_vector = bla.as_vector(unit_vector)

        A = self.get_conservative_diffusive_jacobian_x(U, Q)
        B = self.get_conservative_diffusive_jacobian_y(U, Q)
        return A * unit_vector[0] + B * unit_vector[1]

    def get_mixed_diffusive_jacobian_x(self, U: flowfields):

        if self.mesh.dim == 3:
            raise NotImplementedError("StrainHeat method is not implemented for domain dimension 3!")

        Re = self.root.scaling.reference_reynolds_number
        Pr = self.root.prandtl_number
        mu = self.root.viscosity(U)

        ux, uy = U.u

        A = mu/Re * ngs.CF((
            0, 0, 0, 0, 0,
            2, 0, 0, 0, 0,
            0, 2, 0, 0, 0,
            2*ux, 2*uy, 0, 1/Pr, 0
        ), dims=(4, 5))

        return A

    def get_mixed_diffusive_jacobian_y(self, U: flowfields):

        if self.mesh.dim == 3:
            raise NotImplementedError("StrainHeat method is not implemented for domain dimension 3!")

        Re = self.root.scaling.reference_reynolds_number
        Pr = self.root.prandtl_number
        mu = self.root.viscosity(U)

        ux, uy = U.u

        B = mu/Re * ngs.CF((
            0, 0, 0, 0, 0,
            0, 2, 0, 0, 0,
            0, 0, 2, 0, 0,
            0, 2*ux, 2*uy, 0, 1/Pr
        ), dims=(4, 5))

        return B

    def get_mixed_diffusive_jacobian(self, U: flowfields, unit_vector: ngs.CF) -> ngs.CF:

        if self.mesh.dim == 3:
            raise NotImplementedError("StrainHeat method is not implemented for domain dimension 3!")

        unit_vector = bla.as_vector(unit_vector)
        A = self.get_mixed_diffusive_jacobian_x(U)
        B = self.get_mixed_diffusive_jacobian_y(U)
        return A * unit_vector[0] + B * unit_vector[1]

    def set_initial_conditions(self):
        """ Set initial conditions for the mixed fields based on the initial condition of the conservative fields. """

        if self.mesh.dim == 3:
            raise NotImplementedError("StrainHeat method is not implemented for domain dimension 3!")

        Q_ = {}
        for dom, dc in self.root.dcs.to_pattern(Initial).items():
            eps = self._get_strain_rate_tensor_from_initial_condition(dc.fields.copy(), voigt_notation=True)
            grad_T = self._get_temperature_gradient_from_initial_condition(dc.fields.copy())
            Q_[dom] = ngs.CF((eps, grad_T))
        Q_ = self.mesh.MaterialCF(Q_)

        gfu = self.gfus['Q']
        fes = self.gfus['Q'].space
        Q, P = fes.TnT()

        blf = ngs.BilinearForm(fes)
        blf += Q * P * ngs.dx

        f = ngs.LinearForm(fes)
        f += Q_ * P * ngs.dx

        with ngs.TaskManager():
            blf.Assemble()
            f.Assemble()
            gfu.vec.data = blf.mat.Inverse(freedofs=fes.FreeDofs(), inverse="sparsecholesky") * f.vec

    def _get_strain_rate_tensor_from_initial_condition(self, U: flowfields, voigt_notation: bool = False):
        """ Compute the strain-rate tensor from the initial condition fields."""

        if U.grad_u is not None:
            eps = self.root.strain_rate_tensor(U)

        elif U.u is not None:
            U.grad_u = ngs.CF(
                (U.u[0].Diff(ngs.x),
                 U.u[0].Diff(ngs.y),
                 U.u[1].Diff(ngs.x),
                 U.u[1].Diff(ngs.y)), dims=(2, 2))
            eps = self.root.strain_rate_tensor(U)

        elif all((U.rho, U.rho_u, U.grad_rho, U.grad_rho_u)):
            U.grad_u = self.root.velocity_gradient(U, U)
            eps = self.root.strain_rate_tensor(U)

        elif all((U.rho, U.rho_u)):
            U.grad_rho = ngs.CF((U.rho.Diff(ngs.x), U.rho.Diff(ngs.y)))
            U.grad_rho_u = ngs.CF(
                (U.rho_u[0].Diff(ngs.x),
                 U.rho_u[0].Diff(ngs.y),
                 U.rho_u[1].Diff(ngs.x),
                 U.rho_u[1].Diff(ngs.y)), dims=(2, 2))
            U.grad_u = self.root.velocity_gradient(U, U)
            eps = self.root.strain_rate_tensor(U)

        else:
            raise ValueError(f"Initial condition does not provide sufficient fields to compute the strain-rate tensor!")

        if voigt_notation:
            eps = ngs.CF((eps[0, 0], eps[0, 1], eps[1, 1]))

        return eps

    def _get_temperature_gradient_from_initial_condition(self, U: flowfields):
        """ Compute the temperature gradient from the initial condition fields."""

        if U.grad_T is not None:
            grad_T = U.grad_T

        elif U.T is not None:
            U.grad_T = ngs.CF((U.T.Diff(ngs.x), U.T.Diff(ngs.y)))
            grad_T = self.root.temperature_gradient(U, U)

        elif all((U.rho, U.p, U.grad_p, U.grad_rho)):
            grad_T = self.root.temperature_gradient(U, U)

        elif all((U.rho, U.p)):
            U.grad_rho = ngs.CF((U.rho.Diff(ngs.x), U.rho.Diff(ngs.y)))
            U.grad_p = ngs.CF((U.p.Diff(ngs.x), U.p.Diff(ngs.y)))
            grad_T = self.root.temperature_gradient(U, U)

        else:
            raise ValueError(f"Initial condition does not provide sufficient fields to compute the temperature gradient!")

        return grad_T


class Gradient(ViscousTreatment):

    name: str = "conservative_gradient"

    def add_finite_element_spaces(self, fes: dict[str, ngs.FESpace]) -> None:

        if self.root.dynamic_viscosity.is_inviscid:
            raise TypeError(f"Inviscid configuration does not require mixed method!")

        dim = self.mesh.dim + 2
        order = self.fem.order

        Q = ngs.VectorL2(self.mesh, order=order)
        Q = self.root.dcs.reduce_psponge_layers_order_elementwise(Q)

        fes['Q'] = Q**dim

    def add_mixed_form(self, blf: Integrals, lf: Integrals) -> None:

        Q, P = self.TnT['Q']
        U, _ = self.fem.TnT['U']
        Uhat, _ = self.fem.TnT['Uhat']

        blf['Q']['mixed'] = ngs.InnerProduct(Q, P) * ngs.dx
        blf['Q']['mixed'] += ngs.InnerProduct(U, ngs.div(P)) * ngs.dx
        blf['Q']['mixed'] -= ngs.InnerProduct(Uhat, P*self.mesh.normal) * ngs.dx(element_boundary=True)

    def get_mixed_fields(self, Q: ngs.CoefficientFunction):

        if isinstance(Q, ngs.GridFunction):
            Q = Q.components

        Q_ = flowfields()
        Q_.grad_rho = Q[0]
        Q_.grad_rho_u = Q[slice(1, self.mesh.dim + 1)]
        Q_.grad_rho_E = Q[self.mesh.dim + 1]

        if isinstance(Q, ngs.comp.ProxyFunction):
            Q_.Q = Q

        return Q_
