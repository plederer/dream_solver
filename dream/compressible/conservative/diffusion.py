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



# # #
# Generic Interior penalty Method   #
# # # # # # # # # # # # # # # # # # #

class InteriorPenalty(ViscousTreatment, is_interface=True):
    
    name: str = "interior_penalty"

    def __init__(self, mesh, root=None, **default):

        DEFAULT = {'interior_penalty_coefficient': 1.0,}

        DEFAULT.update(default)
        super().__init__(mesh, root, **DEFAULT)

    @dream_configuration
    def interior_penalty_coefficient(self) -> float:
        r""" Sets the interior penalty constant.

            :getter: Returns the interior penalty constant
            :setter: Sets the interior penalty constant, defaults to 1.0
        """
        return self._interior_penalty_coefficient

    @interior_penalty_coefficient.setter
    def interior_penalty_coefficient(self, alpha: float) -> None:
        if alpha < 0.0:
            raise ValueError("Interior penalty coefficient must be +ve.")
        self._interior_penalty_coefficient = alpha

    def get_scaled_penalty_coefficient(self):
        return self.interior_penalty_coefficient * (self.fem.order + 1)**2 / self.mesh.meshsize

    def add_viscous_farfield_formulation(self, U_infty: ngs.CF, 
                                         blf: Integrals, lf: Integrals, bc: FarField, bnd: str):
        raise NotImplementedError(f"Function must be implemented in a derived class.")

    def get_frozen_diffusion_matrices_conservative(self, U: flowfields):
        
        # Get the relevant nondimensional numbers.
        Re = self.root.scaling.reference_reynolds_number
        Pr = self.root.prandtl_number

        # Extract the velocities.
        vel = self.root.velocity(U)
        u = vel[0]
        v = vel[1]
        
        # Extract the density and energies.
        rho = self.root.density(U)
        ei  = self.root.specific_inner_energy(U)
        ek  = self.root.specific_kinetic_energy(U)

        # Extract the viscosity.
        mu = self.root.viscosity(U)

        # NOTE, we explicitly scale the viscosity with Re, 
        # to obtain the correct nondimensionalization.
        mu /= Re

        # Compute the respective thermal conductivity constant.
        kappa = mu/Pr # NOTE, mu already is divided by Re.
        
        # Get the nondimensionalized specific heat at constant volume.
        cv = self.root.equation_of_state.specific_heat_cv

        # Second viscosity.
        lmb = -2*mu/3

        # Abbreviations.
        ovrho = 1.0/rho
        ovrhocv = ovrho/cv
        lmbp2mu = lmb + 2*mu
       
        # velocity squared. 
        u2 = u*u
        v2 = v*v

        # Form the (frozen) diagonal diffusion matrix: K11.
        KU11_21 = -ovrho*lmbp2mu * u
        KU11_22 =  ovrho*lmbp2mu
        KU11_31 = -ovrho*mu * v
        KU11_33 =  ovrho*mu
        KU11_41 =  ovrhocv*kappa*(ek-ei) - ovrho*( lmbp2mu*u2 + mu*v2 )
        KU11_42 = (ovrho*lmbp2mu - ovrhocv*kappa) * u 
        KU11_43 = (ovrho*mu      - ovrhocv*kappa) * v
        KU11_44 =  ovrhocv*kappa

        KU11 = ngs.CF((0,       0,       0,       0,
                       KU11_21, KU11_22, 0,       0,
                       KU11_31, 0,       KU11_33, 0,
                       KU11_41, KU11_42, KU11_43, KU11_44), dims=(4,4))
        
        # Form the (frozen) diagonal diffusion matrix: K22.
        KU22_21 = -ovrho*mu * u
        KU22_22 =  ovrho*mu
        KU22_31 = -ovrho*lmbp2mu * v
        KU22_33 =  ovrho*lmbp2mu
        KU22_41 =  ovrhocv*kappa*(ek-ei) - ovrho*( mu*u2 + lmbp2mu*v2 )
        KU22_42 = (ovrho*mu      - ovrhocv*kappa) * u 
        KU22_43 = (ovrho*lmbp2mu - ovrhocv*kappa) * v
        KU22_44 =  ovrhocv*kappa

        KU22 = ngs.CF((0,       0,       0,       0,
                       KU22_21, KU22_22, 0,       0,
                       KU22_31, 0,       KU22_33, 0,
                       KU22_41, KU22_42, KU22_43, KU22_44), dims=(4,4))

        # Form the (frozen) off-diagonal diffusion matrix: K12.
        KU12_21 = -ovrho*lmb * v
        KU12_23 =  ovrho*lmb
        KU12_31 = -ovrho*mu  * u
        KU12_32 =  ovrho*mu
        KU12_41 = -ovrho*( lmb + mu ) * u*v
        KU12_42 =  ovrho*mu  * v 
        KU12_43 =  ovrho*lmb * u

        KU12 = ngs.CF((0,       0,       0,       0,
                       KU12_21, 0,       KU12_23, 0,
                       KU12_31, KU12_32, 0,       0,
                       KU12_41, KU12_42, KU12_43, 0), dims=(4,4))

        # Form the (frozen) off-diagonal diffusion matrix: K21.
        KU21_21 = -ovrho*mu  * v
        KU21_23 =  ovrho*mu
        KU21_31 = -ovrho*lmb * u
        KU21_32 =  ovrho*lmb
        KU21_41 = -ovrho*( lmb + mu ) * u*v
        KU21_42 =  ovrho*lmb * v 
        KU21_43 =  ovrho*mu  * u

        KU21 = ngs.CF((0,       0,       0,       0,
                       KU21_21, 0,       KU21_23, 0,
                       KU21_31, KU21_32, 0,       0,
                       KU21_41, KU21_42, KU21_43, 0), dims=(4,4))

        # We're done.
        return KU11, KU12, KU21, KU22

    def get_frozen_diffusion_matrices_conservative_transposed(self, U: flowfields):
        KU11, KU12, KU21, KU22 = self.get_frozen_diffusion_matrices_conservative(U)
        return KU11.trans, KU12.trans, KU21.trans, KU22.trans

    def get_diffusive_flux_from_conservative_jump(self, U: flowfields, Ujump: ngs.CF, unit_vector: ngs.CF) -> ngs.CF:
        r""" Returns the conservative diffusive flux from given states and jump in the conservative variables along the unit normal vector.

            .. math::
                \bm{G}(\bm{U}, \jump{U} \otimes \bm{n})

            :param U: A dictionary containing the flow quantities
            :type U: flowfields
            :param dU: A CoefficientFunction containing the jump in the conservative variables
            :type dU: CoefficientFunction
        """

        dim = unit_vector.dim
        unit_vector = bla.as_vector(unit_vector)

        U.rho = self.root.density(U)
        U.rho_u = self.root.momentum(U)
        U.rho_E = self.root.energy(U)
        U.u = self.root.velocity(U)
        U.p = self.root.pressure(U)
                
        Ujump = bla.outer(Ujump, unit_vector)
        Ujump = flowfields(grad_rho=Ujump[0, :], grad_rho_u=ngs.CF(tuple(Ujump[i, :] for i in range(1, dim+1)), dims=(dim, dim)), grad_rho_E=Ujump[dim+1, :])
                
        Ujump.grad_u = self.root.velocity_gradient(U, Ujump)
        Ujump.grad_rho_Ek = self.root.kinetic_energy_gradient(U, Ujump)
        Ujump.grad_rho_Ei = self.root.inner_energy_gradient(U, Ujump)
        Ujump.grad_p = self.root.pressure_gradient(U, Ujump)
        Ujump.grad_T = self.root.temperature_gradient(U, Ujump)

        return self.root.get_diffusive_flux(U, Ujump)

    def get_surface_viscous_flux_from_linearized_state(self, Uhat: flowfields, gradU: ngs.CF) -> ngs.CF:
        KU11, KU12, KU21, KU22 = self.get_frozen_diffusion_matrices_conservative(Uhat)
        dUdx = gradU[:, 0]; nx = self.mesh.normal[0] 
        dUdy = gradU[:, 1]; ny = self.mesh.normal[1]

        # This returns:  F_n = n_i * K_{ij} * partial_j U.
        return nx*(KU11*dUdx + KU12*dUdy) + ny*(KU21*dUdx + KU22*dUdy)

    def add_diffusion_form(self, blf: Integrals, lf: Integrals):
        bonus = self.fem.bonus_int_order['diffusion']
        dV = ngs.dx(bonus_intorder=bonus['vol'])
        U, V = self.TnT['U']

        # Local solution, which includes the gradient.
        U = self.fem.get_conservative_fields(U, with_gradients=True)

        # Compute the flux of the solution, needed for the volume terms.
        G = self.root.get_diffusive_flux(U, U)

        # Add the volume term.
        blf['U']['diffusion_vol'] = ngs.InnerProduct(G, ngs.grad(V)) * dV

        # Add the remaining surface, symmetrizing and penalty terms.
        self.add_surface_term(blf, lf)
        self.add_symmetrizing_term(blf, lf)
        self.add_penalizing_term(blf, lf)


# # # 
# Interior penalty for Hybridized Discontinuous Galerkin    #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class InteriorPenaltyHDG(InteriorPenalty):

    # XXX: This will be removed, once we are confident in this class.
    def __init__(self, mesh, root=None, **default):
        super().__init__(mesh, root, **default)
        logger.warning("Conservative HDG with IP is still experimental and may not be fully functional!")

    def add_surface_term(self, blf: Integrals, lf: Integrals) -> None:

        bonus = self.fem.bonus_int_order['diffusion']
        dS = ngs.dx(element_boundary=True, bonus_intorder=bonus['bnd'])
        U, V = self.TnT['U']
        Uhat, Vhat = self.TnT['Uhat']
        
        # Extract a nonzero mask for interior faces.
        mask = self.root.fem.get_domain_boundary_mask()

        # Convert the facet solution to flowfields.
        Uhat = self.fem.get_conservative_fields(Uhat)

        # Form the surface term.
        surf = self.get_surface_viscous_flux_from_linearized_state( Uhat, ngs.grad(U) )
        blf['U']['diffusion_surf'] = -ngs.InnerProduct(surf, V) * dS
        blf['Uhat']['diffusion_surf'] = mask * ngs.InnerProduct(surf, Vhat) * dS

    def add_symmetrizing_term(self, blf: Integrals, lf: Integrals) -> None:

        bonus = self.fem.bonus_int_order['diffusion']
        dS = ngs.dx(element_boundary=True, bonus_intorder=bonus['bnd'])
        U, V = self.TnT['U']
        Uhat, _ = self.TnT['Uhat']

        # Get the gradient of the test functions and decompose it in x and y-directions.
        gradV = ngs.grad(V)
        dVdx = gradV[:, 0]
        dVdy = gradV[:, 1]

        # Extract the components of the unit normal vector.
        nx = self.mesh.normal[0]
        ny = self.mesh.normal[1]

        # Define the jump of the solution.
        jumpU = U - Uhat
        
        # Convert the facet solution to flowfields.
        Uhat = self.fem.get_conservative_fields(Uhat)

        # Form the term: G^T * grad(V). Notice, G is evaluated at Uhat, ie. G=G(Uhat).
        KU11T, KU12T, KU21T, KU22T = self.get_frozen_diffusion_matrices_conservative_transposed(Uhat)

        # Form the symmetrizing term.
        symm = nx * (KU11T*dVdx + KU21T*dVdy) + ny * (KU12T*dVdx + KU22T*dVdy) 
        blf['U']['diffusion_symm'] = -ngs.InnerProduct(symm, jumpU) * dS

    def add_penalizing_term(self, blf: Integrals, lf: Integrals) -> None:

        bonus = self.fem.bonus_int_order['diffusion']
        dS = ngs.dx(element_boundary=True, bonus_intorder=bonus['bnd'])
        U, V = self.TnT['U']
        Uhat, Vhat = self.TnT['Uhat']
        
        # Extract a nonzero mask for interior faces.
        mask = self.root.fem.get_domain_boundary_mask()

        # Extract the components of the unit normal vector.
        nx = self.mesh.normal[0]
        ny = self.mesh.normal[1]
        
        # Define the jump of the solution.
        jumpU = U - Uhat
        
        # Convert the solutions to flowfields.
        U = self.fem.get_conservative_fields(U)
        Uhat = self.fem.get_conservative_fields(Uhat)

        # Interior penalty coefficient.
        tau = self.get_scaled_penalty_coefficient() 

        # Get the diffusion matrices, based on the conservative gradients.
        KU11, KU12, KU21, KU22 = self.get_frozen_diffusion_matrices_conservative(Uhat)

        # Form the penalty term, which is based on the contraction of the diffusion matrices.
        KUij = nx * (KU11*nx + KU12*ny) + ny * (KU21*nx + KU22*ny)
        penn = tau * KUij * jumpU
        blf['U']['diffusion_penn'] = ngs.InnerProduct(penn, V) * dS
        blf['Uhat']['diffusion_penn'] = -mask * ngs.InnerProduct(penn, Vhat) * dS

    def add_viscous_farfield_formulation(self, U_infty: ngs.CF, 
                                         blf: Integrals, lf: Integrals, bc: FarField, bnd: str):
        
        logger.warning("Careful, this has not been properly tested.") # TODO: finish me.

        bonus = self.fem.bonus_int_order['diffusion']
        dS = ngs.ds(skeleton=True, definedon=self.mesh.Boundaries(bnd), bonus_intorder=bonus['bnd'])
        U, _ = self.TnT['U']
        Uhat, Vhat = self.TnT['Uhat']
        
        # Choose the linearization state, needed for: K_{ij} = K_{ij}(U0).
        U0 = self.fem.get_conservative_fields(U_infty)

        # # # 
        # Surface term.
        # # 

        # Compute the flux of the solution, which is a function of the solution and external.
        surf = self.get_surface_viscous_flux_from_linearized_state( U0, ngs.grad(U) )
        blf['Uhat'][f"{bc.name}_{bnd}_surf"] = ngs.InnerProduct(surf, Vhat) * dS

        # # # 
        # Penalty term.
        # # 

        # Jump of the solution.
        jumpU = U_infty - Uhat

        # Interior penalty coefficient.
        tau = self.get_scaled_penalty_coefficient() 

        # Get the diffusion matrices, based on the conservative gradients.
        KU11, KU12, KU21, KU22 = self.get_frozen_diffusion_matrices_conservative(U0)

        # Form the penalty term, which is based on the contraction of the diffusion matrices.
        KUij = nx * (KU11*nx + KU12*ny) + ny * (KU21*nx + KU22*ny)
        penn = tau * KUij * jumpU
        blf['Uhat'][f"{bc.name}_{bnd}_penn"] = -ngs.InnerProduct(penn, Vhat) * dS

    def add_adiabatic_wall_formulation(self, blf, lf, bc, bnd):

        logger.warning("Careful, this has not been properly tested.") # TODO: finish me.

        bonus = self.fem.bonus_int_order['diffusion']
        dS = ngs.ds(skeleton=True, definedon=self.mesh.Boundaries(bnd), bonus_intorder=bonus['bnd'])

        n = self.mesh.normal

        U, _ = self.TnT['U']
        Uhat, Vhat = self.TnT['Uhat']

        U = self.fem.get_conservative_fields(U, with_gradients=True)
        Uhat = self.fem.get_conservative_fields(Uhat)

        # Get the inner energy: rho*ei,
        Ei = self.root.inner_energy(U)

        # Compute the boundary state. Note, the velocity (kinetic energy) and heat flux 
        # are set to zero. Remaining variables (density and pressure) are extrapolated.
        Ub = ngs.CF( (U.rho, 0, 0, Ei) ) 

        # Choose the linearization state, needed for: K_{ij} = K_{ij}(U0).
        U0 = self.fem.get_conservative_fields(U)

        # # # 
        # Surface term.
        # # 

        # Compute the flux of the solution, which is a function of the solution and external.
        surf = self.get_surface_viscous_flux_from_linearized_state( U0, ngs.grad(U) )
        blf['Uhat'][f"{bc.name}_{bnd}_surf"] = ngs.InnerProduct(surf, Vhat) * dS

        # # # 
        # Penalty term.
        # # 

        # Jump of the solution.
        jumpU = Ub - Uhat

        # Interior penalty coefficient.
        tau = self.get_scaled_penalty_coefficient() 

        # Get the diffusion matrices, based on the conservative gradients.
        KU11, KU12, KU21, KU22 = self.get_frozen_diffusion_matrices_conservative(U0)

        # Form the penalty term, which is based on the contraction of the diffusion matrices.
        KUij = nx * (KU11*nx + KU12*ny) + ny * (KU21*nx + KU22*ny)
        penn = tau * KUij * jumpU
        blf['Uhat'][f"{bc.name}_{bnd}_penn"] = -ngs.InnerProduct(penn, Vhat) * dS



# # #
# Interior penalty for Standard Discontinuous Galerkin      #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class InteriorPenaltySDG(InteriorPenalty):
    r""" This is based on the implementation in:

    Hartmann, R. and Houston, P., 2008. An optimal order interior penalty discontinuous Galerkin discretization of the compressible Navier–Stokes equations. Journal of Computational Physics, 227(22), pp.9670-9685.
    """
    
    def add_surface_term(self, blf: Integrals, lf: Integrals) -> None:

        bonus = self.fem.bonus_int_order['diffusion']
        dS = ngs.dx(skeleton=True, bonus_intorder=bonus['bnd'])
        U, V = self.TnT['U']

        # Extract local and neighboring solution (with gradients).
        Ui = self.fem.get_conservative_fields(U, with_gradients=True)
        Uj = self.fem.get_conservative_fields(U.Other(), with_gradients=True)

        # Compute the flux of the solution on the current and neighbor element.
        Gi = self.root.get_diffusive_flux(Ui, Ui)
        Gj = self.root.get_diffusive_flux(Uj, Uj)

        # Jump of the test functions.
        jumpV = V - V.Other()

        # Form the surface term.
        surf = (Gi + Gj) * self.mesh.normal / 2 
        blf['U']['diffusion_surf'] = -ngs.InnerProduct(surf, jumpV) * dS

    def add_symmetrizing_term(self, blf: Integrals, lf: Integrals) -> None:

        bonus = self.fem.bonus_int_order['diffusion']
        dS = ngs.dx(skeleton=True, bonus_intorder=bonus['bnd'])
        U, V = self.TnT['U']

        # Extract the components of the unit normal vector.
        nx = self.mesh.normal[0]
        ny = self.mesh.normal[1]

        # Extract local and neighboring solution (with gradients).
        Ui = self.fem.get_conservative_fields(U)
        Uj = self.fem.get_conservative_fields(U.Other())

        # Jump of the conservative solution.
        jumpU = U - U.Other()

        # Get the gradient of the test functions.
        gradVi = ngs.grad(V)
        gradVj = ngs.grad(V.Other())
        
        # Extract the x and y-components of the gradient of the test functions.
        dVidx = gradVi[:, 0]; dVidy = gradVi[:, 1]
        dVjdx = gradVj[:, 0]; dVjdy = gradVj[:, 1]

        # Get the diffusion matrices transposed, based on the conservative gradients.
        KU11Ti, KU12Ti, KU21Ti, KU22Ti = self.get_frozen_diffusion_matrices_conservative_transposed(Ui)
        KU11Tj, KU12Tj, KU21Tj, KU22Tj = self.get_frozen_diffusion_matrices_conservative_transposed(Uj)

        # Form the term: {G^T * grad(V)}, for the current and neighbor solution.
        FTi = nx * (KU11Ti*dVidx + KU21Ti*dVidy) + ny * (KU12Ti*dVidx + KU22Ti*dVidy) 
        FTj = nx * (KU11Tj*dVjdx + KU21Tj*dVjdy) + ny * (KU12Tj*dVjdx + KU22Tj*dVjdy)
  
        # Form the symmetrizing term.
        symm = (FTi + FTj) / 2 
        blf['U']['diffusion_symm'] = -ngs.InnerProduct(symm, jumpU) * dS

    def add_penalizing_term(self, blf: Integrals, lf: Integrals) -> None:

        bonus = self.fem.bonus_int_order['diffusion']
        dS = ngs.dx(skeleton=True, bonus_intorder=bonus['bnd'])
        U, V = self.TnT['U']

        # Extract the components of the unit normal vector.
        nx = self.mesh.normal[0]
        ny = self.mesh.normal[1]

        # Extract local and neighboring solution (with gradients).
        Ui = self.fem.get_conservative_fields(U)
        Uj = self.fem.get_conservative_fields(U.Other())

        # Jump of the conservative solution and test functions.
        jumpU = U - U.Other()
        jumpV = V - V.Other()

        # Interior penalty coefficient.
        tau = self.get_scaled_penalty_coefficient() 

        # Get the diffusion matrices, based on the conservative gradients.
        KU11i, KU12i, KU21i, KU22i = self.get_frozen_diffusion_matrices_conservative(Ui)
        KU11j, KU12j, KU21j, KU22j = self.get_frozen_diffusion_matrices_conservative(Uj)

        # Average the diffusion matrices.
        KU11 = (KU11i + KU11j) / 2
        KU12 = (KU12i + KU12j) / 2
        KU21 = (KU21i + KU21j) / 2
        KU22 = (KU22i + KU22j) / 2

        # Form the penalty term, which is based on the contraction of the diffusion matrices.
        KUij = nx * (KU11*nx + KU12*ny) + ny * (KU21*nx + KU22*ny)
        penn = tau * KUij * jumpU
        blf['U']['diffusion_penn'] = ngs.InnerProduct(penn, jumpV) * dS
    
    def add_viscous_interface_formulation(self, blf: Integrals, lf: Integrals, bc: InterfaceBC, bnd: str):
        
        bonus = self.fem.bonus_int_order['diffusion']
        dS = ngs.ds(skeleton=True, definedon=self.mesh.Boundaries(bnd), bonus_intorder=bonus['bnd'])
        U, V = self.TnT['U']

        # Form the boundary state, written as conservative variables.
        U_infty = ngs.CF(
            (self.root.density(bc.fields),
             self.root.momentum(bc.fields),
             self.root.energy(bc.fields)))

        # Extract local and neighboring solution (with gradients).
        Ui = self.fem.get_conservative_fields(U, with_gradients=True)
        Uj = bc.fields # NOTE, this uses the Other's relations to compute its quantities.

        # # # 
        # Surface term.
        # # 

        # Compute the flux of the solution on the current and neighbor element.
        Gi = self.root.get_diffusive_flux(Ui, Ui)
        Gj = self.root.get_diffusive_flux(Uj, Uj)

        # Form the surface term.
        surf = (Gi + Gj) * self.mesh.normal / 2 
        blf['U'][f"{bc.name}_{bnd}_surf"] = -ngs.InnerProduct(surf, V) * dS

        # # #
        # Symmetrizing term.
        # # 

        # Extract the components of the unit normal vector.
        nx = self.mesh.normal[0]
        ny = self.mesh.normal[1]

        # Jump of the conservative solution.
        jumpU = U - U_infty

        # Get the gradient of the test functions.
        gradVi = ngs.grad(V)
        
        # Extract the x and y-components of the gradient of the test functions.
        dVidx = gradVi[:, 0]; dVidy = gradVi[:, 1]

        # Get the diffusion matrices transposed, based on the conservative gradients.
        KU11Ti, KU12Ti, KU21Ti, KU22Ti = self.get_frozen_diffusion_matrices_conservative_transposed(Ui)
        KU11Tj, KU12Tj, KU21Tj, KU22Tj = self.get_frozen_diffusion_matrices_conservative_transposed(Uj)

        KU11T = (KU11Ti + KU11Tj) / 2
        KU12T = (KU12Ti + KU12Tj) / 2
        KU21T = (KU21Ti + KU21Tj) / 2
        KU22T = (KU22Ti + KU22Tj) / 2
        
        # Form the term: {G^T * grad(V)}, for the current and neighbor solution.
        symm = nx * (KU11T*dVidx + KU21T*dVidy) + ny * (KU12T*dVidx + KU22T*dVidy) 
  
        # Form the symmetrizing term.
        blf['U'][f"{bc.name}_{bnd}_symm"] = -ngs.InnerProduct(symm, jumpU) * dS

        # # # 
        # Penalty term.
        # # 

        # Interior penalty coefficient.
        tau = self.get_scaled_penalty_coefficient() 

        # Get the diffusion matrices, based on the conservative gradients.
        KU11i, KU12i, KU21i, KU22i = self.get_frozen_diffusion_matrices_conservative(Ui)
        KU11j, KU12j, KU21j, KU22j = self.get_frozen_diffusion_matrices_conservative(Uj)

        # Average the diffusion matrices.
        KU11 = (KU11i + KU11j) / 2
        KU12 = (KU12i + KU12j) / 2
        KU21 = (KU21i + KU21j) / 2
        KU22 = (KU22i + KU22j) / 2

        # Form the penalty term, which is based on the contraction of the diffusion matrices.
        KUij = nx * (KU11*nx + KU12*ny) + ny * (KU21*nx + KU22*ny)
        penn = tau * KUij * jumpU
        blf['U'][f"{bc.name}_{bnd}_penn"] = ngs.InnerProduct(penn, V) * dS

    def add_viscous_farfield_formulation(self, U_infty: ngs.CF, 
                                         blf: Integrals, lf: Integrals, bc: FarField, bnd: str):
        
        bonus = self.fem.bonus_int_order['diffusion']
        dS = ngs.ds(skeleton=True, definedon=self.mesh.Boundaries(bnd), bonus_intorder=bonus['bnd'])
        U, V = self.TnT['U']
        gradU = ngs.grad(U)

        # Choose the linearization state, needed for: K_{ij} = K_{ij}(Uhat).
        Uhat = self.fem.get_conservative_fields(U_infty)

        # # # 
        # Surface term.
        # # 

        # Form the surface term.
        surf = self.get_surface_viscous_flux_from_linearized_state(Uhat, gradU)
        blf['U'][f"{bc.name}_{bnd}_surf"] = -ngs.InnerProduct(surf, V) * dS

        # # #
        # Symmetrizing term.
        # # 

        # Extract the components of the unit normal vector.
        nx = self.mesh.normal[0]
        ny = self.mesh.normal[1]

        # Jump of the conservative solution.
        jumpU = U - U_infty

        # Get the gradient of the test functions.
        gradV = ngs.grad(V)
        
        # Extract the x and y-components of the gradient of the test functions.
        dVdx = gradV[:, 0]; dVdy = gradV[:, 1]

        # Get the diffusion matrices transposed, based on the conservative gradients.
        KU11T, KU12T, KU21T, KU22T = self.get_frozen_diffusion_matrices_conservative_transposed(Uhat)

        # Form the term: {G^T * grad(V)}, for the local solution.
        symm = nx * (KU11T*dVdx + KU21T*dVdy) + ny * (KU12T*dVdx + KU22T*dVdy) 
  
        # Form the symmetrizing term.
        blf['U'][f"{bc.name}_{bnd}_symm"] = -ngs.InnerProduct(symm, jumpU) * dS

        # # # 
        # Penalty term.
        # # 

        # Interior penalty coefficient.
        tau = self.get_scaled_penalty_coefficient() 

        # Get the diffusion matrices, based on the conservative gradients.
        KU11, KU12, KU21, KU22 = self.get_frozen_diffusion_matrices_conservative(Uhat)

        # Form the penalty term, which is based on the contraction of the diffusion matrices.
        KUij = nx * (KU11*nx + KU12*ny) + ny * (KU21*nx + KU22*ny)
        penn = tau * KUij * jumpU
        blf['U'][f"{bc.name}_{bnd}_penn"] = ngs.InnerProduct(penn, V) * dS


