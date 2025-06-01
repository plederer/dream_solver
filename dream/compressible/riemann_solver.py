""" Definitions of riemann solvers for compressible flow """
from __future__ import annotations
import typing
import ngsolve as ngs

from dream import bla
from dream.config import Configuration, dream_configuration
from dream.compressible.config import flowfields

if typing.TYPE_CHECKING:
    from .solver import CompressibleFlowSolver


class RiemannSolver(Configuration, is_interface=True):

    root: CompressibleFlowSolver

    def get_convective_stabilisation_matrix_hdg(self, U: flowfields, unit_vector: ngs.CF) -> ngs.CF:
        NotImplementedError("Override this method in the derived class")

    def get_simplified_convective_stabilisation_matrix_hdg(self, U: flowfields, unit_vector: ngs.CF) -> ngs.CF:
        NotImplementedError("Override this method in the derived class")

    def get_convective_numerical_flux_dg(self, Ui: flowfields, Uj: flowfields, unit_vector: ngs.CF) -> ngs.CF:
        NotImplementedError("Override this method in the derived class")


class Upwind(RiemannSolver):
    """ Upwind scheme for the convective flux. """

    name = "upwind"

    def get_convective_stabilisation_matrix_hdg(self, U: flowfields, unit_vector: ngs.CF) -> ngs.CF:
        r""" Returns the convective stabilisation matrix :math:`\mat{\tau}_c` for HDG.

        .. math::
            \bm{\tau}_c := \bm{A}_n^+
        """
        unit_vector = bla.as_vector(unit_vector)
        return self.root.get_conservative_convective_jacobian(U, unit_vector, 'outgoing')

    def get_simplified_convective_stabilisation_matrix_hdg(self, U, unit_vector):
        r""" Returns the simplified convective stabilisation matrix :math:`\mat{\tau}_{cs}` for HDG.

        This stabilisation matrix is used in an inviscid flow to overcome the issue of indefiniteness of the facet 
        variable, when the flow is parallel to the facet :cite:`PellmenreichCharacteristicBoundaryConditions2025`.

        .. math::
            \bm{\tau}_{cs} := \bm{Q}_n + \bm{I}
        """
        Qn = self.root.get_conservative_convective_identity(U, unit_vector, None)
        return Qn + ngs.Id(unit_vector.dim + 2)

    def get_convective_numerical_flux_dg(self, Ui: flowfields, Uj: flowfields, unit_vector: ngs.CF) -> ngs.CF:
        raise ValueError("FVS solver in a standard DG has not been implemented (yet).")


class LaxFriedrich(RiemannSolver):
    """ Lax-Friedrich scheme for the convective flux. """

    name = "lax_friedrich"

    def get_convective_stabilisation_matrix_hdg(self, U: flowfields, unit_vector: ngs.CF) -> ngs.CF:
        r""" Returns the convective stabilisation matrix :math:`\mat{\tau}_c` for HDG.

        .. math::
            \bm{\tau}_c := (|u_n| + c) \bm{I}

        :note: See equation :math:`(35)` in :cite:`vila-perezHybridisableDiscontinuousGalerkin2021`
        """
        unit_vector = bla.as_vector(unit_vector)

        u = self.root.velocity(U)
        c = self.root.speed_of_sound(U)

        lambda_max = bla.abs(bla.inner(u, unit_vector)) + c
        return lambda_max * ngs.Id(unit_vector.dim + 2)

    def get_simplified_convective_stabilisation_matrix_hdg(self, U: flowfields, unit_vector: ngs.CF) -> ngs.CF:
        r""" Returns the simplified convective stabilisation matrix :math:`\mat{\tau}_{cs}` for HDG.

        This stabilisation matrix is used in an inviscid flow to overcome the issue of indefiniteness of the facet 
        variable, when the flow is parallel to the facet :cite:`PellmenreichCharacteristicBoundaryConditions2025`.

        .. math::
            \bm{\tau}_{cs} := \bm{I}
        """
        return ngs.Id(unit_vector.dim + 2)

    def get_convective_numerical_flux_dg(self, Ui: flowfields, Uj: flowfields, unit_vector: ngs.CF) -> ngs.CF:

        # Extract the normal, taken w.r.t. the ith solution.
        n = bla.as_vector(unit_vector)

        # Compute the actual fluxes on the surface (for averaging).
        Fi = self.root.get_convective_flux(Ui)
        Fj = self.root.get_convective_flux(Uj)

        # Compute the normal velocity.
        vni = bla.inner(self.root.velocity(Ui), n)
        vnj = bla.inner(self.root.velocity(Uj), n)

        # Extract the speed of sound.
        ci = self.root.speed_of_sound(Ui)
        cj = self.root.speed_of_sound(Uj)

        # Compute the largest eigenvalues.
        lmbi = bla.abs(vni) + ci
        lmbj = bla.abs(vnj) + cj

        # Deduce the largest eigenvalue w.r.t. the 2 solutions.
        lmb_max = bla.max(lmbi, lmbj)

        # Assemble the numerical flux.
        fs = 0.5*((Fi + Fj)*n - lmb_max*(Uj.U - Ui.U))

        # Return the numerical flux.
        return fs


class Roe(RiemannSolver):
    """ Roe scheme for the convective flux. """

    name = "roe"

    def get_convective_stabilisation_matrix_hdg(self, U: flowfields, unit_vector: ngs.CF) -> ngs.CF:
        r""" Returns the convective stabilisation matrix :math:`\mat{\tau}_c` for HDG.

        .. math::
            \bm{\tau}_c := |\mat{A}_n|

        :note: See equation :math:`(36)` in :cite:`vila-perezHybridisableDiscontinuousGalerkin2021`
        """
        unit_vector = bla.as_vector(unit_vector)

        lambdas = self.root.characteristic_velocities(U, unit_vector, "absolute")
        return self.root.transform_characteristic_to_conservative(bla.diagonal(lambdas), U, unit_vector)

    def get_simplified_convective_stabilisation_matrix_hdg(self, U: flowfields, unit_vector: ngs.CF) -> ngs.CF:
        r""" Returns the simplified convective stabilisation matrix :math:`\mat{\tau}_{cs}` for HDG.

        This stabilisation matrix is used in an inviscid flow to overcome the issue of indefiniteness of the facet 
        variable, when the flow is parallel to the facet :cite:`PellmenreichCharacteristicBoundaryConditions2025`.

        .. math::
            \bm{\tau}_{cs} := \bm{I}
        """
        return ngs.Id(unit_vector.dim + 2)

    def get_convective_numerical_flux_dg(self, Ui: flowfields, Uj: flowfields, unit_vector: ngs.CF) -> ngs.CF:
        raise ValueError("Roe solver in a standard DG has not been implemented (yet).")


class HLL(RiemannSolver):
    """ HLL scheme for the convective flux. """

    name = "hll"

    def get_convective_stabilisation_matrix_hdg(self, U: flowfields, unit_vector: ngs.CF) -> ngs.CF:
        r""" Returns the convective stabilisation matrix :math:`\mat{\tau}_c` for HDG.

        .. math::
            \bm{\tau}_c := \max(u_n + c, 0) \bm{I}

        :note: See equation :math:`(38)` in :cite:`vila-perezHybridisableDiscontinuousGalerkin2021`
        """
        unit_vector = bla.as_vector(unit_vector)

        u = self.root.velocity(U)
        c = self.root.speed_of_sound(U)

        un = bla.inner(u, unit_vector)
        s_plus = bla.max(un + c)

        return s_plus * ngs.Id(unit_vector.dim + 2)

    def get_simplified_convective_stabilisation_matrix_hdg(self, U: flowfields, unit_vector: ngs.CF) -> ngs.CF:
        r""" Returns the simplified convective stabilisation matrix :math:`\mat{\tau}_{cs}` for HDG.

        This stabilisation matrix is used in an inviscid flow to overcome the issue of indefiniteness of the facet 
        variable, when the flow is parallel to the facet :cite:`PellmenreichCharacteristicBoundaryConditions2025`.

        .. math::
            \bm{\tau}_{cs} := (1 + \Ma_n) \bm{I}, \quad \Ma_n := -1 \leq \frac{u_n}{c} \leq 1
        """
        unit_vector = bla.as_vector(unit_vector)

        u = self.root.velocity(U)
        c = self.root.speed_of_sound(U)
        Mn = bla.inner(u, unit_vector)/c

        return (1 + bla.interval(Mn, -1, 1)) * ngs.Id(unit_vector.dim + 2)

    def get_convective_numerical_flux_dg(self, Ui: flowfields, Uj: flowfields, unit_vector: ngs.CF) -> ngs.CF:
        raise ValueError("HLL solver in a standard DG has not been implemented (yet).")


class HLLEM(RiemannSolver):
    """ HLLEM scheme for the convective flux. """

    name = "hllem"

    def __init__(self, mesh, root=None, **default):
        DEFAULT = {"theta_0": 1e-8}
        DEFAULT.update(default)
        super().__init__(mesh, root, **DEFAULT)

    @dream_configuration
    def theta_0(self):
        r""" Defines a threshold value :math:`\theta_0` used to stabilize contact waves, when the eigenvalue tends to zero.
        This can occur if the flow is parallel to the element or domain boundary!

        :note: See Remark :math:`11` in :cite:`vila-perezHybridisableDiscontinuousGalerkin2021`
        """
        return self._theta_0

    @theta_0.setter
    def theta_0(self, theta_0):
        if theta_0 < 0:
            raise ValueError("Theta_0 must be positive")
        self._theta_0 = theta_0

    def get_convective_stabilisation_matrix_hdg(self, U: flowfields, unit_vector: ngs.CF) -> ngs.CF:
        r""" Returns the convective stabilisation matrix :math:`\mat{\tau}_c` for HDG.

        .. math::
            \begin{align*}
            \theta &:= \max\left(\frac{|u_n|}{|u_n| + c}, \theta_0\right), &
            \Theta &:=  \text{diag}(1, \theta, \ldots, \theta, 1), &
            \bm{\tau}_c &:= \bm{P} \bm{\Theta} \bm{P}^{-1}
            \end{align*}

        :note: See equation :math:`(40)` in :cite:`vila-perezHybridisableDiscontinuousGalerkin2021`
        """
        unit_vector = bla.as_vector(unit_vector)

        u = self.root.velocity(U)
        c = self.root.speed_of_sound(U)

        un = bla.inner(u, unit_vector)
        un_abs = bla.abs(un)
        s_plus = bla.max(un + c)

        theta = bla.max(un_abs/(un_abs + c), self.theta_0)
        THETA = bla.diagonal([1] + unit_vector.dim * [theta] + [1])
        THETA = self.root.transform_characteristic_to_conservative(THETA, U, unit_vector)

        return s_plus * THETA

    def get_simplified_convective_stabilisation_matrix_hdg(self, U: flowfields, unit_vector: ngs.CF) -> ngs.CF:
        r""" Returns the simplified convective stabilisation matrix :math:`\mat{\tau}_{cs}` for HDG.

        This stabilisation matrix is used in an inviscid flow to overcome the issue of indefiniteness of the facet 
        variable, when the flow is parallel to the facet :cite:`PellmenreichCharacteristicBoundaryConditions2025`.

        .. math::
            \bm{\tau}_{cs} := (1 + \Ma_n) \bm{I}, \quad \Ma_n := -1 \leq \frac{u_n}{c} \leq 1
        """
        unit_vector = bla.as_vector(unit_vector)

        u = self.root.velocity(U)
        c = self.root.speed_of_sound(U)
        Mn = bla.inner(u, unit_vector)/c

        return (1 + bla.interval(Mn, -1, 1)) * ngs.Id(unit_vector.dim + 2)

    def get_convective_numerical_flux_dg(self, Ui: flowfields, Uj: flowfields, unit_vector: ngs.CF) -> ngs.CF:
        raise ValueError("HLLEM solver in a standard DG has not been implemented (yet).")
