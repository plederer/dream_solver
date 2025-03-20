from __future__ import annotations
import typing
import ngsolve as ngs

from dream import bla
from dream.config import InterfaceConfiguration, configuration
from dream.compressible.config import flowfields

if typing.TYPE_CHECKING:
    from .solver import CompressibleFlowSolver


class RiemannSolver(InterfaceConfiguration, is_interface=True):

    cfg: CompressibleFlowSolver

    def get_convective_stabilisation_matrix_hdg(self, U: flowfields, unit_vector: bla.VECTOR) -> bla.MATRIX:
        NotImplementedError()

    def get_convective_numerical_flux_dg(self, Ui: flowstate, Uj: flowstate, unit_vector: bla.VECTOR) -> ngs.CF:
        NotImplementedError()



class Upwind(RiemannSolver):

    name = "upwind"

    def get_convective_stabilisation_matrix_hdg(self, U: flowfields, unit_vector: bla.VECTOR) -> bla.MATRIX:
        r""" Returns the convective stabilisation matrix for the upwind scheme.
        
        .. math::
            \bm{\tau}_c := \bm{A}_n^+
        """
        unit_vector = bla.as_vector(unit_vector)
        return self.cfg.get_conservative_convective_jacobian(U, unit_vector, 'outgoing')

    def get_convective_numerical_flux_dg(self, Ui: flowstate, Uj: flowstate, unit_vector: bla.VECTOR) -> ngs.CF:
        raise ValueError("FVS solver in a standard DG has not been implemented (yet).")



class LaxFriedrich(RiemannSolver):

    name = "lax_friedrich"
    aliases = ('lf', )

    def get_convective_stabilisation_matrix_hdg(self, U: flowfields, unit_vector: bla.VECTOR) -> bla.MATRIX:
        r""" Returns the convective stabilisation matrix for the upwind scheme.
        
        .. math::
            \bm{\tau}_c := (|u_n| + c) \bm{I}

        .. [1] J. Vila-Pérez, M. Giacomini, R. Sevilla, and A. Huerta. “Hybridisable Discontinuous Galerkin
                Formulation of Compressible Flows”. In: Archives of Computational Methods in Engineering
                28.2 (Mar. 2021), pp. 753–784. doi: 10.1007/s11831-020-09508-z. arXiv: 2009.06396 [physics].
        """
        unit_vector = bla.as_vector(unit_vector)

        u = self.cfg.velocity(U)
        c = self.cfg.speed_of_sound(U)

        lambda_max = bla.abs(bla.inner(u, unit_vector)) + c
        return lambda_max * ngs.Id(unit_vector.dim + 2)

    def get_convective_numerical_flux_dg(self, Ui: flowstate, Uj: flowstate, unit_vector: bla.VECTOR) -> ngs.CF:
       
        # Extract the normal, taken w.r.t. the ith solution.
        n = bla.as_vector(unit_vector)

        # For now, hardcode the local lax friedrich (Rusanov) version.
        Fi = self.cfg.get_convective_flux(Ui) 
        Fj = self.cfg.get_convective_flux(Uj)

        # Compute the normal velocity.
        vni = bla.inner( self.cfg.velocity(Ui), n )
        vnj = bla.inner( self.cfg.velocity(Uj), n )

        # Extract the speed of sound.
        ci = self.cfg.speed_of_sound(Ui)
        cj = self.cfg.speed_of_sound(Uj)

        # Compute the largest eigenvalues.
        lmbi = bla.abs( vni ) + ci
        lmbj = bla.abs( vnj ) + cj

        # Deduce the largest eigenvalue w.r.t. the 2 solutions.
        lmb_max = bla.max( lmbi, lmbj )

        # Assemble the numerical flux.
        fs = 0.5*( (Fi + Fj)*n - lmb_max*(Uj.U - Ui.U) )
       
        # Return the numerical flux.
        return fs



class Roe(RiemannSolver):

    name = "roe"

    def get_convective_stabilisation_matrix_hdg(self, U: flowfields, unit_vector: bla.VECTOR) -> bla.MATRIX:
        r""" Returns the convective stabilisation matrix for the upwind scheme.
        
        .. math::
            \bm{\tau}_c := |A_n|

        .. [1] J. Vila-Pérez, M. Giacomini, R. Sevilla, and A. Huerta. “Hybridisable Discontinuous Galerkin
                Formulation of Compressible Flows”. In: Archives of Computational Methods in Engineering
                28.2 (Mar. 2021), pp. 753–784. doi: 10.1007/s11831-020-09508-z. arXiv: 2009.06396 [physics].
        """
        unit_vector = bla.as_vector(unit_vector)

        lambdas = self.cfg.characteristic_velocities(U, unit_vector, "absolute")
        return self.cfg.transform_characteristic_to_conservative(bla.diagonal(lambdas), U, unit_vector)


    def get_convective_numerical_flux_dg(self, Ui: flowstate, Uj: flowstate, unit_vector: bla.VECTOR) -> ngs.CF:
        raise ValueError("Roe solver in a standard DG has not been implemented (yet).")



class HLL(RiemannSolver):

    name = "hll"

    def get_convective_stabilisation_matrix_hdg(self, U: flowfields, unit_vector: bla.VECTOR) -> bla.MATRIX:
        r""" Returns the convective stabilisation matrix for the upwind scheme.
        
        .. math::
            \bm{\tau}_c := \max(u_n + c, 0) \bm{I}
            
        .. [1] J. Vila-Pérez, M. Giacomini, R. Sevilla, and A. Huerta. “Hybridisable Discontinuous Galerkin
                Formulation of Compressible Flows”. In: Archives of Computational Methods in Engineering
                28.2 (Mar. 2021), pp. 753–784. doi: 10.1007/s11831-020-09508-z. arXiv: 2009.06396 [physics].
        """
        unit_vector = bla.as_vector(unit_vector)

        u = self.cfg.velocity(U)
        c = self.cfg.speed_of_sound(U)

        un = bla.inner(u, unit_vector)
        s_plus = bla.max(un + c)

        return s_plus * ngs.Id(unit_vector.dim + 2)

    def get_convective_numerical_flux_dg(self, Ui: flowstate, Uj: flowstate, unit_vector: bla.VECTOR) -> ngs.CF:
        raise ValueError("HLL solver in a standard DG has not been implemented (yet).")



class HLLEM(RiemannSolver):

    name = "hllem"

    @configuration(default=1e-8)
    def theta_0(self, value):
        r""" Defines a threshold value used to stabilize contact waves, when the eigenvalue tends to zero.
        This can occur if the flow is parallel to the element or domain boundary!

        .. math::
            \theta_0

        .. [1] J. Vila-Pérez, M. Giacomini, R. Sevilla, and A. Huerta. “Hybridisable Discontinuous Galerkin
                Formulation of Compressible Flows”. In: Archives of Computational Methods in Engineering
                28.2 (Mar. 2021), pp. 753–784. doi: 10.1007/s11831-020-09508-z. arXiv: 2009.06396 [physics].
        """
        return float(value)
    
    theta_0: float

    def get_convective_stabilisation_matrix_hdg(self, U: flowfields, unit_vector: bla.VECTOR) -> ngs.CF:
        r""" Returns the convective stabilisation matrix for the upwind scheme.
        
        .. math::
            \begin{align*}
            \theta &:= \max\left(\frac{|u_n|}{|u_n| + c}, \theta_0\right), &
            \Theta &:=  \text{diag}(1, \theta, \ldots, \theta, 1), &
            \bm{\tau}_c &:= \bm{P} \bm{\Theta} \bm{P}^{-1}
            \end{align*}

        .. [1] J. Vila-Pérez, M. Giacomini, R. Sevilla, and A. Huerta. “Hybridisable Discontinuous Galerkin
                Formulation of Compressible Flows”. In: Archives of Computational Methods in Engineering
                28.2 (Mar. 2021), pp. 753–784. doi: 10.1007/s11831-020-09508-z. arXiv: 2009.06396 [physics].
        """
        unit_vector = bla.as_vector(unit_vector)

        u = self.cfg.velocity(U)
        c = self.cfg.speed_of_sound(U)

        un = bla.inner(u, unit_vector)
        un_abs = bla.abs(un)
        s_plus = bla.max(un + c)

        theta = bla.max(un_abs/(un_abs + c), self.theta_0)
        THETA = bla.diagonal([1] + unit_vector.dim * [theta] + [1])
        THETA = self.cfg.transform_characteristic_to_conservative(THETA, U, unit_vector)

        return s_plus * THETA

    def get_convective_numerical_flux_dg(self, Ui: flowstate, Uj: flowstate, unit_vector: bla.VECTOR) -> ngs.CF:
        raise ValueError("HLLEM solver in a standard DG has not been implemented (yet).")


