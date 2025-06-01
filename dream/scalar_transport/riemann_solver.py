""" Definitions of riemann solvers for a scalar transport equation """
from __future__ import annotations
import typing
import ngsolve as ngs

from dream import bla
from dream.config import Configuration, dream_configuration

if typing.TYPE_CHECKING:
    from .solver import ScalarTransportSolver


class RiemannSolver(Configuration, is_interface=True):

    root: ScalarTransportSolver 

    def get_convective_stabilisation_hdg(self, wind: ngs.CF, unit_vector: bla.VECTOR) -> bla.SCALAR:
        r""" Interface for computing an HDG stabilization value. This method must be implemented by subclasses!
        
        See :class:`LaxFriedrich` for concrete implementation.
        
        """
        raise NotImplementedError()

    def get_convective_numerical_flux_dg(self, Ui: transportfields, Uj: transportfields, wind: bla.VECTOR, unit_vector: bla.VECTOR) -> bla.SCALAR:
        r""" Interface for computing an DG numerical flux. This method must be implemented by subclasses!
        
        See :class:`LaxFriedrich` for concrete implementation.
        
        """
        raise NotImplementedError()

class LaxFriedrich(RiemannSolver):
    """ Lax-Friedrich scheme for the convective flux. """

    name = "lax_friedrich"

    def get_convective_stabilisation_hdg(self, wind: ngs.CF, unit_vector: bla.VECTOR) -> bla.SCALAR:
        r""" Returns the convective stabilization value, which in this case is the absolute of the wind in the normal direction.

        .. math::
            \tau_c := |\bm{b} \cdot \bm{n}|
        """
        unit_vector = bla.as_vector(unit_vector)
        return bla.abs( bla.inner(wind, unit_vector) )

    def get_convective_numerical_flux_dg(self, Ui: transportfields, Uj: transportfields, wind: bla.VECTOR, unit_vector: bla.VECTOR) -> bla.SCALAR:
        r""" Returns the convective numerical flux on a surface, which for this linear and scalar equation is identical to standard upwinding.

        .. math::
            f^* := \frac{1}{2} \Big( \bm{f}(\phi_i) + \bm{f}(\phi_j) \Big) \cdot \bm{n} - |\lambda| ( \phi_j - \phi_i ),

        where, :math:`\phi_i` and :math:`\phi_j` correspond to the local solution and its neighboring solution, respectively. Note, for this scalar and linear equation: :math:`\lambda = \bm{b} \cdot \bm{n}`
        """
        n = bla.as_vector(unit_vector)

        Fi = self.root.get_convective_flux(Ui)
        Fj = self.root.get_convective_flux(Uj)

        lmb_max = bla.abs( bla.inner(wind, n) )
        
        fs = 0.5*( (Fi+Fj)*n - lmb_max*(Uj.phi - Ui.U) )

        return fs


