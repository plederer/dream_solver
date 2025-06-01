from __future__ import annotations
import typing
import ngsolve as ngs

from dream import bla
from dream.config import Configuration, dream_configuration

if typing.TYPE_CHECKING:
    from .solver import ScalarTransportSolver


class RiemannSolver(Configuration, is_interface=True):

    root: ScalarTransportSolver 

    def get_convective_stabilisation_matrix_hdg(self, wind: ngs.CF, unit_vector: bla.VECTOR) -> bla.SCALAR:
        NotImplementedError()

    def get_convective_numerical_flux_dg(self, Ui: flowfields, Uj: flowfields, wind: bla.VECTOR, unit_vector: bla.VECTOR) -> bla.SCALAR:
        NotImplementedError()

class LaxFriedrich(RiemannSolver):

    name = "lax_friedrich"

    def get_convective_stabilisation_hdg(self, wind: ngs.CF, unit_vector: bla.VECTOR) -> bla.SCALAR:
        unit_vector = bla.as_vector(unit_vector)
        return bla.abs( bla.inner(wind, unit_vector) )

    def get_convective_numerical_flux_dg(self, Ui: flowfields, Uj: flowfields, wind: bla.VECTOR, unit_vector: bla.VECTOR) -> bla.SCALAR:
        
        n = bla.as_vector(unit_vector)

        Fi = self.root.get_convective_flux(Ui)
        Fj = self.root.get_convective_flux(Uj)

        lmb_max = bla.abs( bla.inner(wind, n) )
        
        fs = 0.5*( (Fi+Fj)*n - lmb_max*(Uj.phi - Ui.U) )

        return fs


