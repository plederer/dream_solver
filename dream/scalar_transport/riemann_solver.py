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
        unit_vector = bla.as_vector(unit_vector)

        return bla.abs( bla.inner(wind, unit_vector) )


    def get_convective_numerical_flux_dg(self, Ui: flowfields, Uj: flowfields, unit_vector: bla.VECTOR) -> bla.SCALAR:
        NotImplementedError()


class Upwind(RiemannSolver):

    name = "upwind"

    def get_convective_numerical_flux_dg(self, Ui: flowfields, Uj: flowfields, unit_vector: bla.VECTOR) -> bla.SCALAR:
        raise ValueError("upwind solver in a standard DG has not been implemented (yet).")


class LaxFriedrich(RiemannSolver):

    name = "lax_friedrich"

    def get_convective_numerical_flux_dg(self, Ui: flowfields, Uj: flowfields, unit_vector: bla.VECTOR) -> bla.SCALAR:
        
        # Return the numerical flux.
        return fs


