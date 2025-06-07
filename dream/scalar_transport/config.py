""" Definitions of boundary/domain conditions for a scalar transport equation. """
from __future__ import annotations
import ngsolve as ngs

from dream import bla
from dream.bla import is_scalar
from dream.config import quantity, dream_configuration, ngsdict
from dream.mesh import (Condition,
                        Periodic,
                        Initial,
                        Force,
                        Perturbation,
                        SpongeLayer,
                        PSpongeLayer,
                        GridDeformation)


class transportfields(ngsdict):
    """ Mutable mapping for flow quantities.

        Literal mathematical symbols as key names are converted to their respective quantities,
        if predefined. Values are converted to NGSolve CoefficientFunctions.

        >>> fields = transportfields(phi=1.0)
        >>> fields
        {'phi': CoefficientFunction((1.0))}
 
    """
    phi = quantity('phi', r"\phi")


class FarField(Condition):
    r""" Farfield condition for a scalar transport equation.

    The farfield condition imposes the value $\phi_{\infty}$ at the prescribed boundaries. This imposes the value, by taking a look at the characteristics which determine if this is an inlet or an outlet. Additionally, this also is imposed via the viscous fluxes, if diffusion is included as well.

    :note: See :func:`add_farfield_formulation <dream.scalar_transport.spatial.HDG.add_farfield_formulation>` for the implementation of the farfield condition in the HDG formulation. Similarly, :func:`add_farfield_formulation <dream.scalar_transport.spatial.DG.add_farfield_formulation>` for a standard DG formulation.

    """

    name = "farfield"
    
    def __init__(self, fields: transportfields | None = None):

        self.fields = fields

        super().__init__()


    


BCS = [Periodic, FarField]
DCS = [Initial]



