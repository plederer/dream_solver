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




class flowfields(ngsdict):
    phi = quantity('scalar_quantity', r"\phi")


BCS = [Periodic]
DCS = [Perturbation, Initial, GridDeformation]



