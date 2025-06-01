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
    phi = quantity('phi', r"\phi")
    grad_phi = quantity('phi gradient', r"\nabla \phi")


class FarField(Condition):

    name = "farfield"
    
    def __init__(self, fields: flowfields | None = None):

        self.fields = fields

        super().__init__()


    


BCS = [Periodic, FarField]
DCS = [Perturbation, Initial, GridDeformation]



