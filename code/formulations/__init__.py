from __future__ import annotations
from enum import Enum
from configuration import
from .base import Formulation
from .conservative2d import ConservativeFormulation2D


class CompressibleFormulations(Enum):
    PRIMITIVE = "primitive"
    CONSERVATIVE = "conservative"


class MixedMethods(Enum):
    NONE = None
    GRADIENT = "gradient"
    STRAIN_HEAT = "strain_heat"


def formulation_factory(mesh, solver_configuration) -> Formulation:

    if mesh.dim == 2:

        if solver_configuration.formulation is CompressibleFormulations.CONSERVATIVE:
            return ConservativeFormulation2D(mesh, solver_configuration)
