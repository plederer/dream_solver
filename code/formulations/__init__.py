from __future__ import annotations
from .base import Formulation
from .conservative2d import ConservativeFormulation2D
from configuration import CompressibleFormulations


def formulation_factory(mesh, solver_configuration) -> Formulation:

    if mesh.dim == 2:

        if solver_configuration.formulation is CompressibleFormulations.CONSERVATIVE:
            return ConservativeFormulation2D(mesh, solver_configuration)
