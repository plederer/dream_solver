from __future__ import annotations
from .interface import CompressibleFormulations, MixedMethods, _Formulation, RiemannSolver, Scaling, FEM
from .conservative import ConservativeFormulation2D
from .primitive import PrimitiveFormulation2D


def formulation_factory(mesh, solver_configuration) -> _Formulation:

    if mesh.dim == 2:

        if solver_configuration.formulation is CompressibleFormulations.CONSERVATIVE:
            return ConservativeFormulation2D(mesh, solver_configuration)
        elif solver_configuration.formulation is CompressibleFormulations.PRIMITIVE:
            return PrimitiveFormulation2D(mesh, solver_configuration)
