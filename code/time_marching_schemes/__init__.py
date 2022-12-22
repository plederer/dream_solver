from configuration import TimeSchemes
from .base import TimeMarchingSchemes, ImplicitEuler, BDF2


def time_marching_factory(mesh, solver_configuration) -> TimeMarchingSchemes:

    time_scheme = solver_configuration.time_scheme

    if time_scheme is TimeSchemes.IE:
        return ImplicitEuler(solver_configuration)
    elif time_scheme is TimeSchemes.BDF2:
        return BDF2(solver_configuration)
