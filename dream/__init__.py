import ngsolve as ngs
from .solver import SolverConfiguration


__all__ = [
    'SolverConfiguration',
]

ngs.Parameter.__repr__ = lambda self: str(self.Get())
