from .solver import IncompressibleFlowSolver
from .config import (flowfields,
                     Inflow,
                     Outflow,
                     Wall,
                     Periodic,
                     Initial,
                     Force)

__all__ = ['IncompressibleFlowSolver',
           'flowfields',
           'Inflow',
           'Outflow',
           'Wall',
           'Periodic',
           'Initial',
           'Force']
