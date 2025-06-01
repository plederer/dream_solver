from .solver import ScalarTransportSolver 
from .spatial import ScalarTransportFiniteElementMethod 
from .config import (flowfields,
                     Periodic,
                     FarField,
                     Initial)

__all__ = ['ScalarTransportSolver',
           'flowfields',
           'Periodic',
           'FarField',
           'Initial']
