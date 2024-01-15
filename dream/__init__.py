import ngsolve as ngs
ngs.Parameter.__repr__ = lambda self: str(self.Get())
# from .solver import CompressibleHDGSolver
# from .configuration import SolverConfiguration

# from .io import *
# from .sensor import PointSensor, BoundarySensor
# from .mesh import *
# from .io import DreAmLogger, ResultsDirectoryTree
# from .bla.state import State


# __all__ = [
#     'CompressibleHDGSolver',
#     'SolverConfiguration',
#     'Loader',
#     'Saver',
#     'ResultsDirectoryTree',
#     'PointSensor',
#     'BoundarySensor',
#     'State',
#     'bcs',
#     'dcs',
#     'State',
#     'BufferCoordinate',
#     'SpongeFunction',
#     'SpongeWeight',
#     'GridDeformationFunction',
#     'LOGGER',
#     'State'
# ]
