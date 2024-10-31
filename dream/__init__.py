import ngsolve as ngs
import logging
from .solver import SolverConfiguration

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
logger.handlers[0].setFormatter(logging.Formatter("%(name)-15s (%(levelname)s) | %(message)s"))
if logger.level == logging.DEBUG:
    logger.handlers[0].setFormatter(logging.Formatter(
        "%(name)-15s (%(levelname)8s) | %(message)s (%(filename)s:%(lineno)s)", "%Y-%m-%d %H:%M:%S"))
__all__ = [
    'SolverConfiguration',
    'logger'
]

ngs.Parameter.__repr__ = lambda self: str(self.Get())
