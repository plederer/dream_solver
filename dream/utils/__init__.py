from .formatter import Formatter
from .geometries import Rectangle1DGrid, RectangleDomain, RectangleGrid, CircularDomain, CircularGrid


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True
        elif shell == 'TerminalInteractiveShell':
            return False
        else:
            return False
    except NameError:
        return False
