import ngsolve as ngs
ngs.Parameter.__repr__ = lambda self: str(self.Get())
