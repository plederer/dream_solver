from dream import ResultsDirectoryTree, Loader, CompressibleHDGSolver
from ngsolve import *

tree = ResultsDirectoryTree()
loader = Loader(tree=tree)
mesh = loader.load_mesh()
cfg = loader.load_configuration("transient_configuration_fine")

solver = CompressibleHDGSolver(mesh, cfg, tree)
solver.formulation.initialize()

loader = solver.get_loader(tree)

solver.drawer.draw()
solver.drawer.draw_acoustic_pressure(1/(cfg.Mach_number**2 * cfg.heat_capacity_ratio), sd=3, autoscale=False, min=-1e-2, max=1e-2)

for t in loader.load_state_time_sequence("transient", sleep_time=0.01, load_step=2):
    # Do whatever here
    print(t)