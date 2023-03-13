from dream import ResultsDirectoryTree, Loader, CompressibleHDGSolver, DreAmLogger
from ngsolve import *
import time
from pathlib import Path

tree = ResultsDirectoryTree("Re150_no_sponge_3", parent_path=Path("/media/jellmenr/DreAm"))
loader = Loader(tree=tree)
mesh = loader.load_mesh()
cfg = loader.load_configuration("transient_configuration_fine")

solver = CompressibleHDGSolver(mesh, cfg, tree)
gfu = GridFunction(solver.formulation.fes)
solver.gfu = gfu

cfg.time_step = 0.01
cfg.time_period = (800, 1000)
solver.draw_solutions()
Draw(solver.formulation.pressure(solver.gfu.components[0]) - 1/(0.3**2 * 1.4), mesh, "p'")
Draw(solver.formulation.velocity(solver.gfu.components[0]) - CF((1,0)), mesh, "u'")
input("Start Time Step:")
for t in cfg.time_period:
    loader.load_state(gfu, f"transient_{t:.2f}")
    Redraw()
    # time.sleep(0.01)