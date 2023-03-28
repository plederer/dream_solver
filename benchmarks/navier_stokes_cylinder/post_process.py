from dream import ResultsDirectoryTree, Loader, CompressibleHDGSolver, DreAmLogger
from ngsolve import *
import time
from pathlib import Path

tree = ResultsDirectoryTree("Re150_no_sponge_2")
loader = Loader(tree=tree)
mesh = loader.load_mesh()
cfg = loader.load_configuration("transient_configuration_fine")

solver = CompressibleHDGSolver(mesh, cfg, tree)
solver.setup()
loader = solver.get_loader()

cfg.time_step = 1
cfg.time_period = (10, 30)

solver.drawer.draw()
solver.drawer.draw_acoustic_pressure(1/(0.3**2 * 1.4))
solver.drawer.draw_particle_velocity(CF((1, 0)))

input("Start Time Step:")
for t in loader.load_state_time_sequence("transient", sleep_time=0, load_step=10):
    print(t)