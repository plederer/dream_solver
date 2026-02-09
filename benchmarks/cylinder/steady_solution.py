from config import (PSEUDO_STATIONARY, 
                    single_transient_routine, 
                    Cylinder)
from dream.io import IOConfiguration
import ngsolve as ngs
from dream.compressible import CompressibleFlowSolver

ngs.SetNumThreads(4)

io = IOConfiguration(None)
io.path = "32x32_drmin0.04_curved"
mesh = io.ngsmesh.load_routine()

HDG = PSEUDO_STATIONARY.copy()
HDG['fem'] = 'conservative_hdg'
HDG['fem.scheme.compile'] = True
HDG['io.vtk.enable'] = True
HDG['io.vtk.rate'] = 1
HDG['io.vtk.subdivision'] = 2
HDG['io.gfu.enable'] = True
HDG['time.timer.interval'] = (0.0, 150.0)
HDG['time.timer.step'] = 0.1
HDG['time.max_time_step'] = 0.1
HDG['fem.solver.method.max_iterations'] = 200

HDG['io.gfu.rate'] = 1
HDG['io.path'] = io.path.joinpath("steady_solution")

cfg = CompressibleFlowSolver(mesh)
simulation = Cylinder(HDG, filename="steady")
simulation.set_conditions(cfg)

mesh.Curve(cfg.fem.order)
single_transient_routine(simulation)
