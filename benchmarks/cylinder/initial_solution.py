import ngsolve as ngs
from dream.compressible import CompressibleFlowSolver
from dream.io import IOConfiguration
from config import (TRANSIENT_CFG,
                    single_transient_routine,
                    Cylinder)

ngs.SetNumThreads(4)

io = IOConfiguration(None)
io.path = "32x32_drmin0.04"
io.ngsmesh.path = io.path.joinpath("meshes")
io.ngsmesh.filename = "gmesh_curved"
mesh = io.ngsmesh.load_routine()

HDG = TRANSIENT_CFG.copy()
HDG['time.timer.interval'] = (0.0, 50.0)
HDG['time.timer.step'] = 0.1
HDG['fem'] = 'conservative_hdg'
HDG['fem.scheme'] = 'sdirk33'
HDG['fem.scheme.compile'] = True
HDG['fem.viscous_treatment'] = 'mixed_strain_temperature_gradient'

HDG['io.sensor.enable'] = True
HDG['io.vtk.enable'] = True
HDG['io.vtk.rate'] = 1
HDG['io.vtk.subdivision'] = 2
HDG['io.gfu.enable'] = True
HDG['io.gfu.rate'] = 1
HDG['io.gfu.time_level_rate'] = 1
HDG['io.path'] = io.path.joinpath("initial_solution")

cfg = CompressibleFlowSolver(mesh)
simulation = Cylinder(HDG, filename="initial")
simulation.set_conditions(cfg)

initial_cfg = CompressibleFlowSolver(mesh)
simulation_initial = Cylinder(HDG, filename="initial")
simulation_initial.set_conditions(initial_cfg)
initial_cfg.io.path = io.path.joinpath("steady_solution")
initial_cfg.io.gfu.filename = f"steady"

mesh.Curve(cfg.fem.order)
single_transient_routine(simulation, initial_cfg=initial_cfg)
