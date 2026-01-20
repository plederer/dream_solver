import ngsolve as ngs
from dream.compressible import CompressibleFlowSolver
from dream.io import IOConfiguration
from config import (TRANSIENT_CFG,
                    single_transient_routine,
                    Cylinder)

ngs.SetNumThreads(4)

io = IOConfiguration(None)
io.path = "meshes/32x32_drmin0.04"
io.ngsmesh.filename = "gmesh"
mesh = io.ngsmesh.load_routine()

steady = {
    'filename': 'steady_2',
    'filepath': io.path.joinpath('steady_solution/states')
}

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

mesh.Curve(cfg.fem.order)
single_transient_routine(simulation, initial=steady)