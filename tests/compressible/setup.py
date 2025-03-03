from ngsolve import Mesh, unit_square
from dream.compressible import CompressibleFlowSolver

mesh = Mesh(unit_square.GenerateMesh(maxh=1))
mip = mesh(0.5, 0.5)

cfg = CompressibleFlowSolver(mesh)
cfg.fem= "conservative"
cfg.fem.order= 2
cfg.fem.method= "hdg"
cfg.fem.mixed_method= "inactive"
cfg.mach_number= 0.3
cfg.equation_of_state= "ideal"
cfg.equation_of_state.heat_capacity_ratio= 1.4
cfg.dynamic_viscosity= "inviscid"
cfg.scaling= "aerodynamic"
cfg.scaling.dimensionful_values= {'length': 1.0, 'density': 1.293, 'velocity': 1.0, 'speed_of_sound': 343.0, 'temperature': 293.15, 'pressure': 101325.0}
cfg.riemann_solver= "lax_friedrich"