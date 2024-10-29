from ngsolve import Mesh, unit_square
from dream.solver import SolverConfiguration

mesh = Mesh(unit_square.GenerateMesh(maxh=1))
mip = mesh(0.5, 0.5)

cfg = SolverConfiguration(mesh)
cfg.pde= "compressible"
cfg.pde.fe= "conservative"
cfg.pde.fe.order= 2
cfg.pde.fe.method= "hdg"
cfg.pde.fe.mixed_method= "inactive"
cfg.pde.mach_number= 0.3
cfg.pde.equation_of_state= "ideal"
cfg.pde.equation_of_state.heat_capacity_ratio= 1.4
cfg.pde.dynamic_viscosity= "inviscid"
cfg.pde.scaling= "aerodynamic"
cfg.pde.scaling.reference_values= {'length': 1.0, 'density': 1.293, 'velocity': 1.0, 'speed_of_sound': 343.0, 'temperature': 293.15, 'pressure': 101325.0}
cfg.pde.riemann_solver= "lax_friedrich"