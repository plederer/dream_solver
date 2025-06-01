from dream import *
from dream.scalar_transport import Initial, transportfields, ScalarTransportSolver
from ngsolve import *
from netgen.occ import OCCGeometry, WorkPlane
from netgen.meshing import IdentificationType
from ngsolve.meshes import MakeStructured2DMesh

# Needed to create simple grids.
from gridmaker import *

# Message output detail from netgen.
ngsglobals.msg_level = 0


# # # # # # # # # #
# Grid Information.
# # # # # # # # # #

# Number of elements per dimension.
nElem1D = 10

# Dimension of the rectangular domain.
xLength = 10.0
yLength = 10.0

# Generate a simple grid.
mesh = CreateSimpleGrid(nElem1D, xLength, yLength)


# # # # # # # # # # # #
# Solver configuration.
# # # # # # # # # # # #

# Base configuration.
cfg = ScalarTransportSolver(mesh)

# Number of threads.
nThread = 4

# Polynomial order of the FEM implementation.
nPoly = 4

# Number of subdivisions, for visualization.
nSubdiv = 3

# Set the number of threads.
SetNumThreads(nThread)


# # # # # # # # # # # #
# Physical parameters.
# # # # # # # # # # # #

cfg.convection_velocity = (1.0, 1.0)
cfg.diffusion_coefficient = 1.0e-03
cfg.is_inviscid = True


# # # # # # # # # # # # #
# Spatial discretization.
# # # # # # # # # # # # #

cfg.riemann_solver = "lax_friedrich"
cfg.fem = "hdg"
cfg.fem.order = nPoly
cfg.fem.interior_penalty_coefficient = 10.0


# # # # # # # # # # # # # #
# Temporal discretization.
# # # # # # # # # # # # # #

cfg.time = "transient"
TEMPORAL = cfg.time
cfg.fem.scheme = "implicit_euler"
#cfg.fem.scheme = "bdf2"
#cfg.fem.scheme = "sdirk22"
TEMPORAL.timer.interval = (0, 50.0)
TEMPORAL.timer.step = 0.05


# # # # # # # # # # #
# Solution strategy.
# # # # # # # # # # #

cfg.linear_solver = "pardiso"


# # # # # # # # # # #
# Optimization flags.
# # # # # # # # # # #

OPTIMIZATION = cfg.optimizations
OPTIMIZATION.static_condensation = True # NOTE, for now this cannot be true!
OPTIMIZATION.compile.realcompile = False


# # # # # # # # # # #
# Initial conditions.
# # # # # # # # # # #

# Obtain the initial condition.
a0 = 1.0
rv = 1.0
x0 = 5.0
y0 = 5.0
U0 = transportfields()
r = sqrt((x-x0)**2 + (y-y0)**2)
U0.phi = 1.0 * (1 + a0 * exp(-r**2/rv**2))


# # # # # # # # # # # # # # # # #
# Boundary and domain conditions.
# # # # # # # # # # # # # # # # #

# cfg.bcs['left|top|bottom|right'] = FarField(state=Uinf)
cfg.bcs['left|right'] = "periodic"
cfg.bcs['top|bottom'] = "periodic"
cfg.dcs['internal'] = Initial(fields=U0)


# # # # # # # # # # # #
# Output/Visualization.
# # # # # # # # # # # #

# Set the spaces and associated grid functions.
cfg.initialize()

# Write output VTK file.
IO = cfg.io
IO.vtk.rate = 10
IO.vtk.subdivision = nSubdiv
IO.log.level = 10


## VTK Visualization data.
fields = cfg.fem.get_fields()
IO.vtk.fields = fields 


# # # # # # # # # # #
# Solve our problem.
# # # # # # # # # # #

# This passes all our configuration to NGSolve to solve.
with TaskManager():
    cfg.solve()



