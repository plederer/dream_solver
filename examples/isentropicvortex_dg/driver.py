from dream import *
from dream.compressible import Initial, CompressibleFlowSolver
from ngsolve import *
from netgen.occ import OCCGeometry, WorkPlane
from netgen.meshing import IdentificationType
from ngsolve.meshes import MakeStructured2DMesh

# Needed to create simple grids.
from gridmaker import *
# Needed to initialize and verify the solution.
from vnv import *

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
cfg = CompressibleFlowSolver(mesh)

# Number of threads.
nThread = 4

# Polynomial order of the FEM implementation.
nPoly = 3

# Number of subdivisions, for visualization.
nSubdiv = 3


# Set the number of threads.
SetNumThreads(nThread)



# # # # # # # # # # # #
# Physical parameters.
# # # # # # # # # # # #

cfg.dynamic_viscosity = "inviscid"
cfg.equation_of_state = "ideal"
cfg.equation_of_state.heat_capacity_ratio = 1.4
cfg.scaling = "acoustic"


# # # # # # # # # # # # #
# Spatial discretization.
# # # # # # # # # # # # #

cfg.riemann_solver = "lax_friedrich"
cfg.fem = "conservative"
cfg.fem.order = nPoly
cfg.fem.method = "dg"
cfg.fem.mixed_method = "inactive"


# # # # # # # # # # # # # #
# Temporal discretization.
# # # # # # # # # # # # # #

cfg.time = "transient"
TEMPORAL = cfg.time
#TEMPORAL.scheme = "explicit_euler"
TEMPORAL.scheme = "ssprk3"
#TEMPORAL.scheme = "crk4"
TEMPORAL.timer.interval = (0, 5.0)
TEMPORAL.timer.step = 0.01


# # # # # # # # # # #
# Solution strategy.
# # # # # # # # # # #

cfg.linear_solver = "pardiso"


# # # # # # # # # # #
# Optimization flags.
# # # # # # # # # # #

OPTIMIZATION = cfg.optimizations
OPTIMIZATION.static_condensation = False
OPTIMIZATION.compile.realcompile = False

#OPTIMIZATION.bonus_int_order.vol = 10
#OPTIMIZATION.bonus_int_order.bnd = 10


# # # # # # # # # # #
# Initial conditions.
# # # # # # # # # # #

# Obtain the initial condition.
Uic = InitialCondition(cfg, TEMPORAL, xLength, yLength)

# Define the initial solution state.
initial = Initial(fields=Uic)


# # # # # # # # # # # # # # # # #
# Boundary and domain conditions.
# # # # # # # # # # # # # # # # #

# cfg.bcs['left|top|bottom|right'] = FarField(state=Uinf)
cfg.bcs['left|right'] = "periodic"
cfg.bcs['top|bottom'] = "periodic"
cfg.dcs['internal'] = initial


# # # # # # # # # # # #
# Output/Visualization.
# # # # # # # # # # # #

# Set the spaces and associated grid functions.
cfg.initialize()

# Get the analytic solution, function of time.
Uexact = AnalyticSolution(cfg, TEMPORAL.timer.t, xLength, yLength)

# Write output VTK file.
IO = cfg.io
IO.vtk = True
IO.vtk.rate = 100
IO.vtk.subdivision = nSubdiv

# VTK Visualization data.
ProcessVTKData(IO, cfg, Uexact)


# # # # # # # # # # #
# Solve our problem.
# # # # # # # # # # #

# This passes all our configuration to NGSolve to solve.
with TaskManager():
    cfg.solve()



