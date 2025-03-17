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
nElem1D = 1

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
cfg.mach_number = 0.5


# # # # # # # # # # # # #
# Spatial discretization.
# # # # # # # # # # # # #

#cfg.riemann_solver = "lax_friedrich"
cfg.riemann_solver = "upwind"
cfg.fem = "conservative"
cfg.fem.order = nPoly
cfg.fem.method = "hdg"
cfg.fem.mixed_method = "inactive"


# # # # # # # # # # # # # #
# Temporal discretization.
# # # # # # # # # # # # # #

cfg.time = "transient"
TEMPORAL = cfg.time
TEMPORAL.scheme = "dirk34_ldd"
#TEMPORAL.scheme = "dirk43_wso2"
#TEMPORAL.scheme = "sdirk22"
#TEMPORAL.scheme = "sdirk54"
#TEMPORAL.scheme = "sdirk33"
#TEMPORAL.scheme = "bdf2"
#TEMPORAL.scheme = "implicit_euler"
TEMPORAL.timer.interval = (0, 1.0)
TEMPORAL.timer.step = 0.5


# # # # # # # # # # #
# Solution strategy.
# # # # # # # # # # #

cfg.linear_solver = "pardiso"
cfg.nonlinear_solver = "pardiso"
cfg.nonlinear_solver.method = "newton"
cfg.nonlinear_solver.method.damping_factor = 1
cfg.nonlinear_solver.max_iterations = 10
cfg.nonlinear_solver.convergence_criterion = 1e-10


# # # # # # # # # # #
# Optimization flags.
# # # # # # # # # # #

OPTIMIZATION = cfg.optimizations
OPTIMIZATION.static_condensation = True
OPTIMIZATION.compile.realcompile = False


# # # # # # # # # # #
# Initial conditions.
# # # # # # # # # # #

# Obtain the initial condition.
Uic = InitialCondition(cfg, TEMPORAL, xLength, yLength)


# # # # # # # # # # # # # # # # #
# Boundary and domain conditions.
# # # # # # # # # # # # # # # # #

# cfg.bcs['left|top|bottom|right'] = FarField(state=Uinf)
cfg.bcs['left|right'] = "periodic"
cfg.bcs['top|bottom'] = "periodic"
cfg.dcs['internal'] = Initial(fields=Uic)

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
IO.vtk.rate = 1
IO.vtk.subdivision = nSubdiv

# VTK Visualization data.
ProcessVTKData(IO, cfg, Uexact)


# # # # # # # # # # #
# Solve our problem.
# # # # # # # # # # #

# This passes all our configuration to NGSolve to solve.
with TaskManager():
    cfg.solve()
