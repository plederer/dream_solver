from dream import *
from dream.compressible import flowstate, Initial
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
cfg = SolverConfiguration(mesh)
cfg.pde = "compressible"
cfg.time = "transient"
cfg.solver = "nonlinear"

# Number of threads.
nThread = 4

# Polynomial order of the FEM implementation.
nPoly = 3

# Number of subdivisions, for visualization.
nSubdiv = 3


# Set the number of threads.
SetNumThreads(nThread)

# Abbreviations.
IO = cfg.io
PDE = cfg.pde
SOLVER = cfg.solver
TEMPORAL = cfg.time
OPTIMIZATION = cfg.optimizations


# # # # # # # # # # # #
# Physical parameters.
# # # # # # # # # # # #

PDE.dynamic_viscosity = "inviscid"
PDE.equation_of_state = "ideal"
PDE.equation_of_state.heat_capacity_ratio = 1.4
PDE.scaling = "acoustic"
PDE.mach_number = 1.0


# # # # # # # # # # # # #
# Spatial discretization.
# # # # # # # # # # # # #

PDE.riemann_solver = "lax_friedrich"
PDE.fem = "conservative"
PDE.fem.order = nPoly
PDE.fem.method = "hdg"
PDE.fem.mixed_method = "inactive"


# # # # # # # # # # # # # #
# Temporal discretization.
# # # # # # # # # # # # # #

TEMPORAL.scheme = "implicit_euler"
TEMPORAL.timer.interval = (0, 1.0)
TEMPORAL.timer.step = 0.01


# # # # # # # # # # #
# Solution strategy.
# # # # # # # # # # #

SOLVER.method = "newton"
SOLVER.method.damping_factor = 1
SOLVER.inverse = "direct"
SOLVER.inverse.solver = "pardiso"
SOLVER.max_iterations = 10
SOLVER.convergence_criterion = 1e-10


# # # # # # # # # # #
# Optimization flags.
# # # # # # # # # # #

OPTIMIZATION.static_condensation = True
OPTIMIZATION.compile.realcompile = False


# # # # # # # # # # #
# Initial conditions.
# # # # # # # # # # #

# Obtain the initial condition.
Uic = InitialCondition(PDE, TEMPORAL, xLength, yLength)

# Define the initial solution state.
initial = Initial(state=Uic)


# # # # # # # # # # # # # # # # #
# Boundary and domain conditions.
# # # # # # # # # # # # # # # # #

# PDE.bcs['left|top|bottom|right'] = FarField(state=Uinf)
PDE.bcs['left|right'] = "periodic"
PDE.bcs['top|bottom'] = "periodic"
PDE.dcs['internal'] = initial


# # # # # # # # # # # #
# Output/Visualization.
# # # # # # # # # # # #

# Set the spaces and associated grid functions.
PDE.initialize_system()

# Get the analytic solution, function of time.
Uexact = AnalyticSolution(PDE, TEMPORAL.timer.t, xLength, yLength)

# Write output VTK file.
IO.vtk = True
IO.vtk.rate = 100
IO.vtk.subdivision = nSubdiv

# VTK Visualization data.
ProcessVTKData(IO, PDE, Uexact)


# # # # # # # # # # #
# Solve our problem.
# # # # # # # # # # #

# This passes all our configuration to NGSolve to solve.
SOLVER.initialize()
with TaskManager():
    SOLVER.solve()
