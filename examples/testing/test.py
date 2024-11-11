from dream import *
from dream.compressible import flowstate, FarField, Initial
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

# Characteristic grid configuration.
isCircle     = False
isStructured = False
isPeriodic   = True
maxElemSize  = 0.1

# Pack the grid data.
gridparam = [isCircle, isStructured, isPeriodic, maxElemSize]

# Generate a simple grid.
mesh = CreateSimpleGrid(gridparam)


# # # # # # # # # # # # 
# Solver configuration.
# # # # # # # # # # # #

# Base configuration.
cfg        =  SolverConfiguration(mesh)
cfg.pde    = "compressible"
cfg.time   = "transient"
cfg.solver = "nonlinear"

# Polynomial order of the FEM implementation.
nPoly      = 4

# Number of threads.
nThread    = 1


# Set the number of threads.
SetNumThreads(nThread)

# Abbreviations.
PDE          = cfg.pde
TEMPORAL     = cfg.time
SOLVER       = cfg.solver
OPTIMIZATION = cfg.optimizations


# # # # # # # # # # # #
# Physical parameters.
# # # # # # # # # # # # 

PDE.dynamic_viscosity                     = "inviscid"
PDE.equation_of_state                     = "ideal"
PDE.equation_of_state.heat_capacity_ratio =  1.4
PDE.scaling                               = "acoustic"
PDE.mach_number                           =  0.03


# # # # # # # # # # # # #
# Spatial discretization.
# # # # # # # # # # # # #

PDE.riemann_solver   = "lax_friedrich"
PDE.fem              = "conservative"
PDE.fem.order        =  nPoly
PDE.fem.method       = "hdg"
PDE.fem.mixed_method = "inactive"


# # # # # # # # # # # # # #
# Temporal discretization. 
# # # # # # # # # # # # # #

TEMPORAL.scheme         = "implicit_euler"
TEMPORAL.timer.interval = (0, 0.1)
TEMPORAL.timer.step     =  0.1


# # # # # # # # # # #
# Solution strategy.
# # # # # # # # # # #

SOLVER.method                = "newton"
SOLVER.method.damping_factor = 1
SOLVER.inverse               = "direct"
SOLVER.inverse.solver        = "pardiso"
SOLVER.max_iterations        = 10
SOLVER.convergence_criterion = 1e-10


# # # # # # # # # # #
# Optimization flags.
# # # # # # # # # # #

OPTIMIZATION.static_condensation = True
OPTIMIZATION.compile.realcompile = False
OPTIMIZATION.bonus_int_order     = {'vol': 4, 'bnd': 4}


# # # # # # # # # # #
# Initial conditions.
# # # # # # # # # # #

# Velocity values (farfield). 
Uinf = cfg.pde.get_farfield_state((1, 0))

Gamma   = 0.1
Rv      = 0.1
r       = sqrt(x**2 + y**2)
p_0     = Uinf.p * (1 + Gamma * exp(-r**2/Rv**2))
rho_0   = Uinf.rho * (1 + Gamma * exp(-r**2/Rv**2))
initial = Initial(state=flowstate(rho=rho_0, u=Uinf.u, p=p_0))


# # # # # # # # # # # # # # # # #
# Boundary and domain conditions.
# # # # # # # # # # # # # # # # #

# cfg.pde.bcs['left|top|bottom|right'] = FarField(state=Uinf)
cfg.pde.bcs['left|right'] = FarField(state=Uinf)
cfg.pde.bcs['top|bottom'] = "periodic"
cfg.pde.dcs['default']    = initial

# Curve the grid, if need be.
if CurvedBoundary(gridparam):
    mesh.Curve(nPoly)


# # # # # # # # # # # #
# Output/Visualization.
# # # # # # # # # # # #

# Setup Spaces and Gridfunctions
cfg.pde.initialize_system()

drawing = cfg.pde.get_drawing_state(p=True)
drawing["p'"] = drawing.p - Uinf.p
cfg.pde.draw(autoscale=False, min=-1e-4, max=1e-4)

#cfg.io.save.vtk = True
#fields = cfg.pde.get_state(p=True)
#cfg.io.save.vtk.fields = cfg.pde.get_state(p=True)


# # # # # # # # # # #
# Solve our problem.
# # # # # # # # # # #

# This passes all our configuration to NGSolve to solve.
cfg.solver.initialize()
with TaskManager():
    cfg.solver.solve()



