from dream import *
from dream.compressible import flowstate, FarField, Initial
from ngsolve import *
from netgen.occ import OCCGeometry, WorkPlane
from netgen.meshing import IdentificationType
from ngsolve.meshes import MakeStructured2DMesh

# DEBUGGING
#import ipdb

# Needed to create simple grids.
from gridmaker import *

# Message output detail from netgen.
ngsglobals.msg_level = 0



# # # # # # # # # #
# Grid Information.
# # # # # # # # # #

# Characteristic grid configuration.
typeElement  = "quadrilateral"
isCircle     = False
isStructured = True
isPeriodic   = True
maxElemSize  = 0.1

# Pack the grid data.
gridparam = [isCircle, isStructured, isPeriodic, maxElemSize, typeElement]

# Generate a simple grid.
mesh = CreateSimpleGrid(gridparam)



# # # # # # # # # # # # 
# Solver configuration.
# # # # # # # # # # # #

# DEBUGGING
# To run it in a debugger, e.g. pdb (or ipdb), compile with -m pdb (or -m ipdb).
# To start the debugger from a certain line, use: pdb.set_trace() or ipdb.set_trace().
#ipdb.set_trace()

# Base configuration.
cfg        =  SolverConfiguration(mesh)
cfg.pde    = "compressible"
cfg.time   = "transient"
cfg.solver = "nonlinear"

# Polynomial order of the FEM implementation.
nPoly      = 4

# Number of threads.
nThread    = 4

# Set the number of threads.
SetNumThreads(nThread)

# Abbreviations.
IO           = cfg.io
PDE          = cfg.pde
SOLVER       = cfg.solver
TEMPORAL     = cfg.time
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
TEMPORAL.timer.interval = (0, 1.0)
TEMPORAL.timer.step     =  0.025



# # # # # # # # # # #
# Solution strategy.
# # # # # # # # # # #

SOLVER.method                = "newton"
SOLVER.method.damping_factor =  1
SOLVER.inverse               = "direct"
SOLVER.inverse.solver        = "pardiso"
SOLVER.max_iterations        =  10
SOLVER.convergence_criterion =  1e-10



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
Uinf = PDE.get_farfield_state((1, 0))

Gamma   = 0.1
Rv      = 0.1
r       = sqrt(x**2 + y**2)
p_0     = Uinf.p * (1 + Gamma * exp(-r**2/Rv**2))
rho_0   = Uinf.rho * (1 + Gamma * exp(-r**2/Rv**2))
initial = Initial(state=flowstate(rho=rho_0, u=Uinf.u, p=p_0))



# # # # # # # # # # # # # # # # #
# Boundary and domain conditions.
# # # # # # # # # # # # # # # # #

# PDE.bcs['left|top|bottom|right'] = FarField(state=Uinf)
PDE.bcs['left|right']  = FarField(state=Uinf)
PDE.bcs['top|bottom']  = "periodic"
if isStructured:
    PDE.dcs['dom']     = initial
else:
    PDE.dcs['default'] = initial

# Curve the grid, if need be.
if CurvedBoundary(gridparam):
    mesh.Curve(nPoly)



# # # # # # # # # # # #
# Output/Visualization.
# # # # # # # # # # # #

# Setup Spaces and Gridfunctions
PDE.initialize_system()

# Solution visualization (native).
drawing       = PDE.get_drawing_state(p=True)
drawing["p'"] = drawing.p - Uinf.p
PDE.draw(autoscale=False, min=-1e-4, max=1e-4)

# Write output VTK file.
IO.save.vtk             = True
IO.save.vtk.subdivision = 1
fields                  = PDE.get_state(p=True)
IO.save.vtk.fields      = PDE.get_state(p=True)



# # # # # # # # # # #
# Solve our problem.
# # # # # # # # # # #

# This passes all our configuration to NGSolve to solve.
SOLVER.initialize()
with TaskManager():
    SOLVER.solve()


