from dream import *
from dream.scalar_transport import Initial, flowfields, ScalarTransportSolver
from ngsolve import *
from netgen.occ import OCCGeometry, WorkPlane
from netgen.meshing import IdentificationType
from ngsolve.meshes import MakeStructured2DMesh
from dream.io import DomainL2Sensor

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
xLength = 2.0
yLength = 2.0

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

cfg.convection_velocity = (1, 1)
cfg.diffusion_coefficient = 1.0e-03
cfg.is_inviscid = True


# # # # # # # # # # # # #
# Spatial discretization.
# # # # # # # # # # # # #

cfg.riemann_solver = "lax_friedrich"
cfg.fem = "dg"
cfg.fem.order = nPoly
cfg.fem.interior_penalty_coefficient = 10.0


# # # # # # # # # # # # # #
# Temporal discretization.
# # # # # # # # # # # # # #

t0 = pi/2
tf = 5*pi/2

cfg.time = "transient"
TEMPORAL = cfg.time
#cfg.fem.scheme = "implicit_euler"
#cfg.fem.scheme = "bdf2"
#cfg.fem.scheme = "sdirk22"
#cfg.fem.scheme = "sdirk33"
#cfg.fem.scheme = "imex_rk_ars443"
cfg.fem.scheme = "explicit_euler"
#cfg.fem.scheme = "ssprk3"
#cfg.fem.scheme = "crk4"
TEMPORAL.timer.interval = (t0, tf)
TEMPORAL.timer.step = 0.0025


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


# # # # # # # # # # #
# Initial conditions.
# # # # # # # # # # #

# Obtain the initial condition.
k = cfg.diffusion_coefficient

x0 = 0.0
y0 = 0.0

b = 1.0/(4.0*k)
a = b/pi

a0 = a/t0
b0 = b/t0

xc0 =  x0*cos(t0) - y0*sin(t0)
yc0 = -x0*sin(t0) + y0*cos(t0)

r2 = (x-xc0)**2 + (y-yc0)**2

U0 = flowfields()
U0.phi = a0*exp( -r2*b0 )



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


# # # # # # # # # # #
# Analytic Solution. 
# # # # # # # # # # #
t = TEMPORAL.timer.t

at = a/t 
bt = b/t 

xc =  x0*cos(t) - y0*sin(t)
yc = -x0*sin(t) + y0*cos(t)

r2 = (x-xc)**2 + (y-yc)**2

#Uex = flowfields()
#Uex.phi = at*exp( -r2*bt )
#fields = cfg.fem.get_fields()
#fields["Exact[phi]"]  = Uex.phi
#fields["Diff[phi]"]  = Uex.phi - fields["phi"] 
#fields = cfg.fem.get_fields()
#fields["phi"] = 


#sensor = DomainL2Sensor( fields, mesh=mesh )
#IO.sensor.add(sensor)


## VTK Visualization data.
fields = cfg.fem.get_fields()
IO.vtk.fields = fields 


# # # # # # # # # # #
# Solve our problem.
# # # # # # # # # # #

# This passes all our configuration to NGSolve to solve.
with TaskManager():
    cfg.solve()



