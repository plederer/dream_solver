from dream import *
from dream.scalar_transport import Initial, flowfields, ScalarTransportSolver, FarField
from ngsolve import *
from netgen.occ import OCCGeometry, WorkPlane
from netgen.meshing import IdentificationType
from ngsolve.meshes import MakeStructured2DMesh
from dream.io import DomainL2Sensor
import numpy as np 
from matplotlib import pyplot as plt

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


def get_analytic_solution(t, k):
   
    # Initial pulse location.
    x0 = 0.0
    y0 = 0.5
   
    # Pulse center trajectory.
    xc =  x0*cos(t) - y0*sin(t)
    yc = -x0*sin(t) + y0*cos(t)
    
    # Radial distance
    r2 = (x-xc)**2 + (y-yc)**2
    # Variance of this pulse.
    s2 = get_variance_pulse(t, k)

    # Generate and return the analytic solution.
    return (s2/pi) * exp( -r2/(4.0*k*t) )

def get_variance_pulse(t, k):
    return 1.0/(4.0*k*t) 


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
cfg.convection_velocity = (-y, x)
cfg.diffusion_coefficient = 1.0e-03
cfg.is_inviscid = False


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

t0 =   pi/2
tf = 5*pi/2

cfg.time = "transient"
cfg.fem.scheme = "implicit_euler"
#cfg.fem.scheme = "bdf2"
#cfg.fem.scheme = "sdirk22"
#cfg.fem.scheme = "sdirk33"
#cfg.fem.scheme = "imex_rk_ars443"
#cfg.fem.scheme = "explicit_euler"
#cfg.fem.scheme = "ssprk3"
#cfg.fem.scheme = "crk4"
cfg.time.timer.interval = (t0, tf)
cfg.time.timer.step = 0.02


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
U0 = flowfields()
U0.phi = get_analytic_solution(t0, cfg.diffusion_coefficient)


# # # # # # # # # # # # # # # # #
# Boundary and domain conditions.
# # # # # # # # # # # # # # # # #


Uic = flowfields()
Uic.phi = 0.0

cfg.bcs['left|top|bottom|right'] = FarField(fields=Uic)
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
Uex = flowfields()
Uex.phi = get_analytic_solution(cfg.time.timer.t, cfg.diffusion_coefficient)

fields = cfg.fem.get_fields()
fields["Exact[phi]"]  = Uex.phi
fields["Diff[phi]"]  = Uex.phi - fields["phi"] 


sensor = DomainL2Sensor( fields, mesh=mesh )
IO.sensor.add(sensor)


## VTK Visualization data.
#fields = cfg.fem.get_fields()
IO.vtk.fields = fields 


# # # # # # # # # # #
# Solve our problem.
# # # # # # # # # # #

## This passes all our configuration to NGSolve to solve.
#with TaskManager():
#    cfg.solve()
        

    
    #for t in cfg.time.start_solution_routine(True):
    #    print( "... time: ", t )
    #    # Step: 1
    #    alpha(t) = u(t)
    #    # Step: 2
    #    sigma(t) = f( u, alpha)


## TESTING
dt = cfg.time.timer.step.Get()
nt = int(round((tf - t0) / dt))

nOrder = 10

iTime = 0
#error = np.zeros( (nt,2), dtype=float )
error = np.zeros( nt, dtype=float )
time  = np.zeros( nt, dtype=float )
for t in cfg.time.start_solution_routine(True):
    
    #uh = get_analytic_solution(t, cfg.diffusion_coefficient)
    s2e = get_variance_pulse(t, cfg.diffusion_coefficient.Get())
    uh = cfg.fem.get_fields("phi").phi
    
    
    xh = ngs.Integrate( x*uh, mesh, order=nOrder )
    yh = ngs.Integrate( y*uh, mesh, order=nOrder )
    
    r2h = (x-xh)**2 + (y-yh)**2

    s2h = ngs.Integrate( r2h*uh, mesh, order=(2*nOrder) )
    
    
    time[iTime] = t
    error[iTime] = s2h*s2e - 1.0
    iTime += 1
    #print( "xh: ", xh )



# Final solution.
#plt.plot(error[:,0], error[:,1])
plt.plot(time, error)
plt.grid()
plt.show()



    
