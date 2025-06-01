from dream import *
from dream.scalar_transport import Initial, transportfields, ScalarTransportSolver
from ngsolve import *
from ngsolve.meshes import Make1DMesh
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
nElem = 10

# Polynomial order of the FEM implementation.
nPoly = 4

# Dimension of the rectangular domain.
xlen = 5.0

# Grid starting and ending points.
x0 = 0.0
x1 = 2.0

# Generate a simple grid.
xcoor = np.linspace(x0, x1, nElem*nPoly, dtype=np.float64)
mesh = Make1DMesh(nElem, periodic=True, mapping=lambda x: x1*x + x0 )


# # # # # # # # # # # #
# Solver configuration.
# # # # # # # # # # # #

# Base configuration.
cfg = ScalarTransportSolver(mesh)

# Number of threads.
nThread = 4

# Number of subdivisions, for visualization.
nSubdiv = 3

# Set the number of threads.
SetNumThreads(nThread)


# # # # # # # # # # # #
# Physical parameters.
# # # # # # # # # # # #

cfg.convection_velocity = (1.0,)
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

t0 = 0.1
tf = 5.0

cfg.time = "transient"
TEMPORAL = cfg.time
#cfg.fem.scheme = "implicit_euler"
cfg.fem.scheme = "bdf2"
#cfg.fem.scheme = "sdirk22"
#cfg.fem.scheme = "sdirk33"
#cfg.fem.scheme = "imex_rk_ars443"
#cfg.fem.scheme = "explicit_euler"
#cfg.fem.scheme = "ssprk3"
#cfg.fem.scheme = "crk4"
TEMPORAL.timer.interval = (t0, tf)
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


# # # # # # # # # # #
# Initial conditions.
# # # # # # # # # # #

# Obtain the initial condition.
k = cfg.diffusion_coefficient

x0 = 0.0
y0 = 0.5

b = 1.0/(4.0*k)
a = b/pi

a0 = a/t0
b0 = b/t0

xc0 =  x0*cos(t0)

r2 = (x-xc0)**2

U0 = transportfields()
U0.phi = a0*exp( -r2*b0 )



# # # # # # # # # # # # # # # # #
# Boundary and domain conditions.
# # # # # # # # # # # # # # # # #

cfg.bcs['left|right'] = "periodic"
cfg.dcs['dom'] = Initial(fields=U0)


# # # # # # # # # # # #
# Output/Visualization.
# # # # # # # # # # # #

# Set the spaces and associated grid functions.
cfg.initialize()



# Visualization.

# Plot the solution over the grid.
gfu = cfg.fem.get_fields('phi')
uvar = gfu.phi( mesh(xcoor) )
data, = plt.plot(xcoor, uvar)
xmin = min(xcoor)
xmax = max(xcoor)
ymin = min(uvar)
ymax = max(uvar)
plt.ylim(ymin, ymax)
plt.xlim(xmin, xmax)
plt.grid()
plt.draw()




# # # # # # # # # # #
# Solve our problem.
# # # # # # # # # # #

# This passes all our configuration to NGSolve to solve.
#with TaskManager():
#    cfg.solve()

dt = TEMPORAL.timer.step.Get()
nt = (tf-t0)/dt

beta = np.zeros( (int(nt)+1,2), dtype=float )
alpha = np.zeros( (int(nt)+1,2), dtype=float )

iTime = 0
for t in cfg.time.start_solution_routine(True):

    # Update the current iteration count.
    iTime += 1
    
    # Plot the solution.
    if iTime%10 == 0:
        #print("simulation time: %f", t)
        gfu = cfg.fem.get_fields('phi')
        uvar = gfu.phi( mesh(xcoor) )
        data.set_data( xcoor, uvar ) 
        plt.pause(0.1)

    beta[iTime-1,0] = ngs.Integrate( gfu.phi, mesh )
    alpha[iTime-1,0] = ngs.Integrate( gfu.phi - beta[iTime-1,0], mesh )


print( beta, alpha )

# Final solution.
gfu = cfg.fem.get_fields('phi')
uvar = gfu.phi( mesh(xcoor) )
data.set_data( xcoor, uvar )
plt.show()




