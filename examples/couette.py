from netgen.geom2d import unit_square, SplineGeometry
from ngsolve import *
from ngsolve.internal import visoptions, viewoptions
import math
import sys
sys.path.insert(1, '../code')
sys.path.insert(1, '../utils')

from HDGSolver import compressibleHDGsolver

ngsglobals.msg_level = 0

H = 1
U = 1
T0 = 0.8
T1 = 0.85
Pr = 0.72
Minf = 0.15
gamma = 1.4 
pinf = 1/(gamma * Minf**2)
mu = 1
Re = 1

print("pinf = ", pinf)
order = 2

#################################################################################
# Geometry, exact solution and boundary solution
geo = SplineGeometry()
geo.AddRectangle((0,0),(H,H), bcs=["left","bottom","right","top"])
mesh = Mesh(geo.GenerateMesh(maxh=0.1))

ybar = y/H
xbar = x/H

u_ex = U * ybar * log(1 + ybar)
v_ex = 0 #U * xbar * log(1 + xbar)

vel_ex = CoefficientFunction((u_ex, v_ex))

# if False:
#     T_ex = 1 / ((gamma - 1) * Minf**2) * (T0 + ybar * (T1-T0) + (gamma-1) * Minf**2/2 * Pr * ybar * (1-ybar))
#     rho_ex = 1/T_ex
# else:
#     # solve for tilde T

T_ex = (T0 + ybar * (T1-T0) + (gamma-1) * Minf**2/2 * Pr * ybar * (1-ybar))
rho_ex = 1/T_ex

#we use the non-dimensional relation T = gamma Minf**2 * p / rho
p_ex = 1/(gamma * Minf**2)

E_ex = p_ex/(rho_ex * (gamma-1)) + 0.5 * (u_ex**2 + v_ex**2)

Draw(vel_ex, mesh, 'vel_ex')
Draw(p_ex, mesh, 'p_ex')
Draw(T_ex, mesh, 'T_ex')
Draw(rho_ex, mesh, 'rho_ex')
Draw(E_ex, mesh, 'E_ex')

force = CoefficientFunction((0,-1/Re * (ybar +2)/(ybar +1)**2,0, -1/Re * ((ybar * (3 + 2 * ybar) * log(ybar + 1) - 2 * ybar - 1)/(ybar + 1)**2 + ybar * log(ybar + 1)/(ybar + 1) + log(ybar + 1)**2) ))

ff_data = {"Re": Re,
           "Pr" : Pr,
           "Minf" : Minf,
           "gamma" : gamma,
           "mu" : mu,
           "R": (gamma - 1)/gamma}

dir_data = CoefficientFunction((rho_ex, vel_ex[0]*rho_ex,vel_ex[1]*rho_ex, E_ex * rho_ex))

bnd_data = {"dirichlet": ["left|right|top|bottom", BoundaryFromVolumeCF(dir_data)]}



hdgsolver = compressibleHDGsolver(
    mesh, 
    order=order, 
    ff_data=ff_data, 
    bnd_data=bnd_data,
    viscid=True)

# we solve for tile T, thus we have p = (gamma-1)/gamma rho * tilde T
uinit = CoefficientFunction((1, 0, 0, pinf/(gamma-1)))
qinit = CoefficientFunction((0, 0, 0, 0, 0, 0, 0, 0), dims=(4, 2))

with TaskManager():
    hdgsolver.SetUp(force, condense=True)
    hdgsolver.SetInitial(uinit, qinit)
    hdgsolver.Solve()

Draw(hdgsolver.velocity, mesh, "u")
Draw(hdgsolver.pressure, mesh, "p")
Draw(hdgsolver.c, mesh, "c")
Draw(hdgsolver.M, mesh, "M")
Draw(hdgsolver.temperature, mesh, "T")
Draw(hdgsolver.energy, mesh, "E")
Draw(hdgsolver.density, mesh, "rho")


print("rho_err = ", sqrt(Integrate((hdgsolver.density-rho_ex)**2, mesh)))
print("E_err = ", sqrt(Integrate((hdgsolver.energy-E_ex)**2, mesh)))
