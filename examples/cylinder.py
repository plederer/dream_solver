from netgen.geom2d import unit_square, SplineGeometry
from ngsolve import *
from ngsolve.internal import visoptions, viewoptions
import math
import sys
sys.path.insert(1, '../code')

from HDGSolver import compressibleHDGsolver

ngsglobals.msg_level = 0

from geometries import * #MakeSmoothRectangle, MakeRectangle, Make_C_type

from meshes import Get_Omesh



# Re = 1
Uinf = 1
Uinf = 1
Minf = 0.3
gamma = 1.4 
pinf = 1

rhoinf = pinf * gamma / (Uinf/Minf)**2
Einf = pinf/(gamma-1)/rhoinf + 0.5 * (1**2)

inf_vals = CF((rhoinf, Uinf * rhoinf, 0, Einf * rhoinf))

order = 2

#################################################################################
# Geometry, exact solution and boundary solution
geo = SplineGeometry()
R = 1
R_farfield = R * 30
# Make_C_type(geo, R, R_farfield, R_farfield * 2, maxh_cyl=0.5)


# geo.AddCircle ( (0, 0), r=R_farfield, leftdomain=1, rightdomain=0, bc="inflow") #, maxh=0.1)

# Make_Circle(geo, R=R_farfield)
# geo.AddCircle ( (0, 0), r=R, leftdomain=0, rightdomain=1, bc="cyl", maxh=0.2)


# mesh = Mesh(geo.GenerateMesh(maxh=1))
# mesh = Mesh(Get_Omesh(R, R_farfield, 64, 35, geom = 1.5))
mesh = Mesh(Get_Omesh(R, R_farfield, 32, 20, geom = 1.5))
print("number of elements = ", mesh.ne)
mesh.Curve(order)
Draw(mesh)

print(mesh.GetBoundaries())
# input()



p_out = pinf


ff_data = {"Minf": Minf,
           "gamma": gamma,
           "R": (gamma - 1),
           "dt": 0.001}

# bnd_data = {"inflow": ["inflow", inf_vals],
#             "inv_wall": "cyl",
#             "outflow": ["outflow", p_out]}


bnd_data = {"inflow": ["inflow|outflow", inf_vals],
            "inv_wall": "cyl"}



hdgsolver = compressibleHDGsolver(mesh, 
                                  order=order,
                                  ff_data=ff_data,
                                  bnd_data=bnd_data,
                                  viscid=False,
                                  stationary=True)


import numpy as np

# U = np.array([1, 1, 0, 1/(gamma-1) + 0.5])
# a = -1
# b = 1
# ab = sqrt(a**2 + b**2)
# N = np.array([a/ab, b/ab])

# A = hdgsolver.FU.jacA(U)
# B = hdgsolver.FU.jacB(U)
# # A = hdgsolver.FU.Aplus(U, N)

# P = hdgsolver.FU.P(U, N)
# Pinv = hdgsolver.FU.Pinv(U, N)
# Lam = hdgsolver.FU.Lam(U, N)

# G = N[0] * A + N[1] * B

# # print(Integrate(P * Lam * Pinv, mesh))
# print(Integrate(G, mesh))
# print("error = ", Integrate(InnerProduct(P * Lam * Pinv - G,P * Lam * Pinv - G), mesh))
# # print(hdgsolver.FU.Aplus(U, N))
# # print("AAAA")
# # input()
# quit()

# U = (x**2 + y**2 - R**2) * 1/(R_farfield**2-R**2)
# start_vals = CF((1, U * 1, 0, pinf/(gamma-1) + 0.5 * U**2))
uinit = inf_vals #CoefficientFunction((1,1,0,pinf/(gamma-1)))
qinit = CoefficientFunction((0,0,0,0,0,0,0,0), dims = (4,2))


hdgsolver.SetUp(condense=True)
hdgsolver.SetInitial(uinit, qinit)

EE = hdgsolver.pressure/pinf * (rhoinf/hdgsolver.density)**gamma - 1
entropy_error = EE #log(IfPos(EE, EE, 1)) * IfPos ((7*R)**2 - y**2, 1,0)
Draw(entropy_error, mesh, "ent_err")



Draw (hdgsolver.velocity,mesh, "u")
Draw (hdgsolver.pressure,mesh, "p")
Draw (hdgsolver.c,mesh, "c")
Draw (hdgsolver.M,mesh, "M")
Draw (hdgsolver.temperature,mesh, "T")
Draw (hdgsolver.energy, mesh, "E")
Draw (hdgsolver.density,mesh, "rho")

# input()
with TaskManager():
    hdgsolver.Solve(maxit=500, maxerr=1e-9, dampfactor=1, printing = True)

err = sqrt(Integrate(EE**2, mesh))
print("entropy_error = ", err)