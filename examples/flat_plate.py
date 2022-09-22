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


Pr = 0.72
Minf = 0.1
Uinf = 1
mu = 1
Re = 1e2
gamma = 1.4 
pinf = 1


rhoinf = pinf * gamma / (Uinf/Minf)**2
Einf = pinf/(gamma-1)/rhoinf + 0.5 * (Uinf**2)

inf_vals = CF((rhoinf, Uinf * rhoinf, 0, Einf * rhoinf))

order = 2

loc_maxh = Re**(-3/4)
print("local maxh = ", loc_maxh)
#################################################################################
# Geometry, exact solution and boundary solution
geo = SplineGeometry()
MakePlate(geo, 1, loc_maxh)
mesh = Mesh(geo.GenerateMesh(maxh=0.1, grading=0.3))
print("number of elements = ", mesh.ne)
Draw(mesh)

ff_data = {"Re": Re,
           "Pr": Pr,
           "Minf": Minf,
           "gamma": gamma,
           "mu": mu,
           "dt": 0.01,
           "Du": True}


bnd_data = {"inflow": ["inflow", inf_vals],
            "ad_wall": "ad_wall",
            "inv_wall": "sym",
            "outflow": ["outflow|top", pinf]
            }



hdgsolver=compressibleHDGsolver(mesh,
                                order=order,
                                ff_data=ff_data,
                                bnd_data=bnd_data,
                                viscid=True,
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

# Uinf_wall = 0
# rhoinf_wall = pinf * gamma / (Uinf_wall/Minf)**2
# Einf_wall = pinf/(gamma-1)/rhoinf + 0.5 * (Uinf_wall**2)


# uinithat = CF((rhoinf_wall, Uinf_wall * rhoinf_wall, 0, Einf_wall * rhoinf_wall))
qinit = CoefficientFunction((0,0,0,0,0,0,0,0), dims = (4,2))


hdgsolver.SetUp(condense=True)
hdgsolver.SetInitial(uinit, qinit)

# EE = hdgsolver.pressure/pinf * (rhoinf/hdgsolver.density)**gamma - 1
# entropy_error = EE #log(IfPos(EE, EE, 1)) * IfPos ((7*R)**2 - y**2, 1,0)
# Draw(entropy_error, mesh, "ent_err")


#  skin frictino coefficient
#  tw = mu * du/dy
# cf = tw / (rho/2 * U_inf**2)

Du = hdgsolver.grad_velocity
tau_w = Du[0, 1]

Draw (hdgsolver.velocity,mesh, "u")
Draw (hdgsolver.pressure,mesh, "p")
Draw (hdgsolver.c,mesh, "c")
Draw (hdgsolver.M,mesh, "M")
Draw (hdgsolver.temperature,mesh, "T")
Draw (hdgsolver.energy, mesh, "E")
Draw (hdgsolver.density,mesh, "rho")


with TaskManager():
    hdgsolver.Solve(maxit=500, maxerr=1e-12, dampfactor=1, printing = True, stop = False)

# err = sqrt(Integrate(EE**2, mesh))
# print("entropy_error = ", err)