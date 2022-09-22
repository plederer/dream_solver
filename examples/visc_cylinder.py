from asyncore import read
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


R = 1
R_farfield = R * 30

Re = 1e3
Uinf = 1
Vinf = 0
Minf = 0.3
gamma = 1.4 
pinf = 1

rhoinf = pinf * gamma / (Uinf/Minf)**2
Einf = pinf/(gamma-1)/rhoinf + 0.5 * (Uinf**2 + Vinf**2)



R2 = 5*R
U0 = IfPos((x**2 + y**2 - R2**2),1, (x**2 + y**2 - R**2) * 1/(R2**2-R**2))
V0 = 0
E0 = pinf/(gamma-1)/rhoinf + 0.5 * (U0**2 + V0**2)

inf_vals = CF((rhoinf, U0 * rhoinf, V0 * rhoinf, E0 * rhoinf))

order = 3

#################################################################################
geo = SplineGeometry()
Make_C_type(geo, R, R_farfield, R_farfield * 2, maxh_cyl=0.2)
mesh = Mesh(geo.GenerateMesh(maxh=5))

print("number of elements = ", mesh.ne)
mesh.Curve(order)
Draw(mesh)

print(mesh.GetBoundaries())
# input()



p_out = pinf


ff_data = {"Minf": Minf,
           "Re": Re,
           "gamma": gamma,
           "dt": 0.5}


bnd_data = {"inflow": ["inflow", inf_vals],
            "outflow": ["outflow", pinf],
            "ad_wall": "cyl"}



hdgsolver = compressibleHDGsolver(mesh, 
                                  order=order,
                                  ff_data=ff_data,
                                  bnd_data=bnd_data,
                                  viscid=True,
                                  stationary=False)


uinit = inf_vals
qinit = CoefficientFunction((0,0,0,0,0,0,0,0), dims = (4,2))


hdgsolver.SetUp(condense=True)
hdgsolver.SetInitial(uinit, qinit)


Draw (hdgsolver.velocity,mesh, "u")
Draw (hdgsolver.pressure,mesh, "p")
Draw (hdgsolver.c,mesh, "c")
Draw (hdgsolver.M,mesh, "M")
Draw (hdgsolver.temperature,mesh, "T")
Draw (hdgsolver.energy, mesh, "E")
Draw (hdgsolver.density,mesh, "rho")



tend = 200 
t = 0

with TaskManager():
    while t < tend:
        t += ff_data["dt"]
        print("time = {}".format(t), end='\r')
        hdgsolver.Solve(maxit=10, maxerr=1e-9, dampfactor=1, printing=False)
