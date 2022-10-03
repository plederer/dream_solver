from asyncore import read
from netgen.geom2d import unit_square, SplineGeometry
from ngsolve import *
from ngsolve.internal import visoptions, viewoptions
import math
import sys
sys.path.insert(1, '../code')
sys.path.insert(1, '../utils')

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
U0 =1#IfPos((x**2 + y**2 - R2**2),1, (x**2 + y**2 - R**2) * 1/(R2**2-R**2))
V0 = 0
E0 = pinf/(gamma-1)/rhoinf + 0.5 * (U0**2 + V0**2)

inf_vals = CF((rhoinf, U0 * rhoinf, V0 * rhoinf, E0 * rhoinf))

order = 3

#################################################################################
geo = SplineGeometry()
mesh = Mesh(Get_Omesh(R, R_farfield, 36, 16, geom = 1.5))
print("number of elements = ", mesh.ne)
mesh.Curve(order)
Draw(mesh)

print(mesh.GetBoundaries())
# input()



p_out = pinf


ff_data = {"Minf": Minf,
           "Re": Re,
           "gamma": gamma,
           "dt": 10}


bnd_data = {"inflow": ["inflow", inf_vals],
            "outflow": ["outflow", pinf],
            "ad_wall": "cyl"}



hdgsolver = compressibleHDGsolver(mesh, 
                                  order=order,
                                  ff_data=ff_data,
                                  bnd_data=bnd_data,
                                  viscid=True,
                                  stationary=True)


uinit = inf_vals
qinit = CoefficientFunction((0,0,0,0,0,0,0,0), dims = (4,2))


hdgsolver.SetUp(condense=True)
hdgsolver.SetInitial(uinit, qinit)


hdgsolver.DrawSolutions()

hdgsolver.InitializeDir("visc_cylinder_data")
hdgsolver.SaveConfig()
hdgsolver.SaveSolution()

with TaskManager():
    hdgsolver.Solve(maxit=100,   maxerr=1e-9, dampfactor=1, printing=True)
    hdgsolver.SaveState(0)