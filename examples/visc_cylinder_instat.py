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


R = 0.5
R_farfield = 2*R * 30

Re_init = 10
Re = 150
Uinf = 1
Vinf = 0
Minf = 0.3
gamma = 1.4 
pinf = 1

rhoinf = pinf * gamma / (Uinf/Minf)**2
Einf = pinf/(gamma-1)/rhoinf + 0.5 * (Uinf**2 + Vinf**2)


R2 = 5*R
U0 = 1
V0 = 0
E0 = pinf/(gamma-1)/rhoinf + 0.5 * (U0**2 + V0**2)

inf_vals = CF((rhoinf, U0 * rhoinf, V0 * rhoinf, E0 * rhoinf))

order = 3

#################################################################################
geo = SplineGeometry()
Make_C_type(geo, R, R_farfield, R_farfield * 2, maxh_cyl=0.1)
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




# initial solution
with TaskManager():
    hdgsolver.SetUp(condense=True)
    hdgsolver.SetInitial(uinit, qinit)    
    hdgsolver.Solve(maxit=100, maxerr=1e-10, dampfactor=1, printing=True)
    hdgsolver.gfu.Save("initial_cyl.dat")
    
input("finished initial solution")


hdgsolver.stationary = False
hdgsolver.FU.dt.Set(0.5)
hdgsolver.FU.Re.Set(Re)
hdgsolver.SetUp(condense=True)

# hdgsolver.gfu.Load("initial_cyl.dat")
hdgsolver.gfu_old.vec.data = hdgsolver.gfu.vec
hdgsolver.gfu_old_2.vec.data = hdgsolver.gfu.vec

tend = 200 
t = 0



Draw (hdgsolver.velocity,mesh, "u")
Draw (hdgsolver.pressure,mesh, "p")
Draw (hdgsolver.c,mesh, "c")
Draw (hdgsolver.M,mesh, "M")
Draw (hdgsolver.temperature,mesh, "T")
Draw (hdgsolver.energy, mesh, "E")
Draw (hdgsolver.density,mesh, "rho")

with TaskManager():
    while t < tend:
        t += hdgsolver.FU.dt.Get() #ff_data["dt"]
        print("time = {}".format(t), end='\r')
        hdgsolver.Solve(maxit=10, maxerr=1e-6, dampfactor=1, printing=False)
