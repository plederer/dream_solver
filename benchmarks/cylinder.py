import os, sys
from ctypes import CDLL, RTLD_GLOBAL

nt = 4
if os.getlogin() != "philip":
    try:
        nt = 24
        CDLL(os.path.join(os.environ["MKLROOT"], "lib/intel64/libmkl_rt.so"), RTLD_GLOBAL)
    except:
        nt = 24
        try:
            CDLL(os.path.join(os.environ["MKL_ROOT"], "lib/intel64/libmkl_rt.so"), RTLD_GLOBAL)
        except: #to be sure
            CDLL('/opt/sw/vsc4/VSC/x86_64/glibc-2.17/skylake/intel/compilers_and_libraries_2019.5.281/linux/mkl/lib/intel64/libmkl_rt.so', RTLD_GLOBAL)


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


print("Using {} threads".format(nt))
SetNumThreads(nt)

# Dimensionless equations with diameter D
R = 1/2

R_farfield = 2 * R * 30

Pr = 0.75
Re = 150
Re_init = 100
Uinf = 1
Vinf = 0
Minf = 0.3
gamma = 1.4 
pinf = 1

rhoinf = pinf * gamma / (Uinf/Minf)**2
Einf = pinf/(gamma-1)/rhoinf + 0.5 * (Uinf**2 + Vinf**2)



R2 = 5*R
U0 = 1 #IfPos((x**2 + y**2 - R2**2),1, (x**2 + y**2 - R**2) * 1/(R2**2-R**2))
V0 = 0
E0 = pinf/(gamma-1)/rhoinf + 0.5 * (U0**2 + V0**2)

inf_vals = CF((rhoinf, U0 * rhoinf, V0 * rhoinf, E0 * rhoinf))

order = 2

#################################################################################

geo = SplineGeometry()
geo.AddCircle ( (0, 0), r=R, leftdomain=0, rightdomain=1, bc="cyl", maxh=0.1)
Make_Circle(geo, R_farfield)
mesh = Mesh(geo.GenerateMesh(maxh=1))

# mesh = Mesh(Get_Omesh(R, R_farfield, 48, 20, geom = 2.5))


print("number of elements = ", mesh.ne)
mesh.Curve(order)
Draw(mesh)

print(mesh.GetBoundaries())




p_out = pinf


ff_data = {"Minf": Minf,
           "Re": Re_init,
           "Pr": Pr,
           "gamma": gamma,
           "dt": 1}


bnd_data = {"inflow": ["inflow", inf_vals],
            "outflow": ["outflow", pinf],
            "ad_wall": "cyl"}



hdgsolver = compressibleHDGsolver(mesh, 
                                  order=order,
                                  ff_data=ff_data,
                                  bnd_data=bnd_data,
                                  viscid=True,
                                  stationary=True,
                                  time_solver="BDF2")


uinit = inf_vals
qinit = CoefficientFunction((0,0,0,0,0,0,0,0), dims = (4,2))




# with TaskManager():
#     hdgsolver.SetUp(condense=True)
#     hdgsolver.SetInitial(uinit, qinit)
#     hdgsolver.Solve(maxit=100, maxerr=1e-10, dampfactor=1, printing=True)
#     hdgsolver.gfu.Save("initial_cyl.dat")
# input("finished initial solution")
# quit()
# file_initial = "solution_t_259.9999999999906.dat"


hdgsolver.stationary = False
hdgsolver.FU.dt.Set(0.1)
hdgsolver.FU.Re.Set(Re)
hdgsolver.SetUp(condense=True)

file_initial = "initial_cyl.dat"
hdgsolver.gfu.Load(file_initial)

nondim_pressure = (hdgsolver.pressure - pinf)/pinf/Uinf**2
Draw(nondim_pressure, mesh, "nond-p")

Draw (hdgsolver.velocity,mesh, "u")
Draw (hdgsolver.pressure,mesh, "p")
Draw (hdgsolver.c,mesh, "c")
Draw (hdgsolver.M,mesh, "M")
Draw (hdgsolver.temperature,mesh, "T")
Draw (hdgsolver.energy, mesh, "E")
Draw (hdgsolver.density,mesh, "rho")



tend = 500 
t = 0


s = 0
with TaskManager():
    while t < tend:
        t += hdgsolver.FU.dt.Get() #["dt"]
        s += 1
        print("step = {}, time = {}".format(s, t), end='\r')

        hdgsolver.Solve(maxit=10, maxerr=1e-6, dampfactor=1, printing=False)
        
        if s % 20 == 0:
            filename = "solution_t_" + str(t) + ".dat"
            hdgsolver.gfu.Save(filename)
