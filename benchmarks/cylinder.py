import os, sys
from ctypes import CDLL, RTLD_GLOBAL

nt = 4
if os.getlogin() != "philip":
    try:
        nt = 12
        CDLL(os.path.join(os.environ["MKLROOT"], "lib/intel64/libmkl_rt.so"), RTLD_GLOBAL)
    except:
        nt = 12
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
D = 1
R = D/2

R_farfield = 2 * R * 30

Pr = 0.75
Re = 150
Re_init = 20
Uinf = 1
Vinf = 0
abs_u = sqrt(Uinf**2 + Vinf**2)
Minf = 0.3
gamma = 1.4 
pinf = 1

rhoinf = pinf * gamma / (abs_u/Minf)**2
Einf = pinf/(gamma-1)/rhoinf + 0.5 * abs_u**2

mu = rhoinf * abs_u * D / Re
mu_init = rhoinf * abs_u * D / Re_init
# mu = abs_u * D / Re
# mu_init = abs_u * D / Re_init


R2 = 5*R
U0 = 1 #IfPos((x**2 + y**2 - R2**2),1, (x**2 + y**2 - R**2) * 1/(R2**2-R**2))
V0 = 0
E0 = pinf/(gamma-1)/rhoinf + 0.5 * (U0**2 + V0**2)

inf_vals = (rhoinf, U0 * rhoinf, V0 * rhoinf, E0 * rhoinf)

order = 3

#################################################################################

geo = SplineGeometry()

if False:
    N = 80
    L = 30
    alpha = 1.9
    hr = (R_farfield - R) * (1/L)**alpha

    mesh = Mesh(Get_Omesh(R, R_farfield, N, L, geom = alpha))
    print("Local mesh size at boundary = ", D * pi / N)
    print("Local mesh size orthogonal to boundary = ", hr)
else:
    Make_Circle(geo, R, R_farfield, loch = 0.04)
    mesh = Mesh(geo.GenerateMesh(maxh=1.5, grading = 0.2))
    

print("Number of elements = ", mesh.ne)
# mesh.Curve(order)
Draw(mesh)
print(mesh.GetBoundaries())
input()

ff_data = {"Minf": Minf,
           "Re": Re_init,
           "Pr": Pr,
           "mu": mu_init,
           "gamma": gamma,
           "dt": 0.1}


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


uinit = CF(tuple(inf_vals))
qinit = CoefficientFunction((0,0,0,0,0,0,0,0), dims = (4,2))

hdgsolver.SetUp(condense=True)
hdgsolver.DrawSolutions()

# hdgsolver.InitializeDir("fine_solution")
hdgsolver.SaveConfig()
hdgsolver.SaveSolution()



with TaskManager():
    hdgsolver.SetInitial(uinit, qinit)
    hdgsolver.Solve(maxit=100, maxerr=1e-10, dampfactor=1, printing=True)
    hdgsolver.SaveState(0)


hdgsolver.stationary = False
hdgsolver.FU.dt.Set(0.1)
hdgsolver.FU.Re.Set(Re)
hdgsolver.FU.mu.Set(mu)
hdgsolver.InitBLF()

# hdgsolver.LoadState(0)

tend = 200 
t = 0
s = 0

scal = 2/rhoinf/abs_u**2/D
hdgsolver.SaveForces(0, "cyl", scal, init=True)

with TaskManager():
    while t < tend:
        t += hdgsolver.FU.dt.Get()
        s += 1
        print("step = {}, time = {}".format(s, t), end='\r')

        hdgsolver.Solve(maxit=10, maxerr=1e-8, dampfactor=1, printing=False)
        hdgsolver.SaveForces(t, "cyl", scal)

        if s % 5 == 0:
            hdgsolver.SaveState(s)
