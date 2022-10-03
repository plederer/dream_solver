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
Re_init = 50
Uinf = 1
Vinf = 0
abs_u = sqrt(Uinf**2 + Vinf**2)
Minf = 0.3
gamma = 1.4 
pinf = 1

rhoinf = pinf * gamma / (abs_u/Minf)**2
Einf = pinf/(gamma-1)/rhoinf + 0.5 * abs_u**2

# mu = rhoinf * abs_u * D / Re
# mu_init = rhoinf * abs_u * D / Re_init
# print("rhoinf = ", rhoinf)
mu = abs_u * D / Re
mu_init = abs_u * D / Re_init

print("mu init = ", mu_init)

R2 = 5*R
U0 = 1 #IfPos((x**2 + y**2 - R2**2),1, (x**2 + y**2 - R**2) * 1/(R2**2-R**2))
V0 = 0
E0 = pinf/(gamma-1)/rhoinf + 0.5 * (U0**2 + V0**2)

inf_vals = CF((rhoinf, U0 * rhoinf, V0 * rhoinf, E0 * rhoinf))

order = 3

#################################################################################

geo = SplineGeometry()

# Make_Circle(geo, R, R_farfield)

# mesh = Mesh(geo.GenerateMesh(maxh=2))
# mesh.ngmesh.Save("cylmesh.vol.gz")
# mesh = Mesh("cylmesh.vol.gz")
# mesh.ngmesh.SetGeometry(geo)

# L = 20
# alpha = 3
# hr = (R_farfield - R) * (1/L)**alpha + R
# print(hr)
# N = int(  D * pi / hr )

# N = N + 4 - N%4
# print("D pi/ N = ", D * pi/N)
# print(N)

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
# mesh = Mesh(Get_Omesh(R, R_farfield, 48, 20, geom = 2.5))


print("Number of elmeents = ", mesh.ne)
# mesh.Curve(order)
Draw(mesh)
print(mesh.GetBoundaries())
p_out = pinf
# input()

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


uinit = inf_vals
qinit = CoefficientFunction((0,0,0,0,0,0,0,0), dims = (4,2))

hdgsolver.SetUp(condense=True)
hdgsolver.DrawSolutions()

# with TaskManager():
#     hdgsolver.SetInitial(uinit, qinit)
#     hdgsolver.Solve(maxit=100, maxerr=1e-10, dampfactor=1, printing=True)
#     # Re_init = Re_init * 2
#     # mu_init = rhoinf * abs_u * D / Re_init
#     # hdgsolver.FU.Re.Set(Re_init)
#     # hdgsolver.FU.mu.Set(mu_init)
#     # hdgsolver.SetUpBLF()
#     # input("first solve")
#     # hdgsolver.Solve(maxit=100, maxerr=1e-10, dampfactor=1, printing=True)
#     hdgsolver.gfu.Save("initial_cyl.dat")
# input("finished initial solution")
# quit()

file_initial = "solution_s_5.dat"


hdgsolver.stationary = False
hdgsolver.FU.dt.Set(0.1)
hdgsolver.FU.Re.Set(Re)
hdgsolver.FU.mu.Set(mu)
hdgsolver.InitBLF()

# file_initial = "initial_cyl.dat"
hdgsolver.gfu.Load(file_initial)

nondim_pressure = (hdgsolver.pressure - pinf)/pinf/Uinf**2
Draw(nondim_pressure, mesh, "nond-p")

# hdgsolver.gfu.Save("./testfolder/test.dat")




tend = 500 
t = 0
input("asdf")

s = 0

import time 

with TaskManager():
    while t < tend:
        t += hdgsolver.FU.dt.Get() #["dt"]
        s += 1
        print("step = {}, time = {}".format(s, t), end='\r')

        # hdgsolver.Solve(maxit=10, maxerr=1e-6, dampfactor=1, printing=False)
        
        if s % 5 == 0:
            filename = "solution_s_" + str(s) + ".dat"
            # hdgsolver.gfu.Save(filename)
            hdgsolver.gfu.Load(filename)
            Redraw(blocking=True)
            time.sleep(0.1)


# >>> import fnmatch
# >>> import os
# >>> fnmatch.filter(os.listdir('.'), 'solution_t_13')