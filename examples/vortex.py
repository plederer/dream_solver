import os
import sys
sys.path.insert(1, '../code')
sys.path.insert(1, '../utils')
from geometries import *
from meshes import Get_Omesh
from HDGSolver import compressibleHDGsolver
import math
from ngsolve.internal import visoptions, viewoptions
from ngsolve import *
from netgen.geom2d import unit_square, SplineGeometry
from ctypes import CDLL, RTLD_GLOBAL


nt = 4
# if os.getlogin() != "philip":
#     try:
#         nt = 12
#         CDLL(os.path.join(os.environ["MKLROOT"], "lib/intel64/libmkl_rt.so"), RTLD_GLOBAL)
#     except:
#         nt = 12
#         try:
#             CDLL(os.path.join(os.environ["MKL_ROOT"], "lib/intel64/libmkl_rt.so"), RTLD_GLOBAL)
#         except: #to be sure
#             CDLL('/opt/sw/vsc4/VSC/x86_64/glibc-2.17/skylake/intel/compilers_and_libraries_2019.5.281/linux/mkl/lib/intel64/libmkl_rt.so', RTLD_GLOBAL)


ngsglobals.msg_level = 0


print("Using {} threads".format(nt))
SetNumThreads(nt)

# Dimensionless equations with diameter D
D = 1
R = D/2

Pr = 0.75
Re = 1e5
Minf = 0.5
gamma = 1.4
mu = 1
rhoinf = 1

pinf = 1/Minf**2/gamma
T_inf = 1 / (gamma - 1)/Minf**2

Uinf = 1
Vinf = 0
Einf = pinf/(gamma-1)/rhoinf + 0.5 * (Uinf**2 + Vinf**2)
inf_vals = (rhoinf, Uinf * rhoinf, Vinf * rhoinf, Einf * rhoinf)


Gamma = 5
rv = 5
psi = Gamma * exp(-(x**2 + y**2)/(2*rv**2))
R_farfield = 2 * R * 10

U0 = Uinf - psi.Diff(y)
V0 = Vinf + psi.Diff(x)
p0 = Gamma * exp(-(x**2 + y**2)/(2*rv**2)) + pinf
p0 = pinf
E0 = p0/(gamma-1)/rhoinf + 0.5 * (U0**2 + V0**2)

zero_vals = (rhoinf, U0 * rhoinf, V0 * rhoinf, E0 * rhoinf)

order = 4

#################################################################################

# geo = SplineGeometry()
geo = CSG2d()
rect = Rectangle(pmin=(-40, -40), pmax=(40, 40), bottom="NRBC", right="NRBC", left="inflow", top="NRBC")
geo.Add(rect)
mesh = Mesh(geo.GenerateMesh(maxh=6))

# , grading = 0.2))
# rect = Rectangle( pmin=(-50,-50), pmax=(50, 50), bc = "inflow")
# rect = Rectangle( pmin=(-50,-50), pmax=(50, 50), bc="inflow")
# rect = Rectangle( pmin=(-70,-70), pmax=(50, 50), bc = "inner")
# rect = Rectangle( pmin=(-50,-50), pmax=(50, 50), bc = "inner")

# R = 1
# R_farfield = 20 * R
# mesh = MakeCircle(geo, R_farfield, addrect=False)
# mesh=Mesh(geo.GenerateMesh(maxh=2))  # , grading = 0.2))
# mesh = Mesh(Get_Omesh(R, R_farfield, 36, 16, geom = 1.5))
print("Number of elements = ", mesh.ne)
# mesh.Curve(order)
print(mesh.GetBoundaries())
Draw(mesh)
# quit()
# V = H1(mesh, dirichlet = "cyl")

# gfu = GridFunction(V)
# gfu.Set(1, BND)
# Draw(gfu)
# input()


ff_data = {"Minf": Minf,
           "Re": Re,
           "Pr": Pr,
           "mu": mu,
           "gamma": gamma,
           "dt": 0.1}


# bnd_data = {"inflow": ["inflow|outflow", inf_vals]}
bnd_data = {"inflow": ["inflow", inf_vals],
            "NRBC": ["NRBC", pinf]}


hdgsolver = compressibleHDGsolver(mesh,
                                  order=order,
                                  ff_data=ff_data,
                                  bnd_data=bnd_data,
                                  viscid=False,
                                  stationary=True,
                                  time_solver="BDF2",
                                  force=inf_vals)


uinit = CF(zero_vals)
qinit = CoefficientFunction((0, 0, 0, 0, 0, 0, 0, 0), dims=(4, 2))

hdgsolver.SetUp(condense=True)


hdgsolver.InitializeDir("test")
hdgsolver.SaveConfig()
hdgsolver.SaveSolution()


with TaskManager():
    hdgsolver.SetInitial(uinit, qinit)
    Redraw()
    # input()
    # hdgsolver.Solve(maxit=100, maxerr=1e-10, dampfactor=1, printing=True, max_dt=10)
    hdgsolver.SaveState(0)


hdgsolver.stationary = False
hdgsolver.InitBLF()
hdgsolver.DrawSolutions()

Draw(hdgsolver.pressure - pinf, mesh, "p'")
Draw(hdgsolver.velocity - CF((1, 0)), mesh, "u'")

# hdgsolver.LoadState(0)

tend = 200
t = 0
s = 0


input()
hdgsolver.gfu_old.vec.data = hdgsolver.gfu.vec
with TaskManager():
    while t < tend:
        t += hdgsolver.FU.dt.Get()
        s += 1
        print("step = {}, time = {}".format(s, t), end='\r')
        hdgsolver.gfu_old_2.vec.data = hdgsolver.gfu_old.vec
        hdgsolver.gfu_old.vec.data = hdgsolver.gfu.vec
        hdgsolver.Solve(maxit=10, maxerr=1e-8, dampfactor=1, printing=False)

        # if s % 5 == 0:
        #     hdgsolver.SaveState(s)
