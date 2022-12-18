import os
import sys
sys.path.insert(1, '../code')
sys.path.insert(1, '../utils')
from netgen.occ import OCCGeometry
from netgen.geom2d import unit_square, SplineGeometry
from ngsolve import *
from HDGSolver import compressibleHDGsolver
from geometries import *



ngsglobals.msg_level = 0
SetNumThreads(8)

dirname = "pressure_nrbc"
maxh = 5
order = 5

# Dimensionless equations with diameter D
D = 1
R = D/2

Pr = 0.72
Re = 1
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

U0 = Uinf  # - psi.Diff(y)
V0 = Vinf  # + psi.Diff(x)
p0 = pinf + Gamma * exp(-(x**2 + y**2)/(2*rv**2))
E0 = p0/(gamma-1)/rhoinf + 0.5 * (U0**2 + V0**2)

zero_vals = (rhoinf, U0 * rhoinf, V0 * rhoinf, E0 * rhoinf)

wp = MakeOCCRectangle((-40, -40), (40, 40), bottom="NRBC", right="NRBC", left="inflow", top="NRBC")
geo = OCCGeometry(wp.Face(), dim=2)
mesh = Mesh(geo.GenerateMesh(maxh=maxh))

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


hdgsolver.InitializeDir(dirname)
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

tend = 70
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
        hdgsolver.Solve(maxit=100, maxerr=1e-8, dampfactor=0.5, printing=True)

        if s % 1 == 0:
            hdgsolver.SaveState(s)
