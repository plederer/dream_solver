from netgen.geom2d import unit_square, SplineGeometry
from ngsolve import *
from ngsolve.internal import visoptions, viewoptions
import math
import sys
sys.path.insert(1, '../code')

from ngsolve.comp import IntegrationRuleSpace
from ngsolve.fem import MinimizationCF, NewtonCF

from HDGSolver import compressibleHDGsolver

ngsglobals.msg_level = 0

H = 1
U = 1
T0 = 0.8
T1 = 0.85
Pr = 0.72
Minf = 1 #0.15
gamma = 1.4 
# pinf = 1/(gamma * Minf**2)
mu = 1
Re = 1

# print("pinf = ", pinf)
# order = 4

#################################################################################
# Geometry, exact solution and boundary solution
geo = SplineGeometry()
geo.AddRectangle((0,0),(H,H), bcs=["left", "bottom", "right", "top"])
# geo.AddRectangle((-H,0),(0,H), bcs=["left","bottom","right","top"])
# geo.AddRectangle((-2,1),(-1,2), bcs=["left","bottom","right","top"])
mesh = Mesh(geo.GenerateMesh(maxh=0.1))

order = 8

order_hdg = 3

fes_ir = IntegrationRuleSpace(mesh, order=order)
V = L2(mesh, order = order)

gfu_ir = GridFunction(fes_ir)
v_ir = fes_ir.TrialFunction()


def eq_rho(c):
    return c**(2/(gamma-1))

def eq_V(c):
    return sqrt(2 * (1-c**2)/(gamma-1))

def eq_V2(c):
    return (2 * (1 - c**2)/(gamma-1))

def eq_J(c):
    return 1/c + 1/(3 * c**3) + 1/(5 * c**5) - 1/2 * log((1+c)/(1-c))
    # return log((1+c)/(1-c))

def eq_psi(c):
    return sqrt(1/(2 * eq_V(c)**2) + eq_rho(c) * (x + eq_J(c)/2))

eq = (x + 0.5 * eq_J(v_ir))**2 + y**2 - 1/(4 * eq_rho(v_ir)**2 * eq_V(v_ir)**4).Compile()
# eq = ( (x) + (v_ir))**2 + y**2 - 4



# gfu_ir.Interpolate(CoefficientFunction(x))

gfu = GridFunction(V)
gfu.Set(0.5)

# ncf = NewtonCF(eq.Compile(realcompile=True, wait=True, maxderiv=1), u, maxiter=2)
ncf = NewtonCF(eq, gfu, maxiter=10000, tol = 1e-14)
# gfu_ir.Interpolate(ncf)


dx_irs = dx(intrules=fes_ir.GetIntegrationRules())

u, v = V.TnT()

m = BilinearForm(V)
m += u * v * dx()
m.Assemble()

L = LinearForm(V)
L += ncf * v * dx_irs
L.Assemble()


gfu.vec.data = m.mat.Inverse(V.FreeDofs()) * L.vec


Draw(gfu_ir, mesh, "gfu_ir")
Draw(gfu, mesh, "gfu")


c = gfu

# theta = asin(eq_psi(c) * eq_V(c) - 0.0001)
theta = 0.5 * asin(2 * eq_rho(c) * eq_V(c)**2 * y)

Draw(theta, mesh, "theta")
# vel_ex = CF((-eq_V(c) * sin(theta), eq_V(c) * cos(theta)))
vel_ex = GridFunction(V**2)
vel_ex.Set(CF((-eq_V(c) * sin(theta), eq_V(c) * cos(theta))))
u_ex = vel_ex[0]
v_ex = vel_ex[1]
p_ex = 1/gamma * c**7
rho_ex = c**5
T_ex = 1/gamma * c**2  # p_ex/rho_ex
E_ex = p_ex/(rho_ex * (gamma-1)) + 0.5 * (u_ex**2 + v_ex**2)

# vel_ex = CF((1, 1))
# u_ex = vel_ex[0]
# v_ex = vel_ex[1]
# p_ex = 1
# rho_ex = 1
# T_ex = p_ex/rho_ex 
# E_ex = p_ex/(rho_ex * (gamma-1)) + 0.5 * (u_ex**2 + v_ex**2)



Draw(vel_ex, mesh, "vel_ex")
Draw(p_ex, mesh, "p_ex")
Draw(rho_ex, mesh, "rho_ex")
Draw(E_ex, mesh, "E_ex")
Draw(T_ex, mesh, "T_ex")

M_ex = sqrt(vel_ex[0]**2 + vel_ex[1]**2) / c
Draw(M_ex, mesh, "mach")

ff_data = {"Re": Re,
           "Pr": Pr,
           "Minf": Minf,
           "gamma": gamma,
           "mu": mu,
           "R": 1 } #gamma - 1}

bnd_data = CoefficientFunction((rho_ex, vel_ex[0]*rho_ex,
                                vel_ex[1]*rho_ex,
                                E_ex * rho_ex))

# bnd_data = CF((1, 1, 1, 1/(gamma-1) + 1))

# Vhat = FacetFESpace(mesh)

# vhat = Vhat.TestFunction()
# L = LinearForm(Vhat)
# L += BoundaryFromVolumeCF(bnd_data)[0] * vhat.Trace() * ds
# L.Assemble()
# print(Norm(L.vec))

# quit()
# bnd_data = CoefficientFunction((1,1,1,1))

# force = CF(((rho_ex*u_ex).Diff(x) + (rho_ex*v_ex).Diff(y),
#              (rho_ex*u_ex*u_ex).Diff(x) + (rho_ex*u_ex*v_ex).Diff(y),
#              (rho_ex*v_ex*u_ex).Diff(x) + (rho_ex*v_ex*v_ex).Diff(y),
#              (rho_ex*E_ex*u_ex).Diff(x) + (rho_ex*E_ex*v_ex).Diff(y)))

# Draw(force, mesh, "force")

# print(gfu(mesh(0.5,0)))

sig = GridFunction(V**4)
sig_u = GridFunction(V**2)
phi = GridFunction(V**2)

gu = CF((Grad(vel_ex.components[0]), Grad(vel_ex.components[1])), dims=(2, 2))
epsu = mu / Re * ((gu + gu.trans) - 2/3 * (gu[0, 0] + gu[1, 1]) * Id(2))
sig.Set(epsu)
sig_u.Set(epsu * vel_ex)
phi.Set(mu/Re/Pr * 1/gamma * 2 * c * Grad(c))


div_sig = CF((Grad(sig.components[0])[0] + Grad(sig.components[1])[1],
              Grad(sig.components[2])[0] + Grad(sig.components[3])[1]))

div_sig_u = Grad(sig_u.components[0])[0] + Grad(sig_u.components[1])[1]

div_phi = Grad(phi.components[0])[0] + Grad(phi.components[1])[1]

force = -CF((0, div_sig, div_sig_u + div_phi))

m0 = GridFunction(V)
m1 = GridFunction(V)

m0.Set(rho_ex * u_ex * u_ex + p_ex)
m1.Set(rho_ex * v_ex * u_ex)
div_m = Grad(m0)[0] + Grad(m1)[1]

print("div = ", Integrate(div_m**2, mesh))
# quit()
# Draw(div_m, mesh, "div_m")

# Draw(test, mesh, "Dtest")

# input()
# force_c = CF((0,
#             (rho_ex*u_ex).Diff(x) + (rho_ex*u_ex).Diff(y),
#             (rho_ex*v_ex).Diff(x) + (rho_ex*v_ex).Diff(y),
#             (rho_ex*E_ex).Diff(x) + (rho_ex*E_ex).Diff(y)))


bnd_names = {"inflow" : "",
             "ss_outflow": "",
             "ad_wall": "",
             "iso_wall":"",
             "inv_wall" : "",
             "dirichlet": "left|bottom|right|top"}

hdgsolver = compressibleHDGsolver(
    mesh,
    order=order_hdg,
    ff_data=ff_data, 
    bnd_data=BoundaryFromVolumeCF((bnd_data)), 
    bnd_names=bnd_names)

# uinit = CoefficientFunction((1,0,0,0.5+pinf/(gamma-1)))

# we solve for tile T, thus we have p = (gamma-1)/gamma rho * tilde T
# input()
# uinit = CoefficientFunction((1,0,0,pinf/(gamma-1)))
# uinit = CoefficientFunction((1,0,0,pinf/(gamma-1)))

# uinit = bnd_data
# qinit = CoefficientFunction((0, 0, sig, sig_u + phi), dims=(4, 2))
# uinit = CF((x, x, x, 1/(gamma-1) + 1))
uinit = CF((1, 0, 1, 1/(gamma-1) + 0.5))

# U = GridFunction(V**4)
# U.components[0].Set(rho_ex)
# U.components[1].Set(rho_ex * u_ex)
# U.components[2].Set(rho_ex * v_ex)
# U.components[3].Set(rho_ex * E_ex)

# qinit = CoefficientFunction(Grad(U), dims=(4, 2))

qinit = CoefficientFunction((0, 0, 0, 0, 0, 0, 0, 0), dims=(4, 2))
# qinit = CoefficientFunction((1, 1, 1, 1, 1, 1, 1, 1), dims=(4, 2))


hdgsolver.SetUp(force, condense=True)
hdgsolver.SetInitial(uinit, qinit)



Draw(hdgsolver.velocity, mesh, "u")
Draw(hdgsolver.pressure, mesh, "p")
Draw(hdgsolver.c, mesh, "c")
Draw(hdgsolver.M, mesh, "M")
Draw(hdgsolver.temperature, mesh, "T")
Draw(hdgsolver.energy, mesh, "E")
Draw(hdgsolver.density, mesh, "rho")


# input()
with TaskManager():
    hdgsolver.Solve(maxit=100, maxerr=1e-10, dampfactor=1, printing = True)



print("rho_err = ", sqrt(Integrate((hdgsolver.density-rho_ex)**2, mesh)))
print("E_err = ", sqrt(Integrate((hdgsolver.energy-E_ex)**2, mesh)))
