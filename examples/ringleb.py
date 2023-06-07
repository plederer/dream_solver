from ngsolve.fem import MinimizationCF, NewtonCF
from ngsolve.comp import IntegrationRuleSpace
from dream import SolverConfiguration, CompressibleHDGSolver
from netgen.occ import OCCGeometry, WorkPlane
from ngsolve import *
from ngsolve.internal import visoptions, viewoptions
import math

ngsglobals.msg_level = 0

H = 1
maxh = 0.2
viscid = False

cfg = SolverConfiguration()
cfg.formulation = "conservative"
cfg.riemann_solver = 'hllem'
if viscid:
    cfg.dynamic_viscosity = "constant"
    cfg.mixed_method = "strain_heat"

cfg.Prandtl_number = 0.72
cfg.Mach_number = 1
cfg.Reynolds_number = 1

cfg.order = 4

cfg.time.simulation = "stationary"
cfg.time.step = 0.0001
cfg.time.max_step = 10

cfg.max_iterations = 100
cfg.convergence_criterion = 1e-12

gamma = cfg.heat_capacity_ratio.Get()
order = 8
order_hdg = 4

########################################################################
# Geometry
face = WorkPlane().Rectangle(H, H).Face()
for bc, edge in zip(['bottom', 'right', 'top', 'left'], face.edges):
    edge.name = bc
mesh = Mesh(OCCGeometry(face, dim=2).GenerateMesh(maxh=maxh))

#######################################################################
# exact solution
fes_ir = IntegrationRuleSpace(mesh, order=order)
V = L2(mesh, order=order)

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

gfu = GridFunction(V)
gfu.Set(0.5)
ncf = NewtonCF(eq, gfu, maxiter=10000, tol=1e-14)

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
theta = 0.5 * asin(2 * eq_rho(c) * eq_V(c)**2 * y)  # more robust

vel_ex = GridFunction(V**2)
vel_ex.Set(CF((-eq_V(c) * sin(theta), eq_V(c) * cos(theta))))
u_ex = vel_ex[0]
v_ex = vel_ex[1]
p_ex = 1/gamma * c**7
rho_ex = c**5
T_ex = 1/gamma * c**2  # p_ex/rho_ex
E_ex = p_ex/(rho_ex * (gamma-1)) + 0.5 * (u_ex**2 + v_ex**2)

Draw(vel_ex, mesh, "vel_ex")
Draw(p_ex, mesh, "p_ex")
Draw(rho_ex, mesh, "rho_ex")
Draw(E_ex, mesh, "E_ex")
Draw(T_ex, mesh, "T_ex")

M_ex = sqrt(vel_ex[0]**2 + vel_ex[1]**2) / c
Draw(M_ex, mesh, "mach")

#######################################################################
# force if viscid test case is used
if viscid:
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
else:
    force = CF((0, 0))
#######################################################################

dir_data = CF((rho_ex, vel_ex[0]*rho_ex, vel_ex[1]*rho_ex, E_ex * rho_ex))
dir_rho = BoundaryFromVolumeCF(rho_ex)
dir_vel = BoundaryFromVolumeCF(vel_ex)
dir_rho_E = BoundaryFromVolumeCF(E_ex * rho_ex)
dir_rho = rho_ex
dir_vel = vel_ex
dir_rho_E = E_ex * rho_ex

solver = CompressibleHDGSolver(mesh, cfg)
solver.boundary_conditions.set_farfield('left|bottom|top|right', dir_vel, dir_rho,  energy=dir_rho_E)
solver.boundary_conditions.set_custom('left|bottom|top|right')

solver.domain_conditions.set_initial((0, 1), 1,  energy=1/(gamma * cfg.Mach_number**2*(gamma - 1)) + 0.5)


with TaskManager():
    solver.setup(force)

    region = mesh.Boundaries('left|bottom|top|right')

    form = solver.formulation
    TnT = form.TnT

    Uhat, Vhat = TnT.PRIMAL_FACET
    solver.blf += (Uhat - dir_data) * Vhat * ds(skeleton=True, definedon=region)

    solver.drawer.draw(energy=True)
    solver.solve_stationary(50)
# for viscid testcase put viscid terms on the right hand side
# hdgsolver.SetUp(force, condense=True)

print("rho_err = ", sqrt(Integrate((solver.formulation.density()-rho_ex)**2, mesh)))
print("E_err = ", sqrt(Integrate((solver.formulation.energy()-E_ex)**2, mesh)))
