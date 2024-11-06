from netgen.occ import OCCGeometry, WorkPlane
from ngsolve import *
from dream import *
from dream.compressible import flowstate, FarField, Initial, Outflow, GRCBC, NSCBC

ngsglobals.msg_level = 0
SetNumThreads(4)

# Setup Mesh
face = WorkPlane().RectangleC(1, 1).Face()
for bc, edge in zip(['bottom', 'right', 'top', 'left'], face.edges):
    edge.name = bc
mesh = Mesh(OCCGeometry(face, dim=2).GenerateMesh(maxh=0.1))

# Setup Configuration
cfg = SolverConfiguration(mesh)
cfg.pde = "compressible"
cfg.pde.dynamic_viscosity = "inviscid"
# cfg.pde.dynamic_viscosity = "constant"
cfg.pde.equation_of_state = "ideal"
cfg.pde.equation_of_state.heat_capacity_ratio = 1.4
cfg.pde.scaling = "aerodynamic"
cfg.pde.mach_number = 0.03
cfg.pde.reynolds_number = 150
cfg.pde.prandtl_number = 0.72
cfg.pde.riemann_solver = "upwind"

cfg.pde.fe = "conservative"
cfg.pde.fe.order = 4
cfg.pde.fe.method = "hdg"
cfg.pde.fe.mixed_method = "inactive"
# cfg.pde.fe.mixed_method = "strain_heat"

cfg.time = "transient"
cfg.time.scheme = "bdf2"
cfg.time.timer.interval = (0, 200)
cfg.time.timer.step = 0.01

cfg.solver = "nonlinear"
cfg.solver.method = "newton"
cfg.solver.method.damping_factor = 1
cfg.solver.inverse = "direct"
cfg.solver.inverse.solver = "pardiso"
cfg.solver.max_iterations = 10
cfg.solver.convergence_criterion = 1e-10

cfg.optimizations.static_condensation = True
cfg.optimizations.compile.realcompile = False
cfg.optimizations.bonus_int_order = {'vol': 4, 'bnd': 4}

# Setup Initial and Boundary Conditions
Uinf = cfg.pde.get_farfield_state((1, 0))
M = cfg.pde.mach_number
gamma = cfg.pde.equation_of_state.heat_capacity_ratio

Mt = 0.01
R = 0.1
r = sqrt((x-0.2)**2 + y**2)
vt = Mt/M * cfg.pde.scaling.velocity_magnitude(M)

psi = vt * R * exp((R**2 - r**2)/(2*R**2))
u_0 = Uinf.u + CF((psi.Diff(y), -psi.Diff(x)))
p_0 = Uinf.p * (1 - (gamma - 1)/2 * Mt**2 * exp((R**2 - r**2)/(R**2)))**(gamma/(gamma - 1))
rho_0 = Uinf.rho * (1 - (gamma - 1)/2 * Mt**2 * exp((R**2 - r**2)/(R**2)))**(1/(gamma - 1))
p_00 = Uinf.p * (1 - (gamma - 1)/2 * Mt**2 * exp(1))**(gamma/(gamma - 1))

cfg.pde.bcs['left|top|bottom'] = FarField(state=Uinf)
# cfg.pde.bcs['right'] = Outflow(state=Uinf)
# cfg.pde.bcs['top|bottom'] = "inviscid_wall"
cfg.pde.bcs['right'] = GRCBC(state=Uinf, target="outflow", relaxation_factor=0.1,
                             tangential_relaxation=cfg.pde.mach_number, is_viscous_fluxes=False)

initial = Initial(state=flowstate(rho=rho_0, u=u_0, p=p_0))
cfg.pde.dcs['default'] = initial


# Setup Spaces and Gridfunctions
cfg.pde.set_system()

drawing = cfg.pde.get_drawing_state(p=True)
drawing['p*'] = (drawing.p - Uinf.p)/(p_00 - Uinf.p)
cfg.pde.draw(autoscale=False, min=-1e-4, max=1e-4)

cfg.solver.set_discrete_system()
with TaskManager():
    cfg.solver.solve()
