
# ------- Import Modules ------- #
from ngsolve import *
from netgen.occ import OCCGeometry, WorkPlane
from dream.compressible import CompressibleFlowSolver, flowfields, FarField, Initial, Outflow, GRCBC, NSCBC

ngsglobals.msg_level = 0
SetNumThreads(8)

# ------- Define Geometry and Mesh ------- #
face = WorkPlane().RectangleC(1, 1).Face()
for bc, edge in zip(['bottom', 'right', 'top', 'left'], face.edges):
    edge.name = bc
mesh = Mesh(OCCGeometry(face, dim=2).GenerateMesh(maxh=0.1))

# ------- Set Configuration ------- #
cfg = CompressibleFlowSolver(mesh)

cfg.time = "transient"
cfg.time.timer.interval = (0, 200)
cfg.time.timer.step = 0.01

cfg.dynamic_viscosity = "inviscid"
# cfg.dynamic_viscosity = "constant"
cfg.equation_of_state = "ideal"
cfg.equation_of_state.heat_capacity_ratio = 1.4
cfg.scaling = "aerodynamic"
cfg.mach_number = 0.03
cfg.reynolds_number = 150
cfg.prandtl_number = 0.72
cfg.riemann_solver = "upwind"

cfg.fem = "conservative"
cfg.fem.method = "hdg"
cfg.fem.scheme = "bdf2"
cfg.fem.order = 4
cfg.fem.mixed_method = "inactive"
# cfg.fem.mixed_method = "strain_heat"


cfg.nonlinear_solver = "pardiso"
cfg.nonlinear_solver.method = "newton"
cfg.nonlinear_solver.method.damping_factor = 1
cfg.nonlinear_solver.max_iterations = 10
cfg.nonlinear_solver.convergence_criterion = 1e-10

cfg.optimizations.static_condensation = True
cfg.optimizations.compile.realcompile = False
cfg.optimizations.bonus_int_order.vol = 4
cfg.optimizations.bonus_int_order.bnd = 4

# ------- Setup Boundary Conditions and Domain Conditions ------- #
Uinf = cfg.get_farfield_state((1, 0))
M = cfg.mach_number
gamma = cfg.equation_of_state.heat_capacity_ratio

Mt = 0.01
R = 0.1
r = sqrt((x-0.2)**2 + y**2)
vt = Mt/M * cfg.scaling.velocity_magnitude(M)

psi = vt * R * exp((R**2 - r**2)/(2*R**2))
u_0 = Uinf.u + CF((psi.Diff(y), -psi.Diff(x)))
p_0 = Uinf.p * (1 - (gamma - 1)/2 * Mt**2 * exp((R**2 - r**2)/(R**2)))**(gamma/(gamma - 1))
rho_0 = Uinf.rho * (1 - (gamma - 1)/2 * Mt**2 * exp((R**2 - r**2)/(R**2)))**(1/(gamma - 1))
p_00 = Uinf.p * (1 - (gamma - 1)/2 * Mt**2 * exp(1))**(gamma/(gamma - 1))

# cfg.bcs['right'] = Outflow(fields=Uinf)
# cfg.bcs['top|bottom'] = "inviscid_wall"
cfg.bcs['right'] = GRCBC(fields=Uinf,
                         target="outflow",
                         relaxation_factor=0.1,
                         tangential_relaxation=cfg.mach_number,
                         is_viscous_fluxes=False)
cfg.bcs['left|top|bottom'] = FarField(fields=Uinf)

initial = Initial(fields=flowfields(rho=rho_0, u=u_0, p=p_0))
cfg.dcs['default'] = initial

# ------- Setup Spaces and Gridfunctions ------- #
cfg.initialize()

# ------- Setup Outputs ------- #
drawing = cfg.fem.get_fields('p', default=False)
drawing['p*'] = (drawing.p - Uinf.p)/(p_00 - Uinf.p)
cfg.io.draw(drawing, autoscale=False, min=-1e-4, max=1e-4)

# ------- Solve System ------- #
with TaskManager():
    cfg.solve()
