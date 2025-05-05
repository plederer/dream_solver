# ------- Import Modules ------- #
from ngsolve import *
from netgen.occ import OCCGeometry, WorkPlane
from netgen.meshing import IdentificationType
from ngsolve.meshes import MakeStructured2DMesh
from dream.compressible import flowfields, FarField, Initial, CompressibleFlowSolver

ngsglobals.msg_level = 0
SetNumThreads(4)

# ------- Define Geometry and Mesh ------- #
circle = False
structured = False
periodic = True
maxh = 0.1

if circle:

    face = WorkPlane().MoveTo(0, -0.5).Arc(0.5, 180).Arc(0.5, 180).Face()
    for bc, edge in zip(['right', 'left'], face.edges):
        edge.name = bc
    mesh = Mesh(OCCGeometry(face, dim=2).GenerateMesh(maxh=maxh))


else:

    if structured:
        N = int(1 / maxh)
        mesh = MakeStructured2DMesh(False, N, N, periodic_y=periodic, mapping=lambda x, y: (x - 0.5, y - 0.5))
    else:
        face = WorkPlane().RectangleC(1, 1).Face()

        for bc, edge in zip(['bottom', 'right', 'top', 'left'], face.edges):
            edge.name = bc
        if periodic:
            periodic_edge = face.edges[0]
            periodic_edge.Identify(face.edges[2], "periodic", IdentificationType.PERIODIC)
        mesh = Mesh(OCCGeometry(face, dim=2).GenerateMesh(maxh=maxh))

# ------- Set Configuration ------- #
cfg = CompressibleFlowSolver(mesh)
cfg.dynamic_viscosity = "inviscid"
cfg.equation_of_state = "ideal"
cfg.equation_of_state.heat_capacity_ratio = 1.4
cfg.scaling = "acoustic"
cfg.mach_number = 0.03
cfg.riemann_solver = "lax_friedrich"

cfg.fem = "conservative"
cfg.fem.order = 4
cfg.fem.method = "hdg"
cfg.fem.mixed_method = "inactive"

cfg.time = "transient"
cfg.time.scheme = "bdf2"
cfg.time.timer.interval = (0, 10)
cfg.time.timer.step = 0.01

cfg.nonlinear_solver = "pardiso"
cfg.nonlinear_solver.method = "newton"
cfg.nonlinear_solver.method.damping_factor = 1
cfg.nonlinear_solver.max_iterations = 10
cfg.nonlinear_solver.convergence_criterion = 1e-10

cfg.optimizations.static_condensation = True
cfg.optimizations.compile.realcompile = False
cfg.optimizations.bonus_int_order.vol = 4
cfg.optimizations.bonus_int_order.bnd = 4

# ------- Curve Mesh ------- #
if circle:
    mesh.Curve(cfg.fem.order)

# ------- Setup Boundary Conditions and Domain Conditions ------- #
Uinf = cfg.get_farfield_state((1, 0))

Gamma = 0.1
Rv = 0.1
r = sqrt(x**2 + y**2)
p_0 = Uinf.p * (1 + Gamma * exp(-r**2/Rv**2))
rho_0 = Uinf.rho * (1 + Gamma * exp(-r**2/Rv**2))
initial = Initial(fields=flowfields(rho=rho_0, u=Uinf.u, p=p_0))
cfg.dcs['default'] = initial

cfg.bcs['left|right'] = FarField(fields=Uinf)
if periodic:
    cfg.bcs['top|bottom'] = "periodic"
else:
    cfg.bcs['top|bottom'] = FarField(fields=Uinf)

# ------- Setup Spaces and Gridfunctions ------- #
cfg.initialize()

# ------- Setup Outputs ------- #
drawing = cfg.get_fields()
drawing["p'"] = drawing.p - Uinf.p
cfg.io.draw(drawing, autoscale=False, min=-1e-4, max=1e-4)
cfg.io.vtk.fields = drawing

# ------- Solve System ------- #
with TaskManager():
    cfg.solve()
