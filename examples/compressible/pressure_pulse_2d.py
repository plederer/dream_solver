from dream import *
from dream.compressible import flowstate, FarField, Initial
from ngsolve import *
from netgen.occ import OCCGeometry, WorkPlane
from netgen.meshing import IdentificationType
from ngsolve.meshes import MakeStructured2DMesh

ngsglobals.msg_level = 0
SetNumThreads(4)

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

# Setup Configuration
cfg = SolverConfiguration(mesh)
cfg.pde = "compressible"
cfg.pde.dynamic_viscosity = "inviscid"
cfg.pde.equation_of_state = "ideal"
cfg.pde.equation_of_state.heat_capacity_ratio = 1.4
cfg.pde.scaling = "acoustic"
cfg.pde.mach_number = 0.03
cfg.pde.riemann_solver = "lf"

cfg.pde.fem = "conservative"
cfg.pde.fem.order = 4
cfg.pde.fem.method = "hdg"
cfg.pde.fem.mixed_method = "inactive"

cfg.time = "transient"
cfg.time.scheme = "ie"
cfg.time.timer.interval = (0, 10)
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
cfg.io.log.to_terminal = True
cfg.io.log.to_file = False
cfg.io.vtk = False
cfg.io.settings = False
cfg.io.settings.pickle = False
cfg.io.settings.txt = False
cfg.io.gfu = False
cfg.io.transient_gfu = False
cfg.io.ngsmesh = False

if circle:
    mesh.Curve(cfg.pde.fem.order)

# Setup Initial and Boundary Conditions
Uinf = cfg.pde.get_farfield_state((1, 0))

Gamma = 0.1
Rv = 0.1
r = sqrt(x**2 + y**2)
p_0 = Uinf.p * (1 + Gamma * exp(-r**2/Rv**2))
rho_0 = Uinf.rho * (1 + Gamma * exp(-r**2/Rv**2))
initial = Initial(state=flowstate(rho=rho_0, u=Uinf.u, p=p_0))


# cfg.pde.bcs['left|top|bottom|right'] = FarField(state=Uinf)
cfg.pde.bcs['left|right'] = FarField(state=Uinf)
# cfg.pde.bcs['top|bottom'] = FarField(state=Uinf)
cfg.pde.bcs['top|bottom'] = "periodic"
cfg.pde.dcs['default'] = initial


# Setup Spaces and Gridfunctions
cfg.pde.initialize_system()

drawing = cfg.pde.get_drawing_fields(p=True)
drawing["p'"] = drawing.p - Uinf.p
cfg.pde.draw(autoscale=False, min=-1e-4, max=1e-4)

cfg.io.vtk.fields = drawing

cfg.solver.initialize()
with TaskManager():
    cfg.solver.solve()
