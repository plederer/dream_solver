from netgen.occ import OCCGeometry, WorkPlane
from ngsolve import *
from dream import *
from ngsolve.meshes import MakeStructured2DMesh

ngsglobals.msg_level = 0
SetNumThreads(8)

circle = False
structured = False
maxh = 0.05

cfg = SolverConfiguration()
cfg.formulation = "conservative"
cfg.scaling = "acoustic"
# cfg.dynamic_viscosity = "constant"
# cfg.dynamic_viscosity = None
# cfg.mixed_method = "strain_heat"
# cfg.mixed_method = None
cfg.riemann_solver = 'hllem'

cfg.Reynolds_number = 166
cfg.Mach_number = 0
cfg.Prandtl_number = 0.72
cfg.heat_capacity_ratio = 1.4

cfg.order = 1
cfg.bonus_int_order_bnd = 0
cfg.bonus_int_order_vol = 0

cfg.time.simulation = "transient"
cfg.time.scheme = "BDF2"
cfg.time.step = 0.01
cfg.time.interval = (0, 100)

cfg.linear_solver = "pardiso"
cfg.damping_factor = 1
cfg.max_iterations = 100
cfg.convergence_criterion = 1e-16

cfg.compile_flag = False
cfg.static_condensation = True

if circle:

    face = WorkPlane().MoveTo(0, -0.5).Arc(0.5, 180).Arc(0.5, 180).Face()
    for bc, edge in zip(['right', 'left'], face.edges):
        edge.name = bc
    mesh = Mesh(OCCGeometry(face, dim=2).GenerateMesh(maxh=maxh))
    mesh.Curve(cfg.order)

else:

    if structured:
        N = int(1 / maxh)
        mesh = MakeStructured2DMesh(False, N, N, mapping=lambda x, y: (x - 0.5, y - 0.5))
    else:
        face = WorkPlane().RectangleC(1, 1).Face()

        for bc, edge in zip(['bottom', 'right', 'top', 'left'], face.edges):
            edge.name = bc
        mesh = Mesh(OCCGeometry(face, dim=2).GenerateMesh(maxh=maxh))
# mesh.Refine()
# mesh.Curve(cfg.order)
gamma = cfg.heat_capacity_ratio
M = cfg.Mach_number

alpha = 0
u_inf = (1, 0)
u_inf = INF.velocity(u_inf, cfg)
rho_inf = INF.density(cfg)
p_inf = INF.pressure(cfg)
farfield = State(u_inf, rho_inf, p_inf)

# # Pressure Pulse
Gamma = 0.02
Rv = 0.1
r = sqrt(x**2 + y**2)
p_0 = p_inf * (1 + Gamma * exp(-r**2/Rv**2))
rho_0 = rho_inf * (1 + Gamma * exp(-r**2/Rv**2))
initial = State(u_inf, rho_inf, p_0)


solver = CompressibleHDGSolver(mesh, cfg)
# solver.boundary_conditions.set(bcs.FarField(farfield), 'left|right|top|bottom')
solver.boundary_conditions.set(bcs.Outflow_NSCBC(p_inf, 0, 1, True), 'left|right|top|bottom')
solver.domain_conditions.set(dcs.Initial(initial))

with TaskManager():
    solver.setup()

    solver.drawer.draw(energy=True)
    solver.drawer.draw_particle_velocity(u_inf)
    solver.drawer.draw_acoustic_pressure(p_inf)

    # Draw(solver.formulation.gfu.components[2], solver.mesh, "uhathat")
    # print(solver.formulation.gfu.components[3].vec)
    # input()
    solver.solve_transient()
