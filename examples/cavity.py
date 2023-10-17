from ngsolve import *
from dream import *
from netgen.occ import OCCGeometry, WorkPlane, Glue

ngsglobals.msg_level = 0
SetNumThreads(8)

air = WorkPlane().MoveTo(0, 0.2).Rectangle(12, 8.3).Face()
for edge, bc in zip(air.edges, ['default', 'right', 'top', 'left']):
    edge.name = bc
boundary_layer = WorkPlane().Rectangle(12, 0.2).Face()
boundary_layer.maxh = 0.1
for edge, bc in zip(boundary_layer.edges, ['wall', 'right', 'default', 'left']):
    edge.name = bc
cavity = WorkPlane().MoveTo(5, -1).Rectangle(2, 1).Face()
cavity.maxh = 0.25
for edge, bc in zip(cavity.edges, ['wall', 'wall', 'default', 'wall']):
    edge.name = bc
geo = Glue([air, boundary_layer, cavity])
mesh = Mesh(OCCGeometry(geo, dim=2).GenerateMesh(maxh=1))

cfg = SolverConfiguration()
cfg.formulation = "conservative"
cfg.scaling = "aerodynamic"
cfg.dynamic_viscosity = "constant"
cfg.mixed_method = "strain_heat"
cfg.riemann_solver = 'hllem'

cfg.Reynolds_number = 166
cfg.Mach_number = 0.42
cfg.Prandtl_number = 0.72
cfg.heat_capacity_ratio = 1.4

cfg.order = 3
cfg.bonus_int_order = {VOL: cfg.order, BND: cfg.order}

cfg.time.simulation = "transient"
cfg.time.scheme = "BDF2"
cfg.time.step = 0.02
cfg.time.interval = (0, 100)

cfg.linear_solver = "pardiso"
cfg.damping_factor = 1
cfg.max_iterations = 100
cfg.convergence_criterion = 1e-12

cfg.compile_flag = True
cfg.static_condensation = True

gamma = cfg.heat_capacity_ratio
M = cfg.Mach_number

rho_inf = 1
alpha = pi/36
u_inf = CF((cos(alpha), sin(alpha)))
p_inf = 1/(M**2 * gamma)
farfield = State(u_inf, rho_inf, p_inf)

solver = CompressibleHDGSolver(mesh, cfg)
solver.boundary_conditions.set(bcs.FarField(farfield), "left|top|right")
solver.boundary_conditions.set(bcs.AdiabaticWall(), 'wall')
solver.domain_conditions.set(dcs.Initial(farfield))


with TaskManager():
    solver.setup()

    formulation = solver.formulation
    TnT = formulation.TnT

    solver.drawer.draw(energy=True)
    solver.drawer.draw_acoustic_pressure(p_inf)
    solver.drawer.draw_particle_velocity(u_inf)

    solver.solve_transient()
