
from ngsolve import *
from dream import *
from netgen.occ import OCCGeometry, WorkPlane
from dream.utils.geometries import Rectangle1DGrid, RectangleDomain, CircularDomain, CircularGrid, RectangleGrid

ngsglobals.msg_level = 0
SetNumThreads(8)

cfg = SolverConfiguration()
cfg.formulation = "conservative"
cfg.scaling = "acoustic"
# cfg.dynamic_viscosity = "constant"
# cfg.dynamic_viscosity = None
# cfg.mixed_method = "strain_heat"
# cfg.mixed_method = None
cfg.riemann_solver = 'lax_friedrich'

cfg.Reynolds_number = 166
cfg.Mach_number = 0.4
cfg.Prandtl_number = 0.72
cfg.heat_capacity_ratio = 1.4

cfg.order = 5
cfg.bonus_int_order = {VOL: cfg.order, BND: cfg.order}

cfg.time.simulation = "transient"
cfg.time.scheme = "BDF2"
cfg.time.step = 0.005
cfg.time.interval = (0, 100)

cfg.linear_solver = "pardiso"
cfg.damping_factor = 1
cfg.max_iterations = 100
cfg.convergence_criterion = 1e-16

cfg.compile_flag = True
cfg.static_condensation = True

sound = RectangleDomain(H=1, W=1, mat="sound", maxh=0.2)
main = Rectangle1DGrid(sound, direction="x")
front = [
    RectangleDomain(W=1, mat="4", maxh=0.3),
    RectangleDomain(W=1, mat="3", maxh=0.3),
    RectangleDomain(W=1, mat="2", maxh=0.3),
    RectangleDomain(W=1, mat="1", maxh=0.3),
    RectangleDomain(W=1, mat="0", maxh=0.3)
]
for domain in front:
    main.add_front(domain)

geo = OCCGeometry(main.get_face(), dim=2)
mesh = Mesh(geo.GenerateMesh())

gamma = cfg.heat_capacity_ratio
M = cfg.Mach_number

alpha = 0
u_inf = cfg.Mach_number * CF((cos(alpha), sin(alpha)))
rho_inf = 1
p_inf = 1
farfield = State(u_inf, rho_inf, p_inf)

# # Pressure Pulse
Gamma = 0.02
Rv = 0.1
p_0 = p_inf * (1 + Gamma * exp(-x**2/Rv**2))
initial = State(u_inf, rho_inf, p_0)

sponge = SpongeFunction(front[0].x_, 1000)
sponge = dcs.SpongeLayer(farfield, sponge)
solver = CompressibleHDGSolver(mesh, cfg)
solver.boundary_conditions.set(bcs.FarField(farfield), "left|right")
solver.boundary_conditions.set(bcs.InviscidWall(), "top|bottom")
solver.domain_conditions.set(dcs.Initial(initial))

for idx, domain in enumerate(front):
    order = cfg.order - idx - 1
    sponge = SpongeFunction.quadratic(domain.x_)
    if order == 0:
        solver.domain_conditions.set(dcs.PSpongeLayer(order, order, sponge, state=farfield), domain.mat)
    else:
        solver.domain_conditions.set(dcs.PSpongeLayer(order, order-1, sponge), domain.mat)

grid = BufferCoordinate.x(0.5, 5.5, 5)
grid = GridDeformationFunction.exponential_thickness(grid, 10, order=10)
solver.domain_conditions.set(dcs.GridDeformation(grid))
# Draw(solver.formulation.dmesh.get_grid_stretching_function(), mesh, "test")

with TaskManager():
    solver.setup()

    solver.drawer.draw(energy=True)
    solver.drawer.draw_particle_velocity(u_inf, autoscale=True, min=-1e-4, max=1e-4)
    solver.drawer.draw_acoustic_pressure(p_inf, autoscale=True, min=-1e-2, max=1e-2)
    solver.solve_transient()
