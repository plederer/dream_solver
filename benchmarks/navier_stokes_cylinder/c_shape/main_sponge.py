from ngsolve import *
from netgen.occ import *
from dream import *
SetNumThreads(8)

R = 0.5
R_bnd = 10 * R
offset = 50 * R
R_farfield = 150 * R + offset
R_sponge = 3000 * R
L_wake = 400 * R

wp = WorkPlane()

# Cylinder
cyl = wp.MoveTo(-offset+R, 0).Rotate(90).Arc(R, 180).LineTo(-offset+R, 0).Face()
for edge in cyl.edges:
    edge.name = "cyl"
cyl.maxh = 0.04


# Boundary Layer
wp.MoveTo(L_wake,0).LineTo(L_wake, R_bnd, "vortex")
wp.LineTo(-offset, R_bnd).Direction(-1, 0).Arc(R_bnd, 90).LineTo(L_wake, 0)
bl = wp.Face()
bl.maxh = 2

# Farfield Layer
wp.MoveTo(L_wake, 0).LineTo(L_wake, R_farfield, "vortex")
wp.LineTo(0, R_farfield).Direction(-1, 0).Arc(R_farfield, 90).LineTo(L_wake, 0)
farfield = wp.Face()
farfield.maxh = 12
farfield -= bl

# Sponge Layer
wp.MoveTo(L_wake, R_farfield).LineTo(L_wake, R_sponge, "vortex").LineTo(0, R_sponge, "farfield").LineTo(0, R_farfield).LineTo(L_wake, R_farfield)
sponge_y = wp.Face()
sponge_y.name = "sponge_y_top"
wp.MoveTo(0, R_sponge).Direction(-1, 0).Arc(R_sponge, 90).LineTo(-R_farfield, 0).Direction(0, 1).Arc(R_farfield, -90).LineTo(0, R_sponge)
sponge_r = wp.Face()
sponge_r.name = "sponge_r"
sponge_r.edges[0].name = "farfield"
sponge = Glue([sponge_y, sponge_r])
sponge.maxh = 150
sponge -= farfield

inner = Glue([sponge, farfield, bl])
geo = inner - cyl
geo = Glue([geo, geo.Mirror(Axes(n=(0, 1, 0), h=(0, 0, 1)))])
geo.faces[4].name = "sponge_y_bottom"
geo = OCCGeometry(geo, dim=2)

mesh = Mesh(geo.GenerateMesh(grading=0.2))

# Enter Directory Name
tree = ResultsDirectoryTree("c_shape_sponge")

# General Solver Configuration
cfg = SolverConfiguration()
cfg.formulation = 'conservative'
cfg.scaling = 'aerodynamic'
cfg.mixed_method = 'strain_heat'
cfg.dynamic_viscosity = 'constant'
cfg.riemann_solver = 'hllem'
cfg.Mach_number = 0.3
cfg.Reynolds_number = 150
cfg.order = 4
cfg.compile_flag = True
cfg.damping_factor = 1
cfg.max_iterations = 100
cfg.time.scheme = 'BDF2'
cfg.linear_solver = 'pardiso'
cfg.bonus_int_order_bnd = cfg.order
cfg.bonus_int_order_vol = cfg.order

farfield = INF.farfield((1,0), cfg)
rho_inf = farfield.density
p_inf = farfield.pressure

# Solver Options
load_stationary = False
draw = True

# Results Directory
saver = Saver(tree)

### Solution process ###
mesh.Curve(cfg.order)




solver = CompressibleHDGSolver(mesh, cfg, tree)
solver.boundary_conditions.set(bcs.FarField(farfield), 'farfield|vortex')
solver.boundary_conditions.set(bcs.CBC(farfield, 'nscbc', 'outflow', True, True, State(pressure=0.1, velocity=0.1)), 'vortex')
solver.boundary_conditions.set(bcs.AdiabaticWall(), 'cyl')
if not load_stationary:
    solver.domain_conditions.set(dcs.Initial(farfield))

x_ = BufferCoordinate.polar(R_farfield, R_sponge)
sponge = SpongeFunction.penta(x_)
solver.domain_conditions.set(dcs.SpongeLayer(farfield, sponge), "sponge_r")

x_ = BufferCoordinate.y(R_farfield, R_sponge)
sponge = SpongeFunction.penta(x_)
solver.domain_conditions.set(dcs.SpongeLayer(farfield, sponge), "sponge_y_top")

x_ = BufferCoordinate.y(-R_farfield, -R_sponge)
sponge = SpongeFunction.penta(x_)
solver.domain_conditions.set(dcs.SpongeLayer(farfield, sponge), "sponge_y_bottom")
dcs.SpongeLayer.fes_order = 5


saver = solver.get_saver()
loader = solver.get_loader()

with TaskManager():
    solver.setup()

if draw:
    solver.drawer.draw()
    solver.drawer.draw_particle_velocity(farfield.velocity)
    solver.drawer.draw_acoustic_pressure(farfield.pressure, sd=3, autoscale=False, min=-1e-4, max=1e-4)

# Solve Stationary
if load_stationary:
    loader.load_configuration('stationary')
    loader.load_state_time_scheme('stationary')
else:
    cfg.time.step = 0.01
    cfg.time.max_step = 10
    with TaskManager():
        solver.solve_stationary()
    saver.save_state_time_scheme('stationary')
    saver.save_configuration(name="stationary")

# Add perturbation to the solution
# r = sqrt((x+4+offset)**2 + (y-0.1)**2)
# p_0 = 1.2 * exp(-r**2)
# psi = 1.2 * exp(-r**2)
# u_0 = CF((psi.Diff(y), -psi.Diff(x)))
# rho_0 = (p_0/p_inf)**(1/cfg.heat_capacity_ratio) * rho_inf
# perturbation = State(u_0, rho_0, p_0)
# solver.domain_conditions.set(dcs.Perturbation(perturbation))
# with TaskManager():
#     solver.solve_perturbation()

# Solver Transient
cfg.time.step = 0.1
cfg.time.interval = (0, 300)
cfg.convergence_criterion = 1e-12
with TaskManager():
    solver.solve_transient()
saver.save_configuration(name=f"transient")
saver.save_state_time_scheme(f"transient_{cfg.time.interval.end}")

# Solver Transient
cfg.time.interval = (300, 700)
cfg.save_state = True

with TaskManager():
    solver.solve_transient()
saver.save_state_time_scheme(f"transient_{cfg.time.interval.end}")
