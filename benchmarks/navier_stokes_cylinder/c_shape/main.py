from ngsolve import *
from netgen.occ import *
from dream import *
SetNumThreads(8)

R = 0.5
R_bnd = 10 * R
offset = 50 * R
R_farfield = 150 * R + offset
L_wake = 200 * R

wp = WorkPlane()

# Cylinder
cyl = wp.MoveTo(-offset+R, 0).Rotate(90).Arc(R, 180).LineTo(-offset+R, 0).Face()
for edge in cyl.edges:
    edge.name = "cyl"
cyl.maxh = 0.05


# Boundary Layer
wp.MoveTo(L_wake,0).LineTo(L_wake, R_bnd, "vortex")
wp.LineTo(-offset, R_bnd).Direction(-1, 0).Arc(R_bnd, 90).LineTo(L_wake, 0)
bl = wp.Face()
bl.maxh = 1.5

# Farfield Layer
wp.MoveTo(L_wake, 0).LineTo(L_wake, R_farfield, "vortex")
wp.LineTo(0, R_farfield, "farfield").Direction(-1, 0).Arc(R_farfield, 90).LineTo(L_wake, 0)
farfield = wp.Face()
farfield.edges[2].name = "farfield"
farfield.maxh = 10

inner = Glue([farfield - bl, bl])
geo = inner - cyl
geo = Glue([geo, geo.Mirror(Axes(n=(0, 1, 0), h=(0, 0, 1)))])
geo = OCCGeometry(geo, dim=2)

mesh = Mesh(geo.GenerateMesh())

# Enter Directory Name
tree = ResultsDirectoryTree()

# General Solver Configuration
cfg = SolverConfiguration()
cfg.formulation = 'conservative'
cfg.scaling = 'aerodynamic'
cfg.mixed_method = 'strain_heat'
cfg.dynamic_viscosity = 'constant'
cfg.riemann_solver = 'hllem'
cfg.Mach_number = 0.3
cfg.Reynolds_number = 150
cfg.order = 2
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
solver.boundary_conditions.set(bcs.NSCBC(farfield, 0.1, 1), 'vortex')
solver.boundary_conditions.set(bcs.AdiabaticWall(), 'cyl')
if not load_stationary:
    solver.domain_conditions.set(dcs.Initial(farfield))

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
r = sqrt((x+4+offset)**2 + (y-0.1)**2)
p_0 = 1.2 * exp(-r**2)
psi = 1.2 * exp(-r**2)
u_0 = CF((psi.Diff(y), -psi.Diff(x)))
rho_0 = (p_0/p_inf)**(1/cfg.heat_capacity_ratio) * rho_inf
perturbation = State(u_0, rho_0, p_0)
solver.domain_conditions.set(dcs.Perturbation(perturbation))
with TaskManager():
    solver.solve_perturbation()

# Solver Transient
cfg.time.step = 0.1
cfg.time.interval = (0, 100)
cfg.convergence_criterion = 1e-12
with TaskManager():
    solver.solve_transient()
saver.save_configuration(name=f"transient")
saver.save_state_time_scheme(f"transient_{cfg.time.interval.end}")

# Solver Transient
cfg.time.interval = (100, 400)
cfg.save_state = True

with TaskManager():
    solver.solve_transient()
saver.save_state_time_scheme(f"transient_{cfg.time.interval.end}")
