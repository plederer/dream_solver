#%%
from ngsolve import *
from netgen.occ import *
from dream import *
SetNumThreads(8)

R = 0.5
offset = 5
name = ""

wp = WorkPlane()

# Cylinder
cyl = wp.Circle(0, 0, R).Face()
cyl.edges.name = "cyl"
cyl.edges.maxh = 0.035

bl = 4
wp.MoveTo(0, bl*R).Direction(-1, 0).Arc(bl*R, 180)
wp.LineTo((20+offset)*R, -bl*R)
wp.LineTo((20+offset)*R, bl*R)
wp.LineTo(0, bl*R)
wake = wp.Face()

wake.faces.maxh = 0.5
wake -= cyl

# domain
domain = wp.MoveTo(offset*R, 0).RectangleC(40*R, 40*R).Face()
domain.faces.maxh = 1.5

for edge, name_ in zip(domain.edges, ['planar', 'outflow', 'planar', 'inflow']):
    edge.name = name_

domain -= cyl

dom = Glue([wake, domain])
dom.faces.name = "inner"

geo = OCCGeometry(dom, dim=2)

# Enter Directory Name
tree = ResultsDirectoryTree()

# General Solver Configuration
cfg = SolverConfiguration()
cfg.formulation = 'conservative'
cfg.scaling = 'aerodynamic'
cfg.mixed_method = 'strain_heat'
cfg.dynamic_viscosity = 'constant'
cfg.riemann_solver = 'farfield'
cfg.Mach_number = 0.1
cfg.Reynolds_number = 200
cfg.order = 4
cfg.compile_flag = True
cfg.static_condensation = True
cfg.damping_factor = 1
cfg.max_iterations = 10
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
mesh = Mesh(geo.GenerateMesh(maxh=1.5))
mesh.Curve(cfg.order)
Draw(mesh)


def test(name: str = ""):

    def wraps(func):

        def wrapper(*args, **kwargs):

            info = cfg.info.copy()

            tree.directory_name = f"Ma{cfg.Mach_number.Get()}/Re{cfg.Reynolds_number.Get()}"

            tree.state_directory_name = func.__name__
            if name:
                tree.state_directory_name += f"_{name}"

            solver = func(*args, **kwargs)
            solver.boundary_conditions.set(bcs.AdiabaticWall(), 'cyl')
            if not load_stationary:
                solver.domain_conditions.set(dcs.Initial(farfield))

            saver = solver.get_saver()
            loader = solver.get_loader()

            saver.state_path
            saver.save_mesh()

            LOGGER.log_to_file = True
            with TaskManager():
                solver.setup()

            if draw:
                solver.drawer.draw()
                solver.drawer.draw_particle_velocity(farfield.velocity)
                solver.drawer.draw_acoustic_pressure(farfield.pressure, sd=3, autoscale=False, min=-1e-4, max=1e-4)

            # Solve Stationary
            if load_stationary:
                loader.load_configuration('stationary_cfg')
                loader.load_state_time_scheme('stationary_0.0')
            else:
                cfg.time.step = 0.01
                cfg.max_iterations = 200
                cfg.time.max_step = 1
                with TaskManager():
                    solver.solve_stationary()
                saver.save_state_time_scheme('stationary')
                saver.save_configuration(name=f"{tree.state_directory_name}/stationary_cfg")

            # Solver Transient
            cfg.time.step = 0.002
            cfg.time.interval = (0, 100)
            cfg.convergence_criterion = 1e-12
            cfg.save_state = False
            with TaskManager():
                solver.solve_transient()
            saver.save_configuration(name=f"{tree.state_directory_name}/transient_cfg")
            saver.save_state_time_scheme(f"transient_{cfg.time.interval.end}")

            # # Solver Transient
            cfg.time.interval = (0.3, 0.4)
            cfg.save_state = True

            with TaskManager():
                solver.solve_transient()
            saver.save_state_time_scheme(f"transient_{cfg.time.interval.end}")

            cfg._info = info
            LOGGER.log_to_file = False


        return wrapper

    return wraps

@test(name)
def grcbc_farfield_inflow_and_outflow():
    solver = CompressibleHDGSolver(mesh, cfg, tree)
    solver.boundary_conditions.set(bcs.CBC(farfield, sigma=State(velocity=0.01, pressure=0.01)), 'inflow')
    solver.boundary_conditions.set(bcs.CBC(farfield, sigma=State(velocity=0.01, pressure=0.01)), 'planar')
    solver.boundary_conditions.set(bcs.CBC(farfield, relaxation="farfield", convective_tangential_flux=True, viscous_fluxes=True, sigma=State(velocity=0.01, pressure=0.01)), 'outflow')
    return solver

@test(name)
def grcbc_farfield_inflow_and_pressure_outflow():
    solver = CompressibleHDGSolver(mesh, cfg, tree)
    solver.boundary_conditions.set(bcs.CBC(farfield, sigma=State(velocity=0.01, pressure=0.01)), 'inflow')
    solver.boundary_conditions.set(bcs.CBC(farfield, sigma=State(velocity=0.01, pressure=0.01)), 'planar')
    solver.boundary_conditions.set(bcs.CBC(farfield, relaxation="outflow", convective_tangential_flux=True, viscous_fluxes=True, sigma=State(velocity=1, pressure=0.01)), 'outflow')
    return solver

@test(name)
def nscbc_farfield_inflow_and_pressure_outflow():
    solver = CompressibleHDGSolver(mesh, cfg, tree)
    solver.boundary_conditions.set(bcs.CBC(farfield, "nscbc"), 'inflow')
    solver.boundary_conditions.set(bcs.CBC(farfield, "nscbc"), 'planar')
    solver.boundary_conditions.set(bcs.CBC(farfield, "nscbc", relaxation="outflow", convective_tangential_flux=True, viscous_fluxes=True), 'outflow')
    return solver

@test(name)
def farfield_inflow_and_outflow():
    solver = CompressibleHDGSolver(mesh, cfg, tree)
    solver.boundary_conditions.set(bcs.FarField(farfield, Qform=True), 'outflow|inflow|planar')
    return solver

@test(name)
def farfield_inflow_and_pressure_outflow():
    solver = CompressibleHDGSolver(mesh, cfg, tree)
    solver.boundary_conditions.set(bcs.FarField(farfield, Qform=True), 'inflow|planar')
    solver.boundary_conditions.set(bcs.Outflow(farfield.pressure), 'outflow')
    return solver

if __name__ == "__main__":
    grcbc_farfield_inflow_and_outflow()
    grcbc_farfield_inflow_and_pressure_outflow()
    nscbc_farfield_inflow_and_pressure_outflow()
    farfield_inflow_and_outflow()
    farfield_inflow_and_pressure_outflow()