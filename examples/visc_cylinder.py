from netgen.geom2d import SplineGeometry
from ngsolve import *
from dream import *
from dream.utils.meshes import Get_Omesh

ngsglobals.msg_level = 0
SetNumThreads(8)

##################################################################
geo = SplineGeometry()
R = 1
R_farfield = R * 30
mesh = Mesh(Get_Omesh(R, R_farfield, 36, 16, geom=1.5))
Draw(mesh)
##################################################################
cfg = SolverConfiguration()

cfg.formulation = "conservative"
cfg.dynamic_viscosity = "constant"
cfg.mixed_method = "strain_heat"
cfg.riemann_solver = "lax_friedrich"

cfg.Reynolds_number = 10
cfg.Prandtl_number = 1  # 0.72
cfg.Mach_number = 0.3
cfg.heat_capacity_ratio = 1.4

cfg.order = 2
cfg.time.simulation = "stationary"
cfg.time.scheme = "IE"
cfg.time.step = 0.1
cfg.time.interval = (0, 200)
cfg.linear_solver = "pardiso"
cfg.damping_factor = 1
cfg.max_iterations = 100
cfg.convergence_criterion = 1e-12
cfg.compile_flag = True
cfg.static_condensation = True

mesh.Curve(cfg.order)
##################################################################
gamma = cfg.heat_capacity_ratio
M_inf = cfg.Mach_number

farfield = INF.farfield((1,0), cfg)
p_inf = INF.pressure(cfg)
u_inf = INF.velocity((1,0), cfg)



solver = CompressibleHDGSolver(mesh, cfg)
solver.boundary_conditions.set(bcs.FarField(farfield), 'inflow')
solver.boundary_conditions.set(bcs.Outflow(p_inf), 'outflow')
solver.boundary_conditions.set(bcs.NSCBC(p_inf), 'outflow')
solver.boundary_conditions.set(bcs.AdiabaticWall(), 'cylinder')
solver.domain_conditions.set(dcs.Initial(farfield))

with TaskManager():
    solver.setup()

    solver.drawer.draw_acoustic_pressure(p_inf)
    solver.drawer.draw_particle_velocity(u_inf)
    solver.drawer.draw_mach_number()
    solver.drawer.draw_deviatoric_stress_tensor()

    solver.solve_transient()
