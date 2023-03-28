from netgen.geom2d import SplineGeometry
from ngsolve import *
from dream import CompressibleHDGSolver, SolverConfiguration
# from dream.utils.geometries import MakeOCCRectangle, MakeOCCCircle, MakeOCCCirclePlane
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
cfg.simulation = "stationary"
cfg.dynamic_viscosity = "constant"
cfg.mixed_method = "strain_heat"
cfg.riemann_solver = "lax_friedrich"

cfg.Reynolds_number = 10
cfg.Prandtl_number = 1  # 0.72
cfg.Mach_number = 0.3
cfg.heat_capacity_ratio = 1.4

cfg.order = 2
cfg.time_scheme = "IE"
cfg.time_step = 0.1
cfg.time_period = (0, 200)
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

u_inf = CF((1, 0))
p_inf = 1
rho_inf = p_inf * gamma / (u_inf[0]/M_inf)**2


solver = CompressibleHDGSolver(mesh, cfg)
solver.boundary_conditions.set_farfield("inflow", rho_inf, u_inf, pressure=p_inf)
solver.boundary_conditions.set_outflow("outflow", pressure=p_inf)
solver.boundary_conditions.set_adiabatic_wall("cylinder")
solver.domain_conditions.set_initial(rho_inf, u_inf, pressure=p_inf)

with TaskManager():
    solver.setup()

    solver.drawer.draw_acoustic_pressure(p_inf)
    solver.drawer.draw_particle_velocity(u_inf)
    solver.drawer.draw_mach_number()
    solver.drawer.draw_deviatoric_stress_tensor()

    solver.solve_transient()
