"""
Inviscid non-lifiting flow over a circular cylinder

We benchmark the compressible Solver in the special case of an
inviscid non-lifting flow around a circular cylinder.
From literature it is well known, that the analytic solution is composed
by the superposition of two potential flows, namely the uniform flow
and doublet flow.

As validation, we compare the numerical pressure coefficient along the
cylinder surface against the analytical solution. The exact pressure coefficient
is given by c_p = 1 - 4 * sin(phi)**2.

Literature:
[1] - J.D. Anderson,
      Fundamentals of Aerodynamics, 6th edition
      New York, NY: McGraw-Hill Education, 2017.
"""
from ngsolve import *
from dream import *
from dream.utils.meshes import Get_Omesh

ngsglobals.msg_level = 0
SetNumThreads(8)

cfg = SolverConfiguration()
cfg.formulation = 'conservative'
cfg.mixed_method = None
cfg.dynamic_viscosity = None
cfg.riemann_solver = "hllem"
cfg.Mach_number = 0.1
cfg.heat_capacity_ratio = 1.4
cfg.order = 3
cfg.damping_factor = 1.0
cfg.time_scheme = 'BDF2'
cfg.time_step = 0.001
cfg.time_step_max = 10
cfg.max_iterations = 300
cfg.convergence_criterion = 1e-12
cfg.linear_solver = 'pardiso'
cfg.simulation = 'stationary'
cfg.compile_flag = True

R = 1
R_farfield = R * 30

rho_inf = 1
u_inf = CF((1, 0))
p_inf = 1/(cfg.Mach_number**2 * cfg.heat_capacity_ratio)

mesh = Mesh(Get_Omesh(R, R_farfield, 28, 12, geom=1.8))
mesh.Curve(cfg.order)

tree = ResultsDirectoryTree()

solver = CompressibleHDGSolver(mesh, cfg, tree)
solver.boundary_conditions.set_farfield('inflow', rho_inf, u_inf, pressure=p_inf)
solver.boundary_conditions.set_outflow('outflow', p_inf)
solver.boundary_conditions.set_inviscid_wall('cylinder')
solver.domain_conditions.set_initial(rho_inf, u_inf, pressure=p_inf)

sensor = PointSensor.from_boundary('cylinder', mesh, 'pressure_coefficient')
sensor.sample_pressure_coefficient(p_inf, reference_velocity=1, reference_density=rho_inf, name="c_p")
solver.add_sensor(sensor)

saver = solver.get_saver()

with TaskManager():
    solver.setup()
    solver.solve_initial()
    solver.draw_solutions()
    solver.solve_stationary()

saver.save_configuration(comment=__doc__)
saver.save_mesh()
saver.save_sensor_data()
