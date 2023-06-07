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
cfg.riemann_solver = "hllem"
cfg.Mach_number = 0.1
cfg.heat_capacity_ratio = 1.4
cfg.order = 3
cfg.damping_factor = 1.0
cfg.time.scheme = 'BDF2'
cfg.time.step = 0.001
cfg.time.max_step = 10
cfg.time.simulation = 'stationary'
cfg.max_iterations = 300
cfg.convergence_criterion = 1e-12
cfg.linear_solver = 'pardiso'
cfg.compile_flag = True

R = 1
R_farfield = R * 30
rho_inf = 1
u_inf = (1, 0)
p_inf = 1/(cfg.Mach_number.Get()**2 * cfg.heat_capacity_ratio.Get())

cfg.info['Cylinder Radius'] = R
cfg.info['Farfield Radius'] = R_farfield
cfg.info['Farfield Density'] = rho_inf
cfg.info['Farfield Velocity'] = u_inf
cfg.info['Farfield Pressure'] = p_inf

mesh = Mesh(Get_Omesh(R, R_farfield, 28, 12, geom=1.8))
mesh.Curve(cfg.order)

tree = ResultsDirectoryTree()

solver = CompressibleHDGSolver(mesh, cfg, tree)
solver.boundary_conditions.set_farfield('inflow', u_inf, rho_inf, p_inf)
solver.boundary_conditions.set_outflow('outflow', p_inf)
solver.boundary_conditions.set_inviscid_wall('cylinder')
solver.domain_conditions.set_initial(u_inf, rho_inf, p_inf)

sensor = PointSensor.from_boundary('cylinder', mesh, 'pressure_coefficient')
sensor.sample_pressure_coefficient(p_inf, reference_velocity=1, reference_density=rho_inf, name="c_p")
solver.add_sensor(sensor)

saver = solver.get_saver()

with TaskManager():
    solver.setup()
    solver.drawer.draw()
    solver.solve_stationary()

saver.save_configuration(comment=__doc__)
saver.save_mesh()
saver.save_sensor_data()
