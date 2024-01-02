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
from dream.mesh.meshes import Get_Omesh

ngsglobals.msg_level = 0
SetNumThreads(8)

cfg = SolverConfiguration()
cfg.formulation = 'conservative'
cfg.scaling = "aeroacoustic"
cfg.riemann_solver = "hllem"
cfg.Mach_number = 0.001
cfg.order = 3
cfg.damping_factor = 1.0
cfg.time.scheme = 'BDF2'
cfg.time.step = 0.001
cfg.time.max_step = 10
cfg.time.simulation = 'stationary'
cfg.max_iterations = 300
cfg.convergence_criterion = 1e-16
cfg.linear_solver = 'pardiso'
cfg.compile_flag = True

R = 1
R_farfield = R * 30

farfield = cfg.get_farfield_state((1, 0))

cfg.info['Cylinder Radius'] = R
cfg.info['Farfield Radius'] = R_farfield
cfg.info['Farfield Density'] = farfield.density
cfg.info['Farfield Velocity'] = farfield.velocity
cfg.info['Farfield Pressure'] = farfield.pressure

mesh = Mesh(Get_Omesh(R, R_farfield, 28, 12, geom=1.8))
mesh.Curve(cfg.order)

tree = ResultsDirectoryTree()

solver = CompressibleHDGSolver(mesh, cfg, tree)
solver.boundary_conditions.set(bcs.FarField(farfield), 'inflow')
solver.boundary_conditions.set(bcs.Outflow(farfield.pressure), 'outflow')
solver.boundary_conditions.set(bcs.InviscidWall(), 'cylinder')
solver.domain_conditions.set(dcs.Initial(farfield))

sensor = PointSensor.from_boundary('cylinder', mesh, 'pressure_coefficient')
sensor.sample_pressure_coefficient(farfield.pressure, reference_velocity=farfield.velocity[0], reference_density=farfield.density, name="c_p")
solver.add_sensor(sensor)

saver = solver.get_saver()

with TaskManager():
    solver.setup()
    solver.drawer.draw()
    solver.solve_stationary()

saver.save_configuration(comment=__doc__)
saver.save_mesh()
saver.save_sensor_data()
