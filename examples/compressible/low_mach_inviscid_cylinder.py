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
from dream.compressible import InviscidWall, FarField, Initial, flowstate
from dream.mesh import get_cylinder_omesh

ngsglobals.msg_level = 0
SetNumThreads(8)

# Define Geometry
ri = 1
ro = ri * 30
mesh = get_cylinder_omesh(ri, ro, 28, 12, geom=1.8)

# Define Flow Parameters
cfg = SolverConfiguration(mesh)
cfg.pde = "compressible"
cfg.pde.fem = 'conservative'
cfg.pde.scaling = "acoustic"
cfg.pde.riemann_solver = "hllem"
cfg.pde.mach_number = 0.001
cfg.pde.equation_of_state.heat_capacity_ratio = 1.4
cfg.pde.fem.order = 5

cfg.solver = "nonlinear"
cfg.solver.method = "newton"
cfg.solver.max_iterations = 300
cfg.solver.inverse = "direct"
cfg.solver.inverse.solver = "pardiso"
cfg.solver.convergence_criterion = 1e-12

cfg.time = "pseudo_time_stepping"
cfg.time.timer.step = 0.001
cfg.time.max_time_step = 10


cfg.optimizations.static_condensation = True


mesh.Curve(cfg.pde.fem.order)
Uinf = cfg.pde.get_farfield_state((1, 0))

# Define Boundary Conditions and Domain Conditions
cfg.pde.bcs['left|right'] = FarField(state=Uinf)
cfg.pde.bcs['cylinder'] = InviscidWall()
cfg.pde.dcs['default'] = Initial(state=Uinf)


# Setup Spaces and Gridfunctions
cfg.pde.initialize_system()

fields = cfg.pde.get_fields()
# cfg.pde.draw(fields, autoscale=False, min=-1e-4, max=1e-4)

cfg.io.sensor = True
cfg.io.sensor.point = "pressure_coefficient"
cfg.io.sensor.point["pressure_coefficient"].points = 'cylinder'
cfg.io.sensor.point["pressure_coefficient"].fields = flowstate(c_p=cfg.pde.pressure_coefficient(fields, Uinf))

cfg.solver.initialize()
with TaskManager():
    cfg.solver.solve()

# Postprocess
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    raise ImportError("This example requires pandas and matplotlib to be installed")

df = cfg.io.sensor.load_as_dataframe('pressure_coefficient')
df.sort_index(axis=1, inplace=True)

# Extract Coords and Pressure Coefficient
coords = np.array([eval(point) for point in df.columns.levels[0]])
cp_h = df.iloc[-1].to_numpy()

# Calculate Angle and Sort
phi_h = np.angle(coords[:, 0] + 1j*coords[:, 1])
cp_h = cp_h[np.argsort(phi_h)]
phi_h.sort()

# Exact solution
phi = np.linspace(-np.pi, np.pi, 100)
cp = 1 - 4 * np.sin(phi)**2

# Draw
fig = plt.figure(figsize=(5, 5), dpi=100)
ax = fig.subplots(1, 1)
ax.plot(phi, cp, color='k')
ax.scatter(phi_h, cp_h, color='red')
ax.set_xlabel(r"$\varphi$")
ax.set_ylabel(r"$c_p$")
ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax.set_xticklabels([r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'])
plt.show()
