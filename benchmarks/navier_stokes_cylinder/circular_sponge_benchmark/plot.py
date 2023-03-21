from dream import ResultsDirectoryTree, Loader, CompressibleHDGSolver, BoundarySensor, Saver, PointSensor
from ngsolve import *
import matplotlib.pyplot as plt
from numpy.fft import *

tree = ResultsDirectoryTree()
loader = Loader(tree=tree)
saver = Saver(tree)
mesh = loader.load_mesh()
cfg = loader.load_configuration("transient_configuration_fine")

solver = CompressibleHDGSolver(mesh, cfg, tree)
solver.formulation.initialize()



sensor = BoundarySensor("cylinder")
sensor.assign_solver(solver)

p_mean = GridFunction(solver.formulation.fes)
loader.load_state(p_mean, "p_mean")

loader = solver.get_loader(tree)
# for t in loader.load_state_time_sequence(sleep_time=0, load_step=1):
#     u_mean.vec.data += solver.formulation.gfu.vec
# u_mean.vec.data /= cfg.time_period.array().size

# saver.save_state(p_mean, "p_mean")

solver.drawer.draw_acoustic_pressure(solver.formulation.pressure(p_mean.components[0]))
Draw(solver.formulation.pressure(p_mean.components[0]), mesh, "p_mean")
# solver.drawer.draw_particle_velocity(solver.formulation.pressure(p_mean.components[0]))

sensor.sample_drag_coefficient(1, 1, 1, (1, 0))
sensor.sample_lift_coefficient(1, 1, 1, (0, 1))

cfg.time_period=(650, 700)
for t in loader.load_state_time_sequence(sleep_time=0, load_step=1):
    # Do whatever here
    print(t)
    sensor.take_single_sample()

t = cfg.time_period.array()
saver.save_sensor_data(sensor, t)
df = sensor.convert_samples_to_dataframe(t)

fig = plt.figure(figsize=(5, 5), dpi=100)
ax = fig.subplots(1, 1)
# ax.set_title(r"Particle velocity $u^{{'}}$")
# ax.set_ylabel(r"$u^{{'}}$")
# ax.set_xlabel(r"$t$")
# ax.set_ylim(-1e-3, 1e-3)
# ax.set_xlim(0.2, 0.7)
ax.plot(0.3 * df.index, df, label=r'$l_{sponge} = 1$')

ax.legend()

plt.show()
