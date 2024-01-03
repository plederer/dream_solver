# %% Initialize Results Directory and Load Configuration
from dream import ResultsDirectoryTree, Loader, CompressibleHDGSolver, BoundarySensor, PointSensor
from ngsolve import *
import matplotlib.pyplot as plt
from numpy.fft import *
import numpy as np

SetNumThreads(8)
# Result Directory
parent_path = "/media/jellmenr/DreAm/simulation_results/navier_stokes_benchmark/c_shape"
directory = "c_shape_sponge_outflow"

tree = ResultsDirectoryTree(directory, parent_path=parent_path)
loader = Loader(tree=tree)

cfg = loader.load_configuration("transient")
mesh = loader.load_mesh()
mesh.Curve(cfg.order)

solver = CompressibleHDGSolver(mesh, cfg, tree)
solver.formulation.initialize()

loader = solver.get_loader()
saver = solver.get_saver()

# %% Sample Lift and Drag Coefficient
cfg.time.interval = (0, 300)

sensor = BoundarySensor("cylinder", name='cylinder')
sensor = BoundarySensor("cyl", name='cyl')
sensor.sample_lift_coefficient(1, 1, 1, (0, 1))
sensor.sample_drag_coefficient(1, 1, 1, (1, 0))
sensor.assign_solver(solver)

points = PointSensor([(0, 0), (-25, 1), (-25, -1), (-15, 1.5), (-15, -1.5), (-15, 0)], name="Components")
points.sample_pressure()
points.sample_voritcity()
points.sample_velocity()
points.assign_solver(solver)

with TaskManager():
    for idx, t in enumerate(loader.load_state_time_sequence(sleep_time=0, load_step=1)):
        sensor.take_single_sample(t)
        points.take_single_sample(t)

        print(f"Time Step = {t}", end='\r')

saver.save_sensor_data(sensor)
saver.save_sensor_data(points)

# %% Plot Lift and Drag
%matplotlib qt5
df = loader.load_sensor_data("cyl")
df = loader.load_sensor_data("Components")
# dff = df.xs((-25, 1), axis=1, level=2)
dff = df.xs('pressure', axis=1, level=0)

t = df.index

fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
# ax.plot(t, df['drag_coefficient'], label=r"$c_D$")
# ax.plot(t, df['lift_coefficient'], label=r"$c_L$")
ax.plot(t, dff-1/(1.4*0.3**2), label=dff.columns)
ax.legend()
# ticks = np.pi * np.linspace(-1, 1, 13)[1:]
# ax.set_xticks(ticks)
# ax.set_thetalim(-np.pi, np.pi)

# ax.set_yticklabels(['', '', '', r"$1 \times 10^{-4}$"])
# ax.set_title(r"$\Delta \tilde{p}_{rms}$ at $r = 75 \cdot ( 1 - \rm{M} \cos(\theta))$")
plt.tight_layout()
plt.show()

# %% Calculate Mean Pressure
recalculate = True
name = "p_mean"
load_step = 1
cfg.time.interval = (300, 400)

p_mean = GridFunction(L2(mesh, order=cfg.order))


def calculate_mean_pressure(load_step: int = 1, name: str = "p_mean"):
    p_mean_load = GridFunction(L2(mesh, order=cfg.order))
    with TaskManager():
        for t in loader.load_state_time_sequence(sleep_time=0, load_step=load_step):
            p_mean_load.Set(solver.formulation.pressure())
            p_mean.vec.data += p_mean_load.vec
            print(f"Time Step = {t}", end='\r')
        p_mean.vec.data /= cfg.time.interval.array().size/load_step
    saver.save_state(p_mean, name)


if recalculate:
    calculate_mean_pressure(load_step, name)
else:
    try:
        loader.load_state(p_mean, name)
    except Exception:
        calculate_mean_pressure(load_step, name)

# %% Calculate RMS Pressure
recalculate = True
name = "p_rms"
load_step = 1
cfg.time.interval = (300, 400)

p_rms = GridFunction(L2(mesh, order=2 * cfg.order))


def calculate_rms_pressure(load_step: int = 1, name: str = "p_rms"):
    p_rms_load = GridFunction(L2(mesh, order=2 * cfg.order))
    with TaskManager():
        for t in loader.load_state_time_sequence(sleep_time=0, load_step=load_step):
            p_rms_load.Set((solver.formulation.pressure() - p_mean)**2)
            p_rms.vec.data += p_rms_load.vec
            print(f"Time Step = {t}", end='\r')
        p_rms.vec.data /= cfg.time.interval.array().size/load_step
    saver.save_state(p_rms, name)


if recalculate:
    calculate_rms_pressure(load_step, name)
else:
    try:
        loader.load_state(p_rms, name)
    except Exception:
        calculate_rms_pressure(load_step, name)


# %% Polar Pressure Projection
%matplotlib qt5

N = 101
theta = np.pi * np.linspace(-1, 1, N)[1:]
R = 75 * (1 - cfg.Mach_number.Get()*np.cos(theta))

coords = np.vstack([R * np.cos(theta), R * np.sin(theta)]).T
p_plot = np.sqrt(np.array([p_rms(mesh(*point)) for point in coords])) * cfg.Mach_number.Get()**2

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 6), dpi=200)
ax.scatter(theta, p_plot, color="white", edgecolor="k", marker='o')
ax.set_rticks([0.25e-4, 0.5e-4, 0.75e-4, 1e-4])  # Less radial ticks
ticks = np.pi * np.linspace(-1, 1, 13)[1:]
ax.set_xticks(ticks)
ax.set_thetalim(-np.pi, np.pi)

ax.set_yticklabels(['', '', '', r"$1 \times 10^{-4}$"])
ax.set_title(r"$\Delta \tilde{p}_{rms}$ at $r = 75 \cdot ( 1 - \rm{M} \cos(\theta))$")
plt.tight_layout()
plt.show()


# %% Radial Pressure Plot over Theta
%matplotlib qt5

file = "transient_349.0"
loader.load_state(name=file)
p = solver.formulation.pressure()
p_acou = p - p_mean
R = np.linspace(0.5, 100, 1000)
for theta in [180 - 50, 180-78.5, 180-120]:
    fig, ax = plt.subplots(figsize=(4.3, 5.8), dpi=200)

    theta_pi = theta * np.pi/180

    p_plot = (np.array([p(mesh(r*np.cos(theta_pi), r * np.sin(theta_pi)))
              for r in R])) * cfg.Mach_number.Get()**2 - 1/1.4
    p_mean_plot = np.array([p_mean(mesh(r*np.cos(theta_pi), r * np.sin(theta_pi)))
                           for r in R]) * cfg.Mach_number.Get()**2 - 1/1.4
    p_acou_plot = np.array([p_acou(mesh(r*np.cos(theta_pi), r * np.sin(theta_pi)))
                           for r in R]) * cfg.Mach_number.Get()**2

    ax.plot(R, p_plot, c='k', ls='--', label=r"$\Delta p$")
    ax.plot(R, p_mean_plot, c='k', ls='-.', label=r"$\Delta p_{mean}$")
    ax.plot(R, p_acou_plot, c='k', ls='-', label=r"$\Delta \tilde{p}$")
    ax.ticklabel_format(axis='both', style='scientific')
    yticks = [-4e-4, -2e-4, 0, 2e-4, 4e-4]
    ax.set_yticks(yticks)
    ax.set_xticks([0, 50, 100])
    ax.set_yticklabels([r"$-4 \times 10^{-4}$", "", "$0$", "", r"$4 \times 10^{-4}$"])
    ax.set_ylim(-4e-4, 4e-4)
    ax.set_ylabel(r"$\Delta \tilde{p}$", rotation='horizontal')
    ax.set_xlabel(r"$r$")
    ax.tick_params(direction='in', right=True, bottom=True)
    ax.legend()
    ax.set_title(rf"$\theta = {theta}^\circ$")
    ax.grid(which='major', axis='y', ls='--')
    plt.tight_layout()
    plt.show()

# %% Radial Pressure Plot over Time
%matplotlib qt5

fig, ax = plt.subplots(figsize=(4.3, 5.8), dpi=200)

for file in [f'transient_{round(i,1)}' for i in np.linspace(349, 351, 3)]:
    loader.load_state(name=file)
    p_acou = solver.formulation.pressure() - p_mean

    R = np.linspace(0.5, 100, 1000)
    p_plot = np.array([p_acou(mesh(0, r)) for r in R]) * cfg.Mach_number.Get()**2

    ax.plot(R, p_plot, label=f"$t={file[-5:]}$")
ax.ticklabel_format(axis='both', style='scientific')
yticks = [-4e-4, -2e-4, 0, 2e-4, 4e-4]
ax.set_yticks(yticks)
ax.set_xticks([0, 50, 100])
ax.set_yticklabels([r"$-4 \times 10^{-4}$", "", "$0$", "", r"$4 \times 10^{-4}$"])
ax.set_ylim(-4e-4, 4e-4)
ax.set_ylabel(r"$\Delta \tilde{p}$", rotation='horizontal')
ax.set_xlabel(r"$r$")
ax.tick_params(direction='in', right=True, bottom=True)
ax.legend()
ax.set_title(r"$\theta = 90^\circ$")
ax.grid(which='major', axis='y', ls='--')
plt.tight_layout()
plt.show()
