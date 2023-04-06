import numpy as np
import matplotlib.pyplot as plt
from dream import Loader

# Load Dataframe
loader = Loader()
df = loader.load_sensor_data('pressure_coefficient')['c_p', '']

# Extract Coords and Pressure Coefficient
coords = np.array([point for point in df])
x, y = coords[:, 0], coords[:, 1]
c_p = df.iloc[-1].to_numpy()

# Calculate Angle and Sort
phi = np.angle(x + 1j*y)
c_p = c_p[np.argsort(phi)]
phi.sort()

# Exact solution
phi_exact = np.linspace(-np.pi, np.pi, 100)
c_p_exact = 1 - 4 * np.sin(phi_exact)**2

# Draw
fig = plt.figure(figsize=(5, 5), dpi=100)
ax = fig.subplots(1, 1)
ax.plot(phi_exact, c_p_exact, color='k')
ax.scatter(phi, c_p, color='red')
ax.set_xlabel(r"$\phi$")
ax.set_ylabel(r"$c_p$")
ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax.set_xticklabels([r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'])
fig.savefig("pressure_coefficient.png")
