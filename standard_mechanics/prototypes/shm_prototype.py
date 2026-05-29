import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

m = 1.0       # mass (kg)
k = 50.0      # spring constant (N/m)
x0 = 0.1      # initial displacement (m)
v0 = 0.0      # initial velocity (m/s)
t_max = 5     # simulation time (s)
dt = 0.01

omega = np.sqrt(k / m)
t = np.arange(0, t_max, dt)
x = x0 * np.cos(omega * t) + (v0 / omega) * np.sin(omega * t)
fig, ax = plt.subplots()
line, = ax.plot([], [], 'r-', lw=2)
ax.set_xlim(0, t_max)
ax.set_ylim(-0.15, 0.15)

def update(frame):
    line.set_data(t[:frame], x[:frame])
    return line,

ani = FuncAnimation(fig, update, frames=len(t), interval=10, blit=True)
plt.show()
