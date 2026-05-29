import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.animation as animation

g = 10
m = 10
b = 0.0  # drag coefficient set to zero (no air resistance)

def equations(t, y):
    x, y_pos, vx, vy = y
    ax = -b / m * vx
    ay = -g - (b / m) * vy
    return [vx, vy, ax, ay]

y0 = [0.0, 0.0, 15.0, 15.0] 
t_span = (0, 3 )
t_eval = np.linspace(t_span[0], t_span[1], 300)

sol = solve_ivp(equations, t_span, y0, t_eval=t_eval, rtol=1e-9, atol=1e-12)
x = np.atleast_1d(np.array(sol.y[0], dtype=float))
y = np.atleast_1d(np.array(sol.y[1], dtype=float))

fig, ax = plt.subplots()    
ax.set_xlim(0, float(np.max(x)) * 1.1)
ax.set_ylim(0, float(np.max(y)) * 1.1)
ax.set_xlabel('X position (m)')
ax.set_ylabel('Y position (m)')
ax.set_title('Projectile Motion with Air Resistance')

ax.plot(x, y, color='lightgray', linestyle='--')

point, = ax.plot([], [], 'ro')
trail, = ax.plot([], [], 'b-', lw=1)

def init():
    point.set_data([], [])
    trail.set_data([], [])
    return point, trail

def update(frame):
    point.set_data([x[frame]], [y[frame]])
    trail.set_data(x[:frame+1], y[:frame+1])
    return point, trail

frames_count = len(x) if len(x) > 1 else 1
ani = animation.FuncAnimation(fig, update, frames=frames_count, init_func=init, interval=30, blit=True, repeat=False)
plt.show()
