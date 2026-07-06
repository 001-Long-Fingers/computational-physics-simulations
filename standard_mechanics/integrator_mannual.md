# Numerical Integrators for Physics Simulations

## What Is an Integrator?

<img width="982" height="814" alt="image" src="https://github.com/user-attachments/assets/aa23df13-77c3-4cd5-8f51-ec00c5156cc2" />


In any physics simulation, you know the current state of an object (position `x`, velocity `v`) and the force/acceleration `a` acting on it. What you don't have is a formula for position at time `t + dt` directly — instead, you have differential equations:

```
dx/dt = v
dv/dt = a
```

An integrator is the numerical method that steps these equations forward in time by a small increment `dt`, turning continuous motion into a discrete sequence of frames/steps. Every physics sim (game engine, orbital mechanics, molecular dynamics, structural analysis) is built on top of one of these.

The core loop always looks like:

```
for each timestep:
    a = compute_acceleration(x, v)
    x, v = integrate(x, v, a, dt)
```

The choice of integrator affects three things: how accurate the result is, how stable it is over long simulations, and how much it costs to compute per step.

---

## The Methods

### 1. Explicit Euler
Uses the *old* velocity to update position, and the *old* acceleration to update velocity, both in the same step. It's a first-order method — the simplest possible approximation of the derivative.

$$x_{n+1} = x_n + v_n \, dt$$
$$v_{n+1} = v_n + a_n \, dt$$

Local truncation error is O(dt²), global error is O(dt).

### 2. Symplectic Euler
A tiny change from Explicit Euler: update velocity first, then use the *new* velocity to update position. This ordering makes it symplectic, meaning it approximately conserves energy over long time periods instead of leaking or gaining energy artificially.

$$v_{n+1} = v_n + a_n \, dt$$
$$x_{n+1} = x_n + v_{n+1} \, dt$$

Same order of accuracy as Explicit Euler (global error O(dt)), but the update order preserves the phase-space volume (Liouville's theorem), which is why energy stays bounded instead of drifting.

### 3. Runge-Kutta 4th Order (RK4)
Instead of taking one slope estimate per step, RK4 samples the derivative four times (at the start, two midpoints, and the end) and combines them in a weighted average. This cancels out a lot of the error, making it fourth-order accurate.

For the state vector y = (x, v) with dy/dt = f(y) = (v, a(x, v)):

$$k_1 = f(y_n)$$
$$k_2 = f\left(y_n + \frac{dt}{2}k_1\right)$$
$$k_3 = f\left(y_n + \frac{dt}{2}k_2\right)$$
$$k_4 = f(y_n + dt \, k_3)$$
$$y_{n+1} = y_n + \frac{dt}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

Expanded for position and velocity separately:

$$x_{n+1} = x_n + \frac{dt}{6}(v_1 + 2v_2 + 2v_3 + v_4)$$
$$v_{n+1} = v_n + \frac{dt}{6}(a_1 + 2a_2 + 2a_3 + a_4)$$

Local truncation error is O(dt⁵), global error is O(dt⁴).

### 4. Verlet Integration
Doesn't store velocity at all — it uses the current and previous position to compute the next one. This makes it slightly awkward to get instantaneous velocity out of, but it has excellent energy conservation and is very cheap per step.

Derived by adding the forward and backward Taylor expansions of position:

$$x_{n+1} = 2x_n - x_{n-1} + a_n \, dt^2$$

Velocity, if needed, is recovered with a central difference:

$$v_n = \frac{x_{n+1} - x_{n-1}}{2 \, dt}$$

Local truncation error is O(dt⁴), global error is O(dt²).

---

## Comparison

| Method | Global Error | Energy Behavior | Cost per Step | Best For |
|---|---|---|---|---|
| Explicit Euler | O(dt) | Gains energy over time (unstable) | 1 acceleration eval | Quick prototypes, not production |
| Symplectic Euler | O(dt) | Conserves energy well | 1 acceleration eval | Game physics, real-time sims |
| RK4 | O(dt⁴) | Very accurate short-term, can still drift over very long runs | 4 acceleration evals | Orbital mechanics, precision sims |
| Verlet | O(dt²) | Excellent energy conservation | 1 acceleration eval | Molecular dynamics, cloth/rope sims |

---

## Advantages and Disadvantages

**Explicit Euler**
- Advantage: trivial to implement, cheapest possible.
- Disadvantage: error grows fast, energy is not conserved, can blow up with large `dt` or stiff systems.

**Symplectic Euler**
- Advantage: same cost as Explicit Euler, but stable and energy-conserving over long runs.
- Disadvantage: still only first-order accurate per step, so short-term accuracy is limited.

**RK4**
- Advantage: much higher short-term accuracy, handles fast-changing forces well.
- Disadvantage: 4x the acceleration evaluations per step, not symplectic, so energy can slowly drift in very long simulations (e.g. long-term orbital sims).

**Verlet**
- Advantage: cheap, extremely good at conserving energy, naturally suited for systems where you don't need velocity directly (particle systems, springs, chains).
- Disadvantage: getting velocity requires an extra calculation, and you need to store/initialize `prev_x` before the loop starts.

---

## Simulation Architecture

Each physical system (SHM, projectile motion, collisions, orbits, etc.) is defined only by its differential equation — the acceleration function. The integrator is kept completely separate from the physics, so any system can be run through any integrator without rewriting either.

The architecture has three layers:

1. **System layer** — defines `a = a(x, v, t)` for that specific physics (e.g. spring force, gravity, drag).
2. **Integrator layer** — the four functions already written (`integrate_explicit_euler`, `integrate_symplectic_euler`, `integrate_rk4`, `integrate_verlet`). These know nothing about what kind of physics they're stepping.
3. **Simulator/driver layer** — holds the current state, calls the system's acceleration function, passes it into the chosen integrator, and stores the resulting state for the next frame.

```python
class Simulator:
    def __init__(self, x0, v0, a_func, integrator, dt):
        self.x = x0
        self.v = v0
        self.a_func = a_func
        self.integrator = integrator
        self.dt = dt
        self.prev_x = x0 - v0 * dt

    def step(self):
        a = self.a_func(self.x, self.v)

        if self.integrator == integrate_rk4:
            self.x, self.v = integrate_rk4(self.x, self.v, self.a_func, self.dt)

        elif self.integrator == integrate_verlet:
            next_x = integrate_verlet(self.x, self.prev_x, a, self.dt)
            self.prev_x = self.x
            self.x = next_x

        else:
            self.x, self.v = self.integrator(self.x, self.v, a, self.dt)

        return self.x, self.v
```

Example system definitions that plug into `a_func`:

```python
def shm_acceleration(x, v, k=1.0, m=1.0):
    return -(k / m) * x

def projectile_acceleration(x, v, g=-9.8):
    return g

def drag_acceleration(x, v, drag_coeff=0.1):
    return -drag_coeff * v
```

Swapping integrators for the same system is then a one-line change:

```python
sim = Simulator(x0=1.0, v0=0.0, a_func=shm_acceleration, integrator=integrate_rk4, dt=0.01)
```

This is what will let each GUI (SHM, Projectile Motion, Collision, and future sims) expose an integrator dropdown — the acceleration function stays fixed per system, only the stepping method changes, so drift and stability can be compared directly on identical physics.

---



For your simulation work, the general pattern will be:
- Prototyping / stability checks → Symplectic Euler
- High-precision short simulations (projectile motion, SHM edge cases) → RK4
- Long-running systems where energy conservation matters (orbits, particle chains) → Verlet
- Baseline comparison / demonstrating instability → Explicit Euler

Each simulator can expose a dropdown or flag to switch integrators, letting you visually compare drift and stability directly against each other on the same system.
