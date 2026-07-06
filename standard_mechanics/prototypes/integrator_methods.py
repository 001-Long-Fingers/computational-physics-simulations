def integrate_explicit_euler(data, v, a, dt):
    next_x = data + v * dt
    next_v = v + a * dt
    return next_x, next_v

def integrate_symplectic_euler(data, v, a, dt):
    next_v = v + a * dt
    next_x = data + next_v * dt
    return next_x, next_v

def integrate_rk4(data, v, a_func, dt):
    x1, v1 = data, v
    a1 = a_func(x1, v1)

    x2 = data + v1 * dt / 2
    v2 = v + a1 * dt / 2
    a2 = a_func(x2, v2)

    x3 = data + v2 * dt / 2
    v3 = v + a2 * dt / 2
    a3 = a_func(x3, v3)

    x4 = data + v3 * dt
    v4 = v + a3 * dt
    a4 = a_func(x4, v4)

    next_x = data + (dt / 6) * (v1 + 2 * v2 + 2 * v3 + v4)
    next_v = v + (dt / 6) * (a1 + 2 * a2 + 2 * a3 + a4)
    return next_x, next_v

def integrate_verlet(data, prev_x, a, dt):
    next_x = 2 * data - prev_x + a * dt * dt
    return next_x
