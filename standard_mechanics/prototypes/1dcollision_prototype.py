import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def collision_simulator(m1, m2, u1, u2, e):

    x1 = -8
    x2 = 8

    block_size = 1.0

    dt = 0.01
    t_max = 10


    v1 = ((m1 - e * m2) * u1 + (1 + e) * m2 * u2) / (m1 + m2)

    v2 = ((m2 - e * m1) * u2 + (1 + e) * m1 * u1) / (m1 + m2)


    Ti1 = 0.5 * m1 * u1**2
    Ti2 = 0.5 * m2 * u2**2

    Tf1 = 0.5 * m1 * v1**2
    Tf2 = 0.5 * m2 * v2**2

    Ti_total = Ti1 + Ti2
    Tf_total = Tf1 + Tf2

    delta_T = Tf_total - Ti_total


    Pi1 = m1 * u1
    Pi2 = m2 * u2

    Pf1 = m1 * v1
    Pf2 = m2 * v2

    Pi_total = Pi1 + Pi2
    Pf_total = Pf1 + Pf2

    delta_P = Pf_total - Pi_total

    if np.isclose(delta_T, 0, atol=1e-5):
        collision_type = "Elastic Collision"

    elif e == 0:
        collision_type = "Perfectly Inelastic Collision"

    else:
        collision_type = "Inelastic Collision"


    print("\n========== COLLISION RESULTS ==========")

    print(f"\nFinal Velocity of Body 1 = {v1:.3f} m/s")
    print(f"Final Velocity of Body 2 = {v2:.3f} m/s")

    print("\n----- Initial Kinetic Energy -----")
    print(f"Ti1 = {Ti1:.3f} J")
    print(f"Ti2 = {Ti2:.3f} J")
    print(f"Total Initial KE = {Ti_total:.3f} J")

    print("\n----- Final Kinetic Energy -----")
    print(f"Tf1 = {Tf1:.3f} J")
    print(f"Tf2 = {Tf2:.3f} J")
    print(f"Total Final KE = {Tf_total:.3f} J")

    print(f"\nChange in KE = {delta_T:.3f} J")

    print("\n----- Initial Momentum -----")
    print(f"Pi1 = {Pi1:.3f} kg m/s")
    print(f"Pi2 = {Pi2:.3f} kg m/s")
    print(f"Total Initial Momentum = {Pi_total:.3f} kg m/s")

    print("\n----- Final Momentum -----")
    print(f"Pf1 = {Pf1:.3f} kg m/s")
    print(f"Pf2 = {Pf2:.3f} kg m/s")
    print(f"Total Final Momentum = {Pf_total:.3f} kg m/s")

    print(f"\nChange in Momentum = {delta_P:.10f} kg m/s")

    print(f"\nNature of Collision: {collision_type}")

    print("\n=======================================\n")

    times = np.arange(0, t_max, dt)

    positions1 = []
    positions2 = []

    vel1 = u1
    vel2 = u2

    collision_done = False
    collision_frame = None

    for i, t in enumerate(times):

        positions1.append(x1)
        positions2.append(x2)

        # Collision Detection
        if not collision_done:

            if abs(x2 - x1) <= block_size:

                vel1 = v1
                vel2 = v2

                collision_done = True
                collision_frame = i

        # Update Positions
        x1 += vel1 * dt
        x2 += vel2 * dt

    positions1 = np.array(positions1)
    positions2 = np.array(positions2)


    fig = plt.figure(figsize=(12, 8))

    ax_anim = plt.subplot(2, 1, 1)

    ax_anim.set_xlim(-20, 20)
    ax_anim.set_ylim(-2, 2)

    ax_anim.set_title("1D Collision Animation")

    ax_anim.set_xlabel("Position")
    ax_anim.set_yticks([])

    ax_anim.axhline(0, color='black')

    # Blocks
    block1 = plt.Rectangle(
        (positions1[0], -0.5),
        block_size,
        1,
        color='blue'
    )

    block2 = plt.Rectangle(
        (positions2[0], -0.5),
        block_size,
        1,
        color='red'
    )

    ax_anim.add_patch(block1)
    ax_anim.add_patch(block2)

    # Velocity Text
    velocity_text = ax_anim.text(
        0.02,
        0.85,
        '',
        transform=ax_anim.transAxes,
        fontsize=11
    )


    ax_graph = plt.subplot(2, 1, 2)

    ax_graph.set_title("Displacement vs Time")

    ax_graph.set_xlabel("Time (s)")
    ax_graph.set_ylabel("Displacement (m)")

    line1, = ax_graph.plot([], [], label='Body 1')
    line2, = ax_graph.plot([], [], label='Body 2')

    ax_graph.legend()

    ax_graph.set_xlim(0, t_max)

    all_positions = np.concatenate((positions1, positions2))

    ax_graph.set_ylim(
        np.min(all_positions) - 2,
        np.max(all_positions) + 2
    )


    def update(frame):

        # Move blocks
        block1.set_x(positions1[frame])
        block2.set_x(positions2[frame])

        # Velocities before/after collision
        if collision_frame is not None and frame >= collision_frame:

            current_v1 = v1
            current_v2 = v2

        else:

            current_v1 = u1
            current_v2 = u2

        velocity_text.set_text(
            f"v1 = {current_v1:.2f} m/s\n"
            f"v2 = {current_v2:.2f} m/s"
        )

        # Update graphs
        line1.set_data(times[:frame], positions1[:frame])
        line2.set_data(times[:frame], positions2[:frame])

        return block1, block2, line1, line2, velocity_text


    ani = FuncAnimation(
        fig,
        update,
        frames=len(times),
        interval=10,
        blit=True
    )

    plt.tight_layout()
    plt.show()


def main():

    print("\n========= 1D COLLISION SIMULATOR =========\n")

    m1 = float(input("Enter mass of Body 1 (m1): "))
    m2 = float(input("Enter mass of Body 2 (m2): "))

    u1 = float(input("Enter initial velocity of Body 1 (u1): "))
    u2 = float(input("Enter initial velocity of Body 2 (u2): "))

    e = float(input("Enter coefficient of restitution (e): "))

    collision_simulator(m1, m2, u1, u2, e)


# ======================================
# DRIVER CODE
# ======================================

if __name__ == "__main__":
    main()