import os
import numpy as np
import mengine as m
import time

env = m.Env(gravity=[0, 0, 0])


def moveto(robot, robot_marker, pos):
    # moves robot and robot reference frame ('robot_marker') to position pos
    robot.set_base_pos_orient(
        pos + np.array([0.1, -0.1, 0.0]), m.get_quaternion([np.radians(90), 0, 0])
    )
    robot_marker.set_base_pos_orient(pos, [0, 0, 0, 1])


def reset():
    # Create environment and ground plane
    env.reset()
    ground = m.Ground(position=[0, 0, -0.02])
    env.set_gui_camera(look_at_pos=[0.5, 0.5, 0], distance=0.7, pitch=-89.99)

    robot_init_position = np.array([0.0, 0, 0.0])
    robot = m.Shape(
        m.Mesh(filename=os.path.join(m.directory, "triangle.obj"), scale=[1, 0.1, 1]),
        static=False,
        position=robot_init_position,
        orientation=m.get_quaternion([np.radians(90), 0, 0]),
        rgba=[0, 1, 0, 0.5],
    )
    # mark robot reference frame
    robot_marker = m.Shape(
        m.Sphere(radius=0.02),
        static=False,
        collision=False,
        position=robot_init_position + np.array([-0.1, 0.1, 0.0]),
        rgba=[1, 1, 1, 1],
    )

    obstacle1 = m.Shape(
        m.Box(half_extents=[0.15, 0.24, 0.01]),
        static=True,
        position=[0.5, 0.5, 0.0],
        rgba=[1, 1, 0, 1],
    )
    obstacle2 = m.Shape(
        m.Box(half_extents=[0.2, 0.18, 0.01]),
        static=True,
        position=[0.9, 0.75, 0.0],
        rgba=[1, 1, 0, 1],
    )

    m.step_simulation(realtime=True)

    # ------ TODO Student answer below -------
    # hints: if robot.get_contact_points() is not None, then robot is in collision.

    c_space = []
    fix_r = m.get_quaternion([np.radians(90), 0, 0])

    for dy in np.arange(1.0, 0.0, -0.005):
        for dx in np.arange(0.2, 1.5, 0.005):
            pos = np.array([dx, dy, 0.0])
            robot.set_base_pos_orient(pos, fix_r)
            robot_marker.set_base_pos_orient(pos + np.array([-0.1, 0.1, 0.0]))
            m.step_simulation(realtime=False)

            if robot.get_contact_points() is not None:
                c_space.append([dx - 0.1, dy + 0.1, 0.0])
                # m.Shape(
                #     m.Sphere(radius=0.005),
                #     static=True,
                #     collision=False,
                #     position=np.array([dx - 0.1, dy + 0.1, 0.0]),
                #     rgba=[1, 0, 0, 0.5],
                # )

            pos = np.array([0.0, 0.0, 0.0])
            robot.set_base_pos_orient(pos, fix_r)
            robot_marker.set_base_pos_orient(pos + np.array([-0.1, 0.1, 0.0]))
            m.step_simulation(realtime=False)

    import matplotlib.pyplot as plt
    import matplotlib.patches as patch

    c_space = np.asarray(c_space)
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.scatter(c_space[:, 0], c_space[:, 1])
    ax.add_patch(
        patch.Rectangle((0.5 - 0.3 / 2, 0.5 - 0.48 / 2), 0.3, 0.48, color="yellow")
    )
    ax.add_patch(
        patch.Rectangle((0.9 - 0.4 / 2, 0.75 - 0.36 / 2), 0.4, 0.36, color="yellow")
    )
    plt.savefig("c_space.png", dpi=300)
    plt.show()



# ------ Student answer above -------


reset()
