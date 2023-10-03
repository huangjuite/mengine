import time
import os
import mengine as m
import numpy as np
import copy


def invertQ(q):
    """
    Invert a quaternion
    """
    q_inv = copy.deepcopy(q)
    q_inv[:3] = -q_inv[:3]
    q_inv = q_inv / (q_inv @ q_inv)

    return q_inv


def line_intersection(p1, p2, q1, q2):
    """
    Find the intersection of two 3D line segments p1-p2 and q1-q2.
    If there is an intersection, returns the point. Otherwise, returns None.
    """
    e = p1 - p2
    e = e / np.linalg.norm(e)

    f = q1 - q2
    f = f / np.linalg.norm(f)

    g = q1 - p1
    fxg = np.cross(f, g)
    fxe = np.cross(f, e)
    h = np.linalg.norm(fxg)
    k = np.linalg.norm(fxe)

    if abs(h) < 1e-5 or abs(k) < 1e-5:
        return None

    sign = 1 if fxg @ fxe > 0 else -1

    M = p1 + (sign * h / k) * e

    return M


def get_se3(trans, q):
    x, y, z, w = q[0], q[1], q[2], q[3]
    se3 = np.identity(4)
    se3[:3, :3] = np.array(
        [
            [w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z],
        ]
    )

    se3[:3, -1] = trans

    return se3


# Create environment and ground plane
env = m.Env()
# ground = m.Ground()
env.set_gui_camera(look_at_pos=[0, 0, 0], yaw=30)

fbl = m.URDF(
    filename=os.path.join(m.directory, "fourbarlinkage.urdf"),
    static=True,
    position=[0, 0, 0.3],
    orientation=[0, 0, 0, 1],
)
fbl.controllable_joints = [0, 1, 2]
# Create a constraint for the 4th joint to create a closed loop
fbl.create_constraint(
    parent_link=1,
    child=fbl,
    child_link=4,
    joint_type=m.p.JOINT_POINT2POINT,
    joint_axis=[0, 0, 0],
    parent_pos=[0, 0, 0],
    child_pos=[0, 0, 0],
)
m.step_simulation(steps=20, realtime=False)

coupler_links = [1, 3, 5]

links = [1, 3]
global_points = []
previous_global_points = []
lines = [None, None]
lines_start_end = [[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]]

for link in links:
    global_points.append(fbl.get_link_pos_orient(link)[0])
    previous_global_points.append(global_points[-1])
    point = m.Shape(
        m.Sphere(radius=0.02),
        static=True,
        position=global_points[-1],
        rgba=[0, 0, 1, 1],
    )

intersect_points_local = []
intersect_points_local_bodies = []

for i in range(10000):
    fbl.control([np.radians(i)] * 3)

    if i > 3:
        for j, (link, global_position, previous_global_position) in enumerate(
            zip(links, global_points, previous_global_points)
        ):
            p_new = fbl.get_link_pos_orient(link)[0]

            ic_vector_of_motion = p_new - previous_global_position
            ic_bisector = np.cross(ic_vector_of_motion, [0, 1, 0])
            ic_bisector = ic_bisector / np.linalg.norm(ic_bisector)
            previous_global_points[j] = p_new

            lines[j] = m.Line(
                p_new - ic_bisector,
                p_new + ic_bisector,
                radius=0.005,
                rgba=[0, 0, 1, 0.5],
                replace_line=lines[j],
            )
            lines_start_end[j] = (p_new - ic_bisector, p_new + ic_bisector)

        if len(intersect_points_local) < 400:
            # stop drawing if we have drawn 500 points
            intersect_point = line_intersection(
                lines_start_end[0][0],
                lines_start_end[0][1],
                lines_start_end[1][0],
                lines_start_end[1][1],
            )

            if intersect_point is not None:
                m.Shape(
                    m.Sphere(radius=0.005),
                    static=True,
                    position=intersect_point,
                    collision=False,
                    rgba=[1, 0, 0, 1],
                )
                # draw moving centrode
                # get intersection point in local frame w.r.t. link 4
                trans, quat = fbl.get_link_pos_orient(3)
                T4 = get_se3(trans, quat)
                local_intersect_point = np.dot(
                    np.linalg.inv(T4), np.hstack((intersect_point, 1))
                )[:3]

                intersect_points_local.append(local_intersect_point)

                # get global coordinates of intersection point
                intersect_point_local_body = m.Shape(
                    m.Sphere(radius=0.01),
                    static=True,
                    position=intersect_point,
                    collision=False,
                    rgba=[0, 1, 0, 1],
                )
                intersect_points_local_bodies.append(intersect_point_local_body)

        # redraw intersection points of moving centrode
        # Hint: You can use Body.set_base_pos_orient(xyz) to update a body's position
        trans, quat = fbl.get_link_pos_orient(3)
        T4 = get_se3(trans, quat)

        for body, point_local in zip(
            intersect_points_local_bodies, intersect_points_local
        ):
            point_global = np.dot(T4, np.hstack((point_local, 1)))[:3]
            body.set_base_pos_orient(point_global)

    m.step_simulation(realtime=True)

    if i == 500 or i == 600 or i == 700:
        print("Please save screenshot and include in writeup")
        input("Press Enter to continue...")
