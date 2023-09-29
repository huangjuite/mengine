import numpy as np
import os
import math
import mengine as m

"""
16-741 Assignment 2 Problem 1.

Attention: quaternions are represented as [x, y, z, w], same as in pybullet.

"""

np.set_printoptions(precision=4, suppress=True)


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix R (3x3) to quaternion q (1x4)."""
    # input: R: rotation matrix
    # output: q: quaternion

    q = np.zeros(4)
    q_tmp = np.zeros(4)

    q_tmp[0] = np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2.0
    q_tmp[1] = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) / 2.0
    q_tmp[2] = np.sqrt(1 - R[0, 0] + R[1, 1] - R[2, 2]) / 2.0
    q_tmp[3] = np.sqrt(1 - R[0, 0] - R[1, 1] + R[2, 2]) / 2.0
    m_arg = np.argmax(q_tmp)

    q0q1 = (R[2, 1] - R[1, 2]) / 4.0
    q0q2 = (R[0, 2] - R[2, 0]) / 4.0
    q0q3 = (R[1, 0] - R[0, 1]) / 4.0
    q1q2 = (R[1, 0] + R[0, 1]) / 4.0
    q1q3 = (R[0, 2] + R[2, 0]) / 4.0
    q2q3 = (R[2, 1] + R[1, 2]) / 4.0

    if m_arg == 0:
        q[0] = q_tmp[0]
        q[1] = q0q1 / q[0]
        q[2] = q0q2 / q[0]
        q[3] = q0q3 / q[0]

    elif m_arg == 1:
        q[1] = q_tmp[1]
        q[0] = q0q1 / q[1]
        q[2] = q1q2 / q[1]
        q[3] = q1q3 / q[1]

    elif m_arg == 2:
        q[2] = q_tmp[2]
        q[0] = q0q2 / q[2]
        q[1] = q1q2 / q[2]
        q[3] = q2q3 / q[2]

    elif m_arg == 3:
        q[3] = q_tmp[3]
        q[0] = q0q3 / q[3]
        q[1] = q1q3 / q[3]
        q[2] = q2q3 / q[3]

    q = -q if q[0] < 0 else q
    w, x, y, z = q

    return np.array([x, y, z, w])


def rodrigues_formula(n, x, theta):
    # Rodrigues' formula for axis-angle: rotate a point x around an axis n by angle theta
    # input: n, x, theta: axis, point, angle
    # output: x_new: new point after rotation
    nx = (
        n * (n @ x)
        + np.sin(theta) * np.cross(n, x)
        - np.cos(theta) * np.cross(n, np.cross(n, x))
    )
    return nx


def axis_angle_to_quaternion(axis: np.ndarray, angle: float) -> np.ndarray:
    """Convert axis-angle representation to quaternion."""
    # input: axis: axis of rotation
    #        angle: angle of rotation (radians)
    # output: q: quaternion
    w = np.cos(angle / 2)
    x, y, z = np.sin(angle / 2) * axis
    return np.array([x, y, z, w])


def hamilton_product(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p3q3 = p[3] * q[3]
    p3q = p[3] * q[:3]
    q3p = q[3] * p[:3]
    pq = np.dot(p[:3], q[:3])
    pxq = np.cross(p[:3], q[:3])
    w = p3q3 - pq
    x, y, z = p3q + q3p + pxq
    return np.array([x, y, z, w])


def unit_tests():
    """Simple unit tests.
    Passing these test cases does NOT ensure your implementation is fully correct.
    """
    # test rotation_matrix_to_quaternion
    q = rotation_matrix_to_quaternion(np.diag([1, -1, -1]))
    try:
        assert np.allclose(q, [1, 0, 0, 0]) or np.allclose(q, [-1, 0, 0, 0])
        print("rotation_matrix_to_quaternion passed test case 1")
    except AssertionError:
        print("rotation_matrix_to_quaternion failed test case 1")

    R = np.array(
        [[-0.545, 0.797, 0.260], [0.733, 0.603, -0.313], [-0.407, 0.021, -0.913]]
    )
    q = rotation_matrix_to_quaternion(R)
    try:
        assert np.allclose(q, [0.437, 0.875, -0.0836, 0.191], atol=1e-3)
        print("rotation_matrix_to_quaternion passed test case 2")
    except AssertionError:
        print("rotation_matrix_to_quaternion failed test case 2")

    # test axis_angle_to_quaternion
    q = axis_angle_to_quaternion(np.array([1, 0, 0]), 0.123)
    try:
        assert np.allclose(q, [0.06146124, 0, 0, 0.99810947])
        print("axis_angle_to_quaternion passed test case")
    except AssertionError:
        print("axis_angle_to_quaternion failed test case")

    # test hamilton_product
    p = np.array([0.437, 0.875, -0.0836, 0.191])
    q = np.array([0.06146124, 0, 0, 0.99810947])
    try:
        assert np.allclose(
            hamilton_product(p, q), [0.4479, 0.8682, -0.1372, 0.1638], atol=1e-3
        )
        print("hamilton_product passed test case")
    except AssertionError:
        print("hamilton_product failed test case")


if __name__ == "__main__":
    # Create environment and ground plane
    env = m.Env()
    ground = m.Ground([0, 0, -0.5])
    env.set_gui_camera(look_at_pos=[0, 0, 0])

    # Axis-angle definition
    n = np.array([0, 0, 1])
    x = np.array([0.2, 0, 0])

    # Create axis
    axis = m.Shape(
        m.Cylinder(radius=0.02, length=0.5),
        static=True,
        position=[0, 0, 0],
        orientation=n,
        rgba=[0.8, 0.8, 0.8, 1],
    )
    # Create point to rotate around axis
    point = m.Shape(m.Sphere(radius=0.02), static=True, position=x, rgba=[0, 0, 1, 0.5])
    point_q = m.Shape(
        m.Sphere(radius=0.02), static=True, position=x, rgba=[0, 1, 0, 0.5]
    )

    # First we want to implement some converstions and the Hamilton product for quaternions.
    print("Running unit tests...")
    unit_tests()

    x_new_report = []
    x_new_q_report = []

    for i in range(501):
        theta = np.radians(i)
        # Rodrigues' formula for axis-angle rotation
        x_new = rodrigues_formula(n, x, theta)

        # Axis-angle to quaternion
        theta = np.radians(i - 10)  # Offset theta so we can see the two points
        q = axis_angle_to_quaternion(n, theta)

        # rotate using quaternion and the hamilton product
        qx = hamilton_product(q, np.hstack((x, 0)))
        qc = q
        qc[:3] *= -1
        x_new_q = hamilton_product(qx, qc)[:3]

        point.set_base_pos_orient(x_new)
        point_q.set_base_pos_orient(x_new_q)

        m.step_simulation(realtime=True)

        if i % 50 == 0 and i < 501:
            x_new_report.append(x_new.tolist())
            x_new_q_report.append(x_new_q[:3].tolist())
        if i == 500:
            print("=" * 30)
            print("Point rotated using rodrigues formula: ")
            for row in x_new_report:
                formatted_row = [f"{elem:.4f}" for elem in row]
                print(formatted_row)
            print("=" * 30)
            print("Point rotated using hamilton product: ")
            for row in x_new_q_report:
                formatted_row = [f"{elem:.4f}" for elem in row]
                print(formatted_row)
            print("=" * 30)
