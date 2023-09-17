import os
import numpy as np
import mengine as m
np.set_printoptions(precision=3, suppress=True)

# NOTE: This problem asks you to convert between the different rotation representations.

# Create environment and ground plane
env = m.Env()
ground = m.Ground([0, 0, -0.5])
env.set_gui_camera(look_at_pos=[0, 0, 0])

# position definition
x = np.array([0.2, 0, 0])

# Create points to rotate
# point rotated using euler angles
point_e = m.Shape(m.Sphere(radius=0.03), static=True,
                  position=x, rgba=[0, 1, 0, 0.2])
# point rotated using axis-angle
point_aa = m.Shape(m.Sphere(radius=0.025), static=True,
                   position=x, rgba=[1, 0, 0, 0.2])
# point rotated using rotation matrix
point_r = m.Shape(m.Sphere(radius=0.02), static=True,
                  position=x, rgba=[0, 0, 1, 0.2])


def rodrigues_formula(n, x, theta):
    # Rodrigues' formula for axis-angle: rotate a point x around an axis n by angle theta
    # input: n, x, theta: axis, point, angle
    # output: x_new: new point after rotation
    # ------ TODO Student answer below -------
    
    nx = n*(n@x) + np.sin(theta)*np.cross(n, x) - np.cos(theta)*np.cross(n, np.cross(n ,x))

    return nx
    # ------ Student answer above -------


def rotate_euler(alpha, beta, gamma, x):
    # Rotate a point x using euler angles (alpha, beta, gamma)
    # input: alpha, beta, gamma: euler angles
    # output: x_new: new point after rotation

    # ------ TODO Student answer below -------
    R0 = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                   [np.sin(alpha), np.cos(alpha), 0],
                   [0, 0, 1]])

    R1 = np.array([[np.cos(beta),0, np.sin(beta)],
                   [0, 1, 0],
                   [-np.sin(beta), 0, np.cos(beta)]])

    R2 = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                   [np.sin(gamma), np.cos(gamma), 0],
                   [0, 0, 1]])
    nx = R0@R1@R2@x

    return nx
    # ------ Student answer above -------


def euler_to_rotation_matrix(alpha, beta, gamma):
    # Convert euler angles (alpha, beta, gamma) to rotation matrix
    # input: alpha, beta, gamma: euler angles
    # output: R: rotation matrix

    # ------ TODO Student answer below -------
    R0 = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                   [np.sin(alpha), np.cos(alpha), 0],
                   [0, 0, 1]])
    R1 = np.array([[np.cos(beta),0, np.sin(beta)],
                   [0, 1, 0],
                   [-np.sin(beta), 0, np.cos(beta)]])
    R2 = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                   [np.sin(gamma), np.cos(gamma), 0],
                   [0, 0, 1]])
    
    return R0 @ R1 @ R2
    # ------ Student answer above -------


def euler_to_axis_angle(alpha, beta, gamma):
    # Convert euler angles (alpha, beta, gamma) to axis-angle representation (n, theta)
    # input: alpha, beta, gamma: euler angles
    # output: n, theta
    # ------ TODO Student answer below -------
    R = euler_to_rotation_matrix(alpha, beta, gamma)

    theta = np.arccos((np.trace(R)-1)/2)
    r = 2*np.sin(theta)
    nx = (R[2,1]- R[1,2]) / r
    ny = (R[0,2]- R[2,0]) / r
    nz = (R[1,0]- R[0,1]) / r

    return np.array([nx, ny, nz]), theta
    # ------ Student answer above -------


x_new_e = np.array([0.2, 0, 0])
x_new_r = np.array([0.2, 0, 0])
x_new_aa = np.array([0.2, 0, 0])

for alpha, beta, gamma in zip([20, -25, 0], [45, 5, 135], [10, 90, -72]):
    alpha = np.radians(alpha)
    beta = np.radians(beta)
    gamma = np.radians(gamma)

    (n, theta) = euler_to_axis_angle(alpha, beta, gamma)
    R = euler_to_rotation_matrix(alpha, beta, gamma)

    # positions of rotated points for each representation
    x_new_e = rotate_euler(alpha, beta, gamma, x)
    x_new_r = R.dot(x)
    x_new_aa = rodrigues_formula(n, x, theta)

    print('-'*20)
    print('Euler angles:', np.degrees(alpha), np.degrees(beta), np.degrees(gamma))
    print('Axis angle:', n, np.degrees(theta))
    print('Rotation matrix:', R)
    print('x_new_e:', x_new_e)
    print('x_new_r:', x_new_r)
    print('x_new_aa:', x_new_aa)
    print('-'*20)

    point_e.set_base_pos_orient(x_new_e)
    point_r.set_base_pos_orient(x_new_r)
    point_aa.set_base_pos_orient(x_new_aa)

    m.step_simulation(realtime=True)
    input("Press enter to continue...")
