import os
import numpy as np
import mengine as m
np.set_printoptions(precision=3, suppress=True)

# NOTE: This assignment asks you to test rotations and translations in varying order.

# Create environment and ground plane
env = m.Env()
ground = m.Ground([0, 0, -0.5])
env.set_gui_camera(look_at_pos=[0, 0, 0])

# position definition
pos = np.array([0.2, 0, 0.0])
# orientation definition as rotation matrix
orient = np.eye(3)

# Create box to transform
box = m.Shape(m.Box(half_extents=[0.1, 0.2, 0.05]), static=True,
              position=pos, orientation=m.get_quaternion(orient), rgba=[0, 1, 0, 1])

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

def apply_transform(pos, orient, d, euler):
    # transform a box using translation d and rotation given by euler angles
    # input: pos, orient: current position and orientation (rotation matrix) of the box
    #        d: target translation
    #        euler: target rotation in euler angles
    # output: pos_new, orient_new: new position and orientation (rotation matrix) of the box
    # ------ TODO Student answer below -------

    R = euler_to_rotation_matrix(euler[0], euler[1], euler[2])
    T = np.identity(4)
    T[:3, :3] = R
    T[:3, 3] = d

    P = np.identity(4)
    P[:3, :3] = orient
    P[:3, 3] = pos

    P = T@P

    return P[:3,3], P[:3, :3]
    # ------ Student answer above -------

# Test cases for rotations
# T_1, T2, T3
d_1 = np.array([0.1, -0.2, 0.05])
d_2 = np.array([-0.01, 0.04, 0.2])
d_3 = np.array([0.0, 0.0, 0.0])

euler_1 = np.radians([20, 45, 10])
euler_2 = np.radians([-15, 7, 23])
euler_3 = np.radians([65, 21, -19])

# 1: T1, T2, T3
pos1, orient1 = apply_transform(pos, orient, d_1, euler_1)
pos2, orient2 = apply_transform(pos1, orient1, d_2, euler_2)
pos_final, orient_final = apply_transform(pos2, orient2, d_3, euler_3)
# update box position and orientation
box.set_base_pos_orient(pos_final, m.get_quaternion(orient_final))
print('-'*20)
print('final position:', pos_final)
print('final orientation:', orient_final)
print('-'*20)
input("Press enter to continue...")

# 2: T2, T1, T3
pos1, orient1 = apply_transform(pos, orient, d_2, euler_2)
pos2, orient2 = apply_transform(pos1, orient1, d_1, euler_1)
pos_final, orient_final = apply_transform(pos2, orient2, d_3, euler_3)
# update box position and orientation
box.set_base_pos_orient(pos_final, m.get_quaternion(orient_final))
print('-'*20)
print('final position:', pos_final)
print('final orientation:', orient_final)
print('-'*20)
input("Press enter to continue...")

# 3: T3, T2, T1
pos1, orient1 = apply_transform(pos, orient, d_3, euler_3)
pos2, orient2 = apply_transform(pos1, orient1, d_2, euler_2)
pos_final, orient_final = apply_transform(pos2, orient2, d_1, euler_1)
# update box position and orientation
box.set_base_pos_orient(pos_final, m.get_quaternion(orient_final))
print('-'*20)
print('final position:', pos_final)
print('final orientation:', orient_final)
print('-'*20)
input("Press enter to continue...")

