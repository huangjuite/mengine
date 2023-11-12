"""
Assignment 4 Problem 3: Antipodal Grasp

NOTE:
    First install open3d using: 'python3 -m pip install open3d'
    On Mac you also need 'brew install libomp'
"""
import os
import time
from typing import Tuple, List
import numpy as np
import open3d as o3d
import mengine as m
from scipy.spatial.transform import Rotation as R


def sample_grasp_ee_poses(obj, num_samples=100) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Sample end effector poses around the object.

    Returns:
        ee_poses: A list of end effector poses (position, orientation)
    """
    # ------ TODO Student answer below -------
    # Note: If you want, the following code can give you a bounding box for the object: obj_min, obj_max = obj.get_AABB()

    # eulers = np.random.random((num_samples, 3)) * np.pi
    # rotations = R.from_euler("zyx", eulers).as_quat()

    eulers = np.random.random((num_samples, 3)) * np.pi
    eulers[:, 1] = np.pi
    eulers[:, 2] = 0
    rotations = R.from_euler("zyx", eulers).as_quat()

    obj_min, obj_max = obj.get_AABB()

    xs = np.random.uniform(obj_min[0] - 0.04, obj_max[0] + 0.04, num_samples)
    ys = np.random.uniform(obj_min[1] - 0.04, obj_max[1] + 0.04, num_samples)
    zs = np.random.uniform(obj_min[2], obj_max[2], num_samples)

    poses = []
    for i in range(num_samples):
        poses.append((np.array([xs[i], ys[i], zs[i]]), rotations[i]))

    return poses
    # ------ Student answer above -------


def get_antipodal_score(robot_joint_angles, pc, normals) -> float:
    """Compute antipodal score for a given robot configuration (joint angles).

    Args:
        robot_joint_angles: Robot joint angles
        pc: Point cloud of target object
        normals: Point cloud normals

    Variables defined outside this function:
        robot: mengine Robot
        antipodal_region: A region that allows us to identify points within the gripper
        gripper_line_vector: A vector that represents the gripper finger axis
        half_extents: Half of the antipodal region bounding box extents

    Returns:
        score: Antipodal score

    Notes:
        1. Check if the robot is in collision or if there is no robot_joint_angles (no IK solution).
        2. Set the robot to robot_joint_angles
        3. Compute the antipodal score.
        4. Restore the robot to its previous configuration.
    """
    score = 0
    # No IK solution or in collision
    if robot_joint_angles is None or robot_in_collision(robot_joint_angles):
        return 0

    # Set robot to joint angles
    prev_joint_angles = robot.get_joint_angles(robot.controllable_joints)
    robot.control(robot_joint_angles, set_instantly=True)

    # ------ TODO Student answer below -------
    # Hint: example code for computing an antipodal grasp score can be found in lecture12_antipodal.py

    # Update our antipodal region of interest
    p, o = robot.get_link_pos_orient(robot.end_effector)
    antipodal_region.set_base_pos_orient(p, o)

    # Get current gripper finger axis
    gripper_line_vector = robot.local_to_global_coordinate_frame(
        [0, 0.2, 0], link=robot.end_effector
    )[0]
    gripper_line_vector = gripper_line_vector - p
    gripper_line_vector = gripper_line_vector / np.linalg.norm(gripper_line_vector)
    # gripper_line = m.Line(p, p+gripper_line_vector*0.2, radius=0.005, rgba=[0, 0, 0, 1], replace_line=gripper_line)

    # Find points that are inside our antipodal_region
    # Transform points to antipodal_region frame
    points = np.array(
        [antipodal_region.global_to_local_coordinate_frame(p)[0] for p in pc]
    )
    # Check if within bounding box of antipodal_region
    left_bound = np.all(points > -half_extents, axis=-1)
    right_bound = np.all(points < half_extents, axis=-1)
    indices = np.logical_and(left_bound, right_bound)
    points_in_gripper = pc[indices]
    normals_in_gripper = normals[indices]

    # Visualize points in gripper
    # limit = min(len(points_in_gripper), len(visual_points))
    # for i in range(limit):
    #     visual_points[i].set_base_pos_orient(points_in_gripper[i])
    # for i in range(limit, len(visual_points)):
    #     visual_points[i].set_base_pos_orient([-10, -10, -10])

    # Compute grasp score
    if len(normals_in_gripper) > 0:
        score = np.mean(np.abs(normals_in_gripper.dot(gripper_line_vector)))
        print("Grasp score:", score)

    # ------ Student answer above -------

    # Restore robot to previous configuration
    robot.control(prev_joint_angles, set_instantly=True)
    return score


def find_best_grasp(obj, **kwargs) -> np.ndarray:
    """
    Find a robot configuration to grasp the given object.

    Args:
        obj: Object to grasp

    Optional Args:
        max_sample: Maximum number of samples to try
        min_score: Minimum antipodal score to accept a grasp

    Returns:
        robot_joint_angles: Robot joint angles to grasp the object
    """
    pc, npc = get_point_cloud(obj)
    sample_ee = sample_grasp_ee_poses(obj, 100)
    scores = []
    joints = []
    for pos, rot in sample_ee:
        joint_angles = robot.ik(robot.end_effector, target_pos=pos, target_orient=rot)
        scores.append(get_antipodal_score(joint_angles, pc, npc))
        joints.append(joint_angles)

    return joints[np.argmax(scores)]
    # ------ Student answer above -------


def get_point_cloud(obj):
    """Returns object's point cloud and normals."""
    # Create two cameras
    camera1 = m.Camera(
        camera_pos=[0, -0.25, 1],
        look_at_pos=obj.get_base_pos_orient()[0],
        fov=60,
        camera_width=1920 // 4,
        camera_height=1080 // 4,
    )
    camera2 = m.Camera(
        camera_pos=[0, 0.25, 1],
        look_at_pos=obj.get_base_pos_orient()[0],
        fov=60,
        camera_width=1920 // 4,
        camera_height=1080 // 4,
    )
    # Show the object
    obj.change_visual(link=obj.base, rgba=[1, 1, 1, 1])
    # Capture a point cloud from the camera
    pc1, rgba1 = camera1.get_point_cloud(body=obj)
    pc2, rgba2 = camera2.get_point_cloud(body=obj)
    pc = np.concatenate([pc1, pc2], axis=0)
    rgba = np.concatenate([rgba1, rgba2], axis=0)

    # Visualize the point cloud
    # m.DebugPoints(pc, points_rgb=rgba[:, :3], size=10)

    # Hide the object
    obj.change_visual(link=obj.base, rgba=[1, 1, 1, 0.75])

    # Create open3d point cloud from array of points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)

    # Estimate normals for each point
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    normals = np.asarray(pcd.normals)

    # o3d.visualization.draw_geometries([pcd])
    return pc, normals


def robot_in_collision(q):
    """Returns True if the robot is in collision at the given joint angles (q).
    For simplicity, we only consider robot collision with table and objects.
    Robot self collision or collision with cubes is optional.
    """
    # set robot to joint angles
    prev_joint_angles = robot.get_joint_angles(robot.controllable_joints)
    robot.control(q, set_instantly=True)

    # robot-obstacle collision
    for obstacle in obstacles:
        if len(robot.get_closest_points(obstacle, distance=0)[-1]) != 0:
            robot.control(prev_joint_angles, set_instantly=True)
            return True

    robot.control(prev_joint_angles, set_instantly=True)
    return False


def moveto(ee_pose=None, joint_angles=None):
    """Move robot to a given ee_pose or joint angles. If both are given, ee_pose is used."""
    if ee_pose is not None:
        joint_angles = robot.ik(
            robot.end_effector,
            target_pos=ee_pose[0],
            target_orient=ee_pose[1],
            use_current_joint_angles=True,
        )
    if joint_angles is None:
        return

    robot.control(joint_angles)
    while (
        np.linalg.norm(robot.get_joint_angles(robot.controllable_joints) - joint_angles)
        > 0.03
    ):
        m.step_simulation(realtime=True)
    return


# Create environment and ground plane
env = m.Env()
ground = m.Ground()

# Create table
table = m.URDF(
    filename=os.path.join(m.directory, "table", "table.urdf"),
    static=True,
    position=[0, 0, 0],
    orientation=[0, 0, 0, 1],
)

# Create bowl
bowl = m.URDF(
    filename="./bowl/object.urdf",
    static=False,
    position=[0, 0, 0.8],
    orientation=[0, 0, 0, 1],
)
# bowl = m.Shape(
#     m.Mesh(filename=os.path.join(m.directory, "ycb", "spam.obj"), scale=[1, 1, 1]),
#     static=False,
#     mass=1.0,
#     position=[0, 0, 0.85],
#     orientation=[0, 0, 0, 1],
#     rgba=None,
#     visual=True,
# )

bowl.set_whole_body_frictions(
    lateral_friction=2000, spinning_friction=2000, rolling_friction=0
)

m.step_simulation(50)

obstacles = [table, bowl]

# Create Panda manipulator
robot = m.Robot.Panda(position=[0.5, 0, 0.76])
robot.motor_gains = 0.01

# Move end effector to a starting position using IK
target_joint_angles = robot.ik(
    robot.end_effector,
    target_pos=[0, 0, 1],
    target_orient=m.get_quaternion(np.array([np.pi, 0, 0])),
)
robot.control(target_joint_angles, set_instantly=True)
robot.set_gripper_position([1] * 2, set_instantly=True)  # Open gripper

# Create a region that will allow us to identify points within the gripper
position, orientation = robot.get_link_pos_orient(robot.end_effector)
half_extents = np.array([0.01, 0.04, 0.01])
antipodal_region = m.Shape(
    m.Box(half_extents),
    static=True,
    collision=False,
    position=position,
    orientation=orientation,
    rgba=[0, 1, 0, 0],
)
gripper_line_vector = robot.local_to_global_coordinate_frame(
    [0, 0.2, 0], link=robot.end_effector
)[0]
# gripper_line = m.Line(position, gripper_line_vector, radius=0.005, rgba=[0, 0, 0, 1])

# visual_points = m.Points([[-10, -10, -10]] * 100, rgba=[0, 0, 0, 1], radius=0.001)

for _ in range(3):
    # Find best grasp
    robot_joint_angles = find_best_grasp(bowl)

    # MOVETO bowl
    moveto(joint_angles=robot_joint_angles)

    # CLOSE gripper
    robot.set_gripper_position([0] * 2, force=5000)
    m.step_simulation(steps=100, realtime=True)

    # MOVE upwards
    pos, ori = robot.get_link_pos_orient(robot.end_effector)
    moveto(ee_pose=(pos + [0, 0, 0.2], ori))
    input("Press enter to next grasping attempt...")

    # OPEN gripper
    robot.set_gripper_position([1] * 2)
    m.step_simulation(steps=50, realtime=True)
