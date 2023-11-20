from imp import lock_held
import os
import time
import numpy as np
import mengine as m
import copy

# use the functions created in part1 of the homework to find a force closure grasp
from part1 import friction_cone_3d, contact_screw_3d, is_force_closure


# Create environment and ground plane
env = m.Env(time_step=0.005)
env.set_gui_camera(look_at_pos=[0, 0, 0.95])
orient = m.get_quaternion([np.pi, 0, 0])


def sample_spherical(npoints, ndim=3):
    p = np.random.random((npoints, 3)) - 0.5
    points = 0.1 * p / np.linalg.norm(p, axis=1, keepdims=1)
    normal = -points

    # import open3d as o3d
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)
    # o3d.visualization.draw_geometries([pcd])

    return points, normal


Rsample = np.array(
    [
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
        [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
        [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
        [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
        [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
    ]
)


def sample_cube(npoints, ndim=3):
    points, normals = [], []
    for n in range(npoints):
        p = np.random.uniform(-0.1, 0.1, (3))
        p[0] = 0.1
        f = np.random.randint(0, Rsample.shape[0])
        # print(f)
        p = Rsample[f] @ p
        points.append(p)
        n = np.array([-1, 0, 0])
        normals.append(Rsample[f] @ n)

    points = np.asarray(points)
    normals = np.asarray(normals)
    # print(points, normals)

    # import open3d as o3d
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)
    # o3d.visualization.draw_geometries([pcd])

    return points, normals


def find_force_closure_grasp(testobj, mu) -> tuple:
    max_iter, iter = 100, 0
    while True:
        iter += 1
        if testobj == "sphere":
            points, normals = sample_spherical(3)
        else:
            points, normals = sample_cube(6)

        if mu != 0:
            contact_points_FC, contact_normals_FC = friction_cone_3d(
                points, normals, mu, 100
            )
        else:
            contact_points_FC, contact_normals_FC = copy.deepcopy(
                points
            ), copy.deepcopy(normals)

        # print(contact_points_FC.shape)
        wrench_friction = contact_screw_3d(contact_points_FC, contact_normals_FC)
        is_FC_friction, z_max_friction = is_force_closure(wrench_friction)

        print(is_FC_friction, z_max_friction)
        if is_FC_friction:
            break
        
        if iter >= max_iter:
            print('no force closure grasp available')
            break

    return points, normals


# Reset simulation env


def reset(
    positions,
    table_friction=0.5,
    obj_mass=100,
    obj_friction=0.5,
    finger_mass=10.0,
    obj_type="sphere",
):
    # Create environment and ground plane
    env.reset()
    ground = m.Ground()

    # Create table and cube
    table = m.URDF(
        filename=os.path.join(m.directory, "table", "table.urdf"),
        static=True,
        position=[0, 0, 0],
        orientation=[0, 0, 0, 1],
        maximal_coordinates=True,
        # scale=1.0,
    )
    table.set_whole_body_frictions(
        lateral_friction=table_friction, spinning_friction=0, rolling_friction=0
    )
    if obj_type == "sphere":
        obj = m.Shape(
            m.Sphere(radius=0.1),
            static=False,
            mass=obj_mass,
            position=[0, 0, 1.2],
            rgba=[0, 1, 0, 1],
        )
    elif obj_type == "cube":
        obj = m.Shape(
            m.Box(half_extents=[0.1] * 3),
            static=False,
            mass=obj_mass,
            position=[0, 0, 1.2],
            orientation=[0, 0, 0, 1],
            rgba=[0, 1, 0, 1],
        )
    else:
        raise TypeError("Object Type unknown. Use either sphere or cube as obj.")

    obj.set_whole_body_frictions(
        lateral_friction=obj_friction, spinning_friction=0, rolling_friction=0
    )

    # Create n-spheres(fingers) to make contact with the cube.
    fingers = []
    for i, p in enumerate(positions):
        fingers.append(
            m.Shape(
                m.Sphere(radius=0.02),
                static=False,
                mass=finger_mass,
                position=obj.local_to_global_coordinate_frame(positions[i])[0],
                rgba=[1, 0, 0, 1],
            )
        )
        fingers[i].set_whole_body_frictions(
            lateral_friction=0.5, spinning_friction=100, rolling_friction=100
        )
    return fingers, obj


# visualize force closure grasps in simulation


def visualize_grasps(grasps):
    for g in grasps:
        # Reset simulator
        fingers, obj = reset(
            g["contact_positions"],
            g["table_friction"],
            g["obj_mass"],
            g["obj_friction"],
            g["finger_mass"],
            g["obj_type"],
        )

        for i in range(100):
            m.clear_all_visual_items()

            for idx, finger in enumerate(fingers):
                # Gravity compensation force
                force = -finger.get_link_mass(finger.base) * env.gravity
                # Apply force in contact point in the direction of the contact normal
                force += np.array(g["force_magnitude"] * g["contact_normals"][idx])
                finger.apply_external_force(
                    link=finger.base,
                    force=force,
                    pos=finger.get_base_pos_orient()[0],
                    local_coordinate_frame=False,
                )

                # Show contact normals
                cp = finger.get_contact_points(bodyB=obj)
                if cp is not None:
                    c = cp[0]
                    m.Line(
                        c["posB"],
                        np.array(c["posB"]) + np.array(c["contact_normal"]) * 0.2,
                        rgb=[1, 0, 0],
                    )

            m.step_simulation()
            # time.sleep(0.5)

            


def main(testobj, friction=True):
    # parameters to play with
    obj_mass = 100
    obj_friction = 0.5
    finger_mass = 10.0
    force_magnitude = 1000
    if friction:
        mu = 0.5
    else:
        mu = 0.0

    contact_positions, contact_normals = find_force_closure_grasp(testobj, mu)

    # spawn fingers slightly away from the obj
    grasps = [
        dict(
            contact_positions=1.1 * contact_positions,
            contact_normals=contact_normals,
            table_friction=0.5,
            obj_type=testobj,
            obj_mass=obj_mass,
            obj_friction=obj_friction,
            force_magnitude=force_magnitude,
            finger_mass=finger_mass,
        )
    ]

    # append grasps with other parameters here:
    # grasps.append()

    visualize_grasps(grasps)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse different test objects.")
    parser.add_argument(
        "testobject",
        type=str,
        help="Call file either with argument 'sphere' or 'cube'.",
    )
    parser.add_argument(
        "friction", type=int, help="Frictionless => False , frictional=>True"
    )
    try:
        args = parser.parse_args()
        testobj = args.testobject
        friction = args.friction
        print(friction)
    except:
        print("No testobj specified. Using sphere as default.")
        testobj = "sphere"
        friction = False
    main(testobj, friction)
