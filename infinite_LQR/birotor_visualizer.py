import numpy as np
from meshcat.geometry import Box, Cylinder, MeshBasicMaterial
import meshcat.transformations as tf


def set_birotor(vis, width, height, radius):
    blue = MeshBasicMaterial(color=0x0000FF)
    green = MeshBasicMaterial(color=0x00C000)

    vis["quadrotor"]["body"].set_object(Box([width, width, height]), green)

    positions = [
        [width / 2, -width / 2, 3 / 4 * height],
        [width / 2, width / 2, 3 / 4 * height],
        [-width / 2, width / 2, 3 / 4 * height],
        [-width / 2, -width / 2, 3 / 4 * height],
    ]

    for i, pos in enumerate(positions):
        vis["quadrotor"]["rotor" + str(i)].set_object(
            Cylinder(height / 2, radius), blue
        )
        vis["quadrotor"]["rotor" + str(i)].set_transform(
            tf.translation_matrix(pos)
            @ tf.quaternion_matrix([np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0])
        )

    return vis


def set_birotor_state(vis, state):
    vis["quadrotor"].set_transform(
        tf.translation_matrix([0, state[0], state[1]])
        @ tf.quaternion_matrix([np.cos(state[2] / 2), np.sin(state[2] / 2), 0, 0])
    )
