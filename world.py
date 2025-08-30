from anki_vector.util import degrees
import numpy as np
from utils import rotation_matrix_z, rotation_matrix_x, rotation_matrix_y
from enum import Enum

class Marker(Enum):
    FRONT_RIGHT = 4
    FRONT_LEFT = 3
    RIGHT = 2
    FRONT = 1
    LEFT = 0
    BOTTOM = 5
    BOTTOM_LEFT = 6
    BOTTOM_RIGHT = 7


class Marker_World:
    
    def __init__(self):
        self.marker_transforms = self.define_marker_world()

    def define_marker_world(self):

        return {
            Marker.LEFT.value: {
                "pos": np.array([-0.2, 0, 0]),
                "rot": rotation_matrix_z(90) @ rotation_matrix_x(90),
                "translation": np.array([-186, 20]),
            },
            Marker.FRONT.value: {
                "pos": np.array([0, 0.2, 0]),
                "rot": rotation_matrix_x(90),
                "translation": np.array([0, 197]),
            },
            Marker.FRONT_LEFT.value: {
                "pos": np.array([-0.2, 0.2, 0]),
                "rot": rotation_matrix_z(45) @ rotation_matrix_x(-90),
                "translation": np.array([-200, 200])
            },
            Marker.FRONT_RIGHT.value: {
                "pos": np.array([0.2, 0.2, 0]),
                "rot": rotation_matrix_z(-45) @ rotation_matrix_x(-90),
                "translation": np.array([197, 196])
            },
            Marker.RIGHT.value: {
                "pos": np.array([0.2, 0, 0]),
                "rot": rotation_matrix_z(-90) @ rotation_matrix_x(90),
                "translation": np.array([205, 20])
            },
            Marker.BOTTOM.value: {
                "pos": np.array([0, -0.2, 0]),
                "rot": rotation_matrix_z(180) @ rotation_matrix_x(90),
                "translation": np.array([-20, -184])
            },
            Marker.BOTTOM_LEFT.value: {
                "pos": np.array([-0.2, -0.2, 0]),
                "rot": rotation_matrix_z(135) @ rotation_matrix_x(-90),
                "translation": np.array([-202, -205])
            },
            Marker.BOTTOM_RIGHT.value: {
                "pos": np.array([0.2, -0.2, 0]),
                "rot": rotation_matrix_z(-135) @ rotation_matrix_x(-90),
                "translation": np.array([209, -207])
            }
        }