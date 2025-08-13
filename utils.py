import numpy as np


def rotation_matrix_y(angle_degrees):
    theta = np.radians(angle_degrees)
    return np.array([
        [ np.cos(theta), 0, np.sin(theta)],
        [ 0,             1, 0            ],
        [-np.sin(theta), 0, np.cos(theta)]
    ])


def rotation_matrix_z(angle_degrees):
    theta = np.radians(angle_degrees)
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,             0,              1]
    ])


def rotation_matrix_x(angle_degrees):
    theta = np.radians(angle_degrees)
    return np.array([
        [1, 0,            0           ],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta),  np.cos(theta)]
    ])


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi