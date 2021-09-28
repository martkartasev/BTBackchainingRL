import numpy as np


def rotation_matrix_y(degrees):
    theta = np.radians(degrees)
    c, s = np.cos(theta), np.sin(theta)
    return np.array(((c, 0, s),
                     (0, 1, 0),
                     (-s, 0, c)
                     )
                    )


def rotation_matrix_y_2d(degrees):
    theta = np.radians(degrees)
    c, s = np.cos(theta), np.sin(theta)
    return np.array(((c, -s),
                     (s, c)
                     )
                    )