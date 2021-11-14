from typing import List

import numpy as np
from numba import njit


@njit
def reconstruct_3d_scene(
    left_image: np.ndarray,
    disparity_image: np.ndarray,
    baseline: int,
    focal_length: int,
    dmin: int
) -> List[tuple]:
    """

    Args:
        left_image:
        disparity_image:
        baseline:
        focal_length:
        dmin:

    Returns:

    """
    rows, cols = left_image.shape[:2]
    points_3d_rgb = []

    for i in range(rows):
        for j in range(cols):
            disparity = disparity_image[i, j] + dmin

            if not disparity:
                continue
            # d = u1 - u2  ->  u2 = u1 - d
            # x = - b * (u1 + u2) / 2 * d = -b * (2 * u1 - d) / 2 * d
            x = round(-baseline * (2 * j - disparity) /(2 * disparity), 4)
            y = -round(baseline * i / disparity, 4)
            z = round(baseline * focal_length / disparity, 4)
            rgb = left_image[i, j]
            points_3d_rgb.append((x, y, z, rgb[0], rgb[1], rgb[2]))

    return points_3d_rgb
