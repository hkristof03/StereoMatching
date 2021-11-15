from typing import Tuple, Callable

import numpy as np
from numba import njit


@njit
def stereo_matching_dp_row(
    row: int,
    left_image: np.ndarray,
    right_image: np.ndarray,
    cost_occlusion: int,
    block_size: int,
    metric: Callable
) -> Tuple[int, np.ndarray]:
    """Calculates a row for the disparity map for two standard stereo image
    pairs. User for multi-processig.

    Args:
        row: Index of a row from the left image
        left_image: Left standard stereo image
        right_image: Right standard stereo image
        cost_occlusion: Penalty for occlusion
        block_size: N of an NxN block to compare image patches
        metric: Similarity measure to compare patches

    Returns:
        A row of the disparity map
    """
    height, width = left_image.shape
    half_size = block_size // 2

    disp_map_left = np.zeros(width, dtype=np.uint8)
    cost_matrix = np.zeros((width, width), dtype=np.float64)
    direction_matrix = np.ones(cost_matrix.shape, dtype=np.uint8)

    for i in range(width):
        cost_matrix[i, 0] = i * cost_occlusion
        cost_matrix[0, i] = i * cost_occlusion

    for i in range(half_size, width - half_size):

        left_patch = left_image[
            row - half_size: row + half_size + 1,
            i - half_size: i + half_size + 1
        ].astype(np.float64)

        for j in range(half_size, width - half_size):
            right_patch = right_image[
                row - half_size: row + half_size + 1,
                j - half_size: j + half_size + 1
            ].astype(np.float64)

            dissim = metric(left_patch, right_patch)

            min1 = cost_matrix[i - 1, j - 1] + dissim
            min2 = cost_matrix[i - 1, j] + cost_occlusion
            min3 = cost_matrix[i, j - 1] + cost_occlusion

            mins = np.array([min1, min2, min3])
            cost_min = np.min(mins)
            idx_min = np.argmin(mins) + 1

            cost_matrix[i, j] = cost_min
            direction_matrix[i, j] = idx_min

    p = width - 1
    q = width - 1

    while p != 0 and q != 0:
        if direction_matrix[p, q] == 1:
            disp_map_left[p] = np.abs(p - q)
            p -= 1
            q -= 1
        elif direction_matrix[p, q] == 2:
            p -= 1
        else:
            q -= 1

    return row, disp_map_left


@njit
def fill_occluded_pixels(image_disparity: np.ndarray) -> np.ndarray:
    """Fills the occluded pixels.

    Args:
        image_disparity: Disparity map for the two standard stereo image pairs

    Returns:
        Filled disparity map
    """
    height, width = image_disparity.shape

    for i in range(height):
        for j in range(width):
            if image_disparity[i, j] == 0:
                left_j = j - 1
                while left_j >= 0:
                    if left_j >= 0 and image_disparity[i, left_j] != 0:
                        image_disparity[i, j] = image_disparity[i, left_j]
                        break
                    left_j -= 1

    return image_disparity
