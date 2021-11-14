from typing import Tuple, List, Callable

import numpy as np
from numba import njit


@njit
def calculate_disparity_map(
    left_image: np.ndarray,
    right_image: np.ndarray,
    metric: Callable,
    block_size: int,
    max_distance: int
) -> np.ndarray:
    """Calculates the disparity map for a given grayscale image pair, which
    was recorded based on standard stereo conditions.

    Args:
        left_image: left grayscale image
        right_image: right grayscale image
        metric: metric to measure similarity between image patches
        block_size: block size to create an nxn block for matching
        max_distance: max distance to look for matching blocks

    Returns:
        Disparity map of same size and the image pairs
    """
    img_x, img_y = left_image.shape
    img_disp = np.zeros((img_x, img_y))
    cbs = int(np.ceil(block_size / 2))

    for tcx in range(img_x):
        for tcy in range(img_y):

            template = left_image[
               max(0, tcx - cbs): min(tcx + cbs, img_x),
               max(0, tcy - cbs): min(tcy + cbs, img_y)
            ]
            blocks = right_image[
                max(0, tcx - cbs): min(tcx + cbs, img_x),
                max(0, tcy - cbs): min(tcy + cbs + max_distance, img_y)
            ]
            _, temp_y = template.shape
            _, bls_y = blocks.shape
            scores = []
            md = 0

            while md + temp_y <= bls_y:
                bl = blocks[:, md: md + temp_y]
                sc = metric(template, bl)
                scores.append(sc)
                md += 1

            img_disp[tcx, tcy] = np.argmin(np.array(scores))

    return img_disp


@njit
def calculate_disparities_for_row(
    row: int,
    left_image: np.ndarray,
    right_image: np.ndarray,
    metric: Callable,
    block_size: int,
    max_distance: int
) -> Tuple[int, np.ndarray]:
    """Calculates the disparities for a given row. This function can be used
    for using multi-processing for the rows to speed up the computation. The
    rows can be ordered according to the first element of the returned tuple.

    Args:
        row: Row in the image to calculate the disparities for
        left_image: left grayscale image
        right_image: right grayscale image
        metric: metric to measure similarity between image patches
        block_size: block size to create an nxn block for matching
        max_distance: max distance to look for matching blocks

    Returns:
        Row index and Disparities of shape [1, img_width]
    """
    img_x, img_y = left_image.shape
    row_disp = np.zeros(img_y)
    cbs = int(np.ceil(block_size / 2))

    for tcy in range(img_y):

        template = left_image[
            max(0, row - cbs): min(row + cbs, img_x),
            max(0, tcy - cbs): min(tcy + cbs, img_y)
        ]
        blocks = right_image[
            max(0, row - cbs): min(row + cbs, img_x),
            max(0, tcy - cbs): min(tcy + cbs + max_distance, img_y)
        ]
        _, temp_y = template.shape
        _, bls_y = blocks.shape
        scores = []
        md = 0

        while md + temp_y <= bls_y:
            bl = blocks[:, md: md + temp_y]
            sc = metric(template, bl)
            scores.append(sc)
            md += 1

        row_disp[tcy] = np.argmin(np.array(scores))

    return row, row_disp
