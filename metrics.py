import numpy as np
from numba import njit


@njit
def normalized_cross_correlation(
    patch1: np.ndarray,
    patch2: np.ndarray
) -> float:
    """Calculates NNC.

    Args:
        patch1: patch from left/right image
        patch2: patch from right/left image

    Returns:
        NNC value
    """
    N = len(patch1.flatten())
    p1m = np.mean(patch1)
    p2m = np.mean(patch2)
    t1 = patch1 - p1m
    t2 = patch2 - p2m
    num = 1 / N * np.sum(t1 * t2)
    denom = np.std(patch1) * np.std(patch2)

    return -num / denom


@njit
def sum_of_squared_differences(
    patch1: np.ndarray,
    patch2: np.ndarray
) -> float:
    """Calculates SSD.

    Args:
        patch1: patch from left/right image
        patch2: patch from right/left image

    Returns:
        SSD values
    """
    return np.sum(np.square(patch1 - patch2))


@njit
def normalized_sum_of_squared_differences(
    patch1: np.ndarray,
    patch2: np.ndarray
) -> float:
    """Calculates Normalized SSD.

    Args:
        patch1: patch from left/right image
        patch2: patch from right/left image

    Returns:
        NSSD value.
    """
    p1m = np.mean(patch1)
    p2m = np.mean(patch2)
    t1 = patch1 - p1m
    t2 = patch2 - p2m
    tt1 = t1 / np.sqrt(np.sum(np.square(t1)))
    tt2 = t2 / np.sqrt(np.sum(np.square(t2)))

    return np.sum(np.square(tt1 - tt2))


@njit
def sum_of_absolute_differences(
    patch1: np.ndarray,
    patch2: np.ndarray
) -> float:
    """Calculates SAD.

    Args:
        patch1: patch from left/right image
        patch2: patch from right/left image

    Returns:
        SAD value.
    """
    return np.sum(np.abs(patch1 - patch2))


@njit
def structural_similarity_index(
    patch1: np.ndarray,
    patch2: np.ndarray
) -> float:
    """Calculates SSIM.

    Args:
        patch1: patch from left/right image
        patch2: patch from right/left image

    Returns:
        SSIM value.
    """
    c = 1e-7
    mp1 = patch1.mean()
    mp2 = patch2.mean()
    cov12 = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
    stdp1 = patch1.std()
    stdp2 = patch2.std()

    num = (2 * mp1 * mp2 + c) * (2 * cov12 + c)
    denom = (mp1 ** 2 + mp2 ** 2 + c) * (stdp1 ** 2 + stdp2 ** 2 + c)
    # * -1 because we take argmin
    ssim = num / denom * -1

    return ssim


def compile_metrics(patch1: np.ndarray, patch2: np.ndarray) -> None:
    """Utility function to precompile numba functions to machine code.

    Args:
        patch1: patch from left/right image
        patch2: patch from right/left image

    Returns:

    """
    _ = normalized_cross_correlation(patch1, patch2)
    _ = sum_of_squared_differences(patch1, patch2)
    _ = normalized_sum_of_squared_differences(patch1, patch2)
    _ = sum_of_absolute_differences(patch1, patch2)
    _ = structural_similarity_index(patch1, patch2)


if __name__ == '__main__':

    x = 13
    bls = np.array([
        list(range(x)),
        list(range(x, 2 * x)),
        list(range(2 * x, 3 * x))
    ])
    temp = np.array([
        [4, 5, 6],
        [17, 18, 19],
        [30, 31, 32]
    ])
    compile_metrics(temp, bls[:, 10:13])
