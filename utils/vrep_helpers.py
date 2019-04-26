from cpg.cpg_gaits import *

VALID_ERROR_CODES = (0, 1)


def distance(a, b):
    return np.sqrt(((a[0] - b[0]) ** 2) + ((a[1] - b[1]) ** 2))


def quat_to_direction(quat):
    w, x, y, z = quat
    v0 = 2 * (x * z + w * y)
    v1 = 2 * (y * z - w * x)
    v2 = 1 - (2 * (x * x + y * y))
    return np.arctan2(v0, v1)
