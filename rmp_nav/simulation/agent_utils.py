import math


def _sign(x):
    return -1 if x < 0 else 1


def clip_within_fov(point, fov):
    angle = math.atan2(point[1], point[0])
    if abs(angle) > fov * 0.5:
        # Out of field of view
        # Project the point to the fov line
        # x0, y0 is the unit vector of the closest fov line
        x0 = math.sqrt(1.0 / (1.0 + math.tan(fov * 0.5)**2)) * _sign(point[0])
        y0 = abs(math.tan(fov * 0.5) * x0) * _sign(point[1])
        lamb = x0 * point[0] + y0 * point[1]
        return lamb * x0, lamb * y0
    else:
        return point
