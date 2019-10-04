import numpy as np
import math
from . import math_utils_cpp


def depth_to_xy(depth, pos=None, heading=0.0, fov=np.pi * 2.0):
    n_bins = depth.shape[0]

    # FIXME: hack
    if abs(fov - np.pi * 2.0) < 1e-5:
        theta = np.linspace(0.0, fov, n_bins, endpoint=False)
        # theta = np.arange(0, fov, fov / n_bins)
    else:
        theta = np.linspace(-fov * 0.5, fov * 0.5, n_bins, endpoint=False)
        # theta = np.arange(-fov * 0.5, fov * 0.5, fov / n_bins)

    xs = np.cos(theta + heading)
    ys = np.sin(theta + heading)
    xy = np.stack([xs * depth, ys * depth], axis=1)
    if pos is None:
        return xy
    return pos + xy


def depth_to_xy_plane(depth, pos=None, heading=0.0, fov=np.pi * 2.0):
    """
    Assume depth is from a 1D depth camera
    """
    assert fov < np.pi

    n = depth.shape[0]
    w = np.tan(fov / 2) * 2.0

    theta = np.arctan2(1.0, w / 2 - np.arange(n, dtype=np.float32) * w / n) - np.pi / 2 + heading
    # The above is equivalent to
    # theta2 = np.zeros(n, np.float32)
    # for i in range(n):
    #     x = w / 2 - i * w / n
    #     theta2[i] = np.arctan2(1.0, x) - np.pi / 2 + heading
    # assert np.allclose(theta, theta2)

    xs = np.cos(theta)
    ys = np.sin(theta)
    xy = np.stack([xs * depth, ys * depth], axis=1)
    if pos is None:
        return xy

    return pos + xy


def xy_to_heading(x, y):
    heading = np.arctan2(y, x)
    if heading < 0:
        heading += 2 * np.pi
    return heading


def rotate_2d(v, angle):
    c = np.cos(angle)
    s = np.sin(angle)
    if isinstance(v, tuple) or len(v.shape) == 1:
        x, y = v
        x2 = c * x - s * y
        y2 = s * x + c * y
        return np.array([x2, y2], np.float32)
    elif len(v.shape) == 2:
        assert v.shape[1] == 2
        m = np.array([[c, -s], [s, c]])
        return np.dot(v, m.T)
    else:
        raise ValueError


def vector_angle(v1, v2):
    '''
    :return: angle in (-pi, pi) from v1 to v2
    '''
    x1, y1 = v1
    x2, y2 = v2
    return np.arctan2(x1 * y2 - y1 * x2, x1 * x2 + y1 * y2)


def compute_normals(points):
    """
    The normal at point i bisects the angle between point i - 1 and point i + 1
    :param points: a Nx2 array
    :return: Nx2 array of unit-length normals.
    """
    # def rotate(v, a):
    #     c, s = math.cos(a), math.sin(a)
    #     x, y = v
    #     x2 = c * x - s * y
    #     y2 = s * x + c * y
    #     return np.array([x2, y2], np.float32)
    #
    # normals = np.zeros((len(points), 2), np.float32)
    #
    # for i in range(len(points)):
    #     p0 = points[i]
    #     p1 = points[(i - 1) % len(points)]
    #     p2 = points[(i + 1) % len(points)]
    #
    #     q1 = p1 - p0
    #     q2 = p2 - p0
    #
    #     angle = vector_angle(q1, q2)
    #
    #     # Convert to clockwise angle
    #     if angle < 0:
    #         angle = -angle
    #     else:
    #         angle = np.pi * 2.0 - angle
    #
    #     n = rotate(q1 / np.linalg.norm(q1), -angle / 2)
    #
    #     normals[i] = n
    # assert np.allclose(normals, normal2, 1e-3, 1e-3), np.sum(np.abs(normals - normal2))

    # Equivalent to above, but much faster.
    normals = math_utils_cpp.compute_normals(points.astype(np.float32))
    return normals


def downsample_1d_depth(depth, resolution):
    """
    :param depth: a 1d array of float. NaN indicates invalid values.
    :param resolution: desired resolution
    :return: a 1d array of float of size @param resolution.
    """
    assert resolution <= len(depth)
    out = []
    for i in range(resolution - 1):  # Exclude the last ray
        p = float(i) / (resolution - 1)
        r = (len(depth) - 1) * p
        w = r - int(r)
        d1 = depth[int(r)]
        d2 = depth[int(r) + 1]
        # If w is close to an integer we shortcut to original depths. This is to avoid getting
        # NaN values due to the other depth being NaN
        if w < 1e-5:
            out.append(d1)
        elif w > 1 - 1e-5:
            out.append(d2)
        else:
            # If d1 or d2 is NaN, the result will also be NaN
            out.append(d1 * (1 - w) + d2 * w)
    out.append(depth[-1])  # Add last ray
    return out
