from __future__ import print_function
import math


# https://stackoverflow.com/questions/5186939/algorithm-for-drawing-a-4-connected-line
def rasterize_line(x0, y0, x1, y1):
    dx = abs(x1 - x0)   # distance to travel in X
    dy = abs(y1 - y0)   # distance to travel in Y

    if x0 < x1:
        ix = 1           # x will increase at each step
    else:
        ix = -1          # x will decrease at each step

    if y0 < y1:
        iy = 1           # y will increase at each step
    else:
        iy = -1          # y will decrease at each step

    e = 0                # Current error

    points = []

    for i in range(dx + dy):
        points.append((x0, y0))
        e1 = e + dy
        e2 = e - dx
        if abs(e1) < abs(e2):
            # Error will be smaller moving on X
            x0 += ix
            e = e1
        else:
            # Error will be smaller moving on Y
            y0 += iy
            e = e2

    points.append((x1, y1))
    return points


def ray_intersect(x, y, x0, y0, x1, y1, x2, y2):
    '''
    :param x, y: ray start position
    :param x0, y0: ray direction
    :param x1, y1, x2, y2: line
    :return: lamb, t
             the intersection is (x1, y1) * lamb + (x2, y2) * (1 - lamb)
             or equivalently (x, y) + t * (x0, y0)
             if the two lines are parallel then return None
    '''
    denom = (x1 - x2) * y0 - (y1 - y2) * x0
    if abs(denom) < 1e-5:
        return None
    lamb = ((x - x2) * y0 - (y - y2) * x0) / denom
    t = ((x1 - x2) * (y2 - y) - (y1 - y2) * (x2 - x)) / denom
    return lamb, t


def point_line_distance(x0, y0, x1, y1, x2, y2):
    return abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / math.sqrt((y2 - y1)**2 + (x2 - x1)**2)


def line_line_distance(x1, y1, x2, y2, x3, y3, x4, y4):
    '''
    Compute the smallest distance between an endpoint to the other line. The foot must lie within
    the line segment.
    '''
    def helper(x0, y0, x1, y1, x2, y2):
        v1 = (x2 - x1, y2 - y1)
        v2 = (x0 - x1, y0 - y1)
        t = (v1[0] * v2[0] + v1[1] * v2[1]) / (v1[0]**2 + v1[1]**2)
        if -1e-3 <= t <= 1+1e-3:
            return point_line_distance(x0, y0, x1, y1, x2, y2)
        return 1e100

    d = min([helper(x1, y1, x3, y3, x4, y4),
            helper(x2, y2, x3, y3, x4, y4),
            helper(x3, y3, x1, y1, x2, y2),
            helper(x4, y4, x1, y1, x2, y2)])
    return d if d < 1e100 else None


if __name__ == '__main__':
    print(rasterize_line(0, 0, 0, 0))
