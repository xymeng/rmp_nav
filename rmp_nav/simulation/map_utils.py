from future.utils import iteritems
from past.builtins import xrange
import math
import numpy as np
import cv2
from . import map_utils_cpp


def a_star(obstacle_map, start_pos, goal_pos):
    height, width = obstacle_map.shape

    # Convert to standard python types. This speeds up path finding significantly.
    obstacle_map = obstacle_map.tolist()
    start_pos = tuple(start_pos)
    goal_pos = tuple(goal_pos)

    def dist_to_goal(x, y):
        # max(x, y) + (sqrt(2)-1)*min(x, y)
        xx = abs(x - goal_pos[0])
        yy = abs(y - goal_pos[1])
        # return np.sqrt(xx**2 + yy**2)
        return max(xx, yy) + (math.sqrt(2) - 1) * min(xx, yy)

    open_dict = {start_pos: (None, 0, dist_to_goal(*start_pos))}
    close_dict = {}

    parents = {}

    def choose_best_node():
        best_g = 1e10
        best_h = 1e10
        best_x = 0
        best_y = 0

        for (x, y), (pa, g, h) in iteritems(open_dict):
            f = g + h
            best_f = best_g + best_h
            # Tie-breaking A*: http://movingai.com/astar.html
            if abs(f - best_f) < 1e-3:
                if best_g < g:
                    best_g = g
                    best_h = h
                    best_x = x
                    best_y = y
            elif f < best_f:
                best_g = g
                best_h = h
                best_x = x
                best_y = y

        return best_x, best_y

    while len(open_dict) > 0:
        x, y = choose_best_node()
        pa, g_score, _ = open_dict.pop((x, y))
        close_dict[(x, y)] = pa

        if (x, y) == goal_pos:
            break

        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue

                x2 = dx + x
                y2 = dy + y
                if 0 <= x2 < width and 0 <= y2 < height:
                    if (x2, y2) in close_dict:
                        continue
                    if obstacle_map[y2][x2] > 0:
                        continue
                    if (x2, y2) not in open_dict:
                        # add and compute score
                        g = g_score + math.sqrt(dx ** 2 + dy ** 2)
                        open_dict[(x2, y2)] = (x, y), g, dist_to_goal(x2, y2)
                        parents[(x2, y2)] = (x, y)
                    else:
                        _, g0, h0 = open_dict[(x2, y2)]
                        g = g_score + math.sqrt(dx ** 2 + dy ** 2)
                        if g < g0:
                            open_dict[(x2, y2)] = (x, y), g, dist_to_goal(x2, y2)
                            parents[(x2, y2)] = (x, y)

    if goal_pos not in parents:
        return None

    path = []
    x, y = goal_pos
    while (x, y) != start_pos:
        path.append((x, y))
        x, y = parents[(x, y)]

    return path[::-1]


def dijkstra(obstacle_map, start_pos):
    """
    Find paths from start_pos to all traversable points.
    A* becomes Dijstra if h = 0
    :return: a dictionary that maps a tuple (x1, y1) to its parent path point (x2, y2).
    """
    height, width = obstacle_map.shape

    # Convert to standard python types. This speeds up path finding significantly.
    obstacle_map = obstacle_map.tolist()
    start_pos = tuple(start_pos)

    open_dict = {start_pos: (None, 0)}
    close_dict = {}

    parents = {}

    def choose_best_node():
        best_g = 1e10
        best_x = 0
        best_y = 0

        for (x, y), (pa, g) in iteritems(open_dict):
            f = g
            best_f = best_g
            # Tie-breaking A*: http://movingai.com/astar.html
            if abs(f - best_f) < 1e-3:  # FIXME: this is probably redundant.
                if best_g < g:
                    best_g = g
                    best_x = x
                    best_y = y
            elif f < best_f:
                best_g = g
                best_x = x
                best_y = y

        return best_x, best_y

    while len(open_dict) > 0:
        x, y = choose_best_node()
        pa, g_score = open_dict.pop((x, y))
        close_dict[(x, y)] = pa

        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue

                x2 = dx + x
                y2 = dy + y
                if 0 <= x2 < width and 0 <= y2 < height:
                    if (x2, y2) in close_dict:
                        continue
                    if obstacle_map[y2][x2] > 0:
                        continue
                    if (x2, y2) not in open_dict:
                        # add and compute score
                        g = g_score + math.sqrt(dx ** 2 + dy ** 2)
                        open_dict[(x2, y2)] = (x, y), g
                        parents[(x2, y2)] = (x, y)
                    else:
                        _, g0 = open_dict[(x2, y2)]
                        g = g_score + math.sqrt(dx ** 2 + dy ** 2)
                        if g < g0:
                            open_dict[(x2, y2)] = (x, y), g
                            parents[(x2, y2)] = (x, y)

    return parents


def dijkstra_fast(obstacle_map, start_pos):
    return map_utils_cpp.dijkstra(obstacle_map, start_pos[0], start_pos[1])


def rasterize(lines, bbox, division=None, resolution=None):
    if division is not None:
        resolution = 1.0 / division
    else:
        division = int(1.0 / resolution)

    x1, x2, y1, y2 = bbox
    origin = (x1, y1)
    width = int((x2 - x1) / resolution)
    height = int((y2 - y1) / resolution)

    def grid_coord(x, y, n_division):
        return int((x - origin[0]) * n_division + 0.5), int((y - origin[1]) * n_division + 0.5)

    canvas = np.zeros((height, width), np.uint8)
    for i in xrange(len(lines)):
        x1, y1, x2, y2 = lines[i]
        cv2.line(canvas, grid_coord(x1, y1, division), grid_coord(x2, y2, division), 255, 2)

    return canvas


def path_length(xys):
    """
    :param xys: N x 2
    :return: path length. float.
    """
    if len(xys) <= 1:
        return 0.0
    return float(np.sum(np.linalg.norm(np.array(xys[1:]) - np.array(xys[:-1]), axis=1, ord=2)))


def cum_path_length(xys):
    """
    :param xys: N x 2
    :return: cumulative path length. size of N - 1.
    """
    return np.cumsum(np.linalg.norm(np.array(xys[1:]) - np.array(xys[:-1]), axis=1, ord=2))
