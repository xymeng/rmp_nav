'''
A map representation that uses 2D lines to represent walls. The actual rendering of the map
does not have to be 2D.

This file is more or less obsolete, as we now almost exclusively use rasterized maps.
'''

from __future__ import print_function
import numpy as np
import range_libc
from . import line_utils
from .map_utils import a_star


class Map(object):
    def __init__(self,
                 lines,
                 initial_pos,
                 path_map=None,
                 path_map_bbox=None,
                 path_map_division=16,
                 path_map_dilation=2,
                 reachable_area_dilation=3,
                 range_max=50.0,
                 name=''):
        '''
        :param lines: a list of (x1, y1, x2, y2) representing walls.
        :param initial_pos: a point that is in the map used to find reachable area.
        '''

        self.lines = lines
        self.range_max = range_max
        self.name = name

        # Compute map bounding box
        self.map_bbox = self.get_map_bbox()

        # A discrete map for path finding
        if path_map is not None and path_map_bbox is not None:
            self.path_map = path_map
            self.path_map_bbox = path_map_bbox
            self.path_map_division = path_map_division
            self.path_map = self._dilate(self.path_map, path_map_dilation)
        else:
            self.path_map_division = path_map_division
            self.path_map_bbox, grid_map = self._make_grid(self.path_map_division)

            # Find reachable area from the player's initial position.
            init_pos_grid = self._grid_coord(
                initial_pos[0], initial_pos[1], self.path_map_division, self.path_map_bbox)

            grid_map = self._dilate(grid_map, path_map_dilation)
            self.path_map = 255 - self._find_reachable_area(grid_map, init_pos_grid)

        # Zeros in path map are free space
        # Visualize path map
        # print 'path map shape', self.path_map.shape
        # import cv2
        # cv2.imshow('path map', self.path_map)  #cv2.resize(self.path_map, dsize=(1024, 1024)))
        # cv2.waitKey(0)

        self.initial_pos = initial_pos

        # Erode the path map to avoid starting too close to walls because it can cause some
        # agents (e.g., long and thin) to get stuck
        self.reachable_area = 255 - self._dilate(self.path_map, reachable_area_dilation)

        # import cv2
        # cv2.imshow('reachable area', self.reachable_area)
        # cv2.waitKey(0)
        # exit(0)

        self.reachable_locs = zip(*np.nonzero(self.reachable_area)[::-1])

        # A discrete map for collision detection
        self.collision_map_division = 4
        self.collision_map_bbox, self.collision_map = self._make_grid(self.collision_map_division)

        # Rasterize the lines for ray tracing using range_libc.
        # This is much faster than doing ray tracing in python.
        self.rasterization_resolution = 0.02
        self._rasterize(self.rasterization_resolution)

        # A matrix of list storing the linedef indices that pass through each bin
        self.line_map = [[[] for i in xrange(self.collision_map.shape[1])]
                         for j in xrange(self.collision_map.shape[0])]

        for i in xrange(len(lines)):
            x1, y1, x2, y2 = lines[i]
            dx1, dy1 = self._grid_coord(x1, y1, self.collision_map_division, self.collision_map_bbox)
            dx2, dy2 = self._grid_coord(x2, y2, self.collision_map_division, self.collision_map_bbox)
            bins = line_utils.rasterize_line(dx1, dy1, dx2, dy2)
            for x, y in bins:
                self.line_map[y][x].append(i)

    def get_reachable_locations(self):
        return self.reachable_locs

    def _rasterize(self, resolution):
        import cv2

        division = int(1.0 / resolution)

        x1, x2, y1, y2 = self.map_bbox
        origin = (x1, y1)
        width = int((x2 - x1) / resolution)
        height = int((y2 - y1) / resolution)

        def grid_coord(x, y, n_division):
            return int((x - origin[0]) * n_division + 0.5), int((y - origin[1]) * n_division + 0.5)

        canvas = np.zeros((height, width), np.uint8)
        for i in xrange(len(self.lines)):
            x1, y1, x2, y2 = self.lines[i]
            cv2.line(canvas, grid_coord(x1, y1, division), grid_coord(x2, y2, division), 255, 2)

        # print canvas.shape
        # cv2.imshow('canvas', canvas)
        # cv2.imwrite('canvas.jpg', canvas)
        # cv2.waitKey(0)
        # exit(0)

        self.rasterized = np.zeros((max(height, width), max(height, width)), np.uint8)
        self.rasterized[:width, :height] = np.transpose(canvas, (1, 0))
        self.omap = range_libc.PyOMap(self.rasterized)
        self.range_scanner = range_libc.PyBresenhamsLine(self.omap, int(self.range_max * division))
        #self.range_scanner = range_libc.PyRayMarching(self.omap, int(self.range_max * division))

    def _find_reachable_area(self, a, loc):
        from collections import deque
        res = np.zeros(a.shape, np.uint8).tolist()
        h, w = a.shape

        if not (0 <= loc[0] < a.shape[1] and 0 <= loc[1] < a.shape[0]):
            print('warning: no reachable area found')
        else:
            q = deque([loc])
            res[loc[1]][loc[0]] = 255

            while len(q) > 0:
                x, y = q.popleft()
                assert res[y][x] == 255, res[y][x]
                for i in (-1, 0, 1):
                    for j in (-1, 0, 1):
                        y2 = y + i
                        x2 = x + j
                        if 0 <= x2 < w and 0 <= y2 < h and res[y2][x2] == 0 and a[y2, x2] == 0:
                            res[y2][x2] = 255
                            q.append((x2, y2))

        return np.array(res, np.uint8)

    def line_segment_intersection(self, x1, y1, x2, y2, lamb_relax=0.0):
        '''
        :param lamb_relax: allow the intersection to be out of the endpoints.
        :return:
        (the line segment index that first intersects the line (or None), t_value),
        (the closest line segment that doesn't intersect the line, distance)
        '''
        dx1, dy1 = self._grid_coord(x1, y1, self.collision_map_division, self.collision_map_bbox)
        dx2, dy2 = self._grid_coord(x2, y2, self.collision_map_division, self.collision_map_bbox)

        assert 0 <= dx1
        assert 0 <= dy1

        bins = line_utils.rasterize_line(dx1, dy1, dx2, dy2)
        tested = set()

        # from map_visualizer import MapVisualizer
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # vis = MapVisualizer(self, ax)
        # vis.draw_map()
        #
        # vis.draw_grid(self.collision_map, self.collision_map_bbox, self.collision_map_division)
        # vis.draw_grid_map_points(bins, self.collision_map_bbox, self.collision_map_division, color='g', alpha=0.1)
        #
        # print 'endpoint', dx1, dy1, dx2, dy2
        # vis.draw_grid_map_points(
        #     [(dx1, dy1), (dx2, dy2)], self.collision_map_bbox, self.collision_map_division, color='b', alpha=0.3)
        # ax.plot((x1, x2), (y1, y2), 'r')

        min_t = 1e10
        closest_intersect = None

        min_d = 1e10
        closest_nonintersect = None

        visited_bins = set()

        # Find the first line that intersects with this line
        for x0, y0 in bins:
            # Search through the neighborhood of the rastered line because line rasterization
            # uses integer endpoints, so the rastered line can be different from the original
            # line.
            for x in (x0 - 1, x0, x0 + 1):
                for y in (y0 - 1, y0, y0 + 1):

                    if (x, y) in visited_bins:
                        continue
                    visited_bins.add((x, y))

                    if not ((x >= 0 and x < self.collision_map.shape[1]) and
                            (y >= 0 and y < self.collision_map.shape[0])):
                        continue

                    for idx in self.line_map[y][x]:
                        if idx in tested:
                            continue
                        tested.add(idx)

                        x3, y3, x4, y4 = self.lines[idx]

                        # ax.plot((x3, x4), (y3, y4), 'r')

                        r = line_utils.ray_intersect(x1, y1, x2 - x1, y2 - y1, x3, y3, x4, y4)

                        if r:
                            lamb, t = r
                            if -lamb_relax <= t <= 1 + lamb_relax and -lamb_relax <= lamb <= 1 + lamb_relax:
                                # An intersection
                                if t < min_t:
                                    min_t = t
                                    closest_intersect = idx
                            else:
                                # No intersection
                                # Choose the smallest vertical distance from a line's endpoint to
                                # the other line. The foot must lie within the line segment.
                                d = line_utils.line_line_distance(x1, y1, x2, y2, x3, y3, x4, y4)
                                if d is not None and d < min_d:
                                    min_d = d
                                    closest_nonintersect = idx

        # plt.show()
        # exit(0)

        return (closest_intersect, min_t), (closest_nonintersect, min_d)

    def ray_intersection(self, x, y, dir_x, dir_y):
        # Convert the ray intersection problem to a line segment intersection problem
        # by clipping the ray within the map bounding box.
        x1, x2, y1, y2 = self.map_bbox
        x1 -= 0.01
        x2 += 0.01
        y1 -= 0.01
        y2 += 0.01
        test_lines = [
            [x1, y1, x1, y2], [x1, y1, x2, y1], [x1, y2, x2, y2], [x2, y1, x2, y2]
        ]

        for line in test_lines:
            r = line_utils.ray_intersect(x, y, dir_x, dir_y, *line)
            if r:
                lamb, t = r
                if 0 <= lamb <= 1 and t > 0:
                    (idx1, min_t), (idx2, d) = self.line_segment_intersection(
                        x, y, x + t * dir_x, y + t * dir_y, lamb_relax=0.001)
                    return (idx1, min_t * t), (idx2, d)
        return None

    def get_map_bbox(self):
        """
        :return: x1, x2, y1, y2
        """
        from itertools import chain
        xs = list(chain.from_iterable([(x1, x2) for x1, y1, x2, y2 in self.lines]))
        ys = list(chain.from_iterable([(y1, y2) for x1, y1, x2, y2 in self.lines]))
        return min(xs), max(xs), min(ys), max(ys)

    def path_coord_to_map_coord(self, x, y):
        '''
        :return: convert coordinates on the path map (integer) into original map coordinates (float)
        '''
        return self._continuous_coord(x, y, self.path_map_division, self.path_map_bbox)

    def _grid_coord(self, x, y, n_division, discrete_bbox):
        '''
        :return: the bin indices where (x, y) falls
        '''
        return int(x * n_division - discrete_bbox[0] + 0.5), \
               int(y * n_division - discrete_bbox[2] + 0.5)

    def _continuous_coord(self, x, y, n_division, discrete_bbox):
        return (x + discrete_bbox[0]) / float(n_division), \
               (y + discrete_bbox[2]) / float(n_division)

    def _make_grid(self, n_division):
        '''
        :param n_division: number of bins for one unit length
        :return:
        '''

        bbox = self.map_bbox

        left, right = int(np.floor(bbox[0] * n_division)) - 1, int(np.ceil(bbox[1] * n_division)) + 1
        bottom, top = int(np.floor(bbox[2] * n_division)) - 1, int(np.ceil(bbox[3] * n_division)) + 1

        grid_bbox = (left, right, bottom, top)
        grid_map = np.zeros((top - bottom + 1, right - left + 1), np.uint8)

        for x1, y1, x2, y2 in self.lines:
            dx1, dy1 = self._grid_coord(x1, y1, n_division, grid_bbox)
            dx2, dy2 = self._grid_coord(x2, y2, n_division, grid_bbox)
            bins = line_utils.rasterize_line(dx1, dy1, dx2, dy2)
            for x, y in bins:
                grid_map[y, x] = 255

        return grid_bbox, grid_map

    # def visible(self, x1, y1, x2, y2, distance_thres=0.0):
    #     '''
    #     :return: check if (x2, y2) is visible from (x1, y1), i.e., the line (x1, y1, x2, y2)
    #     does not intersect with any line (i.e., wall) in the map.
    #     '''
    #     (closest_intersect, _), (_, d) = self.line_segment_intersection(x1, y1, x2, y2)
    #     return closest_intersect is None and (d > distance_thres)

    def visible(self, x1, y1, x2, y2, distance_thres=0.0):
        origin = self.map_bbox[0], self.map_bbox[2]

        def grid_coord(x, y, n_division):
            return int((x - origin[0]) * n_division + 0.5), int((y - origin[1]) * n_division + 0.5)

        resolution = self.rasterization_resolution

        queries = np.zeros((1, 3), dtype=np.float32)
        result = np.zeros(1, dtype=np.float32)

        x, y = grid_coord(x1, y1, int(1.0 / resolution))
        queries[:, 0] = x
        queries[:, 1] = y
        queries[0, 2] = np.arctan2(y2 - y1, x2 - x1)

        self.range_scanner.calc_range_many(queries, result)

        result *= resolution

        dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        return dist - distance_thres < result[0]

    def _dilate(self, src, n_iter):
        import cv2
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated = cv2.dilate(src, kernel, iterations=n_iter)
        return dilated

    def _erode(self, src, n_iter):
        import cv2
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        ret = cv2.erode(src, kernel, iterations=n_iter)
        return ret

    def find_path(self, start_pos, goal_pos):
        bbox, map, n_division = self.path_map_bbox, self.path_map, self.path_map_division

        waypoints = a_star(map,
                           self._grid_coord(start_pos[0], start_pos[1], n_division, bbox),
                           self._grid_coord(goal_pos[0], goal_pos[1], n_division, bbox))

        if waypoints is None:
            return None

        # def visualize():
        #     import matplotlib.pyplot as plt
        #     import matplotlib.patches as patches
        #     from map_visualizer import MapVisualizer
        #
        #     fig = plt.figure()
        #     vis = MapVisualizer(self, fig)
        #
        #     vis.draw_map()
        #     ax = vis.ax
        #
        #     left, right, bottom, top = discrete_bbox
        #
        #     for i in xrange(left, right+1):
        #         x = float(i - 0.5) / n_division
        #         ax.plot([x, x], [float(bottom - 0.5) / n_division, float(top - 0.5) / n_division],
        #                 color='#CCCCCC', linewidth=1.0, alpha=0.5)
        #
        #     for i in xrange(bottom, top+1):
        #         y = float(i - 0.5) / n_division
        #         ax.plot([float(left - 0.5) / n_division, float(right - 0.5) / n_division], [y, y],
        #                 color='#CCCCCC', linewidth=1.0, alpha=0.5)
        #
        #     for i in xrange(discrete_map.shape[0]):
        #         for j in xrange(discrete_map.shape[1]):
        #             if discrete_map[i, j]:
        #                 ax.add_patch(
        #                     patches.Rectangle(
        #                         ((j + left - 0.5) / float(n_division),
        #                          (i + bottom - 0.5) / float(n_division)),
        #                         1.0 / n_division, 1.0 / n_division
        #                     )
        #                 )
        #     for i in xrange(0, len(waypoints), 5):
        #         x, y = waypoints[i]
        #         ax.add_patch(
        #             patches.Rectangle(
        #                 ((x + left - 0.5) / float(n_division),
        #                  (y + bottom - 0.5) / float(n_division)),
        #                 1.0 / n_division, 1.0 / n_division
        #             )
        #         )
        #     plt.show()
        #
        # # visualize()

        res = []
        for x, y in waypoints:
            cx, cy = self._continuous_coord(x, y, self.path_map_division, self.path_map_bbox)
            res.append((cx, cy))

        return res

    def get_1d_depth(self, pos, n_depth_ray, heading=0.0, fov=np.pi * 2.0):
        '''
        Get depth measurement at location x, y with heading and fov. The depth rays are from right
        to left.
        '''

        resolution = self.rasterization_resolution
        division = int(1.0 / resolution)

        x1, x2, y1, y2 = self.map_bbox
        origin = (x1, y1)

        def grid_coord(x, y, n_division):
            return int((x - origin[0]) * n_division + 0.5), int((y - origin[1]) * n_division + 0.5)

        queries = np.zeros((n_depth_ray, 3), dtype=np.float32)
        result = np.zeros(n_depth_ray, dtype=np.float32)

        x, y = grid_coord(pos[0], pos[1], division)
        queries[:, 0] = x
        queries[:, 1] = y

        for i in xrange(n_depth_ray):
            # FIXME: hack
            if abs(fov - np.pi * 2.0) < 1e-5:
                theta = float(i) / n_depth_ray * fov
            else:
                theta = float(i) / n_depth_ray * fov - fov * 0.5

            queries[i, 2] = (theta + heading)

        self.range_scanner.calc_range_many(queries, result)

        result *= self.rasterization_resolution

        return result

    def __repr__(self):
        return self.name
