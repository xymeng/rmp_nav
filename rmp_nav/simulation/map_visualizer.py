from __future__ import print_function
from future.utils import iteritems
from past.builtins import xrange
import matplotlib.pyplot as plt
import numpy as np
from .cached_drawing import CachedPlotter


class OccupancyMapVisualizer:
    def __init__(self, occupancy_map, ax):
        self.occupancy_map = occupancy_map
        self.ax = ax
        self.plotter = CachedPlotter(self.ax)
        self.first_draw = True

    def draw_map(self, draw_dests=False):
        m = self.occupancy_map
        ax = self.ax

        ax.grid(False)
        ax.set_aspect('equal', 'datalim')

        left = m.origin[0]
        bottom = m.origin[1]
        right = left + m.occupancy_grid.shape[1] * m.resolution
        top = bottom + m.occupancy_grid.shape[0] * m.resolution

        bitmap = np.stack([m.occupancy_grid, m.occupancy_grid, m.occupancy_grid], axis=2)
        # stack() returns a new copy so code below won't modify m.occupancy_grid

        if draw_dests:
            for dest, locs in iteritems(m.reachable_locs_per_destination):
                for x, y in locs:
                    mx, my = m.path_coord_to_map_coord(x, y)
                    gx, gy = m.map_coord_to_occupancy_grid_coord(mx, my)
                    bitmap[gy, gx] = dest

        self.plotter.image(
            'occupancy_map', bitmap, origin='lower', extent=(left, right, bottom, top))

        if self.first_draw:
            x_min, x_max, y_min, y_max = m.visible_map_bbox
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])
            self.first_draw = False


def FindMapVisualizer(map):
    from . import map_occupancy_grid
    if isinstance(map, map_occupancy_grid.Map):
        return OccupancyMapVisualizer
    else:
        raise RuntimeError("Cannot find suitable map visualizer")
