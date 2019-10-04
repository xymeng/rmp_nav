import matplotlib
from matplotlib.collections import LineCollection, PatchCollection
import matplotlib.patches as patches
import numpy as np


class CachedPlotter(object):
    def __init__(self, ax):
        self.ax = ax

        self.lines = {}
        self.patches = {}
        self.collections = {}
        self.images = {}
        self.texts = {}

    def get_handle(self, name):
        for d in (self.lines, self.patches, self.collections, self.images):
            if name in d:
                return d[name]
        return None

    def set_visible(self, handle_name, visible):
        h = self.get_handle(handle_name)
        if h:
            h.set_visible(visible)
        else:
            print('warning: cannot find handle name %s' % handle_name)

    def clear(self):
        '''
        remove all graph elements
        '''
        for l in self.lines.values():
            self.ax.lines.remove(l)

        for c in self.collections.values():
            c.remove()

        for p in self.patches.values():
            p.remove()

        for _ in self.texts.values():
            _.remove()

        self.lines = {}
        self.collections = {}
        self.patches = {}
        self.texts = {}

    def plot(self, handle_name, xs, ys, *args, **kwargs):
        if handle_name not in self.lines:
            self.lines[handle_name], = self.ax.plot(xs, ys, *args, **kwargs)
        h = self.lines[handle_name]
        h.set_xdata(xs)
        h.set_ydata(ys)
        self.ax.draw_artist(h)

    def scatter(self, handle_name, xs, ys, *args, **kwargs):
        if handle_name not in self.collections:
            self.collections[handle_name] = self.ax.scatter(xs, ys, *args, **kwargs)
        h = self.collections[handle_name]
        # Use atleast_1d() to handle both sequences or scalars
        xys = np.hstack((np.atleast_1d(xs)[:, np.newaxis], np.atleast_1d(ys)[:, np.newaxis]))
        h.set_offsets(xys)
        self.ax.draw_artist(h)

    def polygon(self, handle_name, points, *args, **kwargs):
        if handle_name not in self.patches:
            p = patches.Polygon(points, *args, **kwargs)
            self.patches[handle_name] = p
            self.ax.add_patch(p)

        h = self.patches[handle_name]
        h.set_xy(points)
        self.ax.draw_artist(h)

    def arc(self, handle_name, xy, width, height, angle, theta1, theta2, **kwargs):
        if handle_name not in self.patches:
            p = patches.Arc(xy, width, height, angle, theta1, theta2, **kwargs)
            self.patches[handle_name] = p
            self.ax.add_patch(p)

        h = self.patches[handle_name]
        h.center = xy
        h.width = width
        h.height = height
        h.angle = angle
        h.theta1 = theta1
        h.theta2 = theta2
        self.ax.draw_artist(h)

    def line_collection(self, handle_name, lines, *args, **kwargs):
        if handle_name not in self.collections:
            collection = LineCollection(lines, *args, **kwargs)
            self.collections[handle_name] = collection
            self.ax.add_collection(collection)

        h = self.collections[handle_name]
        h.set_verts(lines)
        self.ax.draw_artist(h)

    def patch_collection(self, handle_name, patches, *args, **kwargs):
        if handle_name not in self.collections:
            collection = PatchCollection(patches, *args, **kwargs)
            self.collections[handle_name] = collection
            self.ax.add_collection(collection)

        h = self.collections[handle_name]
        h.set_paths(patches)
        self.ax.draw_artist(h)

    def image(self, handle_name, A, **kwargs):
        if handle_name not in self.images:
            im = self.ax.imshow(A, **kwargs)
            self.images[handle_name] = im
        h = self.images[handle_name]
        h.set_data(A)
        self.ax.draw_artist(h)

    def fixed_arrow(self, handle_name, x, y, dx, dy, **kwargs):
        if handle_name not in self.patches:
            p = patches.FancyArrow(x, y, dx, dy, **kwargs)
            self.patches[handle_name] = p
            self.ax.add_patch(p)

        h = self.patches[handle_name]
        self.ax.draw_artist(h)

    def fixed_arrow2(self, handle_name, x1, y1, x2, y2, **kwargs):
        if handle_name not in self.patches:
            p = patches.FancyArrowPatch((x1, y1), (x2, y2), **kwargs)
            self.patches[handle_name] = p
            self.ax.add_patch(p)
        h = self.patches[handle_name]
        self.ax.draw_artist(h)

    def text(self, handle_name, x, y, text, **kwargs):
        if handle_name not in self.texts:
            t = matplotlib.text.Text(x, y, text, **kwargs)
            self.texts[handle_name] = t
            self.ax.add_artist(t)
        h = self.texts[handle_name]
        self.ax.draw_artist(h)

    def camera(self, handle_name, pos, heading, fov, **kwargs):
        x, y = pos

        lines = [
            [(x, y), (x + np.cos(heading - fov * 0.5), y + np.sin(heading - fov * 0.5))],
            [(x, y), (x + np.cos(heading + fov * 0.5), y + np.sin(heading + fov * 0.5))]
        ]

        self.line_collection(handle_name, lines, **kwargs)
