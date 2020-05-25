from matplotlib.patches import Rectangle
from .cached_drawing import CachedPlotter
from .map_visualizer import FindMapVisualizer
import numpy as np


class SimRenderer(object):
    def __init__(self, map, ax, canvas):
        self.ax = ax
        self.canvas = canvas
        self.map = map
        self.map_visualizer = FindMapVisualizer(self.map)(self.map, self.ax)
        self.agents = {}

        self.show_legends = True
        self.h_legend = None

        self.background = None
        self.background_ax_limit = None
        self.background_bbox = None
        self.h_background_rect = None

        self.plotter = CachedPlotter(ax)

    def set_agents(self, agents):
        self.agents = agents

    def clear(self):
        self.plotter.clear()

    def draw_background(self, force_redraw=False):
        '''
        :return: draw background and cache it.
        '''
        # Pan/zoom can change axis limits. In that case we need to redraw the background.
        x1, x2 = self.ax.get_xlim()
        y1, y2 = self.ax.get_ylim()

        if self.background is None or \
                self.background_bbox != self.ax.bbox.__repr__() or \
                self.background_ax_limit != (x1, x2, y1, y2) or \
                force_redraw:

            self.ax.autoscale(False)

            # Cover all graphical elements
            if self.h_background_rect is None:
                self.h_background_rect = Rectangle((x1, y1), x2 - x1, y2 - y1, color='w')
                self.h_background_rect.set_zorder(10**5)
                self.ax.add_patch(self.h_background_rect)
            else:
                self.h_background_rect.set_bounds(x1, y1, x2 - x1, y2 - y1)
                self.h_background_rect.set_visible(True)
                self.ax.draw_artist(self.h_background_rect)

            self.canvas.draw()

            self.map_visualizer.draw_map()

            self.h_background_rect.set_visible(False)

            self.canvas.blit(self.ax.bbox)

            self.background = self.canvas.copy_from_bbox(self.ax.bbox)
            self.background_bbox = self.ax.bbox.__repr__()

            # limits might get changed. Retrieve new limits here.
            x1, x2 = self.ax.get_xlim()
            y1, y2 = self.ax.get_ylim()
            self.background_ax_limit = (x1, x2, y1, y2)
        else:
            self.canvas.restore_region(self.background)

    def set_limits(self, x1, x2, y1, y2):
        self.ax.set_xlim(x1, x2)
        self.ax.set_ylim(y1, y2)

    def get_limits(self):
        x1, x2 = self.ax.get_xlim()
        y1, y2 = self.ax.get_ylim()
        return x1, x2, y1, y2

    def set_viewport(self, x, y, scale):
        """
        :param x, y: center of the viewport
        :param scale: zoom factor w.r.t current viewport
        """
        x_min, x_max, y_min, y_max = self.map.visible_map_bbox

        w = (x_max - x_min) * scale
        h = (y_max - y_min) * scale

        xx1, xx2 = x - w / 2., x + w / 2.
        yy1, yy2 = y - h / 2., y + h / 2.

        self.ax.set_xlim(xx1, xx2)
        self.ax.set_ylim(yy1, yy2)

    def reset_viewport(self):
        x_min, x_max, y_min, y_max = self.map.visible_map_bbox
        self.ax.set_xlim([x_min, x_max])
        self.ax.set_ylim([y_min, y_max])

    def draw_agents(self):
        for name in self.agents:
            v = self.agents[name]
            agent = v['agent']
            visualizer = v['visualizer']
            visualizer.draw_agent_state(agent)

    def render(self, force_redraw=False, blit=True):
        self.draw_background(force_redraw)
        self.draw_agents()
        if blit:
            self.canvas.blit(self.ax.bbox)

    def blit(self):
        self.canvas.blit(self.ax.bbox)

    def get_image(self):
        w, h = self.canvas.get_width_height()
        buf = np.fromstring(self.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape((h, w, 3))
        return buf

    def save(self, filename, **kwargs):
        self.canvas.print_figure(filename, **kwargs)


def visualize(map, traj, markers):
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

    fig = plt.Figure(tight_layout=True)
    ax = fig.add_subplot(111)
    canvas = FigureCanvas(fig)
    canvas.draw()
    cm = matplotlib.cm.get_cmap('Dark2')

    vis = SimRenderer(map, ax, canvas)
    vis.render(True)
    vis.render(True)

    xs, ys = zip(*traj)
    vis.plotter.plot('traj', xs, ys, c='r')

    for i in range(len(markers)):
        vis.plotter.scatter('marker %d' % i, markers[i][0], markers[i][1], marker='o', s=100)

    return vis.get_image()
