from __future__ import print_function
from past.builtins import xrange

import matplotlib.patches as patches
import numpy as np
from ..common.math_utils import rotate_2d
from .cached_drawing import CachedPlotter


class AgentVisualizerBase(object):
    def __init__(self, ax, **kwargs):
        self.ax = ax
        self.option = kwargs

        self.plotter = CachedPlotter(self.ax)
        self.show_traj = True

    def clear(self):
        self.plotter.clear()

    def set_show_trajectory(self, enable):
        self.plotter.set_visible('traj', enable)

    def set_show_accel(self, enable):
        self.plotter.set_visible('accel', enable)

    def set_show_control_point_accels(self, enable):
        self.plotter.set_visible('control_point_accels', enable)

    def set_show_control_point_metrics(self, enable):
        self.plotter.set_visible('control_point_metrics', enable)

    def set_show_obstacles(self, enable):
        self.plotter.set_visible('obstacles', enable)

    def draw_agent_state(self, agent):
        obs = agent.get_obstacles()
        if obs is not None:
            self.draw_obstacles(obs[:, 0], obs[:, 1])
        if agent.waypoints is not None and len(agent.waypoints) > 0:
            self.draw_active_waypoint(*agent.waypoints[agent.wp_idx])

        # We should draw goal_local before agent step, otherwise the position will be wrong.
        if agent.goals_global is not None and len(agent.goals_global) > 0:
            self.draw_goal_global(*agent.goals_global[0])

        self.draw_agent(agent)
        self.draw_control_point_accels(agent)
        self.draw_control_point_metrics(agent)
        self.draw_accel(agent)

    def draw_agent(self, agent):
        """
        Draw the physical shape of the agent
        """
        raise NotImplementedError

    def draw_trajectory(self, xs, ys):
        kwargs = {
            'color': self.option.get('traj_color', 'r'),
            'label': self.option.get('label', None)
        }
        self.plotter.plot('traj', xs, ys, alpha=0.5, **kwargs)

    def draw_obstacles(self, xs, ys):
        kwargs = {
            'color': self.option.get('obstacle_color', 'b')
        }
        self.plotter.plot('obstacles', xs, ys, '+', markeredgewidth=2, **kwargs)

    def draw_active_waypoint(self, x, y):
        kwargs = {
            'color': self.option.get('active_wp_color', 'g'),
            's': self.option.get('active_wp_size', 200.0)
        }
        self.plotter.scatter('active_waypoint', x, y, marker='.', alpha=1.0, **kwargs)

    def draw_goal_global(self, x, y):
        kwargs = {
            'color': self.option.get('goal_color', 'm'),
            's': self.option.get('active_wp_size', 200.0)
        }
        self.plotter.scatter('goal_global', x, y, marker='.', alpha=1.0, **kwargs)

    def draw_accel(self, agent):
        p1 = agent.pos
        p2 = p1 + agent.accel
        self.plotter.plot('accel', (p1[0], p2[0]), (p1[1], p2[1]), 'r')

    def draw_control_point_accels(self, agent):
        accels = None
        if hasattr(agent, 'control_point_accels'):
            accels = agent.control_point_accels

        if accels is None or len(accels) == 0:
            return

        ctrl_point_pos = np.array(agent.get_global_control_points_pos())
        lines = []

        for i in xrange(len(ctrl_point_pos)):
            p = ctrl_point_pos[i]
            q = agent.control_points[i] + accels[i]
            lines.append((p, agent.local_to_global(q)))

        self.plotter.line_collection(
            'control_point_accels', lines, linewidths=0.5, colors=['r'] * (len(lines) - 1))

    def closest_psd(self, A):
        C = (A + A.T) / 2
        eigvals, eigvecs = np.linalg.eig(C)
        D = np.maximum(np.diag(eigvals), np.zeros(A.shape, A.dtype))
        return np.linalg.multi_dot([eigvecs, D, eigvecs.T])

    def draw_control_point_metrics(self, agent):
        metrics = None
        if hasattr(agent, 'control_point_metrics'):
            metrics = agent.control_point_metrics

        if metrics is None or len(metrics) == 0:
            return

        ctrl_point_pos = np.array(agent.get_global_control_points_pos())

        ellipses = []
        for i in xrange(len(ctrl_point_pos)):
            metric = metrics[i]
            eigvals, eigvecs = np.linalg.eig(metric)

            if np.any(np.iscomplex(eigvals)) or np.any(eigvals < 1e-3):
                # Matrix is not positive definite.
                continue

            axes_lengths = 1.0 / np.sqrt(eigvals) * 0.1
            max_len = max(axes_lengths)
            scale = min(2.0 / max_len, 1.0)
            axes_lengths *= scale

            angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0]) + agent.heading
            ellipses.append(patches.Ellipse(
                ctrl_point_pos[i], axes_lengths[0], axes_lengths[1],
                angle=np.rad2deg(angle),
                fill=None))

        self.plotter.patch_collection(
            'control_point_metrics',
            ellipses,
            facecolors=['none']*len(ellipses),
            edgecolors=['green']*len(ellipses),
            linewidths=0.5)


class QuadAgentVisualizer(AgentVisualizerBase):
    def __init__(self, *args, **kwargs):
        super(QuadAgentVisualizer, self).__init__(*args, **kwargs)

    def draw_agent_state(self, agent):
        obs = agent.get_obstacles()
        if obs is not None:
            self.draw_obstacles(obs[:, 0], obs[:, 1])
        if agent.waypoints is not None and len(agent.waypoints) > 0:
            self.draw_active_waypoint(*agent.waypoints[agent.wp_idx])
        self.draw_agent(agent)
        self.draw_control_point_accels(agent)
        self.draw_control_point_metrics(agent)
        self.draw_accel(agent)

    def draw_agent(self, agent):
        ls = []
        ctrl_point_pos = np.array(agent.get_global_control_points_pos())
        for p in ctrl_point_pos:
            ls.append((agent.pos, p))

        heading_color = self.option.get('heading_color', 'r')

        self.plotter.line_collection('quad_arms', ls, colors=[heading_color] + ['k'] * (len(ls) - 1))

        # ls = []
        # for p in ctrl_point_pos:
        #     ls.append((agent.pos, (p - agent.pos) * 5.0 + agent.pos))
        #
        # if self.agent_axis2 is None:
        #     self.agent_axis2 = LineCollection(ls, colors=[heading_color] + ['k'] * (len(ls) - 1), alpha=0.3)
        #     self.ax.add_collection(self.agent_axis2)
        # else:
        #     self.agent_axis2.set_verts(ls)
        #     self.ax.draw_artist(self.agent_axis2)

        # if not self.option.get('draw_control_point', True):
        #     return
        #
        # if self.agent_ctrl_points:
        #     self.agent_ctrl_points.set_offsets(ctrl_point_pos)
        # else:
        #     n = len(ctrl_point_pos)
        #     facecolors = ['r'] + ['none'] * (n - 1)
        #     edgecolors = ['r'] + ['k'] * (n - 1)
        #     self.agent_ctrl_points = self.ax.scatter(
        #         ctrl_point_pos[:, 0], ctrl_point_pos[:, 1], 20.0,
        #         facecolors=facecolors, edgecolors=edgecolors)

        #self.draw_control_point_accels(agent)
        #self.draw_control_point_metrics(agent)


class AgentVisualizer(AgentVisualizerBase):
    def draw_agent(self, agent):
        ctrl_point_pos = np.array(agent.get_global_control_points_pos())

        self.plotter.polygon('boundary', ctrl_point_pos, closed=True, fill=None, edgecolor='gray')

        heading_color = self.option.get('heading_color', 'r')
        n = len(ctrl_point_pos)
        self.plotter.scatter(
            'control_points',
            ctrl_point_pos[:, 0], ctrl_point_pos[:, 1], 10.0,
            facecolors=[heading_color] + ['w'] * (n - 1),
            edgecolors=[heading_color] + ['k'] * (n - 1))


class CarAgentVisualizer(AgentVisualizerBase):
    def draw_steering_velocity(self, agent):
        dsteering = agent.dsteering

        theta1 = np.rad2deg(agent.heading)
        theta2 = np.rad2deg(agent.heading + dsteering)

        if dsteering < 0:
            theta2, theta1 = theta1, theta2

        self.plotter.arc('dsteering', agent.pos, 0.1, 0.1, angle=0.0, theta1=theta1, theta2=theta2)

    def draw_collision_boundary(self, agent):
        boundary = agent.get_boundary_control_points()
        points = np.array([agent.local_to_global(p) for p in boundary])

        self.plotter.polygon(
            'collision_boundary', points, closed=True, fill=None, linewidth=0.5, edgecolor='gray')

    def draw_control_points(self, agent):
        ctrl_point_pos = np.array(agent.get_global_control_points_pos())

        heading_color = self.option.get('heading_color', 'r')
        n = len(ctrl_point_pos)
        self.plotter.scatter(
            'control_points', ctrl_point_pos[:, 0], ctrl_point_pos[:, 1], 10.0,
            facecolors=[heading_color] * n,
            edgecolors=[heading_color] * n)

    def draw_wheels(self, agent):
        wheel_length = agent.L / 4.0
        wheel_width = wheel_length / 8.0

        wheel_points_local = np.array([
            [wheel_length * 0.5, wheel_width * 0.5],
            [-wheel_length * 0.5, wheel_width * 0.5],
            [-wheel_length * 0.5, -wheel_width * 0.5],
            [wheel_length * 0.5, -wheel_width * 0.5]
        ], np.float32)

        steering = agent.steering

        front_left_wheel_points_local = [rotate_2d(p, steering) for p in wheel_points_local]
        front_left_wheel_points_local += np.array([agent.L * 0.5, agent.W * 0.5 + wheel_width * 0.5], np.float32)
        front_left_wheel_points_global = [agent.local_to_global(p) for p in front_left_wheel_points_local]

        self.plotter.polygon('front_left_wheel', front_left_wheel_points_global, closed=True)

        front_right_wheel_points_local = [rotate_2d(p, steering) for p in wheel_points_local]
        front_right_wheel_points_local += np.array([agent.L * 0.5, -agent.W * 0.5 - wheel_width * 0.5], np.float32)
        front_right_wheel_points_global = [agent.local_to_global(p) for p in front_right_wheel_points_local]

        self.plotter.polygon('front_right_wheel', front_right_wheel_points_global, closed=True)

        rear_left_wheel_points_local = wheel_points_local + np.array([-agent.L * 0.5, agent.W * 0.5 + wheel_width * 0.5], np.float32)
        rear_left_wheel_points_global = [agent.local_to_global(p) for p in rear_left_wheel_points_local]

        self.plotter.polygon('rear_left_wheel', rear_left_wheel_points_global, closed=True)

        rear_right_wheel_points_local = wheel_points_local + np.array([-agent.L * 0.5, -agent.W * 0.5 - wheel_width * 0.5], np.float32)
        rear_right_wheel_points_global = [agent.local_to_global(p) for p in rear_right_wheel_points_local]

        self.plotter.polygon('rear_right_wheel', rear_right_wheel_points_global, closed=True)

    def draw_car_model(self, agent):
        body_points_local = np.array([
            [agent.L * 0.5, agent.W * 0.5],
            [-agent.L * 0.5, agent.W * 0.5],
            [-agent.L * 0.5, -agent.W * 0.5],
            [agent.L * 0.5, -agent.W * 0.5]
        ], np.float32)

        body_points_global = [
            agent.local_to_global(body_points_local[i]) for i in xrange(len(body_points_local))
        ]

        self.plotter.polygon('car_body', body_points_global, closed=True, fill=None, edgecolor='gray')

        self.draw_wheels(agent)

    def draw_agent(self, agent):
        self.draw_collision_boundary(agent)
        self.draw_control_points(agent)
        self.draw_car_model(agent)


class CarAgentVisualizerWaypoint(CarAgentVisualizer):
    def draw_agent_state(self, agent):
        super(CarAgentVisualizerWaypoint, self).draw_agent_state(agent)
        self.draw_neural_waypoint(*agent.local_to_global(agent.neural_waypoint))

    def draw_neural_waypoint(self, x, y):
        kwargs = {
            'color': self.option.get('neural_wp_color', (0.8, 0.0, 0.0)),
            's': self.option.get('active_wp_size', 200.0)
        }
        self.plotter.scatter('neural_waypoint', x, y, marker='.', alpha=1.0, **kwargs)


def FindAgentVisualizer(agent_instance):
    cls_name = agent_instance.__class__.__name__
    if cls_name.startswith('Quad'):
        return QuadAgentVisualizer
    elif cls_name.startswith('RCCar'):
        if 'Waypoint' in cls_name:
            return CarAgentVisualizerWaypoint
        else:
            return CarAgentVisualizer
    else:
        return AgentVisualizer
