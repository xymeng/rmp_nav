import numpy as np
from ..common import utils
from .param_utils import Params


def desired_accel_obstacle_vec(o, x, dx, gain=0.8, damping=1.0):
    if len(o) == 0:
        return np.zeros((0, 2), np.float32)

    d = np.linalg.norm(o - x, 2, axis=1)
    v = o - x
    vhat = v / (d[:, np.newaxis] + 0.001)
    accel = -1. / (d[:, np.newaxis] / gain + 0.001) * (.1 * vhat + damping * np.dot(vhat, dx)[:, np.newaxis] * vhat)
    return accel


def desired_accel_goal(xg, x, dx, attraction=10.0, damping=15.0, overshoot_penalty=0.0):
    if xg is None:
        return 0

    overshoot_deaccel = -dx * min(abs(xg[1]) / max(xg[0], 0.01), 10.0) * overshoot_penalty * min(np.linalg.norm(xg), 1.0)

    return attraction * (xg - x) / (np.linalg.norm(xg - x) + 0.01) - damping * dx + overshoot_deaccel


class RMP2:
    def __init__(self, **kwargs):
        # controls weights of obstacle metrics
        self.dist_penalty_offset = kwargs.pop('dist_penalty_offset', 0)
        self.dist_penalty_coeff = kwargs.pop('dist_penalty_coeff', 1.0)
        self.dist_penalty_multiplier = kwargs.pop('dist_penalty_multiplier', 1.0)

        # parameters for computing accelerations
        self.obstacle_repel_gain = kwargs.pop('obstacle_repel_gain', 0.8)
        self.obstacle_damping = kwargs.pop('obstacle_damping', 1.0)
        self.goal_attraction_gain = kwargs.pop('goal_attraction_gain', 10.0)
        self.goal_damping = kwargs.pop('goal_damping', 15.0)

        # Contain additional tuning parameters
        self.params = kwargs.pop('params', Params())

        self.control_point_accels = []
        self.control_point_metrics = []
        self.combined_metric = None

    def compute_goal_ddx(self, x, dx, goals, attraction_gain, damping, overshoot_penalty):
        ddx_goals = np.zeros((len(goals), 2))
        for i in range(len(goals)):
            ddx_goals[i] = desired_accel_goal(
                goals[i], x, dx, attraction_gain, damping, overshoot_penalty)
        return ddx_goals

    def compute_obstacle_ddx(self, x, dx, obstacles, repel_gain, damping, align_with_normal):
        ddx_obstacles = desired_accel_obstacle_vec(
            obstacles, x, dx, repel_gain, damping)

        if align_with_normal:
            # The force from each obstacle is in the same direction of the surface normal
            # of that obstacle point.
            normals = utils.compute_normals(obstacles)
            for i in range(len(ddx_obstacles)):
                ddx_obstacles[i] = np.dot(ddx_obstacles[i], normals[i]) * normals[i]

            ddx_obstacles = np.array(ddx_obstacles, np.float32)

        return ddx_obstacles

    def compute_obstacle_metrics(self, x, obstacles, ddx_obstacles,
                                 dist_penalty_multiplier,
                                 dist_penalty_offset,
                                 dist_penalty_coeff):
        if len(obstacles) == 0:
            return np.zeros((0, 2, 2), np.float32)

        ds = np.linalg.norm(obstacles - x, axis=1)
        ddx_norm = np.linalg.norm(ddx_obstacles, axis=1) + 0.001
        hs = ddx_obstacles / ddx_norm[:, np.newaxis]

        A = np.matmul(hs[:, :, np.newaxis], hs[:, np.newaxis, :])

        a, b, c = dist_penalty_multiplier, dist_penalty_offset, dist_penalty_coeff
        weight = a * np.exp(-(ds - b) * c)
        metrics = weight[:, None, None] * A

        return metrics

    def combine_metrics(self, ddxs, metrics):
        '''
        Compute an equivalent acceleration and metric.
        :return:
        '''
        A = np.sum(metrics, axis=0)

        # Batch matrix multiplication
        # Equivalent to
        # b = []
        # for i in xrange(len(metrics)):
        #     b.append(np.dot(metrics[i], ddxs[i]))
        b = np.squeeze(np.matmul(metrics, ddxs[:, :, np.newaxis]), axis=2)
        b = np.sum(b, axis=0)
        return np.dot(np.linalg.pinv(A), b), A

    def compute_optimal_accel(self, control_point_states, extra_rmps):
        '''
        :param control_point_states: a list of tuples
               (x, dx, goals, obstacles, jacobian, extra_rmps, offset [optional])
        :param extra_rmps: a list of tuples (jacobian, metric, acceleration) to be
                combined.
        :return: the optimal acceleration
        '''

        B = []
        C = []

        self.control_point_accels = []
        self.control_point_metrics = []

        for idx, control_point_state in enumerate(control_point_states):
            x, dx, goals, obstacles, jacobian, extras, props = control_point_state

            rmp_cfg = props.get('rmp_config', {})

            ddx_goals = self.compute_goal_ddx(
                x, dx, goals,
                attraction_gain=rmp_cfg.get('goal_attraction_gain', self.goal_attraction_gain),
                damping=rmp_cfg.get('goal_damping', self.goal_damping),
                overshoot_penalty=rmp_cfg.get('goal_overshoot_penalty', 0.0)
            )

            ddx_obstacles = self.compute_obstacle_ddx(
                x, dx, obstacles,
                repel_gain=rmp_cfg.get('obstacle_repel_gain', self.obstacle_repel_gain),
                damping=rmp_cfg.get('obstacle_damping', self.obstacle_damping),
                align_with_normal=rmp_cfg.get('obstacle_ddx_align_with_normal', False)
            )

            metric_goals = np.zeros((len(goals), 2, 2))
            for i in range(len(goals)):
                metric_goals[i] = np.eye(2)

            metric_obstacles = self.compute_obstacle_metrics(
                x, obstacles, ddx_obstacles,
                dist_penalty_multiplier=rmp_cfg.get('dist_penalty_multiplier', self.dist_penalty_multiplier),
                dist_penalty_offset=rmp_cfg.get('dist_penalty_offset', self.dist_penalty_offset),
                dist_penalty_coeff=rmp_cfg.get('dist_penalty_coeff', self.dist_penalty_coeff)
            )

            ddx_goals *= props.get('goal_accel_gain', 1.0)
            ddx_obstacles *= props.get('obstacle_accel_gain', 1.0)
            metric_goals *= props.get('goal_metric_scale', 1.0)
            metric_obstacles *= props.get('obstacle_metric_scale', 1.0)

            if len(extras) > 0:
                all_ddx = np.concatenate(
                    [ddx_goals, ddx_obstacles, [e[0] for e in extras]], axis=0)
                all_metrics = np.concatenate(
                    [metric_goals, metric_obstacles, [e[1] for e in extras]], axis=0)
            else:
                all_ddx = np.concatenate(
                    [ddx_goals, ddx_obstacles], axis=0)
                all_metrics = np.concatenate(
                    [metric_goals, metric_obstacles], axis=0)

            assert len(all_ddx) == len(all_metrics)

            if 'offset' in props:
                all_ddx += props['offset']

            # Combine all obstacles and goals into a single acceleration and metric
            # for each control point
            ddx_combined, metric_combined = self.combine_metrics(all_ddx, all_metrics)

            ddx_combined *= props.get('combined_accel_gain', 1.0)

            self.control_point_metrics.append(metric_combined)
            self.control_point_accels.append(ddx_combined)

            B.append(np.linalg.multi_dot((jacobian.T, metric_combined, ddx_combined)))
            C.append(np.linalg.multi_dot((jacobian.T, metric_combined, jacobian)))

        for jacobian, metric, ddx in extra_rmps:
            B.append(np.linalg.multi_dot((jacobian.T, metric, ddx)))
            C.append(np.linalg.multi_dot((jacobian.T, metric, jacobian)))

        B = np.sum(B, axis=0)
        C = np.sum(C, axis=0)

        self.combined_metric = C

        accel = np.dot(np.linalg.pinv(C), B)

        return accel
