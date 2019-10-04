from . import rmp
from ..common import math_utils
from .param_utils import Params
import numpy as np


class SolverBase(object):
    def reset(self):
        pass


class AgentLocalClassicRMPSolver(SolverBase):
    def __init__(self, **kwargs):
        self.angular_damping = kwargs.pop('angular_damping', 3.0)
        self.angular_gain = kwargs.pop('angular_gain', 5.0)
        self.heading_alignment_gain = kwargs.pop('heading_alignment_gain', 4.0)

        self.rmp_solver = rmp.RMP2(**kwargs)

    def get_control_point_accels(self):
        return self.rmp_solver.control_point_accels

    def get_control_point_metrics(self):
        return self.rmp_solver.control_point_metrics

    def _get_angular_vel_damping_accel(self, agent):
        damping_factor = self.angular_damping
        gain = self.angular_gain

        vel = agent.get_local_velocity()
        vel_dir = np.arctan2(vel[1], vel[0])

        angular_vel_accel = vel_dir * gain - agent.angular_velocity * damping_factor
        return angular_vel_accel

    def compute_accel_local(self, agent):
        '''
        :param agent: a QuadPointAgent
        :return: the optimal acceleration (accel_x, accel_y, accel_theta)
        '''
        local_tangent_vel = [agent.local_tangent_velocity(p)
                             for p in agent.control_points]

        local_linear_vel = math_utils.rotate_2d(agent.velocity, -agent.heading)
        local_vel = [local_linear_vel + v for v in local_tangent_vel]

        jacobians = agent.get_control_point_jacobians()

        angular_vel_damping_accel = self._get_angular_vel_damping_accel(agent)

        control_point_states = []
        for i in range(len(agent.control_points)):
            extras = []
            if i == 0:
                gs = agent.goals_local
            else:
                gs = []

            # Add an acceleration to encourage the heading to be aligned with velocity.
            # The acceleration is applied to every control point.

            x, y = agent.control_points[i]
            cp_heading = np.arctan2(y, x)

            cp_ortho = np.array(
                [np.cos(cp_heading + np.pi / 2),
                 np.sin(cp_heading + np.pi / 2)]
            )

            accel = self.heading_alignment_gain \
                    * angular_vel_damping_accel \
                    * agent.approx_radius() \
                    * cp_ortho

            #extras.append((accel, np.eye(2)))

            extra_props = {
                'rmp_config': {}
            }

            control_point_states.append(
                (agent.control_points[i],
                 local_vel[i],
                 gs,
                 agent.obstacles_local,
                 jacobians[i],
                 extras,
                 extra_props)
            )

        ret = self.rmp_solver.compute_optimal_accel(
            control_point_states,
            [])

        return ret


class CarAgentLocalClassicRMPSolver(SolverBase):
    def __init__(self, **kwargs):
        self.rmp_solver = rmp.RMP2(**kwargs)
        self.params = kwargs.get('params', Params())

    def get_control_point_accels(self):
        return self.rmp_solver.control_point_accels

    def get_control_point_metrics(self):
        return self.rmp_solver.control_point_metrics

    def compute_accel_local(self, agent, extra_rmps=None):
        control_point_states = []

        if extra_rmps is None:
            extra_rmps = []

        for i in range(len(agent.control_points)):
            control_point_extra_rmps = []

            p = agent.control_points[i]

            prop = agent.control_point_properties[i]
            affected_by_obstacles = prop['affected_by_obstacle']
            affected_by_goals = prop['affected_by_goal']

            if i == 0 and len(agent.goals_local) > 0:
                x, y = agent.control_points[i]
                cp_heading = np.arctan2(y, x)

                cp_ortho = np.array(
                    [np.cos(cp_heading + np.pi / 2),
                     np.sin(cp_heading + np.pi / 2)]
                )

                accel = self.params.get('heading_alignment_gain', required=True) \
                        * (agent.goals_local[0][1]
                           - agent.angular_accel * self.params.get('heading_alignment_damping2', required=True)
                           - agent.angular_velocity * self.params.get('heading_alignment_damping', required=True)
                           - np.sign(agent.angular_velocity) * agent.angular_velocity**2
                           * self.params.get('heading_alignment_damping3', required=True)) \
                        * agent.approx_radius() \
                        * cp_ortho

                control_point_extra_rmps.append(
                    (accel, np.eye(2) * self.params.get('heading_alignment_metric_scale')))

            extra_props = {
                'obstacle_accel_gain': prop['obstacle_accel_gain'],
                'goal_accel_gain': prop['goal_accel_gain'],
                'obstacle_metric_scale': prop['obstacle_metric_scale'],
                'goal_metric_scale': prop['goal_metric_scale'],
                'combined_accel_gain': prop['combined_accel_gain'],
                'rmp_config': prop.get('rmp_config', {})
            }

            if self.params.get('offset_fix', False):
                x, y = agent.control_points[i]
                J = np.array([[1, 0, -y], [0, 1, x]], np.float32)
                offset = np.dot(
                    J, np.array([
                        0,
                        -agent.get_signed_velocity_norm()**2 / agent.L * np.sin(2 * agent.get_beta()),
                        0], np.float32))
                extra_props['offset'] = offset

            state = [
                p,
                agent.local_velocity(p),
                agent.goals_local if affected_by_goals else [],
                agent.obstacles_local if affected_by_obstacles else [],
                agent.get_local_jacobian(p),
                control_point_extra_rmps,
                extra_props
            ]

            control_point_states.append(state)

        accel = self.rmp_solver.compute_optimal_accel(control_point_states, extra_rmps)
        # FIXME: hack!
        if len(accel) == 2:
            dv, dsteering = accel
            dsteering = np.clip(dsteering, *agent.steer_speed_limit)
            return np.array([dv, dsteering], np.float32)
        else:
            return accel


class LocalVisualNeuralMetricSolverV2(SolverBase):
    def __init__(self, weights_file, n_control_points=4, gpu=0):
        from ..neural.regress_rmp_visual.inference import NeuralRMP
        self.neural_solver = NeuralRMP(weights_file,
                                       n_control_points=n_control_points, device='cuda:%d' % gpu)

    def get_control_point_accels(self):
        return self.neural_solver.control_point_accels

    def get_control_point_metrics(self):
        return self.neural_solver.control_point_metrics

    def compute_accel_local(self, agent):
        return self.neural_solver.compute_optimal_accel(
            agent.img,
            agent.goals_local[0],
            agent.get_local_velocity(),
            agent.angular_velocity,
            agent.get_control_point_jacobians(),
            [])
