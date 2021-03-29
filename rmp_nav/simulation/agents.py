from __future__ import print_function
from past.builtins import xrange
import math
import numpy as np
from numpy.linalg import norm
import cv2
from copy import deepcopy
from . import agent_utils, sensors
from ..common.math_utils import rotate_2d, depth_to_xy
from ..common.utils import pprint_dict


class Agent(object):
    def __init__(self, name=''):
        """
        :param name: the name of the agent
        """
        self.name = name
        self.pos = None
        self.heading = None
        self.velocity = None
        self.map = None
        self.step_count = 0

    def save_state(self):
        return deepcopy({
            'pos': self.pos,
            'heading': self.heading,
            'velocity': self.velocity,
        })

    def restore_state(self, state):
        self.pos = deepcopy(state['pos'])
        self.heading = deepcopy(state['heading'])
        self.velocity = deepcopy(state['velocity'])

    def set_pos(self, pos):
        self.pos = np.array(pos, copy=True)

    def set_heading(self, heading):
        self.heading = float(heading)

    def set_map(self, map):
        self.map = map

    def set_velocity(self, pos):
        self.velocity = np.array(pos, copy=True)

    def collide(self):
        '''
        :return: True if the agent collides with walls
        '''
        return NotImplementedError

    def next_visible_waypoint(self, waypoints, cur_wp_idx, dist_thres,
                              pos=None, max_ahead=10, max_dist=None):
        '''
        Find the next visible waypoint starting from @param cur_wp_idx. If none is visible,
        backtrack a closest point. If still none is visible, return @param cur_wp_idx.
        :param dist_thres: threshold of waypoint-obstacle distance when testing visibility.
        :param max_ahead: consider waypoint indices up to this value away.
                          This is to avoid selecting waypoint visible from a window.
        '''
        if pos is None:
            pos = self.pos

        # Not the most efficient
        next_visible_wp = None
        for j in xrange(cur_wp_idx, min(len(waypoints), cur_wp_idx + max_ahead)):
            x, y = waypoints[j]
            if self.map.visible(pos[0], pos[1], x, y, dist_thres):
                next_visible_wp = j
                if max_dist is not None:
                    if (pos[0] - x) ** 2 + (pos[1] - y) ** 2 > max_dist ** 2:
                        break

        if next_visible_wp is None:
            # all further waypoints are not visible, backtrack
            for j in xrange(cur_wp_idx - 1, -1, -1):
                x, y = waypoints[j]
                if self.map.visible(pos[0], pos[1], x, y, dist_thres):
                    next_visible_wp = j
                    break

        if next_visible_wp is None:
            print('warning: no visible waypoint. using current')
            next_visible_wp = cur_wp_idx
        return next_visible_wp

    def reset(self):
        self.pos = np.array([0, 0], np.float32)
        self.velocity = np.array([0, 0], np.float32)
        self.heading = 0.0
        self.step_count = 0

    def stopped(self):
        '''
        :return: True if the agent has stopped (means either it reaches the goal or gets stuck)
        '''
        raise NotImplementedError

    def step(self, step_size, **kwargs):
        '''
        Make one step of simulation.
        :param step_size: time step size.
        '''
        self.step_count += 1


class AgentLocal(Agent):
    def __init__(self, control_points, solver,
                 state_history_len=100,
                 goal_clip_fov=None,  # TODO: change 'goal' to 'waypoint'?
                 max_waypoint_ahead=100,
                 max_waypoint_dist=None,
                 waypoint_visibility_thres=0.06,
                 waypoint_normalize_dist=None,
                 max_vel=None,
                 **kwargs):
        '''
        A general agent that uses local sensor measurements. The obstacles and waypoints
        are w.r.t. the local coordinate system. Note that internally it still has a global
        representation but it is only for convenience, and should not be directly used for making
        actions.
        :param goal_clip_fov: the field of view used to clip the goal points. None to disable.
        :param control_points: user-defined control points. N x 2 array.
        :param state_history_len: the maximum length of state history. Used to determine whether
               the agent has stopped.
        :param max_waypoint_ahead: upper limit of the distance (in waypoint count)
                                   to the next waypoint.
        :param max_waypoint_dist: upper limit of the metric distance to the next waypoint.
        :param max_vel: maximum linear velocity.
        '''

        super(AgentLocal, self).__init__(**kwargs)
        self.control_points = np.array(control_points, copy=True)
        self.solver = solver

        self.angular_velocity = None
        self.wp_idx = None
        self.waypoints = None
        self.accel = None

        self.obstacles_local = None
        self.obstacles_global = None

        self.sensors = []

        # Goals in global / local coordindate frame.
        # Note that goals in goals_global can be different from waypoints[wp_idx] if we postprocess
        # the waypoint.
        self.goals_global = None
        self.goals_local = None

        self.goal_clip_fov = goal_clip_fov
        self.max_waypoint_ahead = max_waypoint_ahead
        self.max_waypoint_dist = max_waypoint_dist

        self.waypoint_visibility_thres = waypoint_visibility_thres
        self.waypoint_normalize_dist = waypoint_normalize_dist

        self.max_vel = max_vel

        self.g = {
            'goal_clip_fov': self.goal_clip_fov,
            'max_waypoint_ahead': self.max_waypoint_ahead,
            'max_waypoint_dist': self.max_waypoint_dist,
            'waypoint_visibility_thres': self.waypoint_visibility_thres,
            'waypoint_normalize_dist': self.waypoint_normalize_dist
        }

        self.control_point_accels = None
        self.control_point_metrics = None

        from collections import deque
        self.state_history = deque([], maxlen=state_history_len)

        self.reset()

    def __repr__(self):
        return '%s options\n%s' % (self.__class__.__name__, pprint_dict(self.g))

    def save_state(self):
        state1 = super(AgentLocal, self).save_state()
        state2 = deepcopy({
            'angular_velocity': self.angular_velocity,
            'waypoints': self.waypoints,  # waypoints may change at every step if replan is enabled
            'wp_idx': self.wp_idx,
            'accel': self.accel,
            'obstacles_local': self.obstacles_local,
            'obstacles_global': self.obstacles_global,
            'goals_local': self.goals_local,
            'goals_global': self.goals_global,
            'control_point_accels': self.control_point_accels,
            'control_point_metrics': self.control_point_metrics
        })
        state1.update(state2)
        return state1

    def restore_state(self, state):
        self.angular_velocity = deepcopy(state['angular_velocity'])
        self.waypoints = deepcopy(state['waypoints'])
        self.wp_idx = deepcopy(state['wp_idx'])
        self.accel = deepcopy(state['accel'])
        self.obstacles_local = deepcopy(state['obstacles_local'])
        self.obstacles_global = deepcopy(state['obstacles_global'])
        self.goals_local = deepcopy(state['goals_local'])
        self.goals_global = deepcopy(state['goals_global'])
        self.control_point_accels = deepcopy(state['control_point_accels'])
        self.control_point_metrics = deepcopy(state['control_point_metrics'])
        super(AgentLocal, self).restore_state(state)

    def reset(self):
        super(AgentLocal, self).reset()
        self.solver.reset()

        self.angular_velocity = 0.0  # Counterclockwise

        self.wp_idx = 0
        self.waypoints = None
        self.accel = np.array([0.0, 0.0])

    def set_map(self, map):
        super(AgentLocal, self).set_map(map)
        for sensor in self.sensors:
            sensor.set_map(map)

    def set_waypoints(self, waypoints):
        if waypoints is None:
            self.waypoints = None
            return
        self.waypoints = np.array(waypoints, copy=True)

    def get_obstacles(self):
        return self.obstacles_global

    def get_local_velocity(self):
        return rotate_2d(self.velocity, -self.heading)

    def get_local_jacobian(self, p):
        '''
        :return: the jacobian of point (x, y) in the local coordinate system
        '''
        return np.array([[1, 0, -p[1]], [0, 1, p[0]]], np.float32)

    def get_control_point_jacobians(self):
        '''
        :return: the jacobians of control points w.r.t. the local coordinate system.
        '''
        return [self.get_local_jacobian(p) for p in self.control_points]

    def global_to_local(self, p):
        '''
        :param p: the point w.r.t the global coordinate system
        :return: the same point w.r.t. the local coordinate system
        '''
        return rotate_2d(p - self.pos, -self.heading)

    def local_to_global(self, p):
        """
        :param p: either a 2-element vector or a Nx2 matrix
        :return:
        """
        return rotate_2d(p, self.heading) + self.pos

    def get_global_control_points_pos(self):
        return [self.local_to_global(p) for p in self.control_points]

    def local_tangent_velocity(self, p):
        return self.angular_velocity * np.array([-p[1], p[0]], np.float32)

    def local_velocity(self, p):
        '''
        :param p: x, y
        :return: the velocity of p w.r.t the local coordinate system
        '''
        v_tangent = self.local_tangent_velocity(p)
        v_linear = rotate_2d(self.velocity, -self.heading)
        return v_linear + v_tangent

    def _measure(self):
        pass

    def _compute_next_waypoint(self):
        if self.waypoints is not None and len(self.waypoints) > 0:
            next_wp = self.next_visible_waypoint(
                self.waypoints, self.wp_idx,
                self.waypoint_visibility_thres,
                max_ahead=self.max_waypoint_ahead,
                max_dist=self.max_waypoint_dist)
            return next_wp
        else:
            return None

    def _postprocess_waypoint(self, wp_idx):
        goal_local = self.global_to_local(self.waypoints[wp_idx])

        if self.goal_clip_fov is not None:
            goal_local = agent_utils.clip_within_fov(goal_local, self.goal_clip_fov)

        if self.waypoint_normalize_dist is not None:
            goal_local = goal_local / np.linalg.norm(goal_local, ord=2) * self.waypoint_normalize_dist

        return goal_local

    def _compute_goals(self):
        next_wp = self._compute_next_waypoint()
        if next_wp is not None:
            self.goals_local = [self._postprocess_waypoint(next_wp)]
            self.goals_global = [self.local_to_global(self.goals_local[0])]
            self.wp_idx = next_wp

    def _apply_accel(self, accel_local, step_size, max_vel=None):
        linear_accel = rotate_2d(accel_local[:2], self.heading)
        angular_accel = accel_local[2]

        self.angular_velocity += angular_accel * step_size
        self.velocity += linear_accel * step_size

        if max_vel is not None:
            vnorm = np.linalg.norm(self.velocity)
            if vnorm > max_vel:
                self.velocity = self.velocity / vnorm * max_vel

        self.accel = linear_accel[:2]

        self.pos += self.velocity * step_size
        self.heading += self.angular_velocity * step_size

    def accel_size(self):
        """
        :return: the number of elements in the acceleration representation. For a point agent
        this is (ddx, ddy, angular accel), but it can be other size for other types of agents.
        """
        return 3

    def step(self, step_size, **kwargs):
        """
        :param kwargs:
            waypoint: overwrites the internally computed waypoint
            measure: False to skip the measurement step
        :return:
        """
        super(AgentLocal, self).step(step_size)

        if kwargs.get('measure', True):
            self._measure()

        wp = kwargs.get('waypoint', None)
        if wp is not None:
            self.goals_local = [wp]
            self.goals_global = [self.local_to_global(wp)]
        else:
            self._compute_goals()

        accel_local = self.solver.compute_accel_local(self)

        if hasattr(self.solver, 'get_control_point_accels'):
            self.control_point_accels = self.solver.get_control_point_accels()

        if hasattr(self.solver, 'get_control_point_metrics'):
            self.control_point_metrics = self.solver.get_control_point_metrics()

        self._apply_accel(accel_local, step_size, max_vel=kwargs.get('max_vel', self.max_vel))

        self.state_history.append(
            (np.array(self.velocity, copy=True), np.array(self.accel, copy=True)))

    def approx_radius(self):
        '''
        :return: the max distance of each control point to the center of the agent.
                 When the agent is ciruclarly symmetric, it is the same as its radius.
        '''
        return max(np.linalg.norm(self.control_points, 2, axis=1))

    def reached_goal(self, relax=1.5):
        """
        :param relax: relaxation on proximity to the goal. 1.0 means it must be within the agent's
                      geometric radius.
        """
        if self.waypoints is None or len(self.waypoints) == 0:
            return False

        goal = self.waypoints[-1]
        return norm(self.pos - goal) < self.approx_radius() * relax

    def stopped(self):
        if len(self.state_history) < self.state_history.maxlen:
            return False

        for vel, accel in self.state_history:
            if not (norm(accel[:2]) < 5e-3 and norm(vel) < 5e-3):
                return False
        return True


class AgentLocalLIDAR(AgentLocal):
    def __init__(self, n_depth_ray, lidar_sensor_pos=(0.0, 0.0), lidar_fov=np.pi*2.0, **kwargs):
        super(AgentLocalLIDAR, self).__init__(**kwargs)
        self.lidar_sensor = sensors.Lidar(n_depth_ray, lidar_fov)
        self.sensors.append(self.lidar_sensor)
        self.depth = None
        self.depth_local = None
        self.lidar_sensor_pos = np.array(lidar_sensor_pos, np.float32)

    def reset(self):
        super(AgentLocalLIDAR, self).reset()
        self.depth = None
        self.depth_local = None

    def _measure(self):
        super(AgentLocalLIDAR, self)._measure()

        depth_local = self.lidar_sensor.measure(
            self.local_to_global(self.lidar_sensor_pos), self.heading)

        self.depth_local = depth_local

        assert np.max(self.depth_local) < 1e3

        obstacles_local = depth_to_xy(depth_local, fov=self.lidar_sensor.fov) + self.lidar_sensor_pos

        self.obstacles_local = obstacles_local
        self.obstacles_global = self.local_to_global(obstacles_local)


class AgentLocalVisual2LIDAR_Gibson(AgentLocal):
    '''
    Use image to predict LIDAR rays, which is then used as input of the configs solver.
    '''
    def __init__(self, n_depth_ray, gpu_idx=0, lidar_fov=np.pi*2.0, h_fov=None, v_fov=None,
                 camera_pos=(0.0, 0.0), camera_z=1.0, **kwargs):
        """
        :param lidar_fov: lidar's fov
        :param h_fov: camera's horizontal fov
        :param v_fov: camera's vertical fov
        """
        self.fov = lidar_fov
        self.n_depth = n_depth_ray
        self.gpu_idx = gpu_idx

        self.h_fov = h_fov
        self.v_fov = v_fov

        self.camera_pos = np.array(camera_pos, np.float32)
        self.camera_z = camera_z

        # These are computed by the neural lidar depth predictor, so not directly measured.
        self.depth = None
        self.depth_local = None

        self.sim_client = None
        self.img = None
        super(AgentLocalVisual2LIDAR_Gibson, self).__init__(**kwargs)

    def save_state(self):
        state1 = super(AgentLocalVisual2LIDAR_Gibson, self).save_state()
        state2 = deepcopy({
            'img': self.img,
            'depth': self.depth,
            'depth_local': self.depth_local,
        })
        state1.update(state2)
        return state1

    def restore_state(self, state):
        self.img = deepcopy(state['img'])
        self.depth = deepcopy(state['depth'])
        self.depth_local = deepcopy(state['depth_local'])
        super(AgentLocalVisual2LIDAR_Gibson, self).restore_state(state)

    def reset(self):
        super(AgentLocalVisual2LIDAR_Gibson, self).reset()
        self.depth = None
        self.depth_local = None

    def set_map(self, map):
        from .gibson_sim_client import GibsonSimClient

        if map.__repr__ == self.map.__repr__:
            # Skip re-initialization if possible
            return

        super(AgentLocalVisual2LIDAR_Gibson, self).set_map(map)

        if self.sim_client is not None:
            self.sim_client.stop()
            self.sim_client = None

        if map is None:
            return

        print('start gibson sim client, scene id', self.map.scene_id)

        self.sim_client = GibsonSimClient()
        self.sim_client.start(
            assets_dir=self.map.assets_dir, scene_id=self.map.scene_id,
            h_fov=self.h_fov, v_fov=self.v_fov, gpu=self.gpu_idx, create_server=True)

    def get_camera_global_pos(self):
        return self.local_to_global(self.camera_pos)

    def _convert_img(self, img):
        '''
        Convert the doom simulator screenbuffer (H x W x 3, uint8) to network input (3 x H x W, float32)
        '''
        img = cv2.resize(img, (224, 224), cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return img

    def _measure(self):
        sim_proc = self.sim_client
        sim_proc.SetZ(self.camera_z)

        cam_pos = self.get_camera_global_pos()
        img = sim_proc.RenderAndGetScreenBuffer(
            cam_pos[0], cam_pos[1], self.heading, None)

        self.img = self._convert_img(img)


class AgentLocalVisualGibson(AgentLocal):
    def __init__(self, gpu_idx=0, h_fov=None, v_fov=None,
                 render_resolution=256,
                 output_resolution=224,
                 camera_pos=(0.0, 0.0), camera_z=1.0,
                 persistent_servers=None,
                 **kwargs):
        """
        :param fov, h_fov, v_fov: If h_fov or v_fov is None, then assume both h_fov and v_fov are
               fov. Otherwise fov is not used.
        """
        super(AgentLocalVisualGibson, self).__init__(**kwargs)

        self.gibson_camera = sensors.GibsonCamera(
            gpu_idx, render_resolution, output_resolution, h_fov, v_fov, persistent_servers)
        self.sensors.append(self.gibson_camera)

        self.img = None
        self.camera_pos = np.array(camera_pos, np.float32)
        self.camera_z = camera_z

    def save_state(self):
        state1 = super(AgentLocalVisualGibson, self).save_state()
        state2 = deepcopy({
            'img': self.img,
        })
        state1.update(state2)
        return state1

    def restore_state(self, state):
        self.img = deepcopy(state['img'])
        super(AgentLocalVisualGibson, self).restore_state(state)

    def reset(self):
        super(AgentLocalVisualGibson, self).reset()
        self.img = None

    def get_camera_global_pos(self):
        return self.local_to_global(self.camera_pos)

    def _measure(self):
        super(AgentLocalVisualGibson, self)._measure()
        cam_pos = self.get_camera_global_pos()
        self.img = self.gibson_camera.measure((cam_pos[0], cam_pos[1], self.camera_z), self.heading)

    def step(self, step_size, **kwargs):
        img = kwargs.pop('img', None)
        if img is not None:
            self.img = img
            kwargs['measure'] = False
        super(AgentLocalVisualGibson, self).step(step_size, **kwargs)


class AgentLocalVisualGibsonWaypoint(AgentLocalVisualGibson):
    """
    This agent uses visual images to predict a waypoint for destination-conditioned navigation.
    """
    def __init__(self, waypoint_solver, **kwargs):
        self.waypoint_solver = waypoint_solver
        self.destination = None
        super(AgentLocalVisualGibsonWaypoint, self).__init__(**kwargs)

    def set_destination(self, dest):
        self.destination = dest

    def reached_goal(self):
        dest = tuple(self.map.get_destination_from_map_coord(self.pos[0], self.pos[1]))
        return tuple(dest) == tuple(self.destination)

    def _compute_goals(self):
        self.neural_waypoint = self.waypoint_solver.compute_waypoint(self)
        self.goals_local = [self.neural_waypoint]
        self.goals_global = [self.local_to_global(self.neural_waypoint)]


class AgentLocalVisualGibsonWaypointRef(AgentLocalVisualGibson):
    """
    The behavior of the agent is identical to AgentLocalVisualGibson.
    It stores a neural waypoint using waypoint solver for comparison.
    For debugging only.
    """
    def __init__(self, waypoint_solver, **kwargs):
        self.destination = None
        self.neural_waypoint = None
        self.waypoint_solver = waypoint_solver
        super(AgentLocalVisualGibsonWaypointRef, self).__init__(**kwargs)

    def set_destination(self, dest):
        self.destination = dest

    def _compute_goals(self):
        super(AgentLocalVisualGibsonWaypointRef, self)._compute_goals()

        # # Re-compute waypoints
        # self.waypoints = self.map.find_path_destination(self.pos, self.destination)
        # self.wp_idx = 0
        #
        # self.goals_local = []
        # if self.waypoints is not None and len(self.waypoints) > 0:
        #     next_wp = self.next_visible_waypoint(
        #         self.waypoints, self.wp_idx,
        #         0.06,
        #         max_ahead=100)
        #     goal_local = self.global_to_local(self.waypoints[next_wp])
        #     if self.goal_clip_fov is not None:
        #         goal_local = agent_utils.clip_within_fov(goal_local, self.goal_clip_fov)
        #     self.goals_local.append(goal_local)
        #     self.wp_idx = next_wp

        self.neural_waypoint = self.waypoint_solver.compute_waypoint(self)


class AgentLocalVisualGibsonWaypointDagger(AgentLocalVisualGibsonWaypoint):
    """
    This agent uses the neural waypoint for navigation, but it also computes the ground truth waypoint
    for dagger training.
    """
    def __init__(self, **kwargs):
        super(AgentLocalVisualGibsonWaypointDagger, self).__init__(**kwargs)
        self.gt_global_waypoint = None

    def _compute_goals(self):
        super(AgentLocalVisualGibsonWaypointDagger, self)._compute_goals()

        # Compute the ground truth waypoint
        # set self.gt_global_waypoint to None if cannot compute the ground truth waypoint

        # Replan a path from current location
        self.waypoints = self.map.find_path_destination(self.pos, self.destination)
        if self.waypoints is None:
            # This could happen if the agent runs into invalid regions.
            self.gt_global_waypoint = None
            return

        self.wp_idx = 0

        # Recompute waypoint
        next_wp = self._compute_next_waypoint()
        if next_wp is None:
            self.gt_global_waypoint = None
            return

        goal_local = self._postprocess_waypoint(next_wp)
        self.gt_global_waypoint = self.local_to_global(goal_local)


class RCCarAgentLocal(AgentLocal):
    def __init__(self, params, noisy_actuation=False, **kwargs):
        self.noisy_actuation = noisy_actuation

        self.steer_range = params.get('steer_range', required=True)
        self.steer_speed_limit = params.get('steer_speed_limit', required=True)

        self.steering = 0.0
        self.dsteering = 0.0  # delta steering. Used for visualization.
        self.angular_accel = 0.0

        self.L = params.get('length', required=True)
        self.W = params.get('width', required=True)

        control_points_table = params.get('control_points', required=True)
        key = list(control_points_table.keys())[0]
        props = list(control_points_table.values())[0]
        prop_keys = [_.strip() for _ in key.split(',')]
        control_point_properties = []
        for prop in props:
            control_point_properties.append(dict(zip(prop_keys, prop)))
        self.control_point_properties = control_point_properties

        control_points = np.array([(prop['x'], prop['y'])
                                   for prop in control_point_properties],
                                  np.float32)

        super(RCCarAgentLocal, self).__init__(control_points=control_points, **kwargs)

    def save_state(self):
        state1 = super(RCCarAgentLocal, self).save_state()
        state2 = deepcopy({
            'steering': self.steering,
            'dsteering': self.dsteering,
            'angular_accel': self.angular_accel
        })
        state1.update(state2)
        return state1

    def restore_state(self, state):
        self.steering = deepcopy(state['steering'])
        self.dsteering = deepcopy(state['dsteering'])
        self.angular_accel = deepcopy(state['angular_accel'])
        super(RCCarAgentLocal, self).restore_state(state)

    def accel_size(self):
        # forward acceleration and steering velocity.
        return 2

    def get_boundary_control_points(self):
        return [self.control_points[i] for i in xrange(len(self.control_points))
                if self.control_point_properties[i]['is_boundary']]

    def get_local_jacobian(self, p):
        jacobian = np.array([[1, 0, -p[1]],
                             [0, 1, p[0]]], np.float32)

        beta = self.get_beta()
        v = self.get_signed_velocity_norm()

        # transform [ddx, ddy, ddtheta] into [dv, dsteering]
        # FIXME: this is not rigorous.
        transform = np.array([
            [1, 0],
            [0, 0],
            [np.sin(beta * 2.0) / self.L,
             4 * v * np.cos(2.0 * beta) / (3 * np.cos(self.steering)**2 + 1)]
        ], np.float32)

        return np.dot(jacobian, transform)

    def get_beta(self):
        return np.arctan(np.tan(self.steering) / 2.0)

    def get_signed_velocity_norm(self):
        '''
        :return: the signed velocity along the heading direction.
        '''
        heading_dir = np.array([np.cos(self.heading), np.sin(self.heading)], np.float32)
        return np.dot(self.velocity, heading_dir)

    def _apply_accel(self, accel_local, step_size, max_vel=None):
        self.pos += self.velocity * step_size
        self.heading += self.angular_velocity * step_size

        dv, dsteering = accel_local
        if self.noisy_actuation:
            dv = dv * (np.random.randn() / 3 + 1.0)
            dsteering = dsteering * (np.random.randn() / 3 + 1.0)

        self.dsteering = dsteering

        heading_dir = np.array([np.cos(self.heading), np.sin(self.heading)], np.float32)

        vel_norm = self.get_signed_velocity_norm()

        self.accel = heading_dir * dv

        self.velocity = heading_dir * vel_norm + self.accel * step_size
        if max_vel is not None:
            vnorm = np.linalg.norm(self.velocity)
            if vnorm > max_vel:
                self.velocity = self.velocity / vnorm * max_vel

        prev_angular_vel = self.angular_velocity

        self.angular_velocity = vel_norm / self.L * np.sin(self.get_beta() * 2.0)

        self.angular_accel = self.angular_velocity - prev_angular_vel

        self.steering += dsteering * step_size

        self.steering = np.clip(self.steering, *self.steer_range)

        # print 'dv %f dsteering %f speed %f steer %f' % (
        #     dv, dsteering, self.get_signed_velocity_norm(), self.steering)

    def collide(self, tolerance=0.075, inflate=1.0):
        """
        :param inflate: inflate robot size.
        :return:
        """
        global_pos = np.array(self.get_global_control_points_pos(), np.float32)
        x1, y1 = self.pos

        global_pos = (global_pos - self.pos) * inflate + self.pos

        lines = np.concatenate([np.array([[x1, y1]] * len(global_pos), np.float32),
                                global_pos], axis=1)
        return not all(self.map.no_touch_batch(lines, tolerance=tolerance))
        # Equivalent to
        # for i in xrange(len(global_pos)):
        #     x2, y2 = global_pos[i]
        #     if not self.map.touch(x1, y1, x2, y2, tolerance=tolerance):
        #         return True
        # return False

    def stopped(self):
        if len(self.state_history) < self.state_history.maxlen:
            return False

        for vel, accel in self.state_history:
            if not (norm(accel[:2]) < 1e-2 and norm(vel) < 1e-2):
                return False

        return True


class RCCarAgentLocalLIDAR(RCCarAgentLocal, AgentLocalLIDAR):
    pass


class RCCarAgentLocalVisualGibson(RCCarAgentLocal, AgentLocalVisualGibson):
    pass


class RCCarAgentLocalVisualGibsonWaypoint(RCCarAgentLocal, AgentLocalVisualGibsonWaypoint):
    pass


class RCCarAgentLocalLIDARVisualGibsonWaypoint(RCCarAgentLocalLIDAR, AgentLocalVisualGibsonWaypoint):
    pass


class RCCarAgentLocalLIDARVisualGibsonWaypointDagger(RCCarAgentLocalLIDAR, AgentLocalVisualGibsonWaypointDagger):
    pass


class RCCarAgentLocalVisualGibsonWaypointDagger(RCCarAgentLocal, AgentLocalVisualGibsonWaypointDagger):
    pass


class RCCarAgentLocalVisualGibsonWaypointRef(RCCarAgentLocal, AgentLocalVisualGibsonWaypointRef):
    pass


class RCCarAgentLocalVisual2LidarGibson(RCCarAgentLocal, AgentLocalVisual2LIDAR_Gibson):
    pass


class TurtleBot(AgentLocal):
    def __init__(self, params, noisy_actuation=False, **kwargs):
        self.noisy_actuation = noisy_actuation

        self.rot_vel_limit = params.get('rot_vel_limit', required=True)
        self.angular_accel = 0.0

        self.R = params.get('radius', required=True)

        control_points_table = params.get('control_points', required=True)
        key = list(control_points_table.keys())[0]
        props = list(control_points_table.values())[0]
        prop_keys = [_.strip() for _ in key.split(',')]
        control_point_properties = []
        for prop in props:
            control_point_properties.append(dict(zip(prop_keys, prop)))
        self.control_point_properties = control_point_properties

        control_points = np.array([(prop['x'], prop['y'])
                                   for prop in control_point_properties],
                                  np.float32)

        super(TurtleBot, self).__init__(control_points=control_points, **kwargs)

    def accel_size(self):
        # forward acceleration and angular acceleration.
        return 2

    def get_boundary_control_points(self):
        return [self.control_points[i] for i in xrange(len(self.control_points))
                if self.control_point_properties[i]['is_boundary']]

    def get_local_jacobian(self, p):
        v = self.get_signed_velocity_norm()
        transform = np.array([
            [1.0, 0.0],
            [0.0, max(v, 0.01)]
        ], np.float32)
        return transform

    def get_signed_velocity_norm(self):
        '''
        :return: the signed velocity along the heading direction.
        '''
        heading_dir = np.array([np.cos(self.heading), np.sin(self.heading)], np.float32)
        return np.dot(self.velocity, heading_dir)

    def _apply_accel(self, accel_local, step_size, max_vel=None):
        self.pos += self.velocity * step_size
        self.heading += self.angular_velocity * step_size

        dv, dtheta = accel_local
        dv = max(dv, 0.0)

        heading_dir = np.array([np.cos(self.heading), np.sin(self.heading)], np.float32)

        vel_norm = self.get_signed_velocity_norm()

        self.accel = heading_dir * dv

        self.velocity = heading_dir * vel_norm + self.accel * step_size
        if max_vel is not None:
            vnorm = np.linalg.norm(self.velocity)
            if vnorm > max_vel:
                self.velocity = self.velocity / vnorm * max_vel

        prev_angular_vel = float(self.angular_velocity)

        self.angular_velocity = dtheta
        self.angular_velocity = np.clip(self.angular_velocity, -self.rot_vel_limit, self.rot_vel_limit)
        self.angular_accel = self.angular_velocity - prev_angular_vel

    def collide(self, tolerance=0.075, inflate=1.0):
        """
        :param inflate: inflate robot size.
        :return:
        """
        global_pos = np.array(self.get_global_control_points_pos(), np.float32)
        x1, y1 = self.pos

        global_pos = (global_pos - self.pos) * inflate + self.pos

        lines = np.concatenate([np.array([[x1, y1]] * len(global_pos), np.float32),
                                np.array(global_pos, np.float32)], axis=1)
        return not all(self.map.no_touch_batch(lines, tolerance=tolerance))
        # Equivalent to
        # for i in xrange(len(global_pos)):
        #     x2, y2 = global_pos[i]
        #     if not self.map.touch(x1, y1, x2, y2, tolerance=tolerance):
        #         return True
        # return False

    def stopped(self):
        if len(self.state_history) < self.state_history.maxlen:
            return False

        for vel, accel in self.state_history:
            if not (norm(accel[:2]) < 1e-2 and norm(vel) < 1e-2):
                return False

        return True


class TurtleBotLocalLIDAR(TurtleBot, AgentLocalLIDAR):
    pass
