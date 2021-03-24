import math
import time
from typing import Tuple

import cv2
import numpy as np
import zmq

import msgpack
import msgpack_numpy
msgpack_numpy.patch()

from rmp_nav.common.math_utils import rotate_2d
from rmp_nav.simulation.map_utils import cum_path_length
from rmp_nav.simulation.gibson2_map import MakeGibson2Map
from rmp_nav.neural.common.dataset import DatasetVisual, DatasetVisualGibson2

from .inference import Client as TrackerClient, find_tracker


class DatasetBase(DatasetVisual):
    def __init__(self, *args,
                 n_frame_min=1, n_frame_max=1, frame_interval=3,
                 demo_camera_z=None,
                 rollout_camera_z=None,
                 demo_fov: Tuple[float, float],
                 rollout_fov: Tuple[float, float],
                 normalize_wp=False,
                 rand_frame_interval=False,
                 device='cuda',
                 debug=False, **kwargs):
        """
        :param args:
        :param traj_embedding_weights_file: Can be None. If None, no embedding will be computed.
        :param n_frame_min:
        :param n_frame_max:
        :param frame_interval:
        :param n_context_frame: number of contextual frames
        :param normalize_wp:
        :param noisy_odom:
        :param rand_frame_interval:
        :param device:
        :param debug:
        :param kwargs:
        """

        super(DatasetBase, self).__init__(*args, **kwargs)

        self.n_frame_min = n_frame_min
        self.n_frame_max = n_frame_max
        self.frame_interval = frame_interval
        self.rand_frame_interval = rand_frame_interval
        self.demo_camera_z = demo_camera_z
        self.rollout_camera_z = rollout_camera_z
        self.demo_fov = demo_fov
        self.rollout_fov = rollout_fov
        self.normalize_waypoint = normalize_wp
        self.g.update({
            'normalize_wp': normalize_wp,
        })
        self.device = device
        self.debug = debug

    def _normalize_waypoint(self, wp, distance):
        if not self.normalize_waypoint:
            return wp
        # Clip waypoint to be within distance away.
        wp_norm = np.linalg.norm(wp)
        if wp_norm > abs(distance):
            return wp / wp_norm * abs(distance)
        else:
            return wp

    def _find_good_start(self, traj):
        i = 0
        for i in range(1, len(traj)):
            dx, dy = traj[i]['pos'] - traj[i - 1]['pos']
            if math.sqrt(dx ** 2 + dy ** 2) < 0.01:
                continue
            heading = traj[i-1]['heading']
            dp = (math.cos(heading) * dx + math.sin(heading) * dy) / math.sqrt(dx ** 2 + dy ** 2)
            if dp > 0.01:
                break
        return i

    def _frame_span(self, n_frame):
        # Returns the total number of dense frames (including start and end frame)
        return n_frame + (self.frame_interval - 1) * (n_frame - 1)

    def _get_sample(self, idx):
        def next_idx():
            return self.rng.randint(len(self))

        while True:
            # First sample a trajectory
            dataset_idx, traj_id = self.traj_ids[idx]

            # Then sample a starting point
            # Find good starting point
            traj = self.fds[dataset_idx][traj_id]
            map_name = traj.attrs['map'].decode('ascii')

            traj = traj[()]  # This is important. Otherwise it's very slow.

            good_start_idx = self._find_good_start(traj)

            n_frame = self.rng.randint(self.n_frame_min, self.n_frame_max + 1)

            if self.rand_frame_interval:
                fi = self.rng.randint(1, self.frame_interval + 1, size=n_frame-1)
            else:
                fi = np.ones(n_frame - 1, np.int32) * self.frame_interval

            frame_span = n_frame + np.sum(fi) - (n_frame - 1)

            if len(traj) - good_start_idx < frame_span:
                # Trajectory too short
                idx = next_idx()
                continue

            start_idx = self.rng.randint(good_start_idx, len(traj) - frame_span + 1)
            end_idx = start_idx + frame_span - 1

            frame_idxs = np.cumsum(fi) + start_idx
            samples = [traj[start_idx]]
            for i in range(n_frame - 1):
                samples.append(traj[frame_idxs[i]])

            dense_samples = traj[start_idx: start_idx + frame_span]

            return dataset_idx, traj_id, traj, map_name, samples, dense_samples

    def __len__(self):
        return len(self.traj_ids)

    def _pad_to_length(self, a, l):
        """
        Pad array a of size N1 x N2 x... to l x N2 x...
        """
        n_pad = l - a.shape[0]
        return np.pad(np.array(a, np.float32), [(0, n_pad)] + [(0, 0)] * (a.ndim - 1), 'constant')

    def _render_samples(self, map_name, samples, **kwargs):
        raise NotImplemented

    def _render_demo_samples(self, map_name, samples):
        return self._render_samples(
            map_name, samples,
            camera_z=self.demo_camera_z, h_fov=self.demo_fov[0], v_fov=self.demo_fov[1])

    def _render_rollout_samples(self, map_name, samples):
        return self._render_samples(
            map_name, samples,
            camera_z=self.rollout_camera_z, h_fov=self.rollout_fov[0], v_fov=self.rollout_fov[1])

    def __getitem__(self, idx):
        self._init_once(idx)
        dataset_idx, traj_id, traj, map_name, samples, dense_samples = \
            self._get_sample(idx)

        demo_imgs, positions, headings, waypoints_global, _ = \
            self._render_demo_samples(map_name, samples)

        if self.demo_camera_z != self.rollout_camera_z or self.demo_fov != self.rollout_fov:
            rollout_imgs, _, _, _, _ = self._render_rollout_samples(map_name, samples)
        else:
            rollout_imgs = demo_imgs

        ref_pos = positions[0]
        ref_heading = headings[0]

        positions_local = [rotate_2d(pos - ref_pos, -ref_heading) for pos in positions]
        traj_len = len(positions_local)

        mask = np.ones(self.n_frame_max, np.float32)
        mask[len(positions_local):] = 0.0

        progress = np.zeros(self.n_frame_max, np.float32)
        path_accum_len = [0.0]
        for i in range(1, len(positions_local)):
            path_accum_len += np.linalg.norm(positions_local[i] - positions_local[i - 1], ord=2)
            progress[i] = path_accum_len
        progress /= path_accum_len
        assert abs(progress[len(positions_local) - 1] - 1.0) < 1e-5

        waypoints_local = np.zeros((self.n_frame_max, 2), np.float32)
        for i in range(len(positions_local)):
            waypoints_local[i] = rotate_2d(waypoints_global[i] - positions[i], -headings[i])
            if self.normalize_waypoint:
                dist_to_goal = np.linalg.norm(positions[i] - positions[-1])
                waypoints_local[i] = self._normalize_waypoint(waypoints_local[i], dist_to_goal)

        return {
            'map_name': map_name,
            'demo_traj_len': traj_len,
            'demo_obs': self._pad_to_length(np.array(demo_imgs), self.n_frame_max),
            'demo_start_ob': np.array([demo_imgs[0]]),
            'demo_goal_ob': np.array([demo_imgs[-1]]),

            'demo_mask': mask,
            'demo_progress': progress,
            'demo_waypoints': waypoints_local,

            'rollout_obs': self._pad_to_length(np.array(rollout_imgs), self.n_frame_max),

            # These are used for visualization, not for training.
            'pos': self._pad_to_length(np.array(positions, np.float32), self.n_frame_max),
            'headings': self._pad_to_length(np.array(headings, np.float32), self.n_frame_max)
        }


class DatasetGibson2Traj(DatasetBase, DatasetVisualGibson2):
    def _render_samples(self, map_name, samples, **kwargs):
        imgs = []

        positions = []
        headings = []
        waypoints = []
        velocities = []

        for sample in samples:
            self._set_agent_state(sample)
            imgs.append(self._render_agent_view(map_name, **kwargs))
            positions.append(sample['pos'])
            waypoints.append(sample['waypoint_global'])
            velocities.append(sample['velocity_global'])
            headings.append(sample['heading'])

        return imgs, positions, headings, waypoints, velocities


def make_maps_gibson2(assets_dir, map_names):
    def load_maps(datadir, map_names, **kwargs):
        maps = {}
        for name in map_names:
            maps[name] = MakeGibson2Map(datadir, name, **kwargs)
            print(maps[name])
        return maps
    map_names = [s.strip() for s in map_names]
    return load_maps(assets_dir, map_names)


class DatasetDagger(DatasetBase):
    def __init__(self,
                 tracker_weights_file=None,
                 tracker_inference_device='cuda',
                 jitter=False,
                 jitter_overlap_thres: Tuple[float, float] = (0.2, 0.2),
                 jitter_collision_inflation=1.0,
                 jitter_collision_tolerance=0.075,
                 jitter_velocity=False,
                 jitter_prob=1.0,
                 steps_margin=50,
                 reach_overlap_thres: Tuple[float, float] = (0.6, 0.6),
                 divergence_thres=1.0,
                 local_inference=False,
                 tracker_server_addr=None,
                 **kwargs):
        """
        :param tracker_weights_file:
        :param jitter:
        :param reach_overlap_thres:
        :param divergence_thres:
        :param local_inference: if True will create the inference models directly.
               Useful with dataset_server.
        :param tracker_server_addr: if specified then will connect to this tracker server.
        :param kwargs:
        """
        super(DatasetDagger, self).__init__(**kwargs)
        self.tracker_weights_file = tracker_weights_file
        self.tracker_inference_device = tracker_inference_device
        self.tracker = None

        self.reach_overlap_thres = reach_overlap_thres

        self.jitter = jitter
        self.jitter_prob = jitter_prob
        self.jitter_overlap_thres = jitter_overlap_thres
        self.jitter_collision_inflation = jitter_collision_inflation
        self.jitter_collision_tolerance = jitter_collision_tolerance
        self.jitter_velocity = jitter_velocity

        self.steps_margin = steps_margin

        self.divergence_thres = divergence_thres
        self.worker_id = 0  # Will be updated by dataloader.
        self.local_inference = local_inference

        self.tracker_server_proc, self.tracker_server_addr = None, None
        self.internal_tracker_server = True

        if tracker_server_addr is not None:
            self.tracker_server_addr = tracker_server_addr
            self.internal_tracker_server = False
        self.tracker_id = None

        self.progress_diff = []  # Debug
        self.screenshot_idx = 0  # Debug
        self.heap_trim_interval = 100.0
        self.last_heap_trim_time = time.time()

    def _launch_tracker_server(self, device):
        addr = 'ipc:///tmp/cbe_inference-frontend-%s' % str(time.time())
        proc = TrackerClient.launch_server(self.tracker_weights_file, addr, device)
        return proc, addr

    def _terminate_tracker_server(self):
        if not self.internal_tracker_server or self.tracker_server_proc is None:
            return
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect(self.tracker_server_addr)
        socket.send(msgpack.packb(['exit'], use_bin_type=True))
        socket.recv()
        self.tracker_server_proc.communicate()
        print('tracker server terminated')

    def reload_tracker(self, new_weights_file):
        self.tracker_weights_file = new_weights_file
        if self.local_inference:
            del self.tracker
            self._make_tracker()
        elif self.internal_tracker_server:
            self._terminate_tracker_server()
            self._make_tracker()
        else:
            # External tracker server
            self.tracker.reload(new_weights_file)

    def _make_tracker(self):
        if self.local_inference:
            self.tracker = find_tracker(self.tracker_weights_file)(
                self.tracker_weights_file, self.tracker_inference_device)
        elif self.internal_tracker_server:
            self.tracker_server_proc, self.tracker_server_addr = self._launch_tracker_server(
                self.tracker_inference_device)
            self.tracker = TrackerClient(self.tracker_weights_file, self.tracker_server_addr)
        else:
            # External tracker server
            print('connect to external tracker server', self.tracker_server_addr)
            self.tracker = TrackerClient(
                self.tracker_weights_file, self.tracker_server_addr, self.tracker_id)

    def _init_once(self, seed):
        if self.first:
            if self.tracker is None:
                self._make_tracker()
        super(DatasetDagger, self)._init_once(seed)

    def __del__(self):
        self._terminate_tracker_server()
        super(type(self), self).__del__()

    def _get_odom(self, prev_pos, prev_heading, cur_pos, cur_heading):
        return rotate_2d(cur_pos - prev_pos, -prev_heading), cur_heading - prev_heading

    def _compute_overlap(self, map, rollout_pos, rollout_heading, demo_pos, demo_heading):
        return map.view_overlap(rollout_pos, rollout_heading, self.rollout_fov[0],
                                demo_pos, demo_heading, self.demo_fov[0], mode='plane')

    def _find_closest_traj_sample_idx(self, pos, traj_points):
        return int(np.argmin(np.linalg.norm(pos - traj_points, 2, axis=1)))

    def _overlap_reach(self, pos, heading, goal_pos, goal_heading, map, thres):
        overlap_ratios = self._compute_overlap(map, pos, heading, goal_pos, goal_heading)
        return overlap_ratios[0] > thres[0] and overlap_ratios[1] > thres[1]

    def _diverge(self, pos, traj_points, thres=3.0):
        """
        Returns true if pos is too far away from any point in traj_points
        """
        idx = self._find_closest_traj_sample_idx(pos, traj_points)
        return np.linalg.norm(pos - traj_points[idx], ord=2) > thres

    def _jitter(self, agent):
        pos, heading = np.copy(agent.pos), float(agent.heading)
        for i in range(5):
            dpos = np.clip(self.rng.randn() * 0.3, -1.0, 1.0)  # 99% prob in (-1, 1)
            dheading = self.rng.randn() * 0.3 * np.deg2rad(45)  # 99% prob in +/- 45deg.
            agent.set_pos(dpos + pos)
            agent.set_heading(dheading + heading)
            if self.agent.collide(tolerance=self.jitter_collision_tolerance,
                                  inflate=self.jitter_collision_inflation):
                continue
            overlaps = self._compute_overlap(agent.map, agent.pos, agent.heading, pos, heading)
            if overlaps[0] < self.jitter_overlap_thres[0] or \
                    overlaps[1] < self.jitter_overlap_thres[1]:
                continue
            if self.debug:
                print('overlap after jittering:', overlaps)
            return
        agent.set_pos(pos)
        agent.set_heading(heading)

    def _gen_rollout(self, positions, headings, waypoints_global, start_ob, goal_ob,
                     embedding, map_name):
        self.tracker.reset2(self.worker_id)

        map = self.maps[map_name]

        positions = np.array(positions)
        headings = np.array(headings)

        agent = self.agent
        agent.reset()

        agent.set_map(map)
        agent.set_pos(positions[0])
        agent.set_heading(headings[0])

        def randomize_2nd_order_state():
            # FIXME: this is only applicable to a car-like vehicle
            # Keep the same direction.
            direction = np.array([math.cos(headings[0]), math.sin(headings[0])], np.float32)
            velocity = self.rng.uniform(0.0, 0.5) * direction
            agent.set_velocity(velocity)
            angular_vel = self.rng.uniform(0, math.pi)
            agent.set_angular_vel(angular_vel)
            # A small hack to set the right steering value.
            agent.steering = agent.infer_steering(agent.get_signed_velocity_norm(), angular_vel)

        if self.jitter:
            if self.rng.uniform() < self.jitter_prob:
                self._jitter(self.agent)

        if self.jitter_velocity:
            randomize_2nd_order_state()

        if self.debug:
            print('map_name:', map_name)
            print('agent_pos:', agent.pos)
            print('agent_heading:', agent.heading)
            print('collision:', agent.collide(self.jitter_collision_tolerance,
                                              self.jitter_collision_inflation))

        heading_diff = headings[0] - agent.heading

        goal_pos = positions[-1]
        goal_heading = headings[-1]
        agent.set_waypoints([goal_pos])  # Used for checking reach condition.

        ob = self.render(agent.pos, agent.heading, map_name,
                         camera_z=self.rollout_camera_z,
                         h_fov=self.rollout_fov[0],
                         v_fov=self.rollout_fov[1])

        cum_path_len = np.array([0.0] + cum_path_length(positions).tolist())
        path_len = cum_path_len[-1]

        obs = []
        pred_progresses = []
        gt_progresses = []
        gt_waypoints = []
        gt_reachable = 0.0

        def step_agent(wp):
            wp_global = agent.local_to_global(wp)
            for i in range(n_step):
                agent.step(0.1, waypoint=agent.global_to_local(wp_global))
                if agent.collide():
                    return False
            return True

        def check_status():
            # Check if reached using overlap estimation
            if self._overlap_reach(
                    agent.pos, agent.heading, goal_pos, goal_heading, map,
                    self.reach_overlap_thres):
                return 'reached'

            if agent.reached_goal(relax=1.0):
                return 'reached'

            # Check divergence
            if np.linalg.norm(agent.pos - positions[closest_idx], ord=2) > self.divergence_thres:
                return 'fail'

            # Check overlap between current observation and the closest trajectory ob
            # FIXME: this might not be a good thing if we train with dynamic obstacles.
            if not self._overlap_reach(
                    agent.pos, agent.heading, positions[closest_idx], headings[closest_idx],
                    map, (0.1, 0.1)):
                return 'fail'

            return 'following'

        max_steps = len(positions) * (1 + self.frame_interval) // 2
        max_steps = max(max_steps * 2, self.steps_margin)  # Give 100% more timesteps.

        step_idx = 0
        while step_idx < max_steps:
            n_step = self.rng.randint(1, self.frame_interval + 1)  # Randomize step size
            noise_scale = math.sqrt(n_step / float(self.frame_interval))

            obs.append(ob)
            closest_idx = self._find_closest_traj_sample_idx(agent.pos, positions)

            gt_progress = cum_path_len[closest_idx] / path_len
            gt_progresses.append(gt_progress)

            gt_wp = agent.global_to_local(waypoints_global[closest_idx])
            if self.normalize_waypoint:
                dist_to_goal = np.linalg.norm(agent.pos - positions[-1])
                gt_wp = self._normalize_waypoint(gt_wp, dist_to_goal)
            gt_waypoints.append(gt_wp)

            pred_progress, wp = self.tracker.step2(start_ob, goal_ob, embedding, ob, self.worker_id)
            pred_progresses.append(pred_progress)

            # prev_pos = np.array(agent.pos, copy=True)
            # prev_heading = float(agent.heading)

            if not step_agent(wp):
                break

            status = check_status()
            if status == 'reached':
                gt_reachable = 1.0
                break
            elif status == 'fail':
                gt_reachable = 0.0
                break

            ob = self.render(agent.pos, agent.heading, map_name,
                             camera_z=self.rollout_camera_z,
                             h_fov=self.rollout_fov[0],
                             v_fov=self.rollout_fov[1])
            step_idx += n_step

        assert len(obs) == len(gt_progresses)
        assert len(obs) == len(gt_waypoints)

        # The number of rollout samples can be larger than n_frame_max.
        # We draw n_frame_max samples here.
        idxs = sorted(self.rng.choice(
            len(obs), size=min(self.n_frame_max, len(obs)), replace=False))

        if self.debug:
            print('max_steps:', max_steps)
            print('idxs', idxs)

        obs_samples = np.array(obs, np.float32)[idxs]
        gt_progresses_samples = np.array(gt_progresses, np.float32)[idxs]
        gt_waypoints_samples = np.array(gt_waypoints, np.float32)[idxs]

        return (obs_samples,
                gt_progresses_samples, gt_waypoints_samples, gt_reachable, heading_diff)

    def __getitem2__(self, idx, extra):
        self.tracker_weights_file = extra['tracker_weights_file']
        self._init_once(idx)  # This will create the tracker.
        # Note that reload only has effect when tracker weights file has changed.
        self.tracker.reload(self.tracker_weights_file)
        return self.__getitem__(idx)

    def __getitem__(self, idx):
        self._init_once(idx)
        if time.time() - self.last_heap_trim_time > self.heap_trim_interval:
            self._malloc_trim()
            self.last_heap_trim_time = time.time()

        dataset_idx, traj_id, traj, map_name, samples, dense_samples = \
            self._get_sample(idx)

        imgs, positions, headings, waypoints_global, _ = \
            self._render_demo_samples(map_name, samples)

        assert len(imgs) <= self.n_frame_max

        embeddings = self.tracker.compute_traj_embedding(imgs)

        start_ob = imgs[0]
        goal_ob = imgs[-1]

        (rollout_obs,
         rollout_progress,
         rollout_waypoints,
         rollout_reachable,
         heading_diff) = self._gen_rollout(
            positions, headings, waypoints_global, start_ob, goal_ob, embeddings[-1], map_name)

        mask = np.ones(self.n_frame_max, np.float32)
        mask[len(rollout_obs):] = 0.0

        demo_mask = np.ones(self.n_frame_max, np.float32)
        demo_mask[len(imgs):] = 0.0

        # This is reference embedding used to condition the tracker.
        return {
            'map_name': map_name,
            'rollout_traj_len': len(rollout_obs),
            'rollout_obs': self._pad_to_length(np.array(rollout_obs), self.n_frame_max),
            'rollout_mask': mask,
            'rollout_progress': self._pad_to_length(np.array(rollout_progress, np.float32),
                                                    self.n_frame_max),
            'rollout_waypoints': self._pad_to_length(np.array(rollout_waypoints, np.float32),
                                                     self.n_frame_max),

            'demo_traj_len': len(imgs),
            'demo_obs': self._pad_to_length(np.array(imgs), self.n_frame_max),
            'demo_start_ob': start_ob,
            'demo_goal_ob': goal_ob,
            'demo_mask': demo_mask,
            'demo_embeddings': self._pad_to_length(embeddings, self.n_frame_max),

            # heading difference between start_ob and rollout_obs[0]
            'heading_diff': np.array([heading_diff], np.float32),
            'reachability': np.array([rollout_reachable], np.float32),
            'pos': self._pad_to_length(np.array(positions, np.float32), self.n_frame_max),
            'headings': self._pad_to_length(np.array(headings, np.float32), self.n_frame_max),
        }


class DatasetGibson2TrajsDagger(DatasetDagger, DatasetGibson2Traj):
    def __init__(self, *args, **kwargs):
        super(DatasetGibson2TrajsDagger, self).__init__(*args, **kwargs)
        self.maps = make_maps_gibson2(self.assets_dir, self.map_names)


class DatasetDaggerRPF(DatasetDagger):
    def __init__(self, n_rollout_frame=64, **kwargs):
        super(DatasetDaggerRPF, self).__init__(**kwargs)
        self.n_rollout_frame = n_rollout_frame

    def _gen_rollout(self, positions, headings, waypoints_global, start_ob, goal_ob,
                     embedding, map_name):
        self.tracker.reset2(self.worker_id)

        map = self.maps[map_name]

        positions = np.array(positions)
        headings = np.array(headings)

        agent = self.agent
        agent.reset()

        agent.set_map(map)
        agent.set_pos(positions[0])
        agent.set_heading(headings[0])

        def randomize_2nd_order_state():
            # FIXME: this is only applicable to a car-like vehicle
            velocity = self.rng.uniform(0.0, 0.5) * np.array([math.cos(headings[0]), math.sin(headings[0])], np.float32)
            agent.set_velocity(velocity)
            angular_vel = self.rng.uniform(0, math.pi)
            agent.set_angular_vel(angular_vel)
            # A small hack to set the right steering value.
            agent.steering = agent.infer_steering(agent.get_signed_velocity_norm(), angular_vel)

        if self.jitter:
            if self.rng.uniform() < self.jitter_prob:
                self._jitter(self.agent)

        if self.jitter_velocity:
            randomize_2nd_order_state()

        if self.debug:
            print('map_name:', map_name)
            print('agent_pos:', agent.pos)
            print('agent_heading:', agent.heading)
            print('collision:', agent.collide(self.jitter_collision_tolerance,
                                              self.jitter_collision_inflation))

        heading_diff = headings[0] - agent.heading

        goal_pos = positions[-1]
        goal_heading = headings[-1]
        agent.set_waypoints([goal_pos])  # Used for checking reach condition.

        ob = self.render(agent.pos, agent.heading, map_name,
                         camera_z=self.rollout_camera_z,
                         h_fov=self.rollout_fov[0],
                         v_fov=self.rollout_fov[1])

        cum_path_len = np.array([0.0] + cum_path_length(positions).tolist())
        path_len = cum_path_len[-1]

        obs = []
        pred_progresses = []
        gt_progresses = []
        gt_waypoints = []
        gt_reachable = 0.0

        def step_agent(wp):
            wp_global = agent.local_to_global(wp)
            for i in range(n_step):
                agent.step(0.1, waypoint=agent.global_to_local(wp_global))
                if agent.collide():
                    return False
            return True

        def check_status():
            # Check if reached using overlap estimation
            if self._overlap_reach(agent.pos, agent.heading, goal_pos, goal_heading,
                                   map, self.reach_overlap_thres):
                return 'reached'

            if agent.reached_goal(relax=1.0):
                return 'reached'

            # Check divergence
            if np.linalg.norm(agent.pos - positions[closest_idx], ord=2) > self.divergence_thres:
                return 'fail'

            # Check overlap between current observation and the closest trajectory ob
            # FIXME: this might not be a good thing if we train with dynamic obstacles.
            if not self._overlap_reach(
                    agent.pos, agent.heading, positions[closest_idx], headings[closest_idx],
                    map, (0.1, 0.1)):
                return 'fail'

            return 'following'

        max_steps = len(positions) * self.frame_interval
        max_steps = max(max_steps * 2, self.steps_margin)  # Give 100% more timesteps.

        step_idx = 0
        while step_idx < max_steps:
            n_step = self.frame_interval
            obs.append(ob)
            closest_idx = self._find_closest_traj_sample_idx(agent.pos, positions)

            gt_progress = cum_path_len[closest_idx] / path_len
            gt_progresses.append(gt_progress)

            gt_wp = agent.global_to_local(waypoints_global[closest_idx])
            if self.normalize_waypoint:
                dist_to_goal = np.linalg.norm(agent.pos - positions[-1])
                gt_wp = self._normalize_waypoint(gt_wp, dist_to_goal)
            gt_waypoints.append(gt_wp)

            pred_progress, wp = self.tracker.step2(None, None, embedding, ob, self.worker_id)
            pred_progresses.append(pred_progress)

            if not step_agent(wp):
                break

            status = check_status()
            if status == 'reached':
                gt_reachable = 1.0
                break
            elif status == 'fail':
                gt_reachable = 0.0
                break

            ob = self.render(agent.pos, agent.heading, map_name,
                             camera_z=self.rollout_camera_z,
                             h_fov=self.rollout_fov[0],
                             v_fov=self.rollout_fov[1])

            step_idx += n_step

        assert len(obs) == len(gt_progresses)
        assert len(obs) == len(gt_waypoints)

        idxs = list(range(0, min(self.n_rollout_frame, len(obs))))

        if self.debug:
            print('max_steps:', max_steps)
            print('idxs', idxs)

        obs_samples = np.array(obs, np.float32)[idxs]
        gt_progresses_samples = np.array(gt_progresses, np.float32)[idxs]
        gt_waypoints_samples = np.array(gt_waypoints, np.float32)[idxs]

        return (obs_samples,
                gt_progresses_samples, gt_waypoints_samples, gt_reachable, heading_diff)

    def __getitem__(self, idx):
        self._init_once(idx)
        if time.time() - self.last_heap_trim_time > self.heap_trim_interval:
            self._malloc_trim()
            self.last_heap_trim_time = time.time()

        dataset_idx, traj_id, traj, map_name, samples, dense_samples = \
            self._get_sample(idx)

        imgs, positions, headings, waypoints_global, _ = \
            self._render_demo_samples(map_name, samples)

        embedding = self.tracker.compute_traj_embedding(imgs)

        (rollout_obs,
         rollout_progress,
         rollout_waypoints,
         rollout_reachable,
         heading_diff) = self._gen_rollout(
            positions, headings, waypoints_global, None, None, embedding, map_name)

        mask = np.ones(self.n_rollout_frame, np.float32)
        mask[len(rollout_obs):] = 0.0

        demo_mask = np.ones(self.n_frame_max, np.float32)
        demo_mask[len(imgs):] = 0.0

        return {
            'map_name': map_name,
            'traj_len': len(rollout_obs),
            'rollout_obs': self._pad_to_length(np.array(rollout_obs), self.n_frame_max),
            'rollout_mask': mask,
            'rollout_waypoints': self._pad_to_length(np.array(rollout_waypoints, np.float32),
                                                     self.n_frame_max),

            'demo_traj_len': len(imgs),
            'demo_obs': self._pad_to_length(np.array(imgs), self.n_frame_max),
            'demo_mask': demo_mask,
            'reachability': np.array([rollout_reachable], np.float32),
        }


class DatasetGibson2TrajsDaggerRPF(DatasetDaggerRPF, DatasetGibson2Traj):
    def __init__(self, *args, **kwargs):
        super(DatasetGibson2TrajsDaggerRPF, self).__init__(*args, **kwargs)
        self.maps = make_maps_gibson2(self.assets_dir, self.map_names)


class DatasetClient(object):
    """
    This allows running the dataset servers on multiple machines to speed up data generation.
    """
    def __init__(self, server_addrs, dataset_len):
        """
        :param server_addr: launch a new server if None.
        """
        self.server_addrs = server_addrs
        self.dataset_len = dataset_len
        self.tracker_weights_file = None
        self.first = True

    def _init_once(self, seed):
        if not self.first:
            return

        self.first = False

        self.context = zmq.Context()
        self.sockets = []

        for addr in self.server_addrs:
            socket = self.context.socket(zmq.REQ)
            while True:
                try:
                    socket.connect(addr)
                    break
                except:
                    time.sleep(1.0)
            print('connected to %s' % addr)
            self.sockets.append(socket)

        self.rng = np.random.RandomState(seed)

    def _send(self, socket, obj, **kwargs):
        socket.send(msgpack.packb(obj, use_bin_type=True), **kwargs)

    def _recv(self, socket):
        return msgpack.unpackb(socket.recv(), raw=False)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        self._init_once(idx)
        # Randomly choose a server
        # TODO: load-balancing
        socket = self.sockets[self.rng.randint(len(self.sockets))]
        self._send(socket,
                   ['__getitem2__', idx, {'tracker_weights_file': self.tracker_weights_file}],
                   flags=zmq.NOBLOCK)
        return self._recv(socket)

    def set_tracker_weights_file(self, new_weights_file):
        self.tracker_weights_file = new_weights_file
