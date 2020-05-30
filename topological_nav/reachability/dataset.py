from torch.utils.data.dataset import Dataset
import glob
import os
import cv2
import bisect
import numpy as np
import time
from typing import Tuple
from copy import deepcopy
import math
import zmq
import msgpack
import msgpack_numpy
msgpack_numpy.patch()

from ....neural.common.dataloaders import DatasetVisualGibson
from ....utils import rotate_2d
from ....gibson_map import MakeGibsonMap
from ..imitation.inference import MotionPolicyClient


def make_maps(assets_dir, map_names):
    def load_maps(datadir, map_names, **kwargs):
        maps = {}
        for name in map_names:
            maps[name] = MakeGibsonMap(datadir, name, **kwargs)
            print(maps[name])
        return maps
    map_names = [s.strip() for s in map_names]
    return load_maps(assets_dir, map_names)


class DatasetSourceTargetPair(DatasetVisualGibson):
    def __init__(self,
                 fov: float,
                 distance_min: float,
                 distance_max: float,
                 sample_diff_traj_prob: float,
                 motion_policy_weights_file,
                 swap_src_tgt_prob: float=0.0,
                 reach_overlap_thres: Tuple[float, float]=(0.9, 0.9),
                 steps_margin: int=50,
                 wp_norm_min_clip: float=2.0,
                 **kwargs):
        super(DatasetSourceTargetPair, self).__init__(**kwargs)

        self.fov = fov
        self.distance_min = distance_min
        self.distance_max = distance_max
        self.sample_diff_traj_prob = sample_diff_traj_prob
        self.swap_src_tgt_prob = swap_src_tgt_prob
        self.motion_policy_weights_file = motion_policy_weights_file
        self.reach_overlap_thres = reach_overlap_thres
        self.steps_margin = steps_margin
        self.wp_norm_min_clip = wp_norm_min_clip

        self.cur_ob, self.cur_goal = None, None
        self.cur_goal_heading = None

        map_name_set = set(self.map_names)
        traj_ids_per_map = {_: [] for _ in self.map_names}
        for dset_idx, traj_id in self.traj_ids:
            map_name = self.traj_id_map[(dset_idx, traj_id)]
            if map_name in map_name_set:
                traj_ids_per_map[map_name].append((dset_idx, traj_id))
        # Map a map name to a list of trajectories
        self.traj_ids_per_map = traj_ids_per_map

        # Map a map name to cumulative sum of trajectory lengths.
        self.traj_len_cumsum_per_map = {
            # Note than when computing cumsum we must ensure the ordering. Hence we must
            # not use .values().
            map_name: np.cumsum([self.traj_len_dict[_] for _ in traj_ids])
            for map_name, traj_ids in traj_ids_per_map.items()}

        self.samples_per_map = {
            map_name: sum([self.traj_len_dict[_] for _ in traj_ids])
            for map_name, traj_ids in traj_ids_per_map.items()}

        # Make sure we count correctly
        assert sum(self.samples_per_map.values()) == len(self)

        self._malloc_trim()
        self.maps = make_maps(self.assets_dir, self.map_names)

        self.heap_trim_interval = 100.0
        self.last_heap_trim_time = time.time()

        self.mp_server_proc, self.mp_server_addr = self._launch_motion_policy_server()

        self.verbose = os.environ.get('MERLIN_VERBOSE', False) == '1'

        # FOR VISUALIZATION
        self.debug = False
        if self.debug:
            import matplotlib
            matplotlib.use('agg')
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
            from .... import sim_renderer
            from ....agent_visualizers import FindAgentVisualizer

            fig = plt.Figure(tight_layout=True)
            ax = fig.add_subplot(111)
            canvas = FigureCanvas(fig)
            canvas.draw()
            cm = matplotlib.cm.get_cmap('Dark2')

            # FIXME: hardcoded env
            vis = sim_renderer.SimRenderer(self.maps['house24'], ax, canvas)
            vis.set_agents({
                self.agent.name: {
                    'agent': self.agent,
                    'visualizer': FindAgentVisualizer(self.agent)(
                        ax,
                        draw_control_point=False,
                        traj_color=cm(0),
                        obstacle_color=cm(0),
                        heading_color=cm(0),
                        active_wp_color=cm(0),
                        label=self.agent.name
                    )}
            })
            vis.render(True)
            vis.render(True)
            self.vis = vis
            self.screenshot_idx = 0

    def _launch_motion_policy_server(self):
        addr = 'ipc:///tmp/motion_policy_inference-frontend-%s' % str(time.time())
        proc = MotionPolicyClient.launch_server(self.motion_policy_weights_file, addr)
        return proc, addr

    def _terminate_motion_policy_server(self):
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect(self.mp_server_addr)
        socket.send(msgpack.packb(['exit'], use_bin_type=True))
        socket.recv()
        self.mp_server_proc.communicate()
        print('motion policy server terminated')

    def _locate_sample_single_map(self, idx, map_name):
        """
        Similar to _locate_sample(), but only considers a single map.
        :param idx: sample index in the range of [0, total number of samples of this map - 1]
        """
        cumsum = self.traj_len_cumsum_per_map[map_name]
        assert 0 <= idx < cumsum[-1], 'Map index %d out of range [0, %d)' % (idx, cumsum[-1])

        trajs = self.traj_ids_per_map[map_name]

        traj_idx = bisect.bisect_right(cumsum, idx)
        dataset_idx, traj_id = trajs[traj_idx]

        if traj_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - cumsum[traj_idx - 1]

        return dataset_idx, traj_id, sample_idx

    def _draw_sample_same_traj(self, idx):
        distance = self.rng.uniform(self.distance_min, self.distance_max)

        while True:
            # Get a good pair of samples
            dataset_idx, traj_id, src_idx = self._locate_sample(idx)

            traj = self.fds[dataset_idx][traj_id]
            map_name = traj.attrs['map'].decode('ascii')

            src_sample = traj[src_idx]
            src_pos = src_sample['pos']

            dst_idx = src_idx + 1

            while dst_idx < len(traj):
                dst_sample = traj[dst_idx]
                dst_pos = dst_sample['pos']
                x, y = dst_pos - src_pos
                if x ** 2 + y ** 2 > distance ** 2:
                    break
                dst_idx += 1

            if dst_idx < len(traj):
                # Found valid dst_sample
                break

            # select another idx if this one doesn't work
            idx = (idx + 1000) % len(self)
        return map_name, src_sample, dst_sample

    def _draw_sample_diff_traj(self, idx):
        """
        These two trajectories are from the same map.
        """
        # Get a good pair of samples
        src_dataset_idx, src_traj_id, src_idx = self._locate_sample(idx)

        src_traj = self.fds[src_dataset_idx][src_traj_id]
        map_name = src_traj.attrs['map'].decode('ascii')

        idx2 = self.rng.randint(self.samples_per_map[map_name])
        dst_dataset_idx, dst_traj_id, dst_idx = self._locate_sample_single_map(idx2, map_name)
        dst_traj = self.fds[dst_dataset_idx][dst_traj_id]

        src_sample = src_traj[src_idx]
        dst_sample = dst_traj[dst_idx]

        return map_name, src_sample, dst_sample

    def _set_cur_ob(self, ob):
        self.cur_ob = ob

    def _get_ob(self, map_name):
        return self._render_agent_view(map_name)

    def _compute_wp(self):
        f = self.motion_policy.g.model_file
        if 'with_heading_diff' in f:
            return self.motion_policy.pred_motion(
                self.cur_ob, self.cur_goal, self.cur_goal_heading - self.agent.heading)
        else:
            return self.motion_policy.pred_motion(self.cur_ob, self.cur_goal)

    def _compute_overlap(self, map, pos1, heading1, pos2, heading2):
        return map.view_overlap(pos1, heading1, self.fov, pos2, heading2, self.fov, mode='plane')

    def reachable(self, map, src_sample, dst_sample):
        overlap_ratios = self._compute_overlap(map,
                                               src_sample['pos'], src_sample['heading'],
                                               dst_sample['pos'], dst_sample['heading'])
        if overlap_ratios[0] < 0.1 or overlap_ratios[1] < 0.1:
            # Shortcut if the two images have small overlap
            if self.debug:
                print('overlap too small')
            return 0.0, []
        elif overlap_ratios[0] > self.reach_overlap_thres[0] and \
                overlap_ratios[1] > self.reach_overlap_thres[1]:
            # Shortcut if the two images have large overlap
            return 1.0, []

        goal_pos = dst_sample['pos']
        goal_heading = dst_sample['heading']

        agent = self.agent
        agent.reset()
        agent.set_map(map)
        agent.set_waypoints([dst_sample['pos']])  # Last waypoint is the goal position
        agent.set_pos(src_sample['pos'])
        agent.set_heading(float(src_sample['heading']))

        if agent.reached_goal():  # Shortcut
            return 1.0, []

        # Estimate roughly the number of steps required to reach the goal
        # Assume a mean velocity of 0.3 m/s
        # In general this is difficult to precisely estimate it, because it heavily depends on the
        # environment geometry. Here we choose a conservative value.
        d = np.linalg.norm(dst_sample['pos'] - src_sample['pos'])
        mean_vel = 0.3
        expected_steps = (d / mean_vel) / 0.1
        step_limit = int(expected_steps + self.steps_margin)

        obs = []  # Rollout observations

        n_step = 2
        n = 0
        reached = 0.0
        stop = False

        while n < step_limit and not stop:
            ob = self._get_ob(map.name)
            obs.append(ob)

            self.cur_ob = ob
            wp = self._compute_wp()

            if np.linalg.norm(wp) < self.wp_norm_min_clip:
                wp = wp / np.linalg.norm(wp) * self.wp_norm_min_clip

            if self.debug:
                self.vis.render(False, blit=True)
                vis_img = cv2.cvtColor(self.vis.get_image(), cv2.COLOR_RGB2BGR)
                cv2.imshow('vis', vis_img)
                cv2.imwrite('/tmp/screenshots/%05d.tiff' % self.screenshot_idx, vis_img)
                self.screenshot_idx += 1
                cv2.waitKey(1)

            wp_global = agent.local_to_global(wp)

            for i in range(n_step):
                agent.step(0.1, waypoint=agent.global_to_local(wp_global))
                if agent.collide():
                    if self.debug:
                        print('collide')
                    # collision usually indicates bad samples (e.g., caused by bad jittering),
                    # so we discard it.
                    return None, None

                if agent.reached_goal():
                    reached = 1.0
                    stop = True
                    break

                # elif agent.collide():
                #     if self.debug:
                #         print('collide')
                #     stop = True
                #     break

            if not stop:
                # Check if reached using overlap estimation
                overlap_ratios = self._compute_overlap(map, agent.pos, agent.heading, goal_pos, goal_heading)

                if overlap_ratios[0] < 0.05 and overlap_ratios[1] < 0.05:
                    if self.debug:
                        print('overlap too small %.2f %.2f' %
                              (overlap_ratios[0], overlap_ratios[1]))
                    reached = 0.0
                    stop = True
                elif overlap_ratios[0] > self.reach_overlap_thres[0] \
                        and overlap_ratios[1] > self.reach_overlap_thres[1]:
                    reached = 1.0
                    stop = True

            n += n_step

        if self.debug:
            # text_location = (5, 64)
            h, w = vis_img.shape[:2]
            box_size = (800, 300)

            if reached > 0.5:
                color = (0, 200, 0)
            else:
                color = (0, 0, 255)

            vis_img[
                h // 2 - box_size[1] // 2: h // 2 + box_size[1] // 2,
                w // 2 - box_size[0] // 2: w // 2 + box_size[0] // 2] = (255, 255, 255)

            cv2.rectangle(vis_img,
                          (w // 2 - box_size[0] // 2, h // 2 - box_size[1] // 2),
                          (w // 2 + box_size[0] // 2, h // 2 + box_size[1] // 2),
                          color=color,
                          thickness=10)

            if reached > 0.5:
                cv2.putText(vis_img, 'reached', (700, h // 2 - 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 3.0, color, 5)
            else:
                cv2.putText(vis_img, 'failed', (750, h // 2 - 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 3.0, color, 5)

            cv2.imshow('vis', vis_img)

            for i in range(15):  # Repeat the last frame to create a pause effect
                cv2.imwrite('/tmp/screenshots/%05d.tiff' % self.screenshot_idx, vis_img)
                self.screenshot_idx += 1
                cv2.waitKey(1)

            if n >= step_limit:
                print('timeout (max steps %d)' % step_limit)

        return reached, obs

    def _init_once(self, seed):
        if self.first:
            self.motion_policy = MotionPolicyClient(self.motion_policy_weights_file,
                                                    self.mp_server_addr)
        super(DatasetSourceTargetPair, self)._init_once(seed)

    def __getitem__(self, idx):
        self._init_once(idx)

        if time.time() - self.last_heap_trim_time > self.heap_trim_interval:
            self._malloc_trim()
            self.last_heap_trim_time = time.time()

        p = self.rng.uniform(0.0, 1.0)
        if p < self.sample_diff_traj_prob:
            map_name, src_sample, dst_sample = self._draw_sample_diff_traj(idx)
        else:
            map_name, src_sample, dst_sample = self._draw_sample_same_traj(idx)

        map = self.maps[map_name]
        src_img, dst_img = self._make(map_name, [src_sample, dst_sample])

        self.cur_goal = dst_img
        self.cur_goal_heading = dst_sample['heading']

        # start_time = time.time()
        r, obs = self.reachable(map, src_sample, dst_sample)
        # print('reachable() time:', time.time() - start_time,
        #       'r:', r,
        #       'src pos: %.2f %.2f' % (src_sample['pos'][0], src_sample['pos'][1]),
        #       'dst pos: %.2f %.2f' % (dst_sample['pos'][0], dst_sample['pos'][1]),
        #       'dist:', np.linalg.norm(dst_sample['pos'] - src_sample['pos']))

        try:
            if self.rng.uniform(0.0, 1.0) < self.swap_src_tgt_prob:
                return dst_img, src_img, np.array([r], np.float32)
            else:
                return src_img, dst_img, np.array([r], np.float32)
        except:
            print('Error during making samples.')
            raise

    def _make(self, map_name, samples, **kwargs):
        imgs = []
        for sample in samples:
            self._set_agent_state(sample)
            imgs.append(self._render_agent_view(map_name))
        return imgs

    def __del__(self):
        self._terminate_motion_policy_server()
        super(DatasetSourceTargetPair, self).__del__()


class DatasetSourceTargetPairV2(DatasetSourceTargetPair):
    """
    Also return images generated during rollouts
    Allow distance_min to be negative.
    Ignore swap_src_tgt_prob
    """
    def __init__(self, n_frame, frame_interval, **kwargs):
        """
        :param n_frame: max number of images to be returned from rollout
        :param frame_interval: frame gap between adjacent images
        """
        super(DatasetSourceTargetPairV2, self).__init__(**kwargs)
        self.n_frame = n_frame
        self.frame_interval = frame_interval

    def _draw_sample_same_traj(self, idx):
        distance = self.rng.uniform(self.distance_min, self.distance_max)

        if distance < 0:
            inc = -1
        else:
            inc = 1

        while True:
            # Get a good pair of samples
            dataset_idx, traj_id, src_idx = self._locate_sample(idx)

            traj = self.fds[dataset_idx][traj_id]
            map_name = traj.attrs['map'].decode('ascii')

            src_sample = traj[src_idx]
            src_pos = src_sample['pos']

            dst_idx = src_idx + inc

            while 0 <= dst_idx < len(traj):
                dst_sample = traj[dst_idx]
                dst_pos = dst_sample['pos']
                x, y = dst_pos - src_pos
                if x ** 2 + y ** 2 > distance ** 2:
                    break
                dst_idx += inc

            if 0 <= dst_idx < len(traj):
                # Found valid dst_sample
                break

            # select another idx if this one doesn't work
            idx = (idx + 1000) % len(self)
        return map_name, src_sample, dst_sample

    def __getitem__(self, idx):
        self._init_once(idx)

        if time.time() - self.last_heap_trim_time > self.heap_trim_interval:
            self._malloc_trim()
            self.last_heap_trim_time = time.time()

        p = self.rng.uniform(0.0, 1.0)
        if p < self.sample_diff_traj_prob:
            map_name, src_sample, dst_sample = self._draw_sample_diff_traj(idx)
        else:
            map_name, src_sample, dst_sample = self._draw_sample_same_traj(idx)

        map = self.maps[map_name]

        r, obs = self.reachable(map, src_sample, dst_sample)

        imgs = self._make(map_name, [src_sample, dst_sample])
        img_dim = imgs[0].shape

        # Skip first frame and take every frame_interval(th) frame
        obs = obs[1::self.frame_interval]

        if len(obs) >= self.n_frame:
            obs = np.array(obs[:self.n_frame])
            mask = [1.0] * self.n_frame
        elif len(obs) > 0:
            zero_imgs = np.zeros((self.n_frame - len(obs),) + img_dim, np.float32)
            mask = [1.0] * len(obs) + [0.0] * zero_imgs.shape[0]
            obs = np.concatenate([np.array(obs), zero_imgs], axis=0)
        else:
            assert len(obs) == 0
            # No image is available due to shortcutting.
            obs = np.zeros((self.n_frame,) + img_dim, np.float32)
            mask = [0.0] * self.n_frame

        assert obs.shape[0] == self.n_frame
        assert len(mask) == self.n_frame

        try:
            return imgs[0], imgs[1], np.array([r], np.float32), obs, np.array(mask, np.float32)
        except:
            print('Error during making samples.')
            raise


class DatasetSourceTargetPairV3(DatasetSourceTargetPair):
    """
    Support negative distance
    Jitter positions
    """
    def __init__(self, **kwargs):
        super(DatasetSourceTargetPairV3, self).__init__(**kwargs)

    def _jitter(self, sample, map):
        self.agent.set_map(map)
        pos, heading = sample['pos'], sample['heading']
        heading_perp = rotate_2d(np.array([math.cos(heading), math.sin(heading)], np.float32),
                                 math.pi / 2)
        new_sample = deepcopy(sample)

        max_try = 5
        for i in range(max_try):
            # Only jitter position sideways
            new_sample['pos'] = pos + heading_perp * np.clip(self.rng.randn() * 0.3, -1.0, 1.0)
            new_sample['heading'] = heading + self.rng.randn() * 0.3 * np.deg2rad(30)
            self._set_agent_state(new_sample)
            if not self.agent.collide(tolerance=-0.3):
                return new_sample
        return sample

    def _draw_sample_same_traj(self, idx):
        distance = self.rng.uniform(self.distance_min, self.distance_max)
        if distance < 0:
            inc = -1
        else:
            inc = 1

        while True:
            # Get a good pair of samples
            dataset_idx, traj_id, src_idx = self._locate_sample(idx)

            traj = self.fds[dataset_idx][traj_id]
            map_name = traj.attrs['map'].decode('ascii')

            src_sample = self._jitter(traj[src_idx], self.maps[map_name])
            src_pos = src_sample['pos']

            dst_idx = src_idx + inc

            while 0 <= dst_idx < len(traj):
                dst_sample = traj[dst_idx]
                dst_pos = dst_sample['pos']
                x, y = dst_pos - src_pos
                if x ** 2 + y ** 2 > distance ** 2:
                    break
                dst_idx += inc

            if 0 <= dst_idx < len(traj):
                # Found valid dst_sample
                break

            # select another idx if this one doesn't work
            idx = (idx + 1000) % len(self)
        return map_name, src_sample, dst_sample

    def _draw_sample_diff_traj(self, idx):
        """
        These two trajectories are from the same map.
        """
        # Get a good pair of samples
        src_dataset_idx, src_traj_id, src_idx = self._locate_sample(idx)

        src_traj = self.fds[src_dataset_idx][src_traj_id]
        map_name = src_traj.attrs['map'].decode('ascii')

        idx2 = self.rng.randint(self.samples_per_map[map_name])
        dst_dataset_idx, dst_traj_id, dst_idx = self._locate_sample_single_map(idx2, map_name)
        dst_traj = self.fds[dst_dataset_idx][dst_traj_id]

        src_sample = self._jitter(src_traj[src_idx], self.maps[map_name])
        dst_sample = dst_traj[dst_idx]
        return map_name, src_sample, dst_sample

    def __getitem__(self, idx):
        self._init_once(idx)

        if time.time() - self.last_heap_trim_time > self.heap_trim_interval:
            self._malloc_trim()
            self.last_heap_trim_time = time.time()

        while True:
            p = self.rng.uniform(0.0, 1.0)
            if p < self.sample_diff_traj_prob:
                map_name, src_sample, dst_sample = self._draw_sample_diff_traj(idx)
            else:
                map_name, src_sample, dst_sample = self._draw_sample_same_traj(idx)

            try:
                src_img, dst_img = self._make(map_name, [src_sample, dst_sample])
                self.cur_goal = dst_img
                self.cur_goal_heading = dst_sample['heading']
                r, obs = self.reachable(self.maps[map_name], src_sample, dst_sample)
                if r is None:
                    continue
                return src_img, dst_img, np.array([r], np.float32)
            except:
                print('Error during making samples.')
                raise


class DatasetSourceTargetPairMultiframeDst(DatasetSourceTargetPairV3):
    """
    Support negative distance
    Jitter positions
    Note that we assume inputs for the reachability estimator and the motion policy are the same.
    """
    def __init__(self, n_frame, frame_interval, future=False, **kwargs):
        super(DatasetSourceTargetPairMultiframeDst, self).__init__(**kwargs)
        self.n_frame = n_frame
        self.frame_interval = frame_interval
        self.future = future

    def _make_sample_seq(self, traj, idx):
        samples = []
        for i in range(self.n_frame):
            samples.append(traj[max(idx - i * self.frame_interval, 0)])
        samples = samples[::-1]
        assert self.motion_policy.g.get('future', False) == self.future
        if self.future:
            for i in range(self.n_frame - 1):
                samples.append(traj[min(idx + (i + 1) * self.frame_interval, len(traj) - 1)])
        return samples

    def _draw_sample_same_traj(self, idx):
        distance = self.rng.uniform(self.distance_min, self.distance_max)
        if distance < 0:
            inc = -1
        else:
            inc = 1

        while True:
            # Get a good pair of samples
            dataset_idx, traj_id, src_idx = self._locate_sample(idx)

            traj = self.fds[dataset_idx][traj_id]
            map_name = traj.attrs['map'].decode('ascii')

            src_sample = self._jitter(traj[src_idx], self.maps[map_name])
            src_pos = src_sample['pos']

            dst_idx = src_idx + inc

            while 0 <= dst_idx < len(traj):
                dst_sample = traj[dst_idx]
                dst_pos = dst_sample['pos']
                x, y = dst_pos - src_pos
                if x ** 2 + y ** 2 > distance ** 2:
                    break
                dst_idx += inc

            if 0 <= dst_idx < len(traj):
                # Found valid dst_sample
                break

            # select another idx if this one doesn't work
            idx = (idx + 1000) % len(self)

        dst_samples = self._make_sample_seq(traj, dst_idx)

        return map_name, src_sample, dst_samples

    def _draw_sample_diff_traj(self, idx):
        """
        These two trajectories are from the same map.
        """
        # Get a good pair of samples
        src_dataset_idx, src_traj_id, src_idx = self._locate_sample(idx)

        src_traj = self.fds[src_dataset_idx][src_traj_id]
        map_name = src_traj.attrs['map'].decode('ascii')

        idx2 = self.rng.randint(self.samples_per_map[map_name])
        dst_dataset_idx, dst_traj_id, dst_idx = self._locate_sample_single_map(idx2, map_name)
        dst_traj = self.fds[dst_dataset_idx][dst_traj_id]

        src_sample = self._jitter(src_traj[src_idx], self.maps[map_name])
        dst_samples = self._make_sample_seq(dst_traj, dst_idx)
        return map_name, src_sample, dst_samples

    def __getitem__(self, idx):
        self._init_once(idx)

        if time.time() - self.last_heap_trim_time > self.heap_trim_interval:
            self._malloc_trim()
            self.last_heap_trim_time = time.time()

        while True:
            p = self.rng.uniform(0.0, 1.0)
            if p < self.sample_diff_traj_prob:
                map_name, src_sample, dst_samples = self._draw_sample_diff_traj(idx)
            else:
                map_name, src_sample, dst_samples = self._draw_sample_same_traj(idx)

            # Destination's position depends on whether there are future frames or not.
            dst_sample = dst_samples[self.n_frame - 1]

            src_img = self._make(map_name, [src_sample])[0]
            dst_imgs = np.array(self._make(map_name, dst_samples))

            if 'n_frame' in self.motion_policy.g:
                self.cur_goal = dst_imgs
            else:
                self.cur_goal = dst_imgs[-1]
            self.cur_goal_heading = dst_sample['heading']
            r, _ = self.reachable(self.maps[map_name], src_sample, dst_sample)
            if r is None:
                continue

            return src_img, dst_imgs, np.array([r], np.float32)


class DatasetSourceTargetPairMultiframeDstRecording(Dataset):
    def __init__(self, recording_dir, resolution, positive=True, negative=True, **kwargs):
        super(DatasetSourceTargetPairMultiframeDstRecording, self).__init__(**kwargs)
        self.recording_dir = recording_dir
        self.resolution = resolution

        self.positive_samples = self._read_images(os.path.join(self.recording_dir, 'positive'))
        self.negative_samples = self._read_images(os.path.join(self.recording_dir, 'negative'))

        if not positive:
            self.positive_samples = []

        if not negative:
            self.negative_samples = []

        self.sample_weights = [1.0 / len(self.positive_samples)] * len(self.positive_samples) + \
                              [1.0 / len(self.negative_samples)] * len(self.negative_samples)

        print('%d positive samples %d negative samples' %
              (len(self.positive_samples), len(self.negative_samples)))

    def _maybe_resize(self, img):
        if img.shape != (self.resolution, self.resolution):
            return cv2.resize(img, dsize=(self.resolution, self.resolution))
        return img

    def _normalize(self, img):
        return (img / 255.0).astype(np.float32).transpose((2, 0, 1))

    def _read_images(self, dir):
        img_files = sorted(glob.glob(os.path.join(dir, '*.png')))
        assert len(img_files) % 2 == 0

        entries = [[None, None] for _ in range(len(img_files) // 2)]

        for fn in img_files:
            img = cv2.cvtColor(cv2.imread(fn), cv2.COLOR_BGR2RGB)

            _1, _2 = os.path.splitext(os.path.basename(fn))[0].split('-')
            idx = int(_2)
            if _1 == 'ob':
                entries[idx][0] = self._normalize(self._maybe_resize(img))
            elif _1 == 'dst':
                # Assume dst images are horizontally arranged.
                assert img.shape[1] % img.shape[0] == 0
                n_frame = img.shape[1] // img.shape[0]
                imgs = [self._normalize(self._maybe_resize(_))
                        for _ in np.split(img, n_frame, axis=1)]
                entries[idx][1] = np.array(imgs)
            else:
                raise RuntimeError('Unknown type')

        # Make sure no unfilled entries
        for _1, _2 in entries:
            assert _1 is not None and _2 is not None

        return entries

    def __getitem__(self, idx):
        if idx < len(self.positive_samples):
            return self.positive_samples[idx][0], self.positive_samples[idx][1], np.array([1.0], np.float32)
        else:
            i = idx - len(self.positive_samples)
            return self.negative_samples[i][0], self.negative_samples[i][1], np.array([0.0], np.float32)

    def __len__(self):
        return len(self.positive_samples) + len(self.negative_samples)


class DatasetSourceTargetPairMultiframeDstNaive(DatasetSourceTargetPairMultiframeDst):
    """
    Implements the "retrieval network" in semi-parametric topological memory
    """

    def __init__(self, positive_interval, negative_interval, randomize_interval=False, **kwargs):
        """
        Note that we abuse the parameter "sample_diff_traj_prob" here. It now refers to the
        probability of sampling positive/negative examples on the SAME trajectory.
        """
        super(DatasetSourceTargetPairMultiframeDstNaive, self).__init__(**kwargs)
        self.positive_interval = positive_interval
        self.negative_interval = negative_interval
        self.randomize = randomize_interval

    def _draw_sample_positive(self, idx):
        while True:
            dataset_idx, traj_id, src_idx = self._locate_sample(idx)
            traj = self.fds[dataset_idx][traj_id]
            map_name = traj.attrs['map'].decode('ascii')

            if self.randomize:
                interval = self.rng.randint(1, self.positive_interval)
            else:
                interval = self.positive_interval

            if self.verbose:
                print('positive interval: %d' % interval)

            dst_idx = src_idx + interval
            if dst_idx >= len(traj):
                idx = (idx + 1000) % len(self)
                continue
            src_sample = traj[src_idx]
            dst_samples = self._make_sample_seq(traj, dst_idx)
            return map_name, src_sample, dst_samples

    def _draw_sample_negative(self, idx):
        while True:
            dataset_idx, traj_id, src_idx = self._locate_sample(idx)
            traj = self.fds[dataset_idx][traj_id]
            map_name = traj.attrs['map'].decode('ascii')

            if self.randomize:
                interval = self.rng.randint(self.negative_interval,
                                            max(len(traj), self.negative_interval + 1))
            else:
                interval = self.negative_interval

            if self.verbose:
                print('negative interval: %d' % interval)

            dst_idx = src_idx + interval
            if dst_idx >= len(traj):
                idx = (idx + 1000) % len(self)
                continue
            src_sample = traj[src_idx]
            dst_samples = self._make_sample_seq(traj, dst_idx)
            return map_name, src_sample, dst_samples

    def __getitem__(self, idx):
        self._init_once(idx)

        if time.time() - self.last_heap_trim_time > self.heap_trim_interval:
            self._malloc_trim()
            self.last_heap_trim_time = time.time()

        while True:
            p = self.rng.uniform(0.0, 1.0)
            if p < self.sample_diff_traj_prob:
                map_name, src_sample, dst_samples = self._draw_sample_positive(idx)
                r = 1.0
            else:
                map_name, src_sample, dst_samples = self._draw_sample_negative(idx)
                r = 0.0

            src_img = self._make(map_name, [src_sample])[0]
            dst_imgs = np.array(self._make(map_name, dst_samples))

            return src_img, dst_imgs, np.array([r], np.float32)


class DatasetSourceTargetPairMultiframeDstLidar(DatasetSourceTargetPairMultiframeDst):
    def __init__(self, depth_dropout_prob=0.0, noisy_depth=False, *args, **kwargs):
        super(DatasetSourceTargetPairMultiframeDstLidar, self).__init__(*args, **kwargs)
        self.depth_dropout_prob = depth_dropout_prob
        self.noisy_depth = noisy_depth

    def _inject_noise(self, depth):
        if not self.noisy_depth:
            return depth
        scales = self.rng.normal(loc=1.0, scale=0.01, size=len(depth))
        return depth * scales

    def _apply_dropout(self, depth):
        if self.depth_dropout_prob == 0.0:
            return depth
        mask = self.rng.binomial(n=1, p=1.0-self.depth_dropout_prob, size=len(depth))
        return depth * mask

    def _get_ob(self, map_name):
        self.agent._measure()
        return self._apply_dropout(self._inject_noise(self.agent.depth_local)).astype(np.float32)

    def _compute_overlap(self, map, pos1, heading1, pos2, heading2):
        return map.view_overlap(pos1, heading1, self.fov, pos2, heading2, self.fov, mode='lidar')

    def _make(self, map_name, samples, **kwargs):
        agent = self.agent
        agent.set_map(self.maps[map_name])

        laser_scans = []

        for sample in samples:
            agent.set_pos(sample['pos'])
            agent.set_heading(sample['heading'])
            agent._measure()
            depth = self._apply_dropout(self._inject_noise(agent.depth_local)).astype(np.float32)
            laser_scans.append(depth)

        return laser_scans


# Obsolete!
class DatasetSourceTargetPairMultiframe(DatasetSourceTargetPair):
    def __init__(self, n_frame, frame_interval, **kwargs):
        super(DatasetSourceTargetPairMultiframe, self).__init__(**kwargs)
        self.n_frame = n_frame
        self.frame_interval = frame_interval

    def _draw_sample_same_traj(self, idx):
        distance = self.rng.uniform(self.distance_min, self.distance_max)

        while True:
            # Get a good pair of samples
            dataset_idx, traj_id, src_idx = self._locate_sample(idx)

            traj = self.fds[dataset_idx][traj_id]
            map_name = traj.attrs['map'].decode('ascii')

            src_sample = traj[src_idx]
            src_pos = src_sample['pos']

            dst_idx = src_idx + 1

            while dst_idx < len(traj):
                dst_sample = traj[dst_idx]
                dst_pos = dst_sample['pos']
                x, y = dst_pos - src_pos
                if x ** 2 + y ** 2 > distance ** 2:
                    break
                dst_idx += 1

            if dst_idx < len(traj):
                # Found valid dst_sample
                break

            # select another idx if this one doesn't work
            idx = (idx + 1000) % len(self)

        src_samples = []
        dst_samples = []

        for i in range(self.n_frame):
            src_samples.append(traj[max(src_idx - i * self.frame_interval, 0)])
            dst_samples.append(traj[max(dst_idx - i * self.frame_interval, 0)])

        src_samples = src_samples[::-1]
        dst_samples = dst_samples[::-1]

        return map_name, src_samples, dst_samples

    def _draw_sample_diff_traj(self, idx):
        # Get a good pair of samples
        src_dataset_idx, src_traj_id, src_idx = self._locate_sample(idx)

        src_traj = self.fds[src_dataset_idx][src_traj_id]
        map_name = src_traj.attrs['map'].decode('ascii')

        idx2 = self.rng.randint(self.samples_per_map[map_name])
        dst_dataset_idx, dst_traj_id, dst_idx = self._locate_sample_single_map(idx2, map_name)
        dst_traj = self.fds[dst_dataset_idx][dst_traj_id]

        src_samples = []
        dst_samples = []

        for i in range(self.n_frame):
            src_samples.append(src_traj[max(src_idx - i * self.frame_interval, 0)])
            dst_samples.append(dst_traj[max(dst_idx - i * self.frame_interval, 0)])

        src_samples = src_samples[::-1]
        dst_samples = dst_samples[::-1]

        return map_name, src_samples, dst_samples

    def __getitem__(self, idx):
        self._init_once(idx)

        if time.time() - self.last_heap_trim_time > self.heap_trim_interval:
            self._malloc_trim()
            self.last_heap_trim_time = time.time()

        # FIXME: it might be better to use rollout images as src samples

        p = self.rng.uniform(0.0, 1.0)
        if p < self.sample_diff_traj_prob:
            map_name, src_samples, dst_samples = self._draw_sample_diff_traj(idx)
        else:
            map_name, src_samples, dst_samples = self._draw_sample_same_traj(idx)

        # Swap src_samples and dst_samples with some probability
        if self.rng.uniform(0.0, 1.0) < self.swap_src_tgt_prob:
            src_samples, dst_samples = dst_samples, src_samples

        src_imgs = np.array(self._make(map_name, src_samples))
        dst_imgs = np.array(self._make(map_name, dst_samples))

        r, obs = self.reachable(self.maps[map_name], src_samples[-1], dst_samples[-1])

        if len(obs) > self.n_frame:
            # Skip the first frame in obs because it is the same as src_samples[-1]
            obs = np.array(obs[1:self.n_frame + 1])
            mask = [1.0] * self.n_frame
        elif len(obs) > 2:  # Skipping the first frame results in at least one frame available.
            zero_imgs = np.zeros((self.n_frame - len(obs) + 1,) + obs[0].shape, obs[0].dtype)
            mask = [1.0] * (len(obs) - 1) + [0.0] * (self.n_frame - len(obs) + 1)
            obs = np.concatenate([np.array(obs[1:]), zero_imgs], axis=0)
        else:
            # No image is available due to shortcutting.
            obs = np.zeros((self.n_frame,) + src_imgs[0].shape, src_imgs[0].dtype)
            mask = [0.0] * self.n_frame

        try:
            return src_imgs, dst_imgs, np.array([r], np.float32), obs, np.array(mask, np.float)
        except:
            print('Error during making samples.')
            raise


if __name__ == '__main__':
    from easydict import EasyDict
    import glob
    import os
    import torch
    from torch.utils.data import DataLoader
    import tabulate
    import cv2
    from lib.python.image_combiner import VStack
    from lib.python.utils import get_project_root

    torch.manual_seed(39123)

    def test_multiframe():
        dataset = DatasetSourceTargetPairMultiframe(
            n_frame=10,
            frame_interval=5,
            fov=np.deg2rad(118.6),
            distance_min=0.0,
            distance_max=3.0,
            sample_diff_traj_prob=0.5,
            swap_src_tgt_prob=0.3,
            motion_policy_weights_file=os.path.dirname(
                __file__) + '/../imitation/experiments/simple/imitation-reverseprob0.3-use_gtwp-model.10',
            maps=None,
            agent_name='classic_240fov_minirccar',
            net_config=EasyDict({
                'input_resolution': 64
            }),
            hd5_files=['/home/xymeng/dev/merlin/data/rmp/gibson/minirccar_agent_local_240fov_house31_dest_replan/train_1.hd5'],
            assets_dir=os.path.dirname(__file__) + '/../../../gibson/assets/dataset',
            camera_pos=(0.065, 0.00),
            camera_z=1.0,
            h_fov=np.deg2rad(118.6),
            v_fov=np.deg2rad(106.9),
            n_filler_server=0, n_sim_per_map=0,
            persistent_server_cfg_file='../gibson_persistent_servers_cfg.yaml')

        loader = DataLoader(dataset, batch_size=1, shuffle=True)

        for idx, (batch_src_imgs, batch_dst_imgs, batch_reachability, batch_obs, batch_mask) in enumerate(loader):
            src_imgs = batch_src_imgs[0].data.cpu().numpy().transpose((0, 2, 3, 1))
            dst_imgs = batch_dst_imgs[0].data.cpu().numpy().transpose((0, 2, 3, 1))
            obs = batch_obs[0].data.cpu().numpy().transpose((0, 2, 3, 1))

            print('reachability', batch_reachability[0].data.cpu().numpy())
            print('mask', batch_mask[0].data.cpu().numpy())

            cv2.imshow('imgs',
                       np.concatenate([np.concatenate(src_imgs, axis=1),
                                       np.concatenate(dst_imgs, axis=1),
                                       np.concatenate(obs, axis=1)]))
            cv2.waitKey(0)

    def test_v2():
        dataset = DatasetSourceTargetPairV2(
            n_frame=10,
            frame_interval=5,
            fov=np.deg2rad(118.6),
            distance_min=-1.0,
            distance_max=3.0,
            sample_diff_traj_prob=0.5,
            swap_src_tgt_prob=0.0,
            motion_policy_weights_file=os.path.dirname(
                __file__) + '/../imitation/experiments/simple/imitation-reverseprob0.3-use_gtwp-model.10',
            maps=None,
            agent_name='classic_240fov_minirccar',
            net_config=EasyDict({
                'input_resolution': 64
            }),
            hd5_files=['/home/xymeng/dev/merlin/data/rmp/gibson/minirccar_agent_local_240fov_house31_dest_replan/train_1.hd5'],
            assets_dir=os.path.dirname(__file__) + '/../../../gibson/assets/dataset',
            camera_pos=(0.065, 0.00),
            camera_z=1.0,
            h_fov=np.deg2rad(118.6),
            v_fov=np.deg2rad(106.9),
            n_filler_server=0, n_sim_per_map=0,
            persistent_server_cfg_file='../gibson_persistent_servers_cfg.yaml')

        loader = DataLoader(dataset, batch_size=1, shuffle=True)

        for idx, (batch_src_imgs, batch_dst_imgs, batch_reachability, batch_obs, batch_mask) in enumerate(loader):
            src_imgs = batch_src_imgs[0].data.cpu().numpy().transpose((1, 2, 0))
            dst_imgs = batch_dst_imgs[0].data.cpu().numpy().transpose((1, 2, 0))
            obs = batch_obs[0].data.cpu().numpy().transpose((0, 2, 3, 1))

            print('reachability', batch_reachability[0].data.cpu().numpy())
            print('mask', batch_mask[0].data.cpu().numpy())

            cv2.imshow('imgs', VStack(src_imgs, dst_imgs, np.concatenate(obs, axis=1)))
            cv2.waitKey(0)

    def test_v3():
        dataset = DatasetSourceTargetPairV3(
            fov=np.deg2rad(118.6),
            distance_min=0.0,
            distance_max=3.0,
            reach_overlap_thres=(0.6, 0.6),
            sample_diff_traj_prob=0.0,
            swap_src_tgt_prob=0.0,
            wp_norm_min_clip=2.0,
            steps_margin=50,
            # motion_policy_weights_file=os.path.dirname(
            #     __file__) + '/../imitation/experiments/house31_unreal/pair/gtwp-normwp-farwp-jitter-weightedloss-fixed-checkwp-model.5',
            motion_policy_weights_file=os.path.dirname(
                __file__) + '/../imitation/experiments/house31/pair/gtwp-normwp-farwp-jitter-weightedloss-checkwp-model.5',
            # motion_policy_weights_file=os.path.dirname(
            #     __file__) + '/../imitation/experiments/house31/pair/with_heading_diff/gtwp-normwp-farwp-jitter-weightedloss-checkwp-model.5',
            maps=None,
            agent_name='classic_240fov_minirccar',
            net_config=EasyDict({
                'input_resolution': 64
            }),
            hd5_files=['/home/xymeng/dev/merlin/data/rmp/gibson/minirccar_agent_local_240fov_house31_farwp/train_1.hd5'],
            assets_dir=os.path.dirname(__file__) + '/../../../gibson/assets/dataset',
            camera_pos=(0.065, 0.00),
            camera_z=1.0,
            h_fov=np.deg2rad(118.6),
            v_fov=np.deg2rad(106.9),
            n_filler_server=0, n_sim_per_map=0,
            persistent_server_cfg_file='../gibson_persistent_servers_cfg.yaml')

        loader = DataLoader(dataset, batch_size=1, shuffle=True)
        sum_r = 0.0
        count = 0

        for idx, (batch_src_imgs, batch_dst_imgs, batch_reachability) in enumerate(loader):
            src_imgs = batch_src_imgs[0].data.cpu().numpy().transpose((1, 2, 0))
            dst_imgs = batch_dst_imgs[0].data.cpu().numpy().transpose((1, 2, 0))

            print(idx)

            r = batch_reachability[0].item()
            print('reachability', r)

            count += 1
            sum_r += r

            print('avg reachability: %.2f' % (sum_r / count))

            cv2.imshow('imgs', VStack(src_imgs, dst_imgs))
            cv2.waitKey(1)

            if count == 1000:
                break

    def test_multiframe_dst_naive():
        dataset = DatasetSourceTargetPairMultiframeDstNaive(
            positive_interval=20,
            negative_interval=100,
            n_frame=6,
            frame_interval=3,
            future=True,
            distance_min=-1,
            distance_max=-1,
            fov=np.deg2rad(118.6),
            sample_diff_traj_prob=0.5,
            motion_policy_weights_file=os.path.dirname(
                __file__) + '/../imitation/experiments/12env_v2/multiframe_dst/future/pair/pred_proximity/pred_heading_diff/conv/gtwp-normwp-farwp-jitter-weightedloss-checkwp-nf6-interval3-dmax3-model.6',
            maps=None,
            agent_name='classic_240fov_minirccar',
            net_config=EasyDict({
                'input_resolution': 64
            }),
            hd5_files=glob.glob('../../../../../../data/rmp/gibson/minirccar_agent_local_240fov_house24_farwp/train_*.hd5'),
            assets_dir=os.path.dirname(__file__) + '/../../../gibson/assets/dataset',
            camera_pos=(0.065, 0.00),
            camera_z=1.0,
            h_fov=np.deg2rad(118.6),
            v_fov=np.deg2rad(106.9),
            n_filler_server=0, n_sim_per_map=0,
            persistent_server_cfg_file='../gibson_persistent_servers_cfg.yaml')

        print(dataset)
        loader = DataLoader(dataset, batch_size=1, shuffle=True)

        sum_r = 0.0
        count = 0

        for idx, (batch_src_imgs, batch_dst_imgs, batch_reachability) in enumerate(loader):
            src_img = batch_src_imgs[0].data.cpu().numpy().transpose((1, 2, 0))
            dst_imgs = batch_dst_imgs[0].data.cpu().numpy().transpose((0, 2, 3, 1))

            r = batch_reachability[0].item()
            print('reachability', r)

            count += 1
            sum_r += r

            print('avg reachability: %.2f' % (sum_r / count))

            # cv2.imshow('imgs', VStack(src_img, np.concatenate(dst_imgs, axis=1)))
            # cv2.waitKey(0)

    def test_multiframe_dst():
        dataset = DatasetSourceTargetPairMultiframeDst(
            n_frame=6,
            frame_interval=3,
            future=True,
            fov=np.deg2rad(118.6),
            distance_min=0.0,
            distance_max=3.0,
            reach_overlap_thres=(0.7, 0.7),
            sample_diff_traj_prob=0.5,
            swap_src_tgt_prob=0.0,
            motion_policy_weights_file=os.path.dirname(
                __file__) + '/../imitation/experiments/12env_v2/multiframe_dst/future/pair/pred_proximity/pred_heading_diff/conv/gtwp-normwp-farwp-jitter-weightedloss-checkwp-nf6-interval3-dmax3-z0228-model.6',
            maps=None,
            agent_name='classic_240fov_minirccar',
            net_config=EasyDict({
                'input_resolution': 64
            }),
            hd5_files=['../../../../../../data/rmp/gibson/minirccar_agent_local_240fov_house24_farwp/train_1.hd5'],
            assets_dir=os.path.dirname(__file__) + '/../../../gibson/assets/dataset',
            camera_pos=(0.065, 0.00),
            camera_z=0.228,
            h_fov=np.deg2rad(118.6),
            v_fov=np.deg2rad(106.9),
            n_filler_server=0, n_sim_per_map=0,
            persistent_server_cfg_file='../gibson_persistent_servers_cfg.yaml')

        print(dataset)

        loader = DataLoader(dataset, batch_size=1, shuffle=True)

        sum_r = 0.0
        count = 0

        for idx, (batch_src_imgs, batch_dst_imgs, batch_reachability) in enumerate(loader):
            src_img = batch_src_imgs[0].data.cpu().numpy().transpose((1, 2, 0))
            dst_imgs = batch_dst_imgs[0].data.cpu().numpy().transpose((0, 2, 3, 1))

            r = batch_reachability[0].item()
            print('reachability', r)

            count += 1
            sum_r += r

            print('avg reachability: %.2f' % (sum_r / count))

            #cv2.imshow('imgs', VStack(src_img, np.concatenate(dst_imgs, axis=1)))
            #cv2.waitKey(0)

    def test_multiframe_dst_lidar():
        dataset = DatasetSourceTargetPairMultiframeDstLidar(
            depth_dropout_prob=0.2,
            noisy_depth=True,
            n_frame=6,
            frame_interval=3,
            future=True,
            fov=np.deg2rad(240.0),
            distance_min=0.0,
            distance_max=3.0,
            reach_overlap_thres=(0.7, 0.7),
            sample_diff_traj_prob=0.0,
            swap_src_tgt_prob=0.0,
            motion_policy_weights_file=os.path.dirname(
                __file__) + '/../imitation/experiments/12env_v2_laserscan/multiframe_dst/future/pair/pred_proximity/pred_heading_diff/conv/gtwp-normwp-farwp-jitter-weightedloss-checkwp-nf6-interval3-dmax3-model.10',
            maps=None,
            agent_name='classic_240fov_minirccar_64',
            net_config=EasyDict({
                'input_resolution': 0
            }),
            hd5_files=[get_project_root() + '/data/rmp/gibson/minirccar_agent_local_240fov_house31_fixedwpdist/train_1.hd5'],
            assets_dir=os.path.dirname(__file__) + '/../../../gibson/assets/dataset',
            n_filler_server=0, n_sim_per_map=0
        )

        print(dataset)

        loader = DataLoader(dataset, batch_size=1, shuffle=True)

        sum_r = 0.0
        count = 0

        for idx, (batch_src_scans, batch_dst_scans, batch_reachability) in enumerate(loader):
            r = batch_reachability[0].item()
            print('reachability', r)

            count += 1
            sum_r += r

            print(batch_src_scans[0])
            print('avg reachability: %.2f' % (sum_r / count))

    def test_multiframe_dst_recording():
        dataset = DatasetSourceTargetPairMultiframeDstRecording(
            'analysis/manual_labels',
            64, negative=False)
        loader = DataLoader(dataset, batch_size=1, shuffle=True)

        for idx, (batch_src_imgs, batch_dst_imgs, batch_reachability) in enumerate(loader):
            src_img = batch_src_imgs[0].data.cpu().numpy().transpose((1, 2, 0))
            dst_imgs = batch_dst_imgs[0].data.cpu().numpy().transpose((0, 2, 3, 1))
            r = batch_reachability[0].item()
            print('reachability', r)
            cv2.imshow('imgs', VStack(src_img, np.concatenate(dst_imgs, axis=1)))
            cv2.waitKey(0)

    # test_v3()
    test_multiframe_dst()
    # test_multiframe_dst_lidar()
    # test_multiframe_dst_naive()
    # test_multiframe_dst_recording()

