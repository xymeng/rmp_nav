import numpy as np
import math
from copy import deepcopy
from rmp_nav.common.math_utils import rotate_2d
from rmp_nav.simulation.gibson_map import MakeGibsonMap
from rmp_nav.neural.common.dataset import DatasetVisualGibson

_DEBUG = False


def make_maps(assets_dir, map_names, **kwargs):
    """
    :return: a dict of (map_name, map)
    """
    def load_maps(datadir, map_names):
        maps = {}
        for name in map_names:
            maps[name] = MakeGibsonMap(datadir, name, **kwargs)
        return maps
    map_names = [s.strip() for s in map_names]
    return load_maps(assets_dir, map_names)


class DatasetSourceTargetPair(DatasetVisualGibson):
    def __init__(self,
                 fov: float,
                 distance_min: float,
                 distance_max: float,
                 overlap_ratio: float,
                 normalize_wp: bool=False,
                 use_gt_wp: bool=False,
                 check_wp: bool=False,
                 jitter: bool=False,
                 **kwargs):
        """
        :param fov: camera horizontal field of view
        :param distance_min: minimum distance of target
        :param distance_max: maximum distance of target
        :param overlap_ratio: minimum overlap betwen source and target.
        :param reverse_prob: probability of swapping source and target image. Could be useful for
               learning backward maneuver.
        :param use_gt_wp: use the groundtruth waypoint provided by the dataset if available. This
               is usually better than relative position. However, if such waypoint is not available,
               e.g., when we reverse the two images, we fall back to relative position.
        :param check_wp: double check that the src sample and dst sample have the same global
               waypoint. This ensures that dst is reachable from src using src's global waypoint.
               But note that this only works if the waypoints are globally stationary! Usually farwp
               datasets work, but don't use fixedwpdist datasets!
        """
        super(DatasetSourceTargetPair, self).__init__(**kwargs)

        self.fov = fov
        self.distance_min = distance_min
        self.distance_max = distance_max
        self.overlap_ratio = overlap_ratio
        self.normalize_waypoint = normalize_wp
        self.use_gt_wp = use_gt_wp
        self.check_wp = check_wp
        self.jitter = jitter

        self.maps = make_maps(self.assets_dir, self.map_names)

    def _compute_overlap(self, src_sample, dst_sample, map_name):
        map = self.maps[map_name]
        overlap_ratios = map.view_overlap(src_sample['pos'], src_sample['heading'], self.fov,
                                          dst_sample['pos'], dst_sample['heading'], self.fov)
        return overlap_ratios

    def _jitter(self, sample, map):
        if not self.jitter:
            return sample

        self.agent.set_map(map)
        pos, heading = sample['pos'], sample['heading']
        heading_perp = rotate_2d(np.array([math.cos(heading), math.sin(heading)], np.float32),
                                 math.pi / 2)
        new_sample = deepcopy(sample)

        while True:
            # Only jitter position sideways
            dpos = heading_perp * np.clip(self.rng.randn() * 0.3, -1.0, 1.0)  # 99% prob in (-1, 1)
            new_sample['pos'] = pos + dpos
            dheading = self.rng.randn() * 0.3 * np.deg2rad(30)
            new_sample['heading'] = heading + dheading
            self._set_agent_state(new_sample)
            if not self.agent.collide():
                if _DEBUG:
                    print('dpos: %s dheading: %s deg' % (dpos, np.rad2deg(dheading)))
                break
        return new_sample

    def _get_sample(self, idx, distance):
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
            src_heading = src_sample['heading']
            src_waypoint = src_sample['waypoint_global']

            dst_idx = src_idx + inc

            skip = False
            last_d = 0.0

            while 0 <= dst_idx < len(traj):
                dst_sample = traj[dst_idx]
                dst_pos = dst_sample['pos']

                if inc > 0:
                    x, y = dst_pos - src_pos
                else:
                    x, y = src_pos - dst_pos

                d = math.sqrt(x ** 2 + y ** 2)

                if d > 0.01 and math.cos(src_heading) * x + math.sin(src_heading) * y < 0:
                    # Skip this sample if the agent is moving backward
                    skip = True
                    break

                if self.check_wp:
                    dst_waypoint = dst_sample['waypoint_global']
                    if np.linalg.norm(src_waypoint - dst_waypoint) > 0.2:
                        # dst_waypoint has changed significantly
                        # This means the destination may not be reachable with src_waypoint, so we
                        # should stop here.
                        dst_idx = max(dst_idx - 1, src_idx)  # use previous dst_idx
                        if last_d < self.distance_min:
                            # previous dst is too close to src. skip this sample
                            skip = True
                        break

                if d > abs(distance):
                    break

                last_d = d
                dst_idx += inc

            if not skip and 0 <= dst_idx < len(traj):
                dst_sample = traj[dst_idx]
                overlap_ratios = self._compute_overlap(src_sample, dst_sample, map_name)
                if overlap_ratios[0] > self.overlap_ratio and \
                        overlap_ratios[1] > self.overlap_ratio:
                    break

            # select another idx if this one doesn't work
            idx = (idx + 1000) % len(self)

        return dataset_idx, traj_id, traj, map_name, src_idx, dst_idx, src_sample, overlap_ratios

    def _normalize_waypoint(self, wp, distance):
        if not self.normalize_waypoint:
            return wp

        wp_norm = np.linalg.norm(wp)
        if wp_norm > abs(distance):
            return wp / wp_norm * abs(distance)
        else:
            return wp

    def __getitem__(self, idx):
        self._init_once(idx)

        distance = self.rng.uniform(self.distance_min, self.distance_max)

        dataset_idx, traj_id, traj, map_name, src_idx, dst_idx, src_sample, overlap_ratios = \
            self._get_sample(idx, distance)

        dst_sample = traj[dst_idx]
        samples = [src_sample, dst_sample]

        heading_diff = dst_sample['heading'] - src_sample['heading']
        heading_diff_xy = np.array([math.cos(heading_diff), math.sin(heading_diff)], np.float32)

        extras = {
            'proximity': np.array(overlap_ratios, np.float32),
            'heading_diff': heading_diff_xy
        }

        try:
            imgs, positions, headings, waypoints, velocities = self._make(
                map_name, samples, attrs=traj.attrs)

            if distance > 0:
                if self.use_gt_wp:
                    wp = rotate_2d(waypoints[0] - positions[0], -headings[0])
                else:
                    wp = rotate_2d(positions[1] - positions[0], -headings[0])

                return imgs[0], imgs[1], self._normalize_waypoint(wp, distance), extras
            else:
                # Always return relative position for negative distance samples
                rel_pos = rotate_2d(positions[1] - positions[0], -headings[0])
                # assert np.linalg.norm(rel_pos) < abs(distance) + 0.1
                return imgs[0], imgs[1], rel_pos, extras
        except:
            print('Error during making samples. dataset',
                  self.hd5_files[dataset_idx], 'traj_id', traj_id, 'sample_idx', dst_idx)
            raise

    def _make(self, map_name, samples, **kwargs):
        imgs = []

        positions = []
        headings = []
        waypoints = []
        velocities = []

        for sample in samples:
            self._set_agent_state(sample)
            imgs.append(self._render_agent_view(map_name))
            positions.append(sample['pos'])
            waypoints.append(sample['waypoint_global'])
            velocities.append(sample['velocity_global'])
            headings.append(sample['heading'])

        return imgs, positions, headings, waypoints, velocities


class DatasetSourceTargetPairMultiframeDst(DatasetSourceTargetPair):
    """
    Return a single-frame src_img and multi-frame dst_img
    """
    def __init__(self, n_frame, frame_interval, future=False, **kwargs):
        """
        :param future: if True returns t_{-n+1}, ... t, t_1, ... t_{n-1}
        :param kwargs:
        """
        super(DatasetSourceTargetPairMultiframeDst, self).__init__(**kwargs)
        self.n_frame = n_frame
        self.frame_interval = frame_interval
        self.future = future

    def __getitem__(self, idx):
        self._init_once(idx)

        distance = self.rng.uniform(self.distance_min, self.distance_max)
        dataset_idx, traj_id, traj, map_name, src_idx, dst_idx, src_sample, overlap_ratios = \
            self._get_sample(idx, distance)

        dst_samples = []
        for i in range(self.n_frame):
            dst_samples.append(traj[max(dst_idx - i * self.frame_interval, 0)])
        dst_samples = dst_samples[::-1]

        if self.future:
            for i in range(self.n_frame - 1):
                dst_samples.append(traj[min(dst_idx + (i + 1) * self.frame_interval, len(traj) - 1)])

        dst_sample = dst_samples[self.n_frame - 1]

        heading_diff = dst_sample['heading'] - src_sample['heading']
        heading_diff_xy = np.array([math.cos(heading_diff), math.sin(heading_diff)], np.float32)

        extras = {
            'proximity': np.array(overlap_ratios, np.float32),
            'heading_diff': heading_diff_xy
        }

        try:
            [src_img], [src_position], [src_heading], [src_waypoint], _ = self._make(
                map_name, [src_sample], attrs=traj.attrs)
            dst_imgs, dst_positions, _, _, _ = self._make(map_name, dst_samples, attrs=traj.attrs)
            dst_imgs = np.array(dst_imgs)

            if distance > 0:
                if self.use_gt_wp:
                    wp = rotate_2d(src_waypoint - src_position, -src_heading)
                else:
                    wp = rotate_2d(dst_positions[self.n_frame - 1] - src_position, -src_heading)
                wp = self._normalize_waypoint(wp, distance)
            else:
                # Always return relative position for negative distance samples
                wp = rotate_2d(dst_positions[self.n_frame - 1] - src_position, -src_heading)
                # assert np.linalg.norm(rel_pos) < abs(distance) + 0.1
            return src_img, dst_imgs, wp, extras
        except:
            print('Error during making samples. dataset',
                  self.hd5_files[dataset_idx], 'traj_id', traj_id, 'sample_idx', dst_idx)
            raise


if __name__ == '__main__':
    from easydict import EasyDict
    import cv2
    import os
    from torch.utils.data import DataLoader
    from rmp_nav.common.image_combiner import VStack
    from rmp_nav.common.utils import get_gibson_asset_dir, get_data_dir, get_config_dir

    gibson_kwargs = {
        'assets_dir': get_gibson_asset_dir(),
        'camera_pos': (0.065, 0.00),
        'camera_z': 0.228,
        'h_fov': np.deg2rad(118.6),
        'v_fov': np.deg2rad(106.9),
        'n_filler_server': 0,
        'n_sim_per_map': 0,
        'persistent_server_cfg_file': os.path.join(get_config_dir(), 'gibson_persistent_servers/local.yaml')
    }

    def test_multiframe_dst():
        dataset = DatasetSourceTargetPairMultiframeDst(
            n_frame=5,
            frame_interval=3,
            overlap_ratio=0.2,
            use_gt_wp=True,
            normalize_wp=True,
            jitter=True,
            check_wp=True,
            future=True,
            fov=np.deg2rad(118.6),
            distance_min=-0.5,
            distance_max=3.0,
            maps=None,
            agent_name='classic_240fov_minirccar',
            net_config=EasyDict({
                'input_resolution': 64
            }),
            hd5_files=[
                os.path.join(get_data_dir(), 'minirccar_agent_local_240fov_space8_farwp_v2', 'train_1.hd5')
            ],
            **gibson_kwargs)

        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        for idx, (batch_src_imgs, batch_dst_imgs, batch_waypoints, batch_extras) in enumerate(loader):
            src_img = batch_src_imgs[0].data.cpu().numpy().transpose((1, 2, 0))
            dst_imgs = batch_dst_imgs[0].data.cpu().numpy().transpose((0, 2, 3, 1))

            print('waypoint:', batch_waypoints[0].data.cpu().numpy(),
                  'proximity:', batch_extras['proximity'][0].data.cpu().numpy())

            cv2.imshow('imgs', VStack(src_img, np.concatenate(dst_imgs, axis=1)))
            cv2.waitKey(0)

    test_multiframe_dst()
