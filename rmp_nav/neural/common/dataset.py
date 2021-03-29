import h5py
import torch.utils.data as data
import itertools
import numpy as np
import math
import time
import bisect
import os
import yaml
import zmq


from rmp_nav.simulation import agent_factory
from rmp_nav.common.math_utils import depth_to_xy
from rmp_nav.common.utils import pprint_dict, get_default_persistent_server_config


_LOG_RENDER_TIME = False
_LOG_EMIT_SAMPLE_TIME = False


def _rotate(x, y, angle):
    c = np.cos(angle)
    s = np.sin(angle)
    x2 = c * x - s * y
    y2 = s * x + c * y
    return x2, y2


def _randomize_goal(rng, goal):
    angle = rng.uniform(0.0, np.pi * 2.0)
    return np.array(_rotate(goal[0], goal[1], angle), np.float32)


def _randomize_velocity(rng, velocity):
    '''
    :return: randomly scaled and rotated velocity. 50% chance of returning the original.
    '''
    if rng.uniform(0.0, 1.0) < 0.5:
        return velocity

    # There are only a few zero-velocity training samples in the data. This can cause
    # the network from not learning the initial acceleration properly. We need to manually
    # inject small-velocity training samples to compensate that.
    gain = rng.uniform(0.0, 2.0)
    angle = rng.uniform(0.0, np.pi * 2.0)

    v = velocity * gain
    return np.array(_rotate(v[0],  v[1], angle), np.float32)


def _randomize_angular_velocity(rng, angular_vel):
    if rng.uniform(0.0, 1.0) < 0.5:
        return angular_vel
    gain = rng.uniform(1.0, 2.0)
    ret = angular_vel * gain
    return ret


def _randomize_lighting(rng, img):
    assert img.dtype == np.float32

    contrast_factor = rng.uniform(0.5, 2.0)
    brightness_factor = rng.uniform(-0.2, 0.2)

    return np.clip(img * contrast_factor + brightness_factor, 0.0, 1.0)


def maybe_decode(s):
    # This is to deal with the breaking change in how h5py 3.0 deals with
    # strings.
    if type(s) == str:
        return s
    else:
        return s.decode('ascii')
    

class DatasetVisual(data.Dataset):
    def __init__(self, hd5_files, agent_name,
                 ignore_goal=False,
                 random_goal=False,
                 no_waypoint=False,
                 random_vel=False,
                 random_lighting=False,
                 net_config=None,
                 load_to_mem=False,
                 maps=None,
                 max_traj=-1):
        '''
        :param maps: an iterable of strs. If specified only these maps will be considered.
        :param max_traj: the max number of trajectories per training file.
        '''

        self.hd5_files = sorted(list(hd5_files))
        self.ignore_goal = ignore_goal
        self.random_goal = random_goal
        self.no_waypoint = no_waypoint
        self.random_vel = random_vel
        self.random_lighting = random_lighting
        self.net_config = net_config
        self.load_to_mem = load_to_mem

        # Used in __repr__
        self.g = {
            'ignore_goal': self.ignore_goal,
            'random_goal': self.random_goal,
            'no_waypoint': self.no_waypoint,
            'random_vel': self.random_vel,
            'random_lighting': self.random_lighting,
            'net_config': self.net_config
        }

        fds = []
        for fn in self.hd5_files:
            try:
                fds.append(h5py.File(fn, 'r'))
            except:
                print('unable to open', fn)
                raise

        def flatten(ll):
            return list(itertools.chain.from_iterable(ll))

        def filter_bad_traj(traj_ids):
            '''
            Ignore very long trajectories (likely agent getting stuck).
            '''
            return [(i, tid) for i, tid in traj_ids if 0 < fds[i][tid].shape[0] < 3000]

        # A list of tuples (dataset_idx, trajectory_id)
        self.traj_ids = flatten(
            zip(itertools.repeat(i),
                list(fds[i].keys())[0: max_traj if max_traj > 0 else len(fds[i])])
            for i in range(len(fds)))
        print('total trajectories:', len(self.traj_ids))

        self.traj_id_to_dset_idx = {traj_id: dset_idx for dset_idx, traj_id in self.traj_ids}

        # Map (dataset_idx, traj_id) to its corresponding map
        traj_id_map = {(dset_idx, traj_id): maybe_decode(fds[dset_idx][traj_id].attrs['map'])
                       for dset_idx, traj_id in self.traj_ids}

        if maps is not None and len(maps) > 0:
            s = set(maps)
            # Filter traj_ids to only include selected maps
            self.traj_ids = [(dset_idx, traj_id) for dset_idx, traj_id in self.traj_ids
                             if traj_id_map[(dset_idx, traj_id)] in s]
            self.map_names = sorted(list(s))
            print('only consider these maps:', self.map_names)
            print('filtered trajectories:', len(self.traj_ids))
        else:
            self.map_names = sorted(list(set(traj_id_map.values())))
        self.traj_id_map = traj_id_map

        self.traj_ids = filter_bad_traj(self.traj_ids)
        print('good trajectories:', len(self.traj_ids))

        # Map (dataset_idx, traj_id) to trajectory length
        self.traj_len_dict = {(i, tid): fds[i][tid].shape[0] for i, tid in self.traj_ids}
        self.traj_len_cumsum = np.cumsum([self.traj_len_dict[_] for _ in self.traj_ids])

        # Compute weights for each sample for weighted random sampling
        n_sample_per_file = [0] * len(fds)
        for i, tid in self.traj_ids:
            n_sample_per_file[i] += fds[i][tid].shape[0]

        self.sample_weights = []
        for i in range(len(n_sample_per_file)):
            self.sample_weights.extend([1.0 / n_sample_per_file[i]] * n_sample_per_file[i])

        assert len(self.sample_weights) == self.traj_len_cumsum[-1]
        assert np.isclose(sum(self.sample_weights), len(fds))

        self.agent = agent_factory.agents_dict[agent_name]()

        self.rng = None
        self.first = True
        self.opened = False

        self.dataset_attrs = []  # The global attribute of each dataset
        for i in range(len(fds)):
            self.dataset_attrs.append(dict(fds[i].attrs.items()))
        #
        # # self.attrs[dset_idx][traj_id] == fds[i][traj_id].attrs
        self.attrs = []  # Per trajactory attributes
        for i in range(len(fds)):
            self.attrs.append({
                traj_id: dict(fds[i][traj_id].attrs.items()) for traj_id in fds[i]
            })

        for fd in fds:
            fd.close()

        # from ctypes import cdll
        # self.libc = cdll.LoadLibrary('libc.so.6')

        # Closing the HDF5 files does not release the memory. This is probably because of the
        # malloc's free strategy. Hence we explicitly trim the heap.
        # self._malloc_trim()

    def __repr__(self):
        return '%s options\n%s' % (self.__class__.__name__, pprint_dict(self.g))

    def _malloc_trim(self):
        """
        Trim heap. This can release significant amount of memory after accessing a large number of
        datasets form a hdf5 file.
        """
        from ctypes import cdll
        cdll.LoadLibrary('libc.so.6').malloc_trim(0)

    def _set_agent_state(self, sample):
        pos, depth_local, waypoint_global, velocity_global, heading, angular_velocity = (
            sample['pos'],
            sample['depth_local'],
            sample['waypoint_global'],
            sample['velocity_global'],
            sample['heading'],
            sample['angular_velocity']
        )

        if max(depth_local) > 1e3:
            print('depth:', depth_local)
            raise ValueError('invalid depth')

        if self.random_vel:
            velocity_global = _randomize_velocity(self.rng, velocity_global)
            angular_velocity = _randomize_angular_velocity(self.rng, angular_velocity)

        obstacles_local = np.array(depth_to_xy(depth_local,
                                               fov=self.agent.lidar_sensor.fov), np.float32)

        self.agent.pos = pos
        self.agent.velocity = velocity_global
        self.agent.heading = heading
        self.agent.angular_velocity = angular_velocity
        self.agent.obstacles_local = obstacles_local

        goal_local = self.agent.global_to_local(waypoint_global)
        if self.random_goal:
            goal_local = _randomize_goal(self.rng, goal_local)

        if self.ignore_goal:
            self.agent.goals_local = []
        else:
            self.agent.goals_local = [goal_local]

    def _render_agent_view(self, map_name, **kwargs):
        self.agent._measure()
        return self.agent.depth_local

    def _make(self, map_name, sample, **kwargs):
        raise NotImplementedError

    def _worker_init(self):
        # Called when __getitem__ is called the first time for a worker.
        pass

    def _locate_sample(self, idx):
        traj_idx = bisect.bisect_right(self.traj_len_cumsum, idx)
        dataset_idx, traj_id = self.traj_ids[traj_idx]

        if traj_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.traj_len_cumsum[traj_idx - 1]

        return dataset_idx, traj_id, sample_idx

    def _emit_sample(self, idx):
        dataset_idx, traj_id, sample_idx = self._locate_sample(idx)

        sample = self.fds[dataset_idx][traj_id][sample_idx]
        map_name = maybe_decode(self.fds[dataset_idx][traj_id].attrs['map'])
        final_goal_global = self.fds[dataset_idx][traj_id][-1]['waypoint_global']

        return self._make(
            map_name, sample,
            final_goal_global=final_goal_global,
            attrs=self.fds[dataset_idx][traj_id].attrs)

    def _discretize_waypoint(self, waypoint, n_discretization):
        def _normalize_angle(a):
            while a < 0.0:
                a += math.pi * 2.0
            while a >= math.pi * 2.0:
                a -= math.pi * 2.0
            return a

        x, y = waypoint
        angle = math.atan2(y, x)
        d_theta = math.pi * 2.0 / n_discretization
        return int(_normalize_angle(angle + d_theta * 0.5) / d_theta)

    def _open_datasets(self):
        if not self.opened:
            driver = None
            if self.load_to_mem:
                driver = 'core'
                print('loading dataset into memory... it may take a while')
            self.fds = [h5py.File(fn, 'r', driver=driver)
                        for fn in self.hd5_files]
            self.opened = True

    def _init_once(self, seed):
        # Should be called after the dataset runs in a separate process
        if self.first:
            self._open_datasets()
            self.rng = np.random.RandomState(12345 + (seed % 1000000) * 666)
            self._worker_init()
            self.first = False

    def locate_traj(self, traj_id):
        return self.fds[traj_id[0]][traj_id[1]]

    def locate_traj_map(self, traj_id):
        return self.traj_id_map[traj_id]

    def render_traj(self, traj, **kwargs):
        return self.render_traj_laserscan(traj)

    def render_traj_laserscan(self, traj, agent=None):
        """
        :param traj: a sequence of samples
        :return: a sequence of observations
        """
        if agent is None:
            agent = self.agent
        laserscans = []
        for sample in traj:
            agent.set_pos(sample['pos'])
            agent.set_heading(sample['heading'])
            agent._measure()
            laserscans.append(agent.depth_local)
        return laserscans

    def render(self, pos, heading, map_name, **kwargs):
        self.agent.set_pos(pos)
        self.agent.set_heading(heading)
        return self._render_agent_view(map_name, **kwargs)

    def __getitem__(self, idx):
        self._init_once(idx)

        if _LOG_EMIT_SAMPLE_TIME:
            if not hasattr(self, 'accum_emit_sample_time'):
                self.accum_emit_sample_time = 0.0
                self.emit_sample_count = 0

            start_time = time.time()
            item = self._emit_sample(idx)
            self.accum_emit_sample_time += time.time() - start_time
            self.emit_sample_count += 1

            if self.emit_sample_count % 100 == 0:
                print('avg emit sample time:', self.accum_emit_sample_time / 100)
                self.accum_emit_sample_time = 0
        else:
            item = self._emit_sample(idx)

        # from ctypes import cdll
        # libc = cdll.LoadLibrary("libc.so.6")
        # libc.malloc_trim(0)

        return item

    def __len__(self):
        return self.traj_len_cumsum[-1]


class DatasetVisualGibson(DatasetVisual):
    def __init__(self,
                 assets_dir,
                 render_resolution=256,
                 camera_pos=(0.0, 0.0), camera_z=1.0, h_fov=None, v_fov=None,
                 gpu_device=None,
                 n_filler_server=0, n_sim_per_map=0,
                 persistent_server_cfg_file=None,
                 **kwargs):
        """
        :param gpu_device: can be either an integer or a list of integers
        :param render_resolution: the resolution of rendered images from Gibson
        :param camera_pos: the camera position w.r.t the agent's coordinate system
        :param camera_z: the height of camera
        :param h_fov, v_fov: control field of view of rendered images. If None
               default to agent.lidar_fov
        :param n_filler_server: number of filler servers to use
        :param n_sim_per_map: number of simulation servers for each map
        :param persistent_server_cfg_file: a yaml file containing configuration of persistent servers.
        """
        if not os.path.isdir(assets_dir):
            raise RuntimeError("directory %s doesn't exist" % assets_dir)
        self.assets_dir = assets_dir

        super(DatasetVisualGibson, self).__init__(**kwargs)

        self.render_resolution = render_resolution
        self.h_fov = h_fov
        self.v_fov = v_fov

        self.camera_pos = np.array(camera_pos, np.float32)
        self.camera_z = camera_z

        if hasattr(gpu_device, '__iter__'):
            self.gpus = list(gpu_device)
        elif gpu_device is not None:
            self.gpus = [gpu_device]
        else:
            self.gpus = []

        self.n_sim_per_map = n_sim_per_map
        self.sim_servers = {}
        self.sim_clients = {}

        # key is map name. value is a list of server address in zmq format.
        self.persistent_server_cfg_file = persistent_server_cfg_file
        self.persistent_servers = {}
        if persistent_server_cfg_file is not None:
            self.persistent_servers = self._parse_persistent_server(persistent_server_cfg_file)

        self.g.update({
            'n_filler_server': n_filler_server,
            'n_sim_per_map': n_sim_per_map,
            'render_resolution': self.render_resolution,
            'persistent_servers': self.persistent_servers
        })

        self.n_filler_server = n_filler_server
        self.filler_servers = []  # Tuples of (process, server_addr)

        self.is_worker = False
        self.start_renderer_servers()

    def render_traj(self, traj, map_name='unspecified'):
        imgs = []
        for sample in traj:
            self._set_agent_state(sample)
            imgs.append(self._render_agent_view(map_name))
        return imgs

    def _parse_persistent_server(self, cfg_file):
        return yaml.load(open(cfg_file), Loader=yaml.SafeLoader)

    def _start_renderer_servers_local(self):
        from rmp_nav.simulation import gibson_sim_client, gibson_filler_server
        
        if self.n_filler_server == 0:
            return

        if self.h_fov is None or self.v_fov is None:
            h_fov, v_fov = self.agent.fov, self.agent.fov
        else:
            h_fov, v_fov = self.h_fov, self.v_fov

        for i in range(self.n_filler_server):
            self.filler_servers.append(gibson_filler_server.LaunchServer(
                self.render_resolution,
                self.gpus[i % len(self.gpus)],
                use_ipc=True,
                ipc_endpoint='/tmp/filler_server.%f' % time.time()))

        idx = 0
        self.sim_servers = {}

        if self.n_sim_per_map == 0:
            return

        for map_name in self.map_names:
            if map_name not in self.sim_servers:
                self.sim_servers[map_name] = []

            for i in range(self.n_sim_per_map):
                filler_server_addr = self.filler_servers[idx % self.n_filler_server][1]

                sim_proc, addr = gibson_sim_client.LaunchServer(
                    self.assets_dir, map_name,
                    self.render_resolution,
                    h_fov, v_fov,
                    gpu=self.gpus[idx % len(self.gpus)],
                    filler_server_addr=filler_server_addr)

                self.sim_servers[map_name].append((sim_proc, addr))
                idx += 1

    def start_renderer_servers(self):
        # Start local servers
        self._start_renderer_servers_local()
        # Add persistent servers
        for map_name, addrs in self.persistent_servers.items():
            if map_name not in self.sim_servers and len(addrs) > 0:
                self.sim_servers[map_name] = []
            for addr in addrs:
                self.sim_servers[map_name].append((None, addr))
        print('started renderer servers:\n%s' % pprint_dict(self.sim_servers))

    def stop_renderer_servers(self):
        import zmq
        context = zmq.Context()
        socket = context.socket(zmq.REQ)

        def stop(proc, addr):
            try:
                socket.connect(addr)
                socket.send_pyobj(['exit'])
                socket.recv_pyobj()  # send and recv must be paired
                proc.communicate()
                socket.disconnect(addr)
            except Exception as e:
                raise RuntimeError()

        for name, servers in self.sim_servers.items():
            for proc, addr in servers:
                if proc is not None:
                    # proc can be None if the server is persistent. In that case we don't kill it.
                    stop(proc, addr)
            print('renderer server for map %s stopped' % name)

        for idx, (proc, addr) in enumerate(self.filler_servers):
            print('stopping filler server %s' % addr)
            stop(proc, addr)
            print('filler server %s stopped' % addr)
        self.filler_servers = []

        socket.close()
        context.term()

    def _start_renderer_clients(self):
        from rmp_nav.simulation.gibson_sim_client import GibsonSimClient
        for map_name, servers in self.sim_servers.items():
            if map_name == 'all':
                # This server can handle all maps. We need to set client's  to make sure
                # requests are correctly routed.
                for sim_proc, addr in servers:
                    for identity in self.map_names:
                        client = GibsonSimClient()
                        client.start(create_server=False, server_addr=addr,
                                     identity=identity.encode())

                        if identity not in self.sim_clients:
                            self.sim_clients[identity] = []

                        self.sim_clients[identity].append(client)
            else:
                if map_name not in self.sim_clients:
                    self.sim_clients[map_name] = []

                for sim_proc, addr in servers:
                    client = GibsonSimClient()
                    client.start(create_server=False, server_addr=addr)
                    self.sim_clients[map_name].append(client)

        print('started renderer clients:\n%s' % pprint_dict(self.sim_clients))

    def _stop_renderer_clients(self):
        for map_name, clients in self.sim_clients.items():
            for client in clients:
                client.stop()

    def _render_agent_view(self, map_name, **kwargs):
        import cv2

        # Randomly choose a simulation client
        client_idx = self.rng.randint(len(self.sim_clients[map_name]))
        sim = self.sim_clients[map_name][client_idx]

        cam_pos = self.agent.local_to_global(self.camera_pos)

        def _render():
            # Send the complete state information in one call because the servers may be load-balanced.
            # Send state information in multiple calls may cause them to be distributed to into multiple
            # servers, which causes inconsistency.
            return sim.RenderAndGetScreenBuffer(
                cam_pos[0], cam_pos[1], self.agent.heading, -1, self.camera_z)

        if _LOG_RENDER_TIME:
            if not hasattr(self, 'accum_render_time'):
                self.accum_render_time = 0
                self.render_count = 0

            start_time = time.time()
            img = _render()
            self.accum_render_time += time.time() - start_time
            self.render_count += 1

            if self.render_count % 100 == 0:
                print('avg render_time', self.accum_render_time / 100)
                self.accum_render_time = 0
        else:
            img = _render()

        # img is H x W x 3 uint8
        img = cv2.resize(img,
                         (self.net_config.input_resolution, self.net_config.input_resolution),
                         cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))

        if self.random_lighting:
            return _randomize_lighting(self.rng, img)

        return img

    def _worker_init(self):
        print('worker_init pid', os.getpid())
        self.is_worker = True
        self._start_renderer_clients()

    def __del__(self):
        if not self.is_worker:
            # This should be only called by the dataset object in the main thread.
            print('stopping render servers (pid %d)' % os.getpid())
            self.stop_renderer_servers()
        else:
            # This is mostly likely never called because worker process gets terminated
            # directly without cleanup.
            print('stopping render clients (pid %d)' % os.getpid())
            self._stop_renderer_clients()


class DatasetVisualRecording(data.Dataset):
    def __init__(self, data_dir, agent_name,
                 ignore_goal=False,
                 random_goal=False,
                 no_waypoint=False,
                 random_vel=False,
                 random_lighting=False,
                 net_config=None,
                 load_to_mem=False,
                 max_traj=-1):
        import glob

        self.data_dir = data_dir
        self.ignore_goal = ignore_goal
        self.random_goal = random_goal
        self.no_waypoint = no_waypoint
        self.random_vel = random_vel
        self.net_config = net_config
        self.load_to_mem = load_to_mem

        assert not random_lighting
        assert not no_waypoint
        assert not load_to_mem
        assert max_traj == -1

        dirs = os.listdir(data_dir)
        samples = []

        for d in dirs:
            if os.path.isdir(os.path.join(data_dir, d)):
                img_files = glob.glob(os.path.join(data_dir, d, '*.tiff'))
                for fn in img_files:
                    basename = os.path.splitext(os.path.basename(fn))[0]
                    meta_file = os.path.join(data_dir, d, '%s.txt' % basename)
                    meta_txt = open(meta_file).read()
                    kvs = [l.split(':') for l in meta_txt.split('\n')]
                    meta = dict()
                    for kv in kvs:
                        if len(kv) == 2:
                            meta[kv[0]] = eval(kv[1])
                    samples.append((fn, meta))

        self.samples = samples
        print('number of samples: ', len(samples))

        self.sample_weights = [1.0 for _ in range(len(samples))]

        self.agent = agent_factory.agents_dict[agent_name]()

        self.rng = None
        self.first = True
        self.opened = False

    def _make(self, sample):
        raise NotImplementedError

    def _set_agent_state(self, meta):
        pos, heading, velocity_global, angular_velocity, goal_local, obstacles_local = (
            np.array(meta['pos'], np.float32),
            meta['heading'],
            np.array(meta['velocity'], np.float32),
            meta['angular_velocity'],
            np.array(meta['goal_local'], np.float32),
            np.array(meta['obstacles_local'], np.float32),
        )

        if self.random_vel:
            velocity_global = _randomize_velocity(self.rng, velocity_global)
            angular_velocity = _randomize_angular_velocity(self.rng, angular_velocity)

        self.agent.pos = pos
        self.agent.velocity = velocity_global
        self.agent.heading = heading
        self.agent.angular_velocity = angular_velocity
        self.agent.obstacles_local = obstacles_local

        if self.random_goal:
            goal_local = _randomize_goal(self.rng, goal_local)

        if self.ignore_goal:
            goal_local = None

        if not self.ignore_goal:
            self.agent.goals_local = [goal_local]

    def __getitem__(self, idx):
        if self.first:
            self.rng = np.random.RandomState(12345 + (idx % 1000000) * 666)
            self.first = False

        return self._make(self.samples[idx])

    def __len__(self):
        return len(self.samples)


class DatasetVisualGibson2(DatasetVisual):
    def __init__(self,
                 assets_dir,
                 output_resolution,
                 camera_pos=(0.0, 0.0), camera_z=1.0, h_fov=None, v_fov=None,
                 gpu_device=None,
                 persistent_server_cfg_file=None,
                 **kwargs):
        """
        :param gpu_device: can be either an integer or a list of integers
        :param render_resolution: the resolution of rendered images from Gibson
        :param camera_pos: the camera position w.r.t the agent's coordinate system
        :param camera_z: the height of camera
        :param h_fov, v_fov: control field of view of rendered images. If None
               default to agent.lidar_fov
        :param n_filler_server: number of filler servers to use
        :param n_sim_per_map: number of simulation servers for each map
        """
        if not os.path.isdir(assets_dir):
            raise RuntimeError("directory %s doesn't exist" % assets_dir)
        self.assets_dir = assets_dir

        super(DatasetVisualGibson2, self).__init__(**kwargs)

        self.output_resolution = output_resolution
        self.h_fov = h_fov
        self.v_fov = v_fov

        self.camera_pos = np.array(camera_pos, np.float32)
        self.camera_z = camera_z

        if hasattr(gpu_device, '__iter__'):
            self.gpus = list(gpu_device)
        elif gpu_device is not None:
            self.gpus = [gpu_device]
        else:
            self.gpus = []

        self.sim_servers = {}
        self.sim_clients = {}

        if persistent_server_cfg_file is None:
            persistent_server_cfg_file = get_default_persistent_server_config()

        self.persistent_server_cfg_file = persistent_server_cfg_file

        # key is map name. value is a list of server address in zmq format.
        self.persistent_servers = {}
        if persistent_server_cfg_file is not None:
            self.persistent_servers = self._parse_persistent_server(persistent_server_cfg_file)

        self.g.update({
            'persistent_servers': self.persistent_servers
        })

        self.is_worker = False

        # Read floor heights. This is important because some maps have nonzero floor heights.
        floor_heights = {}
        for map_name in self.map_names:
            floor_heights[map_name] = self._get_floor_height(map_name)
        self.floor_heights = floor_heights

        # Add persistent servers
        for map_name, addrs in self.persistent_servers.items():
            if map_name not in self.sim_servers and len(addrs) > 0:
                self.sim_servers[map_name] = []
            for addr in addrs:
                self.sim_servers[map_name].append((None, addr))

    def render_traj(self, traj, map_name='unspecified'):
        imgs = []
        for sample in traj:
            self._set_agent_state(sample)
            imgs.append(self._render_agent_view(map_name))
        return imgs

    def _parse_persistent_server(self, cfg_file):
        return yaml.load(open(cfg_file), Loader=yaml.SafeLoader)

    def _get_floor_height(self, map_name):
        meta_file = os.path.join(self.assets_dir, map_name, 'floorplan.yaml')
        if not os.path.isfile(meta_file):
            raise ValueError('Cannot find meta file %s' % meta_file)
        meta = yaml.load(open(meta_file).read(), Loader=yaml.SafeLoader)
        return meta['ref_z']

    def _start_renderer_clients(self):
        from rmp_nav.simulation.gibson2_sim_client import Gibson2SimClient
        
        # There are issues when I let each renderer client create its own context and socket when
        # there are large number of environments. In practice one context and socket is sufficient.
        self.rc_context = zmq.Context()

        for map_name, servers in self.sim_servers.items():
            if map_name == 'all':
                # This server can handle all maps. We need to set client's  to make sure
                # requests are correctly routed.
                for sim_proc, addr in servers:
                    socket = self.rc_context.socket(zmq.REQ)
                    socket.connect(addr)

                    for identity in self.map_names:
                        client = Gibson2SimClient(self.rc_context, socket)
                        client.start(create_server=False, server_addr=addr,
                                     identity=identity.encode())

                        if identity not in self.sim_clients:
                            self.sim_clients[identity] = []

                        self.sim_clients[identity].append(client)
            else:
                if map_name not in self.sim_clients:
                    self.sim_clients[map_name] = []

                for sim_proc, addr in servers:
                    client = Gibson2SimClient(self.rc_context, self.rc_socket)
                    client.start(create_server=False, server_addr=addr)
                    self.sim_clients[map_name].append(client)

        print('started renderer clients:\n%s' % pprint_dict(self.sim_clients))

    def _stop_renderer_clients(self):
        for map_name, clients in self.sim_clients.items():
            for client in clients:
                client.stop()

    def _render_agent_view(self, map_name, camera_z=None, h_fov=None, v_fov=None):
        # Randomly choose a simulation client
        client_idx = self.rng.randint(len(self.sim_clients[map_name]))
        sim = self.sim_clients[map_name][client_idx]

        cam_pos = self.agent.local_to_global(self.camera_pos)

        if camera_z is None:
            camera_z = self.camera_z

        if h_fov is None:
            h_fov = self.h_fov

        if v_fov is None:
            v_fov = self.v_fov

        def _render():
            # Send the complete state information in one call because the servers may be load-balanced.
            # Send state information in multiple calls may cause them to be distributed to into multiple
            # servers, which causes inconsistency.
            return sim.RenderAndGetScreenBuffer(
                cam_pos[0], cam_pos[1], self.agent.heading, camera_z + self.floor_heights[map_name],
                self.output_resolution, self.output_resolution, h_fov, v_fov)

        if _LOG_RENDER_TIME:
            if not hasattr(self, 'accum_render_time'):
                self.accum_render_time = 0
                self.render_count = 0

            start_time = time.time()
            img = _render()
            self.accum_render_time += time.time() - start_time
            self.render_count += 1

            if self.render_count % 100 == 0:
                print('avg render_time', self.accum_render_time / 100)
                self.accum_render_time = 0
        else:
            img = _render()

        # img is H x W x 3 uint8
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))

        if self.random_lighting:
            return _randomize_lighting(self.rng, img)

        return img

    def _worker_init(self):
        print('worker_init pid', os.getpid())
        self.is_worker = True
        self._start_renderer_clients()

    def __del__(self):
        if self.is_worker:
            # This is mostly likely never called because worker process gets terminated
            # directly without cleanup.
            print('stopping render clients (pid %d)' % os.getpid())
            self._stop_renderer_clients()
