from __future__ import print_function
import numpy as np
import cv2
import random
import itertools


class SensorBase(object):
    def __init__(self):
        self.map = None

    def set_map(self, map):
        self.map = map


class Lidar(SensorBase):
    def __init__(self, n_depth_ray, fov=np.pi*2.0):
        super(Lidar, self).__init__()
        self.fov = fov
        self.n_depth = n_depth_ray

    def measure(self, position, heading):
        return self.map.get_1d_depth(position, self.n_depth, heading, self.fov)


class GibsonCamera(SensorBase):
    def __init__(self,
                 gpu_idx=0,
                 render_resolution=256,
                 output_resolution=224,
                 h_fov=None, v_fov=None,
                 persistent_servers=None):
        """
        gpu_idx, render_resolution, h_fov, v_fov are ignored if persistent_servers are specified
        """
        super(GibsonCamera, self).__init__()
        self.sim_client = None
        self.persistent_servers = persistent_servers
        self.gpu_idx = gpu_idx
        self.render_resolution = render_resolution
        self.output_resolution = output_resolution
        self.h_fov = h_fov
        self.v_fov = v_fov

    def set_map(self, map):
        from .gibson_sim_client import GibsonSimClient

        if map.__repr__ == self.map.__repr__:
            # Skip re-initialization if possible
            return

        super(GibsonCamera, self).set_map(map)

        if self.sim_client is not None:
            self.sim_client.stop()
            self.sim_client = None

        if map is None:
            return

        print('start gibson sim client, scene id', self.map.scene_id)

        # Find all usable servers
        usable_servers = []
        if self.persistent_servers is not None:
            for key in (self.map.scene_id, 'all'):
                server_addrs = self.persistent_servers.get(key, [])
                usable_servers.extend(zip(itertools.repeat(key), server_addrs))

        print('usable servers:', usable_servers)

        identity = None

        if len(usable_servers) == 0:
            server_addr = None  # No persistent server available, will create one.
        else:
            # Randomly choose a server
            rng = random.Random()  # We don't care about determinism here.
            key, server_addr = rng.choice(usable_servers)
            if key == 'all':
                identity = self.map.scene_id  # Need to set identity for multi-map load balancer.

        self.sim_client = GibsonSimClient()
        self.sim_client.start(
            assets_dir=self.map.assets_dir,
            scene_id=self.map.scene_id,
            resolution=self.render_resolution,
            h_fov=self.h_fov,
            v_fov=self.v_fov,
            gpu=self.gpu_idx,
            create_server=server_addr is None,
            server_addr=server_addr,
            identity=identity.encode()
        )

    def _convert_img(self, img):
        """
        Convert the doom simulator screenbuffer (H x W x 3, uint8) to network input (3 x H x W, float32)
        """
        img = cv2.resize(img, (self.output_resolution, self.output_resolution), cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return img

    def measure(self, position, heading):
        sim_proc = self.sim_client
        x, y, z = position
        img = sim_proc.RenderAndGetScreenBuffer(x, y, heading, z=z)
        return self._convert_img(img)
