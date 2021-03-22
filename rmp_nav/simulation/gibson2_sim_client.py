from __future__ import print_function
import shlex, subprocess
import zmq
import os
import sys
from rmp_nav.common.utils import pprint_dict

import msgpack
import msgpack_numpy
msgpack_numpy.patch()


def LaunchServer(assets_dir, scene_id, resolution, h_fov, v_fov, gpu, addr=None):
    '''
    :param gpu: this should be the GPU with a display attached so that we can draw using it.
    :return:
    '''
    if addr is None:
        import time
        addr = 'ipc:///tmp/gibson2_sim_server.%f' % time.time()

    server_script = os.path.join(os.path.dirname(__file__), 'gibson2_sim_server.py')
    cmd = '%s %s --assets_dir=%s --scene_id=%s --resolution=%d --h_fov=%f  --v_fov=%f --gpu=%d  --addr=%s' % (
        sys.executable, server_script, assets_dir, scene_id, resolution, h_fov, v_fov, gpu, addr)

    env = os.environ.copy()
    env['OPENBLAS_NUM_THREADS'] = '1'
    env['MKL_NUM_THREADS'] = '1'

    return subprocess.Popen(shlex.split(cmd)), addr


class Gibson2SimClient(object):
    def __init__(self, context=None, socket=None):
        self.process = None
        self.identity = None  # Used to connect to a multi-identity load balancer
        if context is None:
            self.context = zmq.Context()
        else:
            self.context = context

        self.socket = socket
        self.server_addr = None

    def start(self,
              assets_dir=None, scene_id=None, resolution=256, h_fov=0.0, v_fov=0.0, gpu=0,
              create_server=True, server_addr=None, identity=None):
        """
        :param server_addr: if None will create a server internally.
        :return:
        """

        assert self.process is None

        if create_server:
            self.process, self.server_addr = LaunchServer(
                assets_dir, scene_id, resolution, h_fov, v_fov, gpu,
            )
        else:
            assert server_addr is not None
            self.server_addr = server_addr

        self.create_server = create_server
        self.scene_id = scene_id

        if self.socket is None:
            self.external_socket = False
            self.socket = self.context.socket(zmq.REQ)
            self.socket.connect(self.server_addr)
        else:
            self.external_socket = True

        self.identity = identity

    def __repr__(self):
        return pprint_dict({
            'scene_id': self.scene_id,
            'create_server': self.create_server,
            'server_addr': self.server_addr,
            'identity': self.identity
        })

    def _send(self, msg):
        if self.identity is not None:
            self.socket.send_multipart([self.identity, msgpack.packb(msg, use_bin_type=True)])
        else:
            self.socket.send(msgpack.packb(msg, use_bin_type=True))

    def _recv(self):
        return msgpack.unpackb(self.socket.recv(), raw=False)

    def stop(self):
        if self.process is not None:
            self._send(['exit'])
            self._recv()
            self.process.communicate()
            self.process = None

        if not self.external_socket:
            self.socket.disconnect(self.server_addr)
            self.socket = None

    def SetZ(self, z):
        self._send(['SetZ', z])
        _ = self._recv()

    def SetXY(self, x, y):
        self._send(['SetXY', x, y])
        _ = self._recv()

    def SetAngle(self, angle):
        self._send(['SetAngle', angle])
        _ = self._recv()

    def SetFOV(self, fov):
        self._send(['SetFOV', fov])
        _ = self._recv()

    def Render(self):
        self._send(['Render'])
        _ = self._recv()

    def GetScreenBuffer(self):
        self._send(['GetScreenBuffer'])
        return self._recv()

    def RenderAndGetScreenBuffer(self, x, y, angle, z=None, out_width=None, out_height=None, h_fov=None, v_fov=None):
        '''
        Do everything in one call to reduce latency
        Also it is necessary to pass the entire state with a load balancer.
        :param z: default to None which uses current z value.
        '''
        self._send(['RenderAndGetScreenBuffer', x, y, angle, z, out_width, out_height, h_fov, v_fov])
        return self._recv()

    def PlaceBoxObject(self, x, y, z):
        self._send(['PlaceBoxObject', x, y, z])
        return self._recv()

    def PlaceTrashcan(self, x, y, z):
        self._send(['PlaceTrashcan', x, y, z])
        return self._recv()

    def __del__(self):
        if self.process is not None:
            self.stop()
