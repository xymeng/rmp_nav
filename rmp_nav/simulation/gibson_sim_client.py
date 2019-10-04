from __future__ import print_function
import shlex, subprocess
import zmq
import os
import sys
from ..common.utils import pprint_dict

import msgpack
import msgpack_numpy
msgpack_numpy.patch()


def LaunchServer(assets_dir, scene_id, resolution, h_fov, v_fov, gpu, filler_server_addr=None, addr=None):
    '''
    :param gpu: this should be the GPU with a display attached so that we can draw using it.
    :return:
    '''
    if addr is None:
        import time
        addr = 'ipc:///tmp/gibson_sim_server.%f' % time.time()

    server_script = os.path.join(os.path.dirname(__file__), 'gibson_sim_server.py')
    cmd = '%s %s --assets_dir=%s --scene_id=%s --resolution=%d --h_fov=%f  --v_fov=%f --gpu=%d  --addr=%s' % (
        sys.executable, server_script, assets_dir, scene_id, resolution, h_fov, v_fov, gpu, addr)

    if filler_server_addr is not None:
        cmd += ' --filler_server_addr=%s' % filler_server_addr

    env = os.environ.copy()
    env['OPENBLAS_NUM_THREADS'] = '1'
    env['MKL_NUM_THREADS'] = '1'

    return subprocess.Popen(shlex.split(cmd)), addr


class GibsonSimClient(object):
    def __init__(self):
        self.process = None
        self.socket = None
        self.identity = None  # Used to connect to a multi-identity load balancer
        self.context = zmq.Context()
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

        self.socket = self.context.socket(zmq.REQ)
        self.identity = identity
        self.socket.connect(self.server_addr)

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
        self.socket.disconnect(self.server_addr)

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

    def RenderAndGetScreenBuffer(self, x, y, angle, fov=0, z=None):
        '''
        Do everything in one call to reduce latency
        Also it is necessary to pass the entire state with a load balancer.
        :param fov: not used.
        :param z: default to None which uses current z value.
        '''

        self._send(['RenderAndGetScreenBuffer', x, y, angle, fov, z])
        return self._recv()

    def __del__(self):
        if self.process is not None:
            self.stop()


if __name__ == '__main__':
    import numpy as np
    import cv2
    import gflags

    gflags.DEFINE_integer('gpu', 0, '')
    gflags.DEFINE_integer('resolution', 128, '')
    gflags.DEFINE_boolean('show', True, '')

    FLAGS = gflags.FLAGS
    FLAGS(sys.argv)

    client = GibsonSimClient()

    # client.start('gibson/assets/dataset', 'space2', resolution=FLAGS.resolution,
    #              h_fov=np.pi / 3 * 2.0, v_fov=np.pi / 3 * 2.0, gpu=FLAGS.gpu)
    client.start(create_server=False, server_addr='tcp://127.0.0.1:5000', identity='house31'.encode())

    import time
    PRINT_INTERVAL = 100
    start_time = time.time()

    for i in range(10000000):
        img = client.RenderAndGetScreenBuffer(0.0, 0.0, i / 100.0, -1, 1.0)
        if i % PRINT_INTERVAL == 0:
            time_per_frame = (time.time() - start_time) / PRINT_INTERVAL
            print('RenderAndGetScreenBuffer: %.3f sec %.2f fps' % (time_per_frame, 1.0 / time_per_frame))
            start_time = time.time()
        # if FLAGS.show:
        #     img = cv2.resize(img, (64, 64))
        #     cv2.imshow('im', img)
        #     cv2.waitKey(1)

    client.stop()
