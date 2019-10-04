from __future__ import print_function
from .gibson_renderer import Renderer, heading_to_quat
import zmq
import sys
import gflags
import numpy as np
import cv2
import os

from ..gibson.data.datasets import ViewDataSet3D
from ..common.utils import get_project_root

import msgpack
import msgpack_numpy
msgpack_numpy.patch()


gflags.DEFINE_string('assets_dir', '', 'Will use the default path if unspecified.')
gflags.DEFINE_string('scene_id', '', '')
gflags.DEFINE_float('h_fov', None, '')
gflags.DEFINE_float('v_fov', None, '')
gflags.DEFINE_integer('resolution', 256, '')
gflags.DEFINE_integer('gpu', 0, '')
gflags.DEFINE_boolean('use_filler', True, 'False to disable the filler.')
gflags.DEFINE_string('filler_server_addr', '',
                     'If not specified the filler will be created internally.')
gflags.DEFINE_string('addr', '', 'The address of this server. In zmq format.')
gflags.DEFINE_boolean('bind', True,
                      'Bind the socket to the specified address instead of connecting to it. '
                      'This is the case for REQ/REP pattern. If you use ROUTER/DEALER then you'
                      'might need to set it to False.')


FLAGS = gflags.FLAGS
FLAGS(sys.argv)


context = zmq.Context()
socket = context.socket(zmq.REP)

if FLAGS.bind:
    socket.bind(FLAGS.addr)
else:
    socket.connect(FLAGS.addr)


def find_free_port():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return int(port)


class Server(object):
    def __init__(self):
        self.process = None
        self.socket = None
        self.renderer = None
        self.render_proc = None

        self.h_fov = None
        self.v_fov = None
        self.aspect_ratio = None
        self.x = None
        self.y = None
        self.z = None
        self.angle = None

    def start(self, datapath, scene_id, resolution, h_fov, v_fov, gpu, filler_server_addr=None):
        '''
        :param datapath:
        :param scene_id: e.g. space2
        :param resolution: resolution of rendered image
        :param h_fov, v_fov: in radians
        :param filler_server_addr: the address of the filler server.
        :return:
        '''
        import easydict

        self.h_fov = h_fov
        self.v_fov = v_fov

        fov = max(self.v_fov, self.h_fov)
        self.aspect_ratio = np.tan(0.5 * self.h_fov) / np.tan(0.5 * self.v_fov)

        env = easydict.EasyDict({
            'config': {'resolution': resolution, 'use_filler': True, 'display_ui': True, 'fov': fov}
        })

        d = ViewDataSet3D(root=datapath,
                          transform=np.array,
                          mist_transform=np.array,
                          off_3d=False,
                          off_pc_render=True,
                          train=False,
                          overwrite_fofn=True,
                          env=env)

        scene_dict = dict(zip(d.scenes, range(len(d.scenes))))
        scene = scene_dict[scene_id]
        uuids, rts = d.get_scene_info(scene)

        targets = []
        sources = []
        source_depths = []
        poses = []

        import time
        start_time = time.time()

        print('total number of images:', len(uuids))
        print('%s loading images...' % scene_id)

        for k, v in uuids:
            data = d[v]
            target = data[1]
            target_depth = data[3]
            pose = data[-1][0]

            target = cv2.resize(target, None, fx=0.5, fy=0.5)
            target_depth = cv2.resize(target_depth, None, fx=0.5, fy=0.5)

            targets.append(target)

            poses.append(pose)
            sources.append(target)
            source_depths.append(target_depth)

        print('%s loading completed %f sec' % (scene_id, time.time() - start_time))

        port = find_free_port()

        def launch_render_proc():
            bin_dir = os.path.join(get_project_root(), 'rmp_nav/gibson/core/channels/depth_render')
            use_egl = os.environ.get('USE_EGL', 1)
            screen = os.environ.get('SCREEN', 0)
            cmd = './depth_render --port {} --modelpath {} --GPU {} -w {} -h {} -f {} --egl {} --screen {}'.format(
                port,
                get_project_root() + '/rmp_nav/gibson/assets/dataset/%s' % scene_id,
                gpu,
                env.config.resolution,
                env.config.resolution,
                fov / np.pi * 180.0,
                use_egl,
                screen)

            # Some of the libraries are in the binary directory. Append that directory
            # to the load path.
            osenv = os.environ.copy()

            ld_lib_path = env.get('LD_LIBRARY_PATH', '')
            osenv['LD_LIBRARY_PATH'] = ld_lib_path + ':%s' % bin_dir

            import shlex, subprocess
            self.render_proc = subprocess.Popen(
                shlex.split(cmd), shell=False, cwd=bin_dir, env=osenv)

        launch_render_proc()

        self.renderer = Renderer(
            port, sources, source_depths, target, rts,
            windowsz=env.config.resolution,
            env=env,
            use_filler=FLAGS.use_filler,
            filler_server_addr=filler_server_addr,
            internal_filler_gpu=gpu  # Only used if filler_server_addr is not specified
        )

    def stop(self):
        if self.render_proc is not None:
            self.render_proc.kill()
            self.render_proc = None
            self.renderer = None

    def SetZ(self, z):
        self.z = z

    def SetXY(self, x, y):
        self.x = x
        self.y = y

    def SetAngle(self, angle):
        self.angle = angle

    def SetFOV(self, fov):
        # fov is fixed after initialization.
        pass

    def Render(self):
        pose = [
            np.array([self.x, self.y, self.z]),
            heading_to_quat(self.angle)
        ]
        renderer = self.renderer

        all_dist, all_pos = renderer.getAllPoseDist(pose)
        top_k = renderer.find_best_k_views(all_dist)

        renderer.setNewPose(pose)
        renderer.renderOffScreen(pose, top_k)

    def GetScreenBuffer(self):
        return self._crop_to_fit_aspect_ratio(self.renderer.show_rgb)

    def RenderAndGetScreenBuffer(self, x, y, angle, fov, z=None):
        """
        :param fov: unused. Only for compatibility.
        :return:
        """
        if not hasattr(self, 'accum_time'):
            self.accum_count = 0
            self.accum_time = 0

        import time
        start_time = time.time()

        self.SetXY(x, y)
        if z is not None:
            self.SetZ(z)

        self.SetAngle(angle)
        self.Render()
        img = self.GetScreenBuffer()

        self.accum_time += time.time() - start_time
        self.accum_count += 1
        if self.accum_count == 100:
            print('sim server frame time:', (self.accum_time / self.accum_count))
            self.accum_time = 0
            self.accum_count = 0

        return img

    def _crop_to_fit_aspect_ratio(self, img):
        # Crop the rendered image into the desired aspect ratio
        crop_h, crop_w = img.shape[:2]

        if self.aspect_ratio is not None:
            if self.aspect_ratio >= 1:
                crop_h = int(crop_w // self.aspect_ratio)
            else:
                crop_w = int(crop_h * self.aspect_ratio)

        h_offset = (img.shape[0] - crop_h) // 2
        w_offset = (img.shape[1] - crop_w) // 2

        return img[h_offset:h_offset+crop_h, w_offset:w_offset+crop_w]

    def __del__(self):
        self.stop()


server = Server()

if FLAGS.assets_dir != '':
    assets_dir = FLAGS.assets_dir
else:
    assets_dir = get_project_root() + '/rmp_nav/gibson/assets/dataset'

server.start(assets_dir, FLAGS.scene_id, FLAGS.resolution, FLAGS.h_fov, FLAGS.v_fov, FLAGS.gpu,
             FLAGS.filler_server_addr)


def exec_cmd(msg):
    func_name = msg[0]
    args = msg[1:]
    ret = getattr(server, func_name)(*args)
    return ret


while True:
    msg = msgpack.unpackb(socket.recv(), raw=False)
    ret = None

    if msg[0] == 'exit':
        socket.send_pyobj('ok')
        break

    elif msg[0] == 'cmd_list':
        for m in msg[1:]:
            ret = exec_cmd(m)
    else:
        ret = exec_cmd(msg)

    socket.send(msgpack.packb(ret, use_bin_type=True))


if FLAGS.bind:
    socket.unbind(FLAGS.addr)
else:
    socket.disconnect(FLAGS.addr)

socket.close()
context.term()
print('gibson simulation server terminated')
