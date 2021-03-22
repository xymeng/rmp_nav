import cv2
import zmq
import sys
import gflags
import numpy as np
import math
import os
from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer, MeshRendererSettings

import msgpack
import msgpack_numpy
msgpack_numpy.patch()

from rmp_nav.common.utils import get_gibson2_asset_dir


gflags.DEFINE_string('assets_dir', '', '')
gflags.DEFINE_string('scene_id', '', '')
gflags.DEFINE_float('h_fov', None, '')
gflags.DEFINE_float('v_fov', None, '')
gflags.DEFINE_integer('resolution', 256, '')
gflags.DEFINE_boolean('shadow', False, 'Enable shadow in rendering.')
gflags.DEFINE_integer('gpu', 0, '')
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

        self.box_obj = None
        self.trashcan_obj = None

    def start(self, datapath, scene_id, resolution, h_fov, v_fov, gpu, enable_shadow):
        '''
        :param datapath:
        :param scene_id: e.g. space2
        :param resolution: resolution of rendered image
        :param resolution: output resolution
        :param h_fov, v_fov: in radians
        :param gpu: gpu index.
        :return:
        '''
        self.h_fov = h_fov
        self.v_fov = v_fov

        fov = max(self.v_fov, self.h_fov)
        self.aspect_ratio = np.tan(0.5 * self.h_fov) / np.tan(0.5 * self.v_fov)
        model_path = os.path.join(datapath, scene_id, 'mesh_z_up.obj')

        settings = MeshRendererSettings(enable_shadow=enable_shadow, msaa=False, enable_pbr=False, optimized=True)
        renderer = MeshRenderer(width=resolution, height=resolution, device_idx=gpu, rendering_settings=settings)
        renderer.load_object(model_path)
        renderer.add_instance(0)
        renderer.set_fov(np.rad2deg(fov))

        self.datapath = datapath
        self.renderer = renderer

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
        self.renderer.set_fov(fov)

    def Render(self):
        camera_pose = np.array([self.x, self.y, self.z])
        view_direction = np.array([math.cos(self.angle), math.sin(self.angle), 0.0])
        self.renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 1])
        out = self.renderer.render(('rgb',))
        img = np.concatenate(out, axis=1)[:, :, :3]
        return img

    def RenderAndGetScreenBuffer(self, x, y, angle, z=None, out_width=None, out_height=None, h_fov=None, v_fov=None):
        """
        :param z:
        :param out_width, out_height: optionally resizing the images. Useful for saving network bandwidth.
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

        if h_fov is not None and v_fov is not None:
            # Renderer always renders a square image
            self.renderer.set_fov(np.rad2deg(max(h_fov, v_fov)))
            aspect_ratio = np.tan(0.5 * h_fov) / np.tan(0.5 * v_fov)
        else:
            aspect_ratio = self.aspect_ratio

        img = self.Render()
        img = self._crop_to_fit_aspect_ratio(img, aspect_ratio)

        if out_width is not None and out_height is not None:
            # FIXME: assume downsizing
            img = cv2.resize(img, (out_width, out_height), interpolation=cv2.INTER_AREA)

        img = (img * 255).astype(np.uint8)

        self.accum_time += time.time() - start_time
        self.accum_count += 1
        if self.accum_count == 100:
            print('sim server frame time:', (self.accum_time / self.accum_count))
            self.accum_time = 0
            self.accum_count = 0

        return img

    def PlaceBoxObject(self, x, y, z):
        renderer = self.renderer
        if self.box_obj is None:
            # Note that scale and transform_pos are used to transform the mesh.
            renderer.load_object(self.datapath + '/objects/box/box.obj',
                                 scale=(0.10, 0.10, 0.3), transform_pos=(0.0, -0.25, 0.5))
            renderer.add_instance(1)
            self.box_obj = renderer.instances[-1]

        self.box_obj.set_position((x, y, z))

    def PlaceTrashcan(self, x, y, z):
        renderer = self.renderer
        if self.trashcan_obj is None:
            # Note that scale and transform_pos are used to transform the mesh, not for positioning the object.
            # 0.3m x 0.3m x 1.0m
            renderer.load_object(self.datapath + '/objects/trashcan/model.obj',
                                 scale=(0.026, 0.026, 0.057), transform_pos=(0.0, -0.15, 0.0))
            renderer.add_instance(1)
            self.trashcan_obj = renderer.instances[-1]

        self.trashcan_obj.set_position((x, y, z))

    def _crop_to_fit_aspect_ratio(self, img, aspect_ratio):
        # Crop the rendered image into the desired aspect ratio
        crop_h, crop_w = img.shape[:2]

        if aspect_ratio >= 1:
            crop_h = int(crop_w // self.aspect_ratio)
        else:
            crop_w = int(crop_h * self.aspect_ratio)

        h_offset = (img.shape[0] - crop_h) // 2
        w_offset = (img.shape[1] - crop_w) // 2

        return img[h_offset:h_offset+crop_h, w_offset:w_offset+crop_w]

    def __del__(self):
        self.stop()


server = Server()

if not FLAGS.assets_dir:
    assets_dir = get_gibson2_asset_dir()
else:
    assets_dir = FLAGS.assets_dir

server.start(assets_dir, FLAGS.scene_id, FLAGS.resolution, FLAGS.h_fov, FLAGS.v_fov, FLAGS.gpu, FLAGS.shadow)


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
print('gibson2 simulation server terminated')
