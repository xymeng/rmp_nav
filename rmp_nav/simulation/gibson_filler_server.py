from __future__ import print_function
import zmq
import sys
import gflags
from .gibson_filler import Filler
import os
import subprocess
import shlex
import numpy as np
from rmp_nav.gibson.core import render as render_package
import ctypes as ct


def find_free_port():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return int(port)


def LaunchServer(resolution, gpu=0, port=None, use_ipc=False, ipc_endpoint=''):
    '''
    :param gpu: this should be the GPU with a display attached so that we can draw using it.
    :return:
    '''
    if not use_ipc:
        if port is None:
            port = find_free_port()
        addr = 'tcp://localhost:%d' % port
    else:
        addr = 'ipc://%s' % ipc_endpoint

    cmd = '%s %s --resolution=%d --addr=%s --gpu=%d' % (
        sys.executable, __file__, resolution, addr, gpu)

    return subprocess.Popen(shlex.split(cmd)), addr


def render_pc(cuda_pc, gpu, resolution, opengl_arr, rgbs, depths, pose, poses, show_pc, fov):
    poses_after = [pose.dot(np.linalg.inv(poses[i])).astype(np.float32) for i in range(len(rgbs))]

    cuda_pc.render(ct.c_int(gpu),
                   ct.c_int(len(rgbs)),
                   ct.c_int(rgbs[0].shape[0]),
                   ct.c_int(rgbs[0].shape[1]),
                   ct.c_int(resolution),
                   ct.c_int(resolution),
                   rgbs.ctypes.data_as(ct.c_void_p),
                   depths.ctypes.data_as(ct.c_void_p),
                   np.asarray(poses_after, dtype=np.float32).ctypes.data_as(ct.c_void_p),
                   show_pc.ctypes.data_as(ct.c_void_p),
                   opengl_arr.ctypes.data_as(ct.c_void_p),
                   ct.c_float(fov))


if __name__ == '__main__':
    import time

    try:
        cuda_pc = np.ctypeslib.load_library(
            os.path.join(os.path.dirname(render_package.__file__), 'render_cuda_f'), '.')
    except:
        print("Error: cuda renderer is not loaded, rendering will not work")
        raise

    gflags.DEFINE_integer('resolution', None, '')
    gflags.DEFINE_string('addr', '', '')
    gflags.DEFINE_integer('gpu', 0, '')

    FLAGS = gflags.FLAGS
    FLAGS(sys.argv)

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(FLAGS.addr)

    res = FLAGS.resolution

    filler = Filler(res, gpu=FLAGS.gpu)

    prefilled_img = np.zeros((res, res, 3), dtype=np.uint8)

    while True:
        msg = socket.recv_pyobj()
        ret = None

        if msg[0] == 'exit':
            socket.send_pyobj('ok')
            break

        elif msg[0] == 'args':
            opengl_arr, imgs, depths, pose, poses, fov = msg[1:]
            render_pc(cuda_pc, FLAGS.gpu, FLAGS.resolution, opengl_arr, imgs, depths, pose, poses,
                      prefilled_img, fov)
            ret = filler.fill(prefilled_img, opengl_arr)

        success = False
        while not success:
            try:
                socket.send_pyobj(ret, zmq.NOBLOCK)
                success = True
            except zmq.error.Again as e:
                time.sleep(0.01)

    socket.unbind(FLAGS.addr)
    socket.close()
    context.term()
    print('gibson filler server %s terminated' % FLAGS.addr)
