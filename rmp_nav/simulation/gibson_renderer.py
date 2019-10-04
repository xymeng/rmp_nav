# This file is modified from the original gibson renderer

import numpy as np
import ctypes as ct
import os
import argparse
import transforms3d
import zmq

from ..gibson import assets
from ..gibson.core.render import utils
from ..gibson.data.datasets import ViewDataSet3D
from .gibson_filler import Filler


assets_file_dir = os.path.dirname(assets.__file__)


class Renderer:
    ROTATION_CONST = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])

    def __init__(self, port, imgs, depths, target, target_poses,
                 use_filler=True,
                 filler_server_addr=None,
                 internal_filler_gpu=None,
                 windowsz=256,
                 env=None):
        self.env = env
        self.roll, self.pitch, self.yaw = 0, 0, 0
        self.quat = [1, 0, 0, 0]
        self.x, self.y, self.z = 0, 0, 0
        self.fps = 0
        self.mousex, self.mousey = 0.5, 0.5
        self.org_pitch, self.org_yaw, self.org_roll = 0, 0, 0
        self.org_x, self.org_y, self.org_z = 0, 0, 0
        self.clickstart = (0,0)
        self.mousedown  = False
        self.overlay    = False
        self.show_depth = False
        self._context_phys = zmq.Context()
        self._context_mist = zmq.Context()
        self._context_dept = zmq.Context()      ## Channel for smoothed depth
        self._context_norm = zmq.Context()      ## Channel for smoothed depth
        self._context_semt = zmq.Context()

        self.env = env

        self.port = port
        self.socket_mist = self._context_mist.socket(zmq.REQ)
        self.socket_mist.connect("tcp://localhost:{}".format(port))

        self.target_poses = target_poses
        self.pose_locations = np.array([tp[:3,-1] for tp in self.target_poses])

        self.relative_poses = [np.dot(np.linalg.inv(tg), self.target_poses[0]) for tg in target_poses]

        self.imgs = imgs
        self.depths = depths
        self.target = target
        self.model = None
        self.old_topk = set([])
        self.k = 5
        self.use_filler = use_filler

        self.showsz = windowsz
        self.capture_count = 0

        self.show = np.zeros((self.showsz, self.showsz, 3), dtype='uint8')
        self.show_rgb = np.zeros((self.showsz, self.showsz, 3), dtype='uint8')
        self.show_semantics = np.zeros((self.showsz, self.showsz, 3), dtype='uint8')

        self.show_prefilled = np.zeros((self.showsz, self.showsz, 3), dtype='uint8')
        self.surface_normal = np.zeros((self.showsz, self.showsz, 3), dtype='uint8')

        self.semtimg_count = 0

        self.filler_server_addr = filler_server_addr
        self.internal_filler_gpu = internal_filler_gpu

        def init_render_lib():
            from rmp_nav.gibson import core
            try:
                self.cuda_pc = np.ctypeslib.load_library(
                    os.path.join(os.path.dirname(core.render.__file__),
                                 'render_cuda_f'), '.')
            except:
                print("Error: cuda renderer is not loaded, rendering will not work")
                raise

        if use_filler:
            if not filler_server_addr:
                print('filler server addr not specified. creating filler internally.')
                init_render_lib()
                self.filler = Filler(windowsz, gpu=internal_filler_gpu)
            else:
                self._context_filler_server = zmq.Context()
                self.socket_filler = self._context_filler_server.socket(zmq.REQ)
                try:
                    self.socket_filler.connect(filler_server_addr)
                except:
                    print('bad filler server addr', filler_server_addr)
                    raise
        else:
            init_render_lib()

    def _close(self):
        self._context_dept.destroy()
        self._context_mist.destroy()
        self._context_norm.destroy()
        self._context_phys.destroy()

    def _getViewerRelativePose(self):
        cpose = np.eye(4)
        cpose[:3, :3] = transforms3d.quaternions.quat2mat(self.quat)
        cpose[0, -1] = self.x
        cpose[1, -1] = self.y
        cpose[2, -1] = self.z
        return cpose

    def render(self, rgbs, depths, pose, poses, target_pose, show):
        v_cam2world = target_pose
        p = (v_cam2world).dot(np.linalg.inv(pose))
        p = p.dot(np.linalg.inv(Renderer.ROTATION_CONST))
        s = utils.mat_to_str(p)

        self.socket_mist.send_string(s)
        mist_msg = self.socket_mist.recv()

        wo, ho = self.showsz * 4, self.showsz * 3

        # Calculate height and width of output image, and size of each square face
        h = wo//3
        w = 2*h
        n = ho//3

        pano = False
        if pano:
            opengl_arr = np.frombuffer(mist_msg, dtype=np.float32).reshape((h, w))
        else:
            opengl_arr = np.frombuffer(mist_msg, dtype=np.float32).reshape((n, n))

        def _render_pc(opengl_arr, imgs_pc, show_pc):
            poses_after = [
                pose.dot(np.linalg.inv(poses[i])).astype(np.float32)
                for i in range(len(imgs_pc))]

            self.cuda_pc.render(
                ct.c_int(self.internal_filler_gpu),
                ct.c_int(len(imgs_pc)),
                ct.c_int(imgs_pc[0].shape[0]),
                ct.c_int(imgs_pc[0].shape[1]),
                ct.c_int(self.showsz),
                ct.c_int(self.showsz),
                imgs_pc.ctypes.data_as(ct.c_void_p),
                depths.ctypes.data_as(ct.c_void_p),
                np.asarray(poses_after, dtype=np.float32).ctypes.data_as(ct.c_void_p),
                show_pc.ctypes.data_as(ct.c_void_p),
                opengl_arr.ctypes.data_as(ct.c_void_p),
                ct.c_float(self.env.config["fov"])
            )

        if self.filler_server_addr:
            self.socket_filler.send_pyobj(
                ['args', opengl_arr, rgbs, depths, pose, poses, self.env.config["fov"]])
            show[:] = self.socket_filler.recv_pyobj()
        else:
            _render_pc(opengl_arr, rgbs, show)
            if self.use_filler:
                show[:] = self.filler.fill(show, opengl_arr)

    def setNewPose(self, pose):
        new_pos, new_quat = pose[0], pose[1]
        self.x, self.y, self.z = new_pos
        self.quat = new_quat
        v_cam2world = self.target_poses[0]
        v_cam2cam = self._getViewerRelativePose()
        # Using OpenBLAS can result in horrible high cpu usage.
        #if not hasattr(self, 'render_cpose'):
        self.render_cpose = np.linalg.inv(np.linalg.inv(v_cam2world).dot(v_cam2cam).dot(Renderer.ROTATION_CONST))
        
    def getAllPoseDist(self, pose):
        pose_distances = np.linalg.norm(self.pose_locations - pose[0].reshape(1,3), axis = 1)
        return pose_distances, self.pose_locations

    def renderOffScreen(self, pose, k_views=None):
        if k_views is not None:
            all_dist, _ = self.getAllPoseDist(pose)
            k_views = (np.argsort(all_dist))[:self.k]

        if set(k_views) != self.old_topk:
            self.imgs_topk = np.array([self.imgs[i] for i in k_views])
            self.depths_topk = np.array([self.depths[i] for i in k_views]).flatten()

            self.relative_poses_topk = [self.relative_poses[i] for i in k_views]
            self.old_topk = set(k_views)

        self.render(self.imgs_topk, self.depths_topk, self.render_cpose.astype(np.float32),
                    self.relative_poses_topk, self.target_poses[0], self.show)

        self.show = np.reshape(self.show, (self.showsz, self.showsz, 3))
        self.show_rgb = self.show

    def find_best_k_views(self, all_dist):
        least_order = (np.argsort(all_dist))
        return least_order[:5]
    
    
def heading_to_quat(heading):
    '''
    :param heading: a scalar indicating the horizontal heading.
           The reference direction is x = 1, y = 0
    :return:
    '''
    # wxyz
    hack = (np.cos(np.pi / 2 * 0.5), np.sin(np.pi / 2 * 0.5), 0, 0)
    # Rotate np.pi / 2 clockwise because gibson's reference heading direction is x = 0, y = 1.
    heading_q = (np.cos((heading - np.pi / 2) * 0.5), 0, 0, np.sin((heading - np.pi / 2) * 0.5))
    return transforms3d.quaternions.qmult(heading_q, hack)


if __name__ == '__main__':
    import cv2
    import torch
    from torch import nn
    from .gibson.learn.completion import CompletionNet

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--datapath', required=True, help='dataset path')
    parser.add_argument('--model_id', type=str, default=0, help='model id')
    parser.add_argument('--model', type=str, default='', help='path of model')

    opt = parser.parse_args()

    model = None
    if opt.model != '':
        comp = CompletionNet(norm=nn.BatchNorm2d)
        comp = torch.nn.DataParallel(comp).cuda()
        comp.load_state_dict(torch.load(opt.model))
        model = comp.module
        model.eval()

    print(model)

    fov = np.pi / 3 * 2.0

    import easydict
    env = easydict.EasyDict({
        'config': {'resolution': 256, 'use_filler': True, 'display_ui': True, 'fov': fov}
    })

    d = ViewDataSet3D(root=opt.datapath,
                      transform=np.array,
                      mist_transform=np.array,
                      seqlen=2, off_3d=False, train=False, env=env)

    scene_dict = dict(zip(d.scenes, range(len(d.scenes))))
    if not opt.model_id in scene_dict.keys():
        print("model not found")
    else:
        scene_id = scene_dict[opt.model_id]

    uuids, rts = d.get_scene_info(scene_id)

    targets = []
    sources = []
    source_depths = []
    poses = []

    for k, v in uuids:
        data = d[v]
        source = data[0][0]
        target = data[1]
        target_depth = data[3]
        source_depth = data[2][0]
        pose = data[-1][0].numpy()

        target = cv2.resize(target, None, fx=0.5, fy=0.5)
        target_depth = cv2.resize(target_depth, None, fx=0.5, fy=0.5)

        targets.append(target)

        poses.append(pose)
        sources.append(target)
        source_depths.append(target_depth)

    pose = [np.array([-14.30079059,   4.72863504,   0.3]),
            [0.0019725768752885675, -0.0009562987956104478, 0.7095259744305187, 0.7046758730377446]]

    # quaternion is in wxyz format

    # Fix camera orientation to make it point forward.
    hack = (np.cos(np.pi / 2 * 0.5), np.sin(np.pi / 2 * 0.5), 0, 0)

    heading = -np.pi / 4
    heading_q = (np.cos(heading * 0.5), 0, 0, np.sin(heading * 0.5))

    pose = [np.array([0, 0, 1.0]), transforms3d.quaternions.qmult(heading_q, hack)]

    port = 12345

    def launch_render_proc():
        file_dir = os.path.dirname(os.path.abspath(__file__))
        bin_dir = os.path.join(file_dir, 'gibson/core/channels/depth_render')

        use_egl = os.environ.get('USE_EGL', 1)

        cmd = './depth_render --port {} --modelpath {} --GPU {} -w {} -h {} -f {} --egl {}'.format(
            port, '../../../assets/dataset/%s' % opt.model_id, 1,
            env.config.resolution, env.config.resolution, fov / np.pi * 180.0, use_egl)

        # Some of the libraries are in the binary directory. Append that directory
        # to the load path.
        osenv = os.environ.copy()

        ld_lib_path = env.get('LD_LIBRARY_PATH', '')
        osenv['LD_LIBRARY_PATH'] = ld_lib_path + ':%s' % bin_dir

        import shlex, subprocess
        return subprocess.Popen(
            shlex.split(cmd), shell=False, cwd=bin_dir, env=osenv)

    render_proc = launch_render_proc()

    # port, imgs, depths, target, target_poses, scale_up
    renderer = Renderer(port, sources, source_depths, target, rts,
                        windowsz=env.config.resolution, env=env, use_filler=False)

    all_dist, all_pos = renderer.getAllPoseDist(pose)
    top_k = renderer.find_best_k_views(all_dist)

    renderer.setNewPose(pose)
    renderer.renderOffScreen(pose, top_k)

    for i in range(100):
        cv2.imshow('rgb', renderer.show_rgb)
        cv2.waitKey(1)
