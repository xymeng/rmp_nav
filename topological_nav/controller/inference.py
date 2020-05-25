import copy
import os
import sys
import time
import yaml
import torch
import torch.nn.functional as F
import math
import random
import subprocess
import zmq
import numpy as np
import msgpack
import msgpack_numpy
msgpack_numpy.patch()

from . import networks
from rmp_nav.common.utils import pprint_dict


def make_nets(specs, device):
    ret = {}
    for net_name, spec in specs.items():
        net_class = getattr(networks, spec['class'])
        net_args = spec.get('net_kwargs', {})
        net = net_class(no_weight_init=True, **net_args).to(device)
        ret[net_name] = net
    return ret


class ControllerBase(object):
    def __init__(self, weights_file, device='cuda'):
        state_dict = torch.load(weights_file, map_location='cpu')
        print('loaded %s' % weights_file)
        g = state_dict.get('global_args', {})
        print('global args:')
        print(pprint_dict(g))

        self.g = g
        self.weights_file = weights_file

        if isinstance(g.model_spec, dict):
            nets = make_nets(g.model_spec, device)
        else:
            nets = make_nets(yaml.load(open(g.model_spec).read(), Loader=yaml.SafeLoader), device)

        for name, net in nets.items():
            net.load_state_dict(state_dict['nets'][name])
            net.train(False)

        self.device = device
        self.nets = nets


class ControllerMultiframeDst(ControllerBase):
    @property
    def n_frame(self):
        return self.g.n_frame

    @property
    def frame_interval(self):
        return self.g.frame_interval

    def predict_proximity(self, ob, goal_seq):
        nets = self.nets
        ob_th = torch.as_tensor(ob, dtype=torch.float, device=self.device)
        goal_seq_th = torch.as_tensor(goal_seq, dtype=torch.float, device=self.device)

        if self.g.model_variant == 'future_pair':
            assert len(goal_seq) == self.n_frame * 2 - 1, len(goal_seq)
            ob_th2 = ob_th.unsqueeze(0).expand_as(goal_seq_th)
            pair_features = nets['img_pair_encoder'](ob_th2, goal_seq_th).view(-1)
            proximity_pred = torch.sigmoid(nets['proximity_regressor'](pair_features.unsqueeze(0)))
        elif self.g.model_variant == 'future_pair_conv':
            ob_th2 = ob_th.unsqueeze(0).expand_as(goal_seq_th)
            pair_features = nets['img_pair_encoder'](ob_th2, goal_seq_th).view(1, len(goal_seq), -1)
            conv_feature = nets['conv_encoder'](pair_features.transpose(1, 2))
            proximity_pred = torch.sigmoid(nets['proximity_regressor'](conv_feature).squeeze(0))
            return proximity_pred.data.cpu().numpy()
        else:
            raise RuntimeError('Unsupport model variant %s' % self.g.model_variant)

        return proximity_pred.data.cpu().numpy()[0]

    def predict_waypoint(self, ob, goal_seq):
        nets = self.nets

        with torch.no_grad():
            ob_th = torch.as_tensor(ob, dtype=torch.float, device=self.device)
            goal_seq_th = torch.as_tensor(goal_seq, dtype=torch.float, device=self.device)

            if self.g.model_variant == 'attention':
                assert len(goal_seq) == self.n_frame
                ob_feature = nets['img_encoder'](ob_th.unsqueeze(0))
                goal_features = nets['img_encoder'](goal_seq_th)
                goal_temporal_feature = nets['seq_encoder'](goal_features.unsqueeze(0))
                final_feature = torch.cat([ob_feature, goal_temporal_feature], dim=1)
                wp_pred = nets['wp_regressor'](final_feature).squeeze(0)
                assert wp_pred.size() == (2,)
                return wp_pred.data.cpu().numpy()
            elif self.g.model_variant == 'future':
                assert len(goal_seq) == self.n_frame * 2 - 1, len(goal_seq)
                ob_feature = nets['img_encoder'](ob_th.unsqueeze(0))
                goal_features = nets['img_encoder'](goal_seq_th)
                past_features = goal_features[:self.n_frame]
                future_features = goal_features[self.n_frame:]
                past_temporal_features = nets['seq_encoder'](past_features.unsqueeze(0))
                future_temporal_features = nets['seq_encoder'](future_features.unsqueeze(0))
                final_features = torch.cat([ob_feature,
                                            past_temporal_features,
                                            future_temporal_features], dim=-1)
                wp_pred = nets['wp_regressor'](final_features).squeeze(0)
                return wp_pred.data.cpu().numpy()
            elif self.g.model_variant == 'future_stack':
                assert len(goal_seq) == self.n_frame * 2 - 1, len(goal_seq)
                img_stack = torch.cat([ob_th.unsqueeze(0).unsqueeze(0),
                                       goal_seq_th.unsqueeze(0)], dim=1)
                features = nets['stack_encoder'](img_stack)
                wp_pred = nets['wp_regressor'](features).squeeze(0)
                return wp_pred.data.cpu().numpy()
            elif self.g.model_variant == 'future_pair':
                assert len(goal_seq) == self.n_frame * 2 - 1, len(goal_seq)
                ob_th2 = ob_th.unsqueeze(0).expand_as(goal_seq_th)
                pair_features = nets['img_pair_encoder'](ob_th2, goal_seq_th).view(-1)
                wp_pred = nets['wp_regressor'](pair_features.unsqueeze(0)).squeeze(0)
                return wp_pred.data.cpu().numpy()
            elif self.g.model_variant == 'future_pair_conv':
                assert len(goal_seq) == self.n_frame * 2 - 1, len(goal_seq)
                ob_th2 = ob_th.unsqueeze(0).expand_as(goal_seq_th)
                pair_features = nets['img_pair_encoder'](ob_th2, goal_seq_th).view(1, len(goal_seq), -1)
                conv_feature = nets['conv_encoder'](pair_features.transpose(1, 2))
                wp_pred = nets['wp_regressor'](conv_feature).squeeze(0)
                return wp_pred.data.cpu().numpy()
            elif self.g.model_variant == 'future_pair_featurized':
                assert len(goal_seq) == self.n_frame * 2 - 1, len(goal_seq)
                goal_features = nets['img_encoder'](goal_seq_th)
                ob_feature = nets['img_encoder'](ob_th.unsqueeze(0)).expand_as(goal_features).contiguous()
                pair_features = nets['feature_pair_encoder'](ob_feature, goal_features).view(-1)
                wp_pred = nets['wp_regressor'](pair_features.unsqueeze(0)).squeeze(0)
                return wp_pred.data.cpu().numpy()
            else:
                raise RuntimeError('Unsupported model variant %s' % self.g.model_variant)


class Client(object):
    """
    Run inference in a separate process. This is useful if you want to use it in multi-process
    dataloaders, where gpu cannot be easily used in forked processes (unless you don't initialize
    cuda before forking).
    """
    def __init__(self, weights_file, server_addr=None):
        """
        :param server_addr: launch a new server if None.
        """
        self.weights_file = weights_file
        state_dict = torch.load(weights_file, map_location='cpu')
        self.g = copy.deepcopy(state_dict.get('global_args', {}))

        if server_addr is None:
            self.addr = 'ipc:///tmp/controller-frontend-%s' % str(time.time())
            self.proc = self.launch_server(weights_file, self.addr)
        else:
            self.addr = server_addr
            self.proc = None

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)

        while True:
            try:
                self.socket.connect(self.addr)
                break
            except:
                time.sleep(1.0)
        print('connected to %s' % self.addr)

    @classmethod
    def _find_constructor(cls, weights_file):
        cons = find_controller(weights_file)
        return '.'.join([cons.__module__, cons.__name__])

    @classmethod
    def launch_server(cls, weights_file, addr):
        osenv = os.environ.copy()
        osenv['MKL_NUM_THREADS'] = '1'
        osenv['OPENBLAS_NUM_THREADS'] = '1'
        osenv['OMP_NUM_THREADS'] = '1'
        print('class_path', cls._find_constructor(weights_file))
        return subprocess.Popen([
            sys.executable, '-u', '-m',
            # TODO: can we programmatically infer the module path?
            'topological_nav.controller.inference_server',
            '--weights_file', weights_file,
            '--class_path', cls._find_constructor(weights_file),
            '--addr', addr
        ], env=osenv)

    # FIXME: assume that fields are valid.
    @property
    def n_frame(self):
        return self.g.n_frame

    @property
    def frame_interval(self):
        return self.g.frame_interval

    def _send(self, obj, **kwargs):
        self.socket.send(msgpack.packb(obj, use_bin_type=True), **kwargs)

    def _recv(self):
        return msgpack.unpackb(self.socket.recv(), raw=False)

    def predict_waypoint(self, *args):
        self._send(['predict_waypoint'] + list(args), flags=zmq.NOBLOCK)
        return self._recv()

    def __del__(self):
        if self.proc is not None:
            self._send(['exit'])
            # self._recv()


def find_controller(weights_file):
    # FIXME: this purely based on heuristics and may not be correct!
    state_dict = torch.load(weights_file, map_location='cpu')
    g = state_dict.get('global_args', {})

    if 'multiframe_dst' in g.model_file:
        return ControllerMultiframeDst

    return None
