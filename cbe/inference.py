import torch
import copy
import yaml
import zmq
import subprocess
import os
import sys
import time
import numpy as np
import msgpack
import msgpack_numpy
msgpack_numpy.patch()

from rmp_nav.common.utils import pprint_dict

from . import networks


def make_nets(specs, device):
    ret = {}
    for net_name, spec in specs.items():
        net_class = getattr(networks, spec['class'])
        net_args = spec.get('net_kwargs', {})
        net = net_class(no_weight_init=True, **net_args).to(device)
        ret[net_name] = net
    return ret


class TrackerBase(object):
    def __init__(self, weights_file, device='cuda'):
        self.weights_file = weights_file
        if weights_file:
            self._load(weights_file, device)

        self.device = device
        self.h = None

    def _as_tensor(self, data, dtype):
        return torch.as_tensor(data, dtype=dtype).to(device=self.device, non_blocking=True)

    def _load(self, weights_file, device):
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

        self.nets = nets

    def reload(self, new_weights_file):
        # Reload weights. Used for dagger training.
        if self.weights_file == new_weights_file:
            return

        if self.nets is not None:
            del self.nets

        print('reload')
        self._load(new_weights_file, self.device)


class TrackerCBE(TrackerBase):
    def __init__(self, *args, **kwargs):
        super(TrackerCBE, self).__init__(*args, **kwargs)
        self.state_cache = {}

    def count_parameters(self):
        assert self.g.model_variant == 'context_lite_v5'
        nets = ['embedding_img_pair_encoder',
                'embedding_recurrent',
                'embedding_bottleneck',
                'img_encoder',
                'feature_map_pair_encoder',
                'conv_encoder',
                'recurrent',
                'progress_regressor',
                'waypoint_regressor']
        total_params = 0
        for _ in nets:
            n = sum(p.numel() for p in self.nets[_].parameters())
            print(_, n)
            total_params += n
        return total_params

    def compute_traj_embedding(self, obs):
        nets = self.nets
        with torch.no_grad():
            obs_th = self._as_tensor(np.array(obs), torch.float)
            obs2_th = torch.cat([obs_th[0][None], obs_th[:-1]])
            obs_features = nets['embedding_img_pair_encoder'](obs_th, obs2_th)
            hs, _ = nets['embedding_recurrent'](obs_features.unsqueeze(1))
            return nets['embedding_bottleneck'](hs)[:, 0].cpu().numpy()

    def _make_context_feature(self, ob, target):
        nets = self.nets
        _, ob_c, ob_h, ob_w = ob.size()

        context_len = target.size(1)

        mv = self.g.model_variant

        if mv == 'default' or mv == 'context_lite_v5':
            target_feature = nets['img_encoder'](target.view(context_len, ob_c, ob_h, ob_w))
            # target_feature is of size context_len x d x s x s
            ob_feature = nets['img_encoder'](ob).expand_as(target_feature)
            pair_feature = nets['feature_map_pair_encoder'](ob_feature, target_feature)  # context_len x d
            conv_feature = nets['conv_encoder'](pair_feature.unsqueeze(0).transpose(1, 2))
            assert conv_feature.dim() == 2, conv_feature.size()
            return conv_feature

        else:
            raise RuntimeError('Unknown model variant %s' % mv)

    def _embedding_encode_step(self, ob, caller_id):
        # Encode one step
        state = self.state_cache.get(caller_id, dict())

        nets = self.nets
        with torch.no_grad():
            ob_th = self._as_tensor(np.array(ob), torch.float).unsqueeze(0)

            last_ob = state.get('last_ob', None)
            if last_ob is None:
                last_ob = ob_th

            obs_features = nets['embedding_img_pair_encoder'](ob_th, last_ob)
            state['last_ob'] = ob_th

            h = state.get('h', None)
            hs, new_h = nets['embedding_recurrent'](obs_features.unsqueeze(1), h)

            state['h'] = new_h
            self.state_cache[caller_id] = state

            assert hs.size(0) == 1
            h = hs[-1, 0]

            return self.nets['embedding_bottleneck'](h[None])[0].cpu().numpy()

    def encode_step(self, odom, ob, caller_id='default'):
        return self._embedding_encode_step(odom, ob, caller_id)

    def _step(self, start, goal, traj_embedding, ob, h, init_start_feature, caller_id):
        nets = self.nets
        with torch.no_grad():
            ob_th = self._as_tensor(ob, torch.float).unsqueeze(0)
            embedding_th = self._as_tensor(traj_embedding, torch.float).unsqueeze(0)
            start_th = self._as_tensor(start, torch.float).unsqueeze(0)
            goal_th = self._as_tensor(goal, torch.float).unsqueeze(0)
            rollout_embedding = self._as_tensor(self._embedding_encode_step(ob, caller_id), torch.float).unsqueeze(0)

            if self.g.attractor:
                if self.g.n_context_frame == 0:
                    # when n_context_frame = 0, the context dimension is not present.
                    # We add it back here.
                    start_th = start_th.unsqueeze(1)
                    goal_th = goal_th.unsqueeze(1)

                if init_start_feature is None:
                    init_start_feature = self._make_context_feature(ob_th, start_th)

                ob_goal_feature = self._make_context_feature(ob_th, goal_th)

                if self.g.get('no_embedding', False):
                    feature = torch.cat([init_start_feature, ob_goal_feature], dim=-1)
                else:
                    feature = torch.cat(
                        [init_start_feature, ob_goal_feature, rollout_embedding, embedding_th],
                        dim=-1)
            else:
                feature = torch.cat([rollout_embedding, embedding_th], dim=-1)

            out, next_h = nets['recurrent'](feature.unsqueeze(1), h)

            pred_progress = nets['progress_regressor'](out.view(-1))
            pred_waypoint = nets['waypoint_regressor'](out.view(-1))

            return pred_progress.item(), pred_waypoint.squeeze(0).data.cpu().numpy(), \
                   next_h, init_start_feature

    def step(self, start, goal, traj_embedding, ob):
        return self.step2(start, goal, traj_embedding, ob, 'default')

    def step2(self, start, goal, traj_embedding, ob, caller_id):
        # Keep track of latent state using caller_id
        state = self.state_cache.get(caller_id, {})
        h = state.get('tracker_h', None)
        init_start_feature = state.get('init_start_feature', None)

        pred_progress, pred_wp, h, init_start_feature = self._step(
            start, goal, traj_embedding, ob, h, init_start_feature, caller_id)

        self.state_cache[caller_id]['tracker_h'] = h
        self.state_cache[caller_id]['init_start_feature'] = init_start_feature

        return pred_progress, pred_wp

    def reset(self):
        self.reset2('default')

    def reset2(self, caller_id):
        if caller_id in self.state_cache:
            self.state_cache.pop(caller_id)


class TrackerRPF(TrackerBase):
    def __init__(self, *args, **kwargs):
        super(TrackerRPF, self).__init__(*args, **kwargs)
        self.state_cache = {}

    def compute_traj_embedding(self, obs):
        # obs is N x C x H x W
        # In RPF, the embedding is featurized observations
        nets = self.nets
        with torch.no_grad():
            obs_th = self._as_tensor(np.array(obs), torch.float)
            obs_features = nets['img_encoder'](obs_th)
            return obs_features.cpu().numpy()

    def _step(self, start, goal, traj_embedding, ob, h, caller_id):
        nets = self.nets
        with torch.no_grad():
            ob_th = self._as_tensor(ob, torch.float).unsqueeze(0)
            embedding_th = self._as_tensor(traj_embedding, torch.float).unsqueeze(0)
            idxs = torch.arange(len(traj_embedding), dtype=torch.float, device=self.device).unsqueeze(0)

            state = self.state_cache.get(caller_id, dict())

            eta = state.get('eta', 0)
            h = state.get('h', None)
            ws = torch.exp(-torch.abs(idxs - eta))  # 1 x traj_len

            ob_feature = nets['img_encoder'](ob_th)

            feature_dim = embedding_th.dim() - 2

            attn_features = embedding_th * ws.view(ws.size() + feature_dim * (1,))
            attn_features = torch.sum(attn_features, dim=1)

            if self.g.model_variant == 'default':
                features = torch.cat([attn_features, ob_feature], dim=-1)
            elif self.g.model_variant == 'feature_map':
                features = nets['feature_map_pair_encoder'](attn_features, ob_feature)
                # This conv_encoder just acts like a MLP. This is to keep the overall feature
                # extraction part the same as context_lite_v5
                features = nets['conv_encoder'](features.unsqueeze(-1))

            h = nets['recurrent_cell'](features, h)
            pred_waypoint = nets['waypoint_regressor'](h)
            pred_increment = torch.tanh(nets['increment_regressor'](h)) + 1
            eta = eta + pred_increment

            eta = torch.clamp(eta, 0.0, len(traj_embedding) - 1)

            state['h'] = h
            state['eta'] = eta
            self.state_cache[caller_id] = state

            return 0.0, pred_waypoint.squeeze(0).data.cpu().numpy(), h

    def step(self, start, goal, traj_embedding, ob):
        return self.step2(start, goal, traj_embedding, ob, 'default')

    def step2(self, start, goal, traj_embedding, ob, caller_id):
        # Keep track of latent state using caller_id
        state = self.state_cache.get(caller_id, dict())
        h = state.get('h', None)
        pred_progress, pred_wp, h = self._step(start, goal, traj_embedding, ob, h, caller_id)
        return pred_progress, pred_wp

    def reset(self):
        self.reset2('default')

    def reset2(self, caller_id):
        if caller_id in self.state_cache:
            self.state_cache.pop(caller_id)


class Client(object):
    """
    Run inference in a separate process. This is useful if you want to use it in multi-process
    dataloaders, where gpu cannot be easily used in forked processes (unless you don't initialize
    cuda before forking).
    """
    def __init__(self, weights_file, server_addr=None, identity=None):
        """
        :param server_addr: launch a new server if None.
        """
        if weights_file:
            self.weights_file = weights_file
            state_dict = torch.load(weights_file, map_location='cpu')
            self.g = copy.deepcopy(state_dict.get('global_args', {}))

        if server_addr is None:
            self.addr = 'ipc:///tmp/tracker_inference-frontend-%s' % str(time.time())
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

        # If not None it will be used to connect to a worker with the same identity on the server side.
        self.identity = identity

    @classmethod
    def launch_server(cls, weights_file, addr, devices='cuda', n_worker=1, with_id=False):
        osenv = os.environ.copy()
        osenv['MKL_NUM_THREADS'] = '1'
        osenv['OPENBLAS_NUM_THREADS'] = '1'
        osenv['OMP_NUM_THREADS'] = '1'
        return subprocess.Popen([
            sys.executable, '-u', '-m',
            # TODO: can we programmatically infer the module path?
            'cbe.inference_server',
            '--weights_file', weights_file,
            '--addr', addr,
            '--devices', devices,
            '--n_worker', '%d' % n_worker,
            '--with_id', '%s' % with_id
        ], env=osenv)

    # FIXME: assume that fields are valid.
    @property
    def n_frame(self):
        return self.g.n_frame

    @property
    def frame_interval(self):
        return self.g.frame_interval

    def _send(self, msg, **kwargs):
        if self.identity is not None:
            self.socket.send_multipart([self.identity, msgpack.packb(msg, use_bin_type=True)], **kwargs)
        else:
            self.socket.send(msgpack.packb(msg, use_bin_type=True), **kwargs)

    def _recv(self):
        return msgpack.unpackb(self.socket.recv(), raw=False)

    def reload(self, weights_file):
        self._send(['reload', weights_file], flags=zmq.NOBLOCK)
        return self._recv()

    def step2(self, *args):
        self._send(['step2'] + list(args), flags=zmq.NOBLOCK)
        return self._recv()

    def step3(self, *args):
        self._send(['step3'] + list(args), flags=zmq.NOBLOCK)
        return self._recv()

    def reset(self):
        self._send(['reset'], flags=zmq.NOBLOCK)
        return self._recv()

    def reset2(self, *args):
        self._send(['reset2'] + list(args), flags=zmq.NOBLOCK)
        return self._recv()

    def compute_traj_embedding(self, *args):
        self._send(['compute_traj_embedding'] + list(args), flags=zmq.NOBLOCK)
        return self._recv()

    def __del__(self):
        if self.proc is not None:
            self._send(['exit'])
            # self._recv()


def find_tracker(weights_file):
    # FIXME: this is purely based on heuristics and may not be correct!
    state_dict = torch.load(weights_file, map_location='cpu')
    g = state_dict.get('global_args', {})

    if 'rpf' in g.model_file:
        return TrackerRPF

    return TrackerCBE
