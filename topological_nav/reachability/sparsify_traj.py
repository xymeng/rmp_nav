from easydict import EasyDict
import numpy as np
import torch
import tabulate
import time
import os
import sys
import subprocess
import copy
import zmq
import yaml
import msgpack
import msgpack_numpy
msgpack_numpy.patch()

from . import networks


def make_nets(specs, device):
    ret = {}
    for net_name, spec in specs.items():
        net_class = getattr(networks, spec['class'])
        net_args = spec.get('net_kwargs', {})
        net = net_class(no_weight_init=True, **net_args).to(device)
        ret[net_name] = net
    return ret


class TrajSparsifier(object):
    def __init__(self, weights_file, device='cuda'):
        super(TrajSparsifier, self).__init__()

        state_dict = torch.load(weights_file, map_location='cpu')
        print('loaded %s' % weights_file)
        g = state_dict.get('global_args', {})
        print('global args:')
        print(tabulate.tabulate(g.items()))

        if isinstance(g.model_spec, dict):
            nets = make_nets(g.model_spec, device)
        else:
            nets = make_nets(yaml.load(open(g.model_spec).read()), device)

        for name, net in nets.items():
            net.load_state_dict(state_dict['nets'][name])
            net.train(False)

        self.device = device
        self.g = g
        self.nets = nets

    def get_ob_repr(self, imgs, idx):
        return imgs[idx]

    def get_dst_repr(self, imgs, idx):
        """
        :param imgs: images representing the trajectory
        :param idx:  index of the dst image
        :return: the representation of the dst image. By default it is simply the imgs[idx], but it
                 could be overridden to support other representations (e.g., multiple frames).
        """
        return imgs[idx]

    def sparsify(self, img_seqs, thres):
        seqs = []

        src_idx = 0
        dst_idx = 1

        while dst_idx < len(img_seqs):
            idx2 = dst_idx
            rs = []

            while idx2 < len(img_seqs):
                value = self.predict_reachability(img_seqs[src_idx], img_seqs[idx2])[0][0]
                rs.append((idx2, value))
                if value > thres:
                    idx2 += 1
                else:
                    break

            print('%d:' % src_idx, rs)

            idx2 -= 1
            idx2 = max(idx2, dst_idx)

            seqs.append(idx2)

            src_idx = idx2
            dst_idx = idx2 + 1

        # Add the last image if needed
        if seqs[-1] < len(img_seqs) - 1:
            seqs.append(len(img_seqs) - 1)

        return seqs

    def predict_reachability(self, ob, goal):
        with torch.no_grad():
            ob_th = torch.as_tensor(ob, dtype=torch.float, device=self.device)
            goal_th = torch.as_tensor(goal, dtype=torch.float, device=self.device)
            ob_feature = self.nets['img_encoder'](ob_th.unsqueeze(0))
            goal_feature = self.nets['img_encoder'](goal_th.unsqueeze(0))
            feature = torch.cat([ob_feature, goal_feature], dim=1)
            reachability_pred = torch.sigmoid(self.nets['reachability_regressor'](feature))
            return reachability_pred.item()

    def predict_reachability_batch(self, obs, goals):
        raise NotImplementedError


class TrajSparsifierMultiframeDst(TrajSparsifier):
    @property
    def n_frame(self):
        return self.g.n_frame

    @property
    def frame_interval(self):
        return self.g.frame_interval

    def get_dst_repr(self, imgs, idx):
        """
        :param imgs: trajectory images
        :param idx: position of the target
        :return: a model-dependent representation of the target
        """
        return self._make_dst_seq(imgs, idx)

    def get_dst_repr_single(self, img):
        """
        Create a dst repr with only a single image.
        """
        if self.g.get('future', False):
            return [img] * (self.g.n_frame * 2 - 1)
        else:
            return [img] * self.g.n_frame

    def _make_dst_seq(self, imgs, idx):
        goal_seq = []
        for i in range(self.g.n_frame):
            goal_seq.append(imgs[max(idx - i * self.g.frame_interval, 0)])
        goal_seq = goal_seq[::-1]

        if self.g.get('future', False):
            for i in range(self.g.n_frame - 1):
                goal_seq.append(imgs[min(idx + (i + 1) * self.g.frame_interval, len(imgs) - 1)])
        return goal_seq

    def _make_dst_seq2(self, imgs, idx):
        goal_seq = []
        for i in range(self.g.n_frame):
            goal_seq.append(imgs[max(idx - i * self.g.frame_interval, 0)])
        goal_seq = goal_seq[::-1]

        if self.g.get('future', False):
            for i in range(self.g.n_frame - 1):
                goal_seq.append(imgs[min(idx + (i + 1) * self.g.frame_interval, len(imgs) - 1)])
        return goal_seq

    def compute_sparsification_ratio(self, seq_len, sparse_idxs):
        """
        :param seq_len: length of the original sequence
        :param sparse_idxs: sparse indices
        """
        mask = [0.0 for _ in range(seq_len)]
        for idx in sparse_idxs:
            for j in range(self.g.n_frame):
                mask[max(idx - j * self.g.frame_interval, 0)] = 1.0
            if self.g.get('future', False):
                for j in range(self.g.n_frame - 1):
                    mask[min(idx + (j + 1) * self.g.frame_interval, seq_len - 1)] = 1.0
        return sum(mask) / float(seq_len)

    def sparsify(self, imgs, thres):
        # TODO: should move all images to gpu here.
        seqs = []

        # First image is always kept
        seqs.append(0)

        src_idx = 0
        dst_idx = 1

        while dst_idx < len(imgs):
            idx2 = dst_idx
            rs = []

            while idx2 < len(imgs):
                goal_seq = self._make_dst_seq(imgs, idx2)
                value = self.predict_reachability(imgs[src_idx], goal_seq)
                rs.append((idx2, value))
                if value > thres:
                    idx2 += 1
                else:
                    break

            print('%d:' % src_idx, rs)

            idx2 -= 1
            idx2 = max(idx2, dst_idx)

            seqs.append(idx2)

            src_idx = idx2
            dst_idx = idx2 + 1

        # Add the last image if needed
        if seqs[-1] < len(imgs) - 1:
            seqs.append(len(imgs) - 1)

        return seqs

    def predict_reachability(self, ob, goal_seq):
        # import time
        # start_time = time.time()
        ret = self.predict_reachability_batch([ob], [goal_seq])[0]
        # print('predict_reachability time:', time.time() - start_time)
        return ret

    def predict_reachability_batch(self, obs, goals, batch_size=64):
        nets = self.nets
        with torch.no_grad():
            model_variant = self.g.get('model_variant', 'default')

            def helper(ob_batch, goal_batch):
                # as_tensor() is very slow when passing in a list of np arrays, but is 30X faster
                # when wrapping the list with np.array().
                if isinstance(ob_batch[0], np.ndarray):
                    ob_batch = np.array(ob_batch)
                elif isinstance(ob_batch[0], torch.Tensor):
                    if not isinstance(ob_batch, torch.Tensor):
                        ob_batch = torch.stack(ob_batch)
                else:
                    raise RuntimeError('Unsupported datatype: %s' % type(ob_batch[0]))

                if isinstance(goal_batch[0][0], np.ndarray):
                    goal_batch = np.array(goal_batch)
                elif isinstance(goal_batch[0][0], torch.Tensor):
                    if not isinstance(ob_batch, torch.Tensor):
                        goal_batch = torch.stack(goal_batch)
                else:
                    raise RuntimeError('Unsupported datatype: %s' % type(goal_batch[0]))

                ob_batch = torch.as_tensor(ob_batch).to(non_blocking=True, device=self.device)
                goal_batch = torch.as_tensor(goal_batch).to(non_blocking=True, device=self.device)

                win_size, c, h, w = goal_batch.size()[1:]

                if model_variant == 'future_pair':
                    assert goal_batch.size(1) == self.g.n_frame * 2 - 1
                    ob_batch2 = ob_batch.unsqueeze(1).expand_as(goal_batch).contiguous()
                    pair_features = nets['img_pair_encoder'](
                        ob_batch2.view(-1, c, h, w),
                        goal_batch.view(-1, c, h, w)).view(ob_batch.size(0), -1)
                    pred_reachability = torch.sigmoid(
                        nets['reachability_regressor'](pair_features)).squeeze(1)
                    return pred_reachability

                elif model_variant == 'future_pair_conv':
                    assert goal_batch.size(1) == self.g.n_frame * 2 - 1
                    ob_batch2 = ob_batch.unsqueeze(1).expand_as(goal_batch).contiguous()
                    pair_features = nets['img_pair_encoder'](
                        ob_batch2.view(-1, c, h, w),
                        goal_batch.view(-1, c, h, w)).view(ob_batch.size(0), win_size, -1)
                    conv_feature = nets['conv_encoder'](pair_features.transpose(1, 2))
                    pred_reachability = torch.sigmoid(
                        nets['reachability_regressor'](conv_feature)).squeeze(1)
                    return pred_reachability

                else:
                    raise RuntimeError('Unsupported model variant %s' % model_variant)

            assert len(obs) == len(goals)
            n = len(obs)

            results = []
            n_remaining = n
            while n_remaining > 0:
                results.append(helper(obs[n - n_remaining: n - n_remaining + batch_size],
                                      goals[n - n_remaining: n - n_remaining + batch_size]))
                n_remaining -= batch_size
            return torch.cat(results, dim=0).data.cpu().numpy()


class TrajSparisfierClient(object):
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
            self.addr = 'ipc:///tmp/reachability_inference-frontend-%s' % str(time.time())
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

    @staticmethod
    def launch_server(weights_file, addr):
        osenv = os.environ.copy()
        osenv['MKL_NUM_THREADS'] = '1'
        osenv['OPENBLAS_NUM_THREADS'] = '1'
        osenv['OMP_NUM_THREADS'] = '1'
        return subprocess.Popen([
            sys.executable, '-u', '-m',
            # TODO: can we programmatically infer the module path?
            'topological_nav.reachability.inference_server',
            '--weights_file', weights_file,
            '--addr', addr
        ])

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

    def predict_reachability(self, *args):
        self._send(['predict_reachability'] + list(args), flags=zmq.NOBLOCK)
        return self._recv()

    def __del__(self):
        if self.proc is not None:
            self._send(['exit'])


def find_sparsifier(weights_file: str):
    if 'multiframe_dst' in weights_file:
        return TrajSparsifierMultiframeDst
    else:
        raise ValueError('Cannot find sparsifier for %s' % weights_file)
