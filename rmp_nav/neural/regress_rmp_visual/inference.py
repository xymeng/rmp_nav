import torch
import numpy as np
import yaml
from . import networks
from . import math_utils
from ...common.utils import pprint_dict


def _combine(accel_pred, metric_pred, jacobians, extra_metrics):
    B = []
    C = []

    for i in range(len(jacobians)):
        jacobian = jacobians[i]
        metric = metric_pred[i]
        accel = accel_pred[i]
        B.append(np.linalg.multi_dot((jacobian.T, metric, accel)))
        C.append(np.linalg.multi_dot((jacobian.T, metric, jacobian)))

    for jacobian, metric, ddx in extra_metrics:
        B.append(np.linalg.multi_dot((jacobian.T, metric, ddx)))
        C.append(np.linalg.multi_dot((jacobian.T, metric, jacobian)))

    B = np.sum(B, axis=0)
    C = np.sum(C, axis=0)

    accel = np.dot(np.linalg.pinv(C), B)
    return accel


def make_nets(specs, device):
    ret = {}
    for net_name, spec in specs.items():
        net_class = getattr(networks, spec['class'])
        net_args = spec.get('net_kwargs', {})
        net = net_class(**net_args).to(device)
        ret[net_name] = net
    return ret


class NeuralRMP(object):
    def __init__(self, weights_file, n_control_points=4, device='cuda'):
        state_dict = torch.load(weights_file, map_location='cpu')
        print('loaded %s' % weights_file)
        g = state_dict.get('global_args', {})
        print('global args:')
        print(pprint_dict(g))
        self.g = g

        if isinstance(g.model_spec, dict):
            nets = make_nets(g.model_spec, device)
        else:
            nets = make_nets(yaml.load(open(g.model_spec).read()), device)

        for name, net in nets.items():
            net.load_state_dict(state_dict['nets'][name])
            net.train(False)

        self.device = device
        self.nets = nets

        self.n_control_points = n_control_points
        self.control_point_accels = None
        self.control_point_metrics = None

    def _run_model(self, img, wp_local, vel_local, angular_vel):
        with torch.no_grad():
            nets = self.nets
            img_th = torch.as_tensor(img, dtype=torch.float32, device=self.device)
            wp_th = torch.as_tensor(wp_local, dtype=torch.float32, device=self.device)
            vel_th = torch.as_tensor(vel_local, dtype=torch.float32, device=self.device)
            angular_vel_th = torch.as_tensor(angular_vel, dtype=torch.float32, device=self.device)

            img_feature = nets['img_encoder'](img_th)
            goal_feature = nets['wp_encoder'](wp_th)
            vel_feature = nets['vel_encoder'](vel_th)
            angular_vel_feature = nets['angular_vel_encoder'](angular_vel_th)

            feature = torch.cat([img_feature, goal_feature, vel_feature, angular_vel_feature],
                                dim=-1)

            pred_accel, pred_metric = nets['rmp_regressor'](feature)

            pred_accel = pred_accel.data.cpu().numpy().reshape((self.n_control_points, 2)) / self.g.accel_output_scale
            pred_metric = pred_metric.data.cpu().numpy().reshape((self.n_control_points, 2, 2)) / self.g.metric_output_scale

            if self.g.get('log_metric', False):
                pred_metric = np.array([math_utils.exp_psd_matrix(m) for m in pred_metric], np.float32)

            return pred_accel, pred_metric

    def compute_optimal_accel(
            self, img, wp_local, vel_local, angular_vel, jacobians, extra_metrics):
        '''
        :param extra_metrics: a list of tuples (jacobian, metric, acceleration) to be combined.
        :return:
        '''

        accel_pred, metric_pred = self._run_model(
            img[None, :].astype(np.float32),
            wp_local[None, :].astype(np.float32) * self.g.waypoint_scale,
            vel_local[None, :].astype(np.float32),
            np.atleast_1d(angular_vel)[None, :].astype(np.float32))

        self.control_point_accels = accel_pred
        self.control_point_metrics = metric_pred
        return _combine(accel_pred, metric_pred, jacobians, extra_metrics)
