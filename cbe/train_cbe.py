import torch
from torch import optim
import gflags
import glob
import numpy as np
import os
import sys
import subprocess
import yaml
import visdom

from rmp_nav.common.utils import get_gibson2_asset_dir

from .dataset import DatasetGibson2Traj, DatasetClient
from . import networks
from . import train_fixture
from . import args
from .args import g


def make_nets(specs, device):
    ret = {}
    for net_name, spec in specs.items():
        net_class = getattr(networks, spec['class'])
        net_args = spec.get('net_kwargs', {})
        net = net_class(**net_args).to(device)
        ret[net_name] = {
            'net': net,
            'opt': getattr(optim, spec['opt'])(net.parameters(), **spec['opt_kwargs'])
        }
    return ret


if __name__ == '__main__':
    torch.manual_seed(931238)
    torch.set_num_threads(1)

    torch.backends.cudnn.benchmark = True

    gflags.DEFINE_string('model_variant', 'default', '')
    gflags.DEFINE_string('dataset_variant', 'gibson2', '')
    gflags.DEFINE_string('dataset_inference_device', 'cuda',
                         'device for the dataset to run neural models to generate training data.')

    gflags.DEFINE_string('robot', 'minirccar_240fov_rmp_v2', '')

    gflags.DEFINE_integer('n_frame_min', 16, '')
    gflags.DEFINE_integer('n_frame_max', 64, '')
    gflags.DEFINE_integer('frame_interval', 3, 'Maximum frame gap.')
    gflags.DEFINE_boolean('rand_frame_interval', True, 'Randomize frame gaps between 1 and frame_interval.')

    gflags.DEFINE_float('demo_camera_z', 1.0, 'Camera height for demonstration trajs.')
    gflags.DEFINE_float('rollout_camera_z', 1.0, 'Camera height for rollout trajs.')
    gflags.DEFINE_float('demo_hfov', 118.6, '')
    gflags.DEFINE_float('demo_vfov', 106.9, '')
    gflags.DEFINE_float('rollout_hfov', 118.6, '')
    gflags.DEFINE_float('rollout_vfov', 106.9, '')

    gflags.DEFINE_boolean('normalize_wp', True, 'Normalize waypoints.')
    gflags.DEFINE_string('progress_loss', 'l1', '')
    gflags.DEFINE_integer('n_context_frame', 0, 'Context length. Only used for context models.')
    gflags.DEFINE_boolean('attractor', True, 'Use start and goal image as attractors.')
    gflags.DEFINE_boolean('no_embedding', False, 'Disable the embedding. For ablation study only.')

    gflags.DEFINE_boolean('dagger', True, 'Enable dagger training.')
    gflags.DEFINE_integer('dagger_epoch', 1, 'Start dagger training from this epoch.')
    gflags.DEFINE_integer('dagger_init', 100000, 'Initial number of dagger samples. (First epoch is 0).')
    gflags.DEFINE_integer('dagger_inc', 100000, 'Increment of dagger samples per epoch.')
    gflags.DEFINE_integer('dagger_max', 200000, 'Increment of dagger samples per epoch.')
    gflags.DEFINE_boolean('dagger_jitter', True, 'Jitter initial location and orientation during dagger training.')

    FLAGS = gflags.FLAGS
    FLAGS(sys.argv)

    g['cmd_line'] = ' '.join(sys.argv)

    args.fill(FLAGS.FlagDict())

    g.model_spec = yaml.load(open(g.model_spec).read(), Loader=yaml.SafeLoader)
    args.set_s(FLAGS.set)

    print('global args:')
    print(args.repr())

    vis = visdom.Visdom(env=g.get('visdom_env', 'main'), server=g.visdom_server, port=g.visdom_port)
    if not g.resume:
        vis.close()
    vis.text(args.repr('html'), win='global args', opts={'title': 'global args'})

    nets = make_nets(g.model_spec, g.train_device)
    dataset_files = 'train_1.hd5' if g.trial else '*.hd5'
    dagger_constructor = None

    dv = g.dataset_variant

    camera_kwargs = {
        'camera_pos': (g.camera_x, g.camera_y),
        'camera_z': g.camera_z,
        'demo_camera_z': g.demo_camera_z,
        'rollout_camera_z': g.rollout_camera_z,
        'demo_fov': (np.deg2rad(g.demo_hfov), np.deg2rad(g.demo_vfov)),
        'rollout_fov': (np.deg2rad(g.rollout_hfov), np.deg2rad(g.rollout_vfov)),
    }

    if dv == 'gibson2':
        gibson_kwargs = {
            'maps': None if g.maps == '' else g.maps.split(','),
            'output_resolution': g.resolution,
            'assets_dir': get_gibson2_asset_dir(),
            'persistent_server_cfg_file': g.persistent_server_cfg
        }

        dataset = DatasetGibson2Traj(
            device=g.dataset_inference_device,
            n_frame_min=g.n_frame_min,
            n_frame_max=g.n_frame_max,
            frame_interval=g.frame_interval,
            rand_frame_interval=g.rand_frame_interval,
            normalize_wp=g.normalize_wp,
            hd5_files=glob.glob(os.path.join(g.dataset_dir, dataset_files)),
            agent_name=g.robot,
            **gibson_kwargs, **camera_kwargs)

        def dagger_constructor(epoch):
            hosts = {
                'localhost': 'tcp://localhost:5002',
                # If you have additional machines running the dataset server, you can put them here.
            }

            dagger_dataset = DatasetClient(sorted(hosts.values()), len(dataset))

            weights_file = '%s.%d' % (g.model_file, epoch)

            for host in hosts:
                print('copy %s to %s' % (weights_file, host))
                subprocess.run(['scp', '-o', 'StrictHostKeyChecking=no', weights_file, '%s:/tmp/' % host], check=True)

            # TODO: send weights file to every server machine
            dagger_dataset.set_tracker_weights_file('/tmp/' + os.path.basename(weights_file))
            return dagger_dataset

    else:
        raise RuntimeError('Unknown dataset variant', dv)

    train_funcs = {
        'default': train_fixture.train_simple,
    }

    print('dataset options:\n%r' % dataset)

    train_funcs[g.model_variant](
        nets={
            name: spec['net'] for name, spec in nets.items()
        },
        net_opts={
            name: spec['opt'] for name, spec in nets.items()
        },
        dataset=dataset,
        dagger_dataset_constructor=dagger_constructor,
        vis=vis,
        global_args=g
    )
