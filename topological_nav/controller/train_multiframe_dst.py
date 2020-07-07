import gflags
import glob
import os
import sys
import yaml
from easydict import EasyDict
import numpy as np
import torch
from torch import optim
import visdom
from .dataset import DatasetSourceTargetPairMultiframeDst
from . import networks
from . import train_fixture
from . import args
from .args import g
from rmp_nav.common.utils import get_gibson_asset_dir


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

    gflags.DEFINE_integer('n_frame', 10, '')
    gflags.DEFINE_integer('frame_interval', 5, '')
    gflags.DEFINE_string('model_variant', 'attention', '')
    gflags.DEFINE_boolean('weight_loss', False, '')
    gflags.DEFINE_float('weight_loss_min_clip', 0.1, '')

    FLAGS = gflags.FLAGS
    FLAGS(sys.argv)

    g['cmd_line'] = ' '.join(sys.argv)

    args.fill(FLAGS.FlagDict())

    g.model_spec = yaml.load(open(g.model_spec).read(), Loader=yaml.SafeLoader)
    args.set_s(FLAGS.set)

    if g.model_variant.startswith('future'):
        g['future'] = True
    else:
        g['future'] = False

    print('global args:')
    print(args.repr())

    vis = visdom.Visdom(env=g.get('visdom_env', 'main'), server=g.visdom_server, port=g.visdom_port)
    if not g.resume:
        vis.close()
    vis.text(args.repr('html'), win='global args', opts={'title': 'global args'})

    nets = make_nets(g.model_spec, g.train_device)
    dataset_files = 'train_1.hd5' if g.trial else '*.hd5'

    dataset = DatasetSourceTargetPairMultiframeDst(
        n_frame=g.n_frame,
        frame_interval=g.frame_interval,
        fov=np.deg2rad(g.hfov),
        distance_min=g.dmin,
        distance_max=g.dmax,
        overlap_ratio=g.overlap_ratio,
        use_gt_wp=g.use_gt_wp,
        normalize_wp=g.normalize_wp,
        check_wp=g.check_wp,
        jitter=g.jitter,
        future=g.future,
        maps=None if g.maps == '' else g.maps.split(','),
        agent_name='classic_240fov_minirccar',
        net_config=EasyDict({
            'input_resolution': g.resolution
        }),
        hd5_files=glob.glob(os.path.join(g.dataset_dir, dataset_files)),
        assets_dir=get_gibson_asset_dir(),
        camera_pos=(g.camera_x, g.camera_y),
        camera_z=g.camera_z,
        h_fov=np.deg2rad(g.hfov),
        v_fov=np.deg2rad(g.vfov),
        n_filler_server=0, n_sim_per_map=0,
        persistent_server_cfg_file=g.persistent_server_cfg)

    train_funcs = {
        'attention': train_fixture.train_multiframedst,
        'concat_early': train_fixture.train_multiframedst,
        'future': train_fixture.train_multiframedst,
        'future_stack': train_fixture.train_multiframedst,
        'future_stack_v2': train_fixture.train_multiframedst,
        'future_pair': train_fixture.train_multiframedst,
        'future_pair_featurized': train_fixture.train_multiframedst,
        'future_pair_featurized_v2': train_fixture.train_multiframedst,
        'future_pair_conv': train_fixture.train_multiframedst,
    }

    train_funcs[g.model_variant](
        nets={
            name: spec['net'] for name, spec in nets.items()
        },
        net_opts={
            name: spec['opt'] for name, spec in nets.items()
        },
        dataset=dataset,
        vis=vis,
        global_args=g
    )
