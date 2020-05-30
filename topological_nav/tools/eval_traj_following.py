import numpy as np
import gflags
import sys
import glob
from easydict import EasyDict
import os
from rmp_nav.neural.common.dataset import DatasetVisualGibson
from rmp_nav.common.utils import get_project_root, get_data_dir, get_gibson_asset_dir, get_config_dir
from rmp_nav.simulation import agent_factory
from topological_nav.reachability import model_factory
from topological_nav.tools.eval_traj_following_common import EvaluatorReachability


gflags.DEFINE_string('env', 'house31', '')
gflags.DEFINE_string('model', 'model_12env_v2_future_pair_proximity_z0228', '')
gflags.DEFINE_boolean('dry_run', False, '')
gflags.DEFINE_float('sparsify_thres', 0.99, '')
gflags.DEFINE_integer('start_idx', 0, '')
gflags.DEFINE_integer('n_traj', 100, '')
gflags.DEFINE_float('clip_velocity', 0.5, 'Limit the max velocity.')
gflags.DEFINE_boolean('visualize', True, '')
gflags.DEFINE_boolean('save_screenshot', False, '')
gflags.DEFINE_float('zoom', 1.0, '')
FLAGS = gflags.FLAGS
FLAGS(sys.argv)

_env_dict = {}


def register(f):
    _env_dict[f.__name__] = f
    return f


common_kwargs = {
    'agent_name': 'classic_240fov_minirccar',
    'net_config': EasyDict({'input_resolution': 64}),
    'assets_dir': get_gibson_asset_dir(),
    'n_filler_server': 0,
    'n_sim_per_map': 0,
    'persistent_server_cfg_file': get_config_dir() + '/gibson_persistent_servers/local.yaml'
}


@register
def house31(**kwargs):
    return DatasetVisualGibson(
        hd5_files=glob.glob(get_data_dir() + '/minirccar_agent_local_240fov_house31_farwp_v2/train_1.hd5'),
        **kwargs)


model = model_factory.get(FLAGS.model)()
sparsifier = model['sparsifier']
motion_policy = model['motion_policy']
traj_follower = model['follower']
agent = agent_factory.agents_dict[model['agent']]()

common_kwargs['camera_pos'] = (sparsifier.g.get('camera_x', 0.065),
                               sparsifier.g.get('camera_y', 0.0))
common_kwargs['camera_z'] = sparsifier.g.get('camera_z', 1.0)
common_kwargs['h_fov'] = np.deg2rad(sparsifier.g.get('hfov', 118.6))
common_kwargs['v_fov'] = np.deg2rad(sparsifier.g.get('vfov', 106.9))
common_kwargs['net_config']['input_resolution'] = sparsifier.g.get('resolution', 64)


e = EvaluatorReachability(dataset=_env_dict[FLAGS.env](**common_kwargs),
                          sparsifier=sparsifier,
                          motion_policy=motion_policy,
                          follower=traj_follower,
                          agent=agent,
                          agent_reverse=None,
                          sparsify_thres=FLAGS.sparsify_thres,
                          clip_velocity=FLAGS.clip_velocity,
                          visualize=FLAGS.visualize,
                          save_screenshot=FLAGS.save_screenshot,
                          zoom=FLAGS.zoom,
                          dry_run=FLAGS.dry_run)

e.run(start_idx=FLAGS.start_idx, n_traj=FLAGS.n_traj)
