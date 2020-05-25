import numpy as np
from easydict import EasyDict
import os
import glob
import sys
import gflags
from rmp_nav.simulation import agent_factory
from rmp_nav.common.utils import get_model_dir, get_gibson_asset_dir, get_config_dir, get_data_dir
from topological_nav.controller import inference
from topological_nav.controller.dataset import DatasetSourceTargetPairMultiframeDst
from topological_nav.tools.eval_controller_common import EvaluatorMultiframeDst


gflags.DEFINE_string('env', 'space8', '')
gflags.DEFINE_integer('start_idx', 0, '')
gflags.DEFINE_integer('n_traj', 200, '')
gflags.DEFINE_boolean('visualize', True, '')
gflags.DEFINE_boolean('save_screenshot', False, '')
gflags.DEFINE_float('zoom', 1.0, '')
FLAGS = gflags.FLAGS
FLAGS(sys.argv)


weights_file = os.path.join(
    get_model_dir(),
    'topological_nav/controller/multiframe_dst/gtwp-normwp-farwp-jitter-weightedloss-checkwp-nf6-interval3-dmax3-z0228-model.8')


_env_dict = {}


def register(f):
    _env_dict[f.__name__] = f
    return f


mp = inference.Client(weights_file)


common_kwargs = {
    'overlap_ratio': 0.3,
    'fov': np.deg2rad(mp.g.get('hfov', 118.6)),
    'distance_min': -1.0,
    'distance_max': mp.g.dmax,
    'agent_name': 'classic_240fov_minirccar',
    'net_config': EasyDict({'input_resolution': mp.g.get('resolution', 64)}),
    'assets_dir': get_gibson_asset_dir(),
    'camera_z': mp.g['camera_z'],
    'camera_pos': (mp.g.get('camera_x', 0.065), mp.g.get('camera_y', 0.0)),
    'h_fov': np.deg2rad(mp.g.get('hfov', 118.6)),
    'v_fov': np.deg2rad(mp.g.get('vfov', 106.9)),
    'n_filler_server': 0,
    'n_sim_per_map': 0,
    'persistent_server_cfg_file': os.path.join(get_config_dir(), 'gibson_persistent_servers/local.yaml')
}


@register
def space8(**kwargs):
    return DatasetSourceTargetPairMultiframeDst(
        hd5_files=glob.glob(
            os.path.join(get_data_dir(), 'minirccar_agent_local_240fov_space8_farwp_v2/train_*.hd5')),
        **kwargs)


agent = agent_factory.agents_dict['classic_240fov_minirccar']()
agent_rev = None


common_kwargs['n_frame'] = mp.g.n_frame
common_kwargs['frame_interval'] = mp.g.frame_interval
common_kwargs['future'] = mp.g.get('future', False)
e = EvaluatorMultiframeDst(mp,
                           _env_dict[FLAGS.env](**common_kwargs),
                           agent=agent, agent_reverse=agent_rev,
                           visualize=FLAGS.visualize,
                           zoom=FLAGS.zoom,
                           save_screenshot=FLAGS.save_screenshot)

e.run(start_idx=FLAGS.start_idx, n_traj=FLAGS.n_traj, seed=12345)
print('done.')
