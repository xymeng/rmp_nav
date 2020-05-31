import glob
from easydict import EasyDict
import os
import numpy as np
from rmp_nav.neural.common.dataset import DatasetVisualGibson
from rmp_nav.common.utils import get_project_root, get_data_dir, get_gibson_asset_dir, get_config_dir


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
        hd5_files=glob.glob(get_data_dir() + '/minirccar_agent_local_240fov_house31_farwp_v2/train_*.hd5'),
        **kwargs)


@register
def space8(**kwargs):
    return DatasetVisualGibson(
        hd5_files=glob.glob(get_data_dir() + '/minirccar_agent_local_240fov_space8_farwp_v2/train_*.hd5'),
        **kwargs)


@register
def space8_pairwise_destination(**kwargs):
    return DatasetVisualGibson(
        hd5_files=glob.glob(get_data_dir() + '/minirccar_agent_local_240fov_space8_pairwise_destination/trajs.hd5'),
        **kwargs)


def make(env_name, sparsifier):
    common_kwargs['camera_pos'] = (sparsifier.g.get('camera_x', 0.065),
                                   sparsifier.g.get('camera_y', 0.0))
    common_kwargs['camera_z'] = sparsifier.g.get('camera_z', 1.0)
    common_kwargs['h_fov'] = np.deg2rad(sparsifier.g.get('hfov', 118.6))
    common_kwargs['v_fov'] = np.deg2rad(sparsifier.g.get('vfov', 106.9))
    common_kwargs['net_config']['input_resolution'] = sparsifier.g.get('resolution', 64)
    return _env_dict[env_name](**common_kwargs)
