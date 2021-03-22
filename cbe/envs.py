import glob
import numpy as np

from rmp_nav.common.utils import (get_default_persistent_server_config, get_gibson2_asset_dir,
                                  get_data_dir)
from rmp_nav.simulation.gibson2_map import MakeGibson2Map
from .dataset import DatasetGibson2Traj


_env_dict = {}


def register(f):
    _env_dict[f.__name__] = f
    return f


gibson2_common_kwargs = {
    'agent_name': 'minirccar_240fov_rmp_v2',
    'output_resolution': 64,
    'camera_pos': (0.065, 0.00),
    'camera_z': 1.0,
    'demo_fov': (np.deg2rad(118.6), np.deg2rad(106.9)),
    'rollout_fov': (np.deg2rad(118.6), np.deg2rad(106.9)),
    'assets_dir': get_gibson2_asset_dir(),
    'persistent_server_cfg_file': get_default_persistent_server_config()
}


def make_gibson2_maps(map_names):
    def load_maps(datadir, map_names, **kwargs):
        maps = []
        for name in map_names:
            maps.append(MakeGibson2Map(datadir, name, **kwargs))
        return maps
    map_names = [s.strip() for s in map_names]
    print('maps:', map_names)
    map_param = {}
    return load_maps(get_gibson2_asset_dir(), map_names, **map_param)


@register
def calavo(**kwargs):
    kwargs2 = gibson2_common_kwargs.copy()
    kwargs2.update(kwargs)
    dset = DatasetGibson2Traj(
        hd5_files=glob.glob(get_data_dir() + '/gibson2/pairwise_destination_testenv/Calavo.hd5'),
        **kwargs2)
    maps = {map_name: make_gibson2_maps([map_name])[0] for map_name in dset.map_names}
    return dset, maps


# @register
# def frierson(**kwargs):
#     kwargs2 = kwargs.copy()
#     kwargs2.update(gibson2_common_kwargs)
#     dset = DatasetGibson2Traj(
#         hd5_files=glob.glob(get_project_root() + '/data/rmp/gibson2/pairwise_destination_testenv/Frierson.hd5'),
#         **kwargs2)
#     maps = {map_name: make_gibson2_maps([map_name])[0] for map_name in dset.map_names}
#     return dset, maps
#
#
# @register
# def kendall(**kwargs):
#     kwargs2 = kwargs.copy()
#     kwargs2.update(gibson2_common_kwargs)
#     dset = DatasetGibson2Traj(
#         hd5_files=glob.glob(get_project_root() + '/data/rmp/gibson2/pairwise_destination_testenv/Kendall.hd5'),
#         **kwargs2)
#     maps = {map_name: make_gibson2_maps([map_name])[0] for map_name in dset.map_names}
#     return dset, maps
#
#
# @register
# def ooltewah(**kwargs):
#     kwargs2 = kwargs.copy()
#     kwargs2.update(gibson2_common_kwargs)
#     dset = DatasetGibson2Traj(
#         hd5_files=glob.glob(get_project_root() + '/data/rmp/gibson2/pairwise_destination_testenv/Ooltewah.hd5'),
#         **kwargs2)
#     maps = {map_name: make_gibson2_maps([map_name])[0] for map_name in dset.map_names}
#     return dset, maps
#
#
# @register
# def sultan(**kwargs):
#     kwargs2 = kwargs.copy()
#     kwargs2.update(gibson2_common_kwargs)
#     dset = DatasetGibson2Traj(
#         hd5_files=glob.glob(get_project_root() + '/data/rmp/gibson2/pairwise_destination_testenv/Sultan.hd5'),
#         **kwargs2)
#     maps = {map_name: make_gibson2_maps([map_name])[0] for map_name in dset.map_names}
#     return dset, maps
#
#
# @register
# def env18(**kwargs):
#     kwargs2 = gibson2_common_kwargs.copy()
#     kwargs2.update(kwargs)
#     dset = DatasetGibson2Traj(
#         hd5_files=glob.glob(get_project_root() + '/data/rmp/gibson2/minirccar_agent_local_240fov_18env_slow/train_1.hd5'),
#         **kwargs2)
#     maps = {map_name: make_gibson2_maps([map_name])[0] for map_name in dset.map_names}
#     return dset, maps
#
#
@register
def testenvs(**kwargs):
    kwargs2 = gibson2_common_kwargs.copy()
    kwargs2.update(kwargs)
    dset = DatasetGibson2Traj(
        hd5_files=glob.glob(get_data_dir() + '/gibson2/pairwise_destination_testenv/*.hd5'),
        **kwargs2)
    maps = {map_name: make_gibson2_maps([map_name])[0] for map_name in dset.map_names}
    return dset, maps

#
# def _test_envs_slam_helper(resolution, **kwargs):
#     kwargs.update(gibson2_common_kwargs)
#     kwargs['output_resolution'] = resolution
#     dset = DatasetGibson2SLAM(
#         hd5_files=glob.glob(
#             get_project_root() + '/data/rmp/gibson2/pairwise_destination_testenv/*.hd5'),
#         **kwargs)
#     maps = {map_name: make_gibson2_maps([map_name])[0] for map_name in dset.map_names}
#     return dset, maps


def get(env_name):
    return _env_dict[env_name]
