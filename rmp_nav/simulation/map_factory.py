from .gibson2_map import MakeGibson2Map
from .gibson_map import MakeGibsonMap
from ..common.utils import get_gibson_asset_dir, get_gibson2_asset_dir


def make_map(map_type, map_name, **map_kwargs):
    if map_type == 'gibson':
        return MakeGibsonMap(get_gibson_asset_dir(), map_name, **map_kwargs)
    elif map_type == 'gibson2':
        return MakeGibson2Map(get_gibson2_asset_dir(), map_name, **map_kwargs)
    else:
        raise ValueError('Unknown map type:', map_type)
