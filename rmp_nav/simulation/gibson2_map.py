import os
import numpy as np
import cv2
import yaml
from . import map_occupancy_grid


class Gibson2Map(map_occupancy_grid.Map):
    def __init__(self, assets_dir, scene_id, floor_height, **kwargs):
        self.assets_dir = assets_dir
        self.scene_id = scene_id
        self.floor_height = floor_height
        super(Gibson2Map, self).__init__(**kwargs)


def MakeGibson2Map(assets_dir, scene_id, ignore_destination=True, **kwargs):
    """
    :param ignore_destination: if True will ignore the destination map to speed up loading.
    :param kwargs:
    :return:
    """
    file_dir = os.path.join(assets_dir, scene_id)
    meta_file = os.path.join(file_dir, 'floorplan.yaml')

    if not os.path.isfile(meta_file):
        raise ValueError('Cannot find meta file %s' % meta_file)

    meta = yaml.load(open(meta_file).read(), Loader=yaml.SafeLoader)
    floorplan_file = meta['floorplan']
    resolution = float(meta['resolution'])
    init_pos = meta.get('init_pos', np.array([0.0, 0.0], np.float32))
    path_map_file = meta.get('path_map', None)
    path_map_resolution = meta.get('path_map_resolution', None)
    destination_map_file = meta.get('destination_map', None)

    floor_height = meta['ref_z']

    floorplan_file_path = os.path.join(file_dir, floorplan_file)
    floorplan = cv2.imread(floorplan_file_path, cv2.IMREAD_COLOR)[:, :, 0]
    floorplan = cv2.flip(floorplan, 0)

    h, w = floorplan.shape[:2]

    if path_map_file is None:
        path_map = (floorplan < 128).astype(np.uint8) * 255
        path_map_resolution = resolution
    else:
        path_map_file_path = os.path.join(file_dir, path_map_file)
        path_map = cv2.imread(path_map_file_path, cv2.IMREAD_COLOR)[:, :, 0]
        path_map = cv2.flip(path_map, 0)

    if not ignore_destination and destination_map_file is not None:
        dest_map_file_path = os.path.join(file_dir, destination_map_file)
        dest_map = cv2.cvtColor(cv2.imread(dest_map_file_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        dest_map = cv2.flip(dest_map, 0)
    else:
        dest_map = None

    return Gibson2Map(
        assets_dir, scene_id, floor_height,
        occupancy_grid=floorplan,
        resolution=resolution,
        origin=(-w//2 * resolution, -h//2 * resolution),
        initial_pos=init_pos,
        path_map=path_map,
        path_map_division=int(1.0 / path_map_resolution),
        destination_map=dest_map,
        name=scene_id,
        background_traversable=True,
        **kwargs)
