from svgpathtools import svg2paths
import os
import numpy as np
import cv2
from . import map_2dline, map_occupancy_grid, map_utils


def _paths_to_lines(paths, w, h, s):
    lines = []
    for path in paths:
        for l in path:
            x1, y1, x2, y2 = l.start.real, l.start.imag, l.end.real, l.end.imag
            x1, y1 = _pixel_to_world_coord(w, h, s, x1, y1)
            x2, y2 = _pixel_to_world_coord(w, h, s, x2, y2)
            lines.append((x1, y1, x2, y2))
    return lines


def _gen_path_map(path_map_file, w, h, s):
    if path_map_file.endswith('.svg'):
        paths, _ = svg2paths(path_map_file)
        lines = _paths_to_lines(paths, w, h, s)
        bbox = (-s, s, -s, s)
        discrete_bbox = (-w / 2, w / 2, -h / 2, h / 2)
        division = w // (2 * s)
        return 255 - map_utils.rasterize(
            lines, bbox, division=division), discrete_bbox, division
    elif path_map_file.endswith('.tiff'):
        path_map = cv2.imread(path_map_file, cv2.IMREAD_COLOR)
        if path_map is None:
            raise RuntimeError('failed to read path map %s' % path_map_file)
        path_map = path_map[:, :, 0]
        path_map = cv2.flip(path_map, 0)
        assert path_map.shape[:2] == (h, w)
        discrete_bbox = (-w / 2, w / 2, -h / 2, h / 2)
        division = w // (2 * s)
        return path_map, discrete_bbox, division
    else:
        raise RuntimeError()


def _pixel_to_world_coord(w, h, s, x, y):
    return (x - w * 0.5) / (w * 0.5) * s, ((h - y) - h * 0.5) / (h * 0.5) * s


def _load_destination_map(filename, w, h):
    m = cv2.imread(filename, cv2.IMREAD_COLOR)
    if m is None:
        raise RuntimeError('failed to read goal map %s' % filename)
    m = cv2.cvtColor(m, cv2.COLOR_BGR2RGB)
    m = cv2.flip(m, 0)
    assert m.shape[:2] == (h, w)
    return m


class GibsonMap(object):
    def __init__(self, assets_dir, scene_id, **kwargs):
        self.assets_dir = assets_dir
        self.scene_id = scene_id
        super(GibsonMap, self).__init__(**kwargs)


class GibsonMapLine(GibsonMap, map_2dline.Map):
    pass


class GibsonMapRasterized(GibsonMap, map_occupancy_grid.Map):
    pass


def MakeGibsonMap(assets_dir, scene_id, **kwargs):
    # FIXME: remove hardcoded values.
    W = 4000
    H = 4000
    S = 100

    init_pos = (W / 2, H / 2)

    file_dir = os.path.join(assets_dir, scene_id)
    meta_file = os.path.join(file_dir, 'floorplan.yaml')
    floorplan_file = None
    path_map_file = None
    destination_map_file = None

    if os.path.isfile(meta_file):
        import yaml
        meta = yaml.load(open(meta_file).read(), Loader=yaml.SafeLoader)
        init_pos = meta.get('init_pos', init_pos)
        floorplan_file = meta.get('floorplan', None)
        path_map_file = meta.get('path_map', None)
        destination_map_file = meta.get('destination_map', None)
    else:
        print('map metadata file %s does not exist' % meta_file)

    # Convert init_pos into the standard coordinate space.
    init_pos = init_pos[0], init_pos[1] + 1

    if path_map_file is not None:
        path_map, path_map_bbox, path_map_division = _gen_path_map(
            os.path.join(assets_dir, scene_id, path_map_file), W, H, S)
    else:
        path_map, path_map_bbox, path_map_division = None, None, 16

    if destination_map_file is not None:
        destination_map = _load_destination_map(os.path.join(assets_dir, scene_id, destination_map_file), W, H)
    else:
        destination_map = None

    def load_floorplan_svg(floorplan_file):
        paths, attributes = svg2paths(floorplan_file)
        lines = _paths_to_lines(paths, W, H, S)

        return GibsonMapLine(
            assets_dir, scene_id,
            lines=lines,
            initial_pos=_pixel_to_world_coord(W, H, S, *init_pos),
            path_map=path_map,
            path_map_bbox=path_map_bbox,
            path_map_division=path_map_division,
            name=scene_id,
            **kwargs)

    def load_floorplan_rasterized(floorplan_file):
        floorplan = cv2.imread(floorplan_file, cv2.IMREAD_COLOR)[:, :, 0]
        floorplan = cv2.flip(floorplan, 0)

        return GibsonMapRasterized(
            assets_dir, scene_id,
            occupancy_grid=floorplan,
            resolution=S / (W / 2.0),
            origin=(-S, -S),
            initial_pos=_pixel_to_world_coord(W, H, S, *init_pos),
            path_map=path_map,
            path_map_division=path_map_division,
            destination_map=destination_map,
            name=scene_id,
            **kwargs
        )

    if floorplan_file is None:
        floorplan_file = os.path.join(assets_dir, scene_id, 'floorplan.svg')
    else:
        floorplan_file = os.path.join(assets_dir, scene_id, floorplan_file)

    if floorplan_file.endswith('.svg'):
        return load_floorplan_svg(floorplan_file)
    elif floorplan_file.endswith('.tiff'):
        return load_floorplan_rasterized(floorplan_file)
    else:
        raise RuntimeError()
