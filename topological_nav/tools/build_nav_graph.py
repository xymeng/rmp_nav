import numpy as np
import cv2
import os
from topological_nav.reachability.planning import NavGraph, update_nav_graph, NavGraphSPTM
from rmp_nav.simulation.gibson_map import MakeGibsonMap
from rmp_nav.common.utils import get_gibson_asset_dir, pprint_dict


def _make_maps(map_names):
    def load_maps(datadir, map_names, **kwargs):
        maps = []
        for name in map_names:
            maps.append(MakeGibsonMap(datadir, name, **kwargs))
        return maps
    map_names = [s.strip() for s in map_names]
    map_param = {}
    return load_maps(get_gibson_asset_dir(), map_names, **map_param)


class DatasetRealTrace(object):
    def __init__(self, data_dir, map_name):
        import glob
        self.data_dir = data_dir
        self.map_name = map_name

        dirs = os.listdir(data_dir)
        sample_dict = dict()

        for d in dirs:
            if os.path.isdir(os.path.join(data_dir, d)):
                samples = []
                img_files = sorted(glob.glob(os.path.join(data_dir, d, '*.tiff')))
                for fn in img_files:
                    basename = os.path.splitext(os.path.basename(fn))[0]
                    meta_file = os.path.join(data_dir, d, '%s.yaml' % basename)
                    metadata = yaml.load(open(meta_file).read(), yaml.SafeLoader)
                    samples.append((fn, metadata))
                sample_dict[(0, d)] = samples

        self.sample_dict = sample_dict
        print('number of samples: ', sum([len(_) for _ in self.sample_dict.values()]))

        self.traj_ids = list(self.sample_dict.keys())

    def locate_traj(self, traj_id):
        raw_samples = self.sample_dict[traj_id]
        samples = []
        for fn, metadata in raw_samples:
            samples.append((metadata['pos'], metadata['heading'], fn))

        dtype = np.dtype([
            ('pos', (np.float32, 2)),
            ('heading', np.float32),
            ('filenames', 'U128')
        ])
        return np.array(samples, dtype=dtype)

    def locate_traj_map(self, traj_id):
        return map_name

    def render_traj(self, traj, **kwargs):
        imgs = []
        for fn in traj['filenames']:
            img = cv2.imread(fn, cv2.IMREAD_COLOR)
            img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255).astype(np.float32)
            img = img.transpose((2, 0, 1))
            imgs.append(img)
        return imgs


if __name__ == '__main__':
    import gflags
    import sys
    import yaml
    from topological_nav.reachability import model_factory
    from topological_nav.tools import eval_envs

    gflags.DEFINE_string('env', '', '')
    gflags.DEFINE_string('graph_config', '', '')
    gflags.DEFINE_string('save_file', '', '')
    gflags.DEFINE_string('device', 'cuda', '')

    FLAGS = gflags.FLAGS
    FLAGS(sys.argv)

    graph_config = yaml.load(open(FLAGS.graph_config).read(), Loader=yaml.SafeLoader)
    print('graph_config:\n%s' % pprint_dict(graph_config))

    model = model_factory.get(graph_config['model'])(device=FLAGS.device)

    graph_type = graph_config.get('type', 'ours')

    if graph_type == 'ours':
        nav_graph = NavGraph(model['sparsifier'], model['motion_policy'], **graph_config['graph_kwargs'])
    elif graph_type == 'SPTM':
        nav_graph = NavGraphSPTM(model['sparsifier'], model['motion_policy'], **graph_config['graph_kwargs'])
    else:
        raise RuntimeError('Unknown graph type: %s' % graph_type)

    data_source = graph_config.get('data_source', 'gibson')

    if data_source == 'gibson':
        dataset = eval_envs.make(FLAGS.env, model['sparsifier'])
        map_name = dataset.map_names[0]
        map = _make_maps([map_name])[0]
        dataset._init_once(0)
        dataset.agent.set_map(map)

    elif data_source == 'real':
        map_name = graph_config['map']
        map = _make_maps([map_name])[0]
        dataset = DatasetRealTrace(data_dir=graph_config['data_dir'], map_name=map_name)

    else:
        raise RuntimeError('Unknown data source %s' % data_source)

    traj_ids = sorted(dataset.traj_ids, key=lambda _: _[1])
    # You can either specify traj_idxs or traj_ids in the config YAML.
    filtered_traj_ids = set(graph_config.get('traj_ids', []))
    if filtered_traj_ids:
        traj_ids_to_add = []
        for _ in traj_ids:
            if _[1] in filtered_traj_ids:
                traj_ids_to_add.append(_)
    else:
        traj_idxs = graph_config.get('traj_idxs', list(range(len(traj_ids))))
        traj_ids_to_add = [traj_ids[_] for _ in traj_idxs]

    try:
        nav_graph.load(FLAGS.save_file)
        print('loaded', FLAGS.save_file)
        print(nav_graph.extra['id_mapping'])
    except FileNotFoundError:
        os.makedirs(os.path.dirname(FLAGS.save_file), exist_ok=True)
        pass

    traj_info_dict = {}
    update_nav_graph(nav_graph, dataset, traj_ids_to_add, traj_info_dict,
                     **graph_config.get('builder_kwargs', {}))

    print('number of trajs: %d' % len(nav_graph.trajs))
    print('number of traj images: %d' % sum([len(_['samples']) for _ in traj_info_dict.values()]))
    print('number of nodes: %d' % len(nav_graph.graph.nodes))
    print('number of edges: %d' % len(nav_graph.graph.edges))
    print('number of anchors: %d' % sum([len(_) for _ in nav_graph.trajs.values()]))

    print('save graph to %s' % FLAGS.save_file)
    nav_graph.extra['model'] = graph_config['model']
    nav_graph.extra['env'] = FLAGS.env
    nav_graph.extra['traj_info'] = traj_info_dict  # Useful for visualization, etc.

    nav_graph.save(FLAGS.save_file)
    nav_graph.visualize(map, traj_info_dict,
                        save_file=os.path.dirname(FLAGS.save_file) + '/graph.svg')
