from easydict import EasyDict as edict
import gflags
import tabulate
from rmp_nav.common.utils import get_default_persistent_server_config


gflags.DEFINE_string('dataset_dir', '', '')
gflags.DEFINE_string('maps', '', 'Comma separated map names')
gflags.DEFINE_string('model_spec', '', '')
gflags.DEFINE_string('model_file', '', '')
gflags.DEFINE_boolean('resume', False, 'Resume training.')
gflags.DEFINE_integer('resolution', 64, 'Image resolution')
gflags.DEFINE_float('camera_x', 0.065, 'Camera x position in agent local coord system')
gflags.DEFINE_float('camera_y', 0.0, 'Camera y position in agent local coord system')
gflags.DEFINE_float('camera_z', 1.0, 'Camera z position in agent local coord system')
gflags.DEFINE_integer('batch_size', 64, '')
gflags.DEFINE_integer('samples_per_epoch', 200000, '')
gflags.DEFINE_integer('max_epochs', 10, '')
gflags.DEFINE_integer('lr_decay_epoch', 1, '')
gflags.DEFINE_float('lr_decay_rate', 0.7, '')
gflags.DEFINE_integer('n_dataset_worker', 2, '')
gflags.DEFINE_string('train_device', 'cuda', '')
gflags.DEFINE_integer('log_interval', 10, '')
gflags.DEFINE_integer('save_interval', 100, '')
gflags.DEFINE_string('persistent_server_cfg', get_default_persistent_server_config(), '')
gflags.DEFINE_boolean('trial', False, 'True to enable trial run (smaller datasets).')
gflags.DEFINE_string('visdom_env', 'main', '')
gflags.DEFINE_string('visdom_server', 'http://localhost', '')
gflags.DEFINE_integer('visdom_port', 5001, '')
gflags.DEFINE_integer('vis_interval', 100, '')
gflags.DEFINE_string('set', '', 'A comma separated assignment string that overwrites flag values.')


defaults = {}


def helper(d):
    ret = {}
    for k, v in d.items():
        if isinstance(v, dict):
            ret[k] = helper(v)
        else:
            ret[k] = v
    return tabulate.tabulate(ret.items())


# The global flag tree. Tree can be augmented by loading external config files.
g = edict(defaults)


# Since g's values can be accessed as attributes, we cannot add additional methods to it.
# Use the following methods to manipulate g.


def fill(args):
    for key in args.keys():
        g[key] = args[key].value


def set_s(set_str):
    ss = set_str.split(',')
    for s in ss:
        if s == '':
            continue
        field, value = s.split('=')
        try:
            value = eval(value, {'__builtins__': None})
        except:
            # Cannot convert the value. Treat it as it is.
            pass
        attrs = field.split('.')
        node = g
        for i in range(len(attrs) -1):
            node = node[attrs[i]]
        node[attrs[-1]] = value


def repr(fmt='plain'):
    def helper(d):
        ret = {}
        for k, v in d.items():
            if isinstance(v, dict):
                ret[k] = helper(v)
            else:
                ret[k] = v
        return tabulate.tabulate(ret.items(), tablefmt=fmt)
    return helper(g)
