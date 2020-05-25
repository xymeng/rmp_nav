import os
import glob
import numpy as np
import tabulate


def pprint_dict(x):
    """
    :param x: a dict
    :return: a string of pretty representation of the dict
    """
    def helper(d):
        ret = {}
        for k, v in d.items():
            if isinstance(v, dict):
                ret[k] = helper(v)
            else:
                ret[k] = v
        return tabulate.tabulate(ret.items())
    return helper(x)


def str_to_dict(s, delim=',', kv_delim='='):
    ss = s.split(delim)
    d = {}
    for s in ss:
        if s == '':
            continue
        field, value = s.split(kv_delim)
        try:
            value = eval(value, {'__builtins__': None})
        except:
            # Cannot convert the value. Treat it as it is.
            pass
        d[field] = value
    return d


def module_grad_stats(module):
    headers = ['layer', 'max', 'min']

    def maybe_max(x):
        return x.max() if x is not None else 'None'

    def maybe_min(x):
        return x.min() if x is not None else 'None'

    data = [
        (name, maybe_max(param.grad), maybe_min(param.grad))
        for name, param in module.named_parameters()
    ]
    return tabulate.tabulate(data, headers, tablefmt='psql')


def save_model(state, step, dir, filename):
    import torch
    path = os.path.join(dir, '%s.%d' % (filename, step))
    torch.save(state, path)


def load_model(dir, filename, step=None, load_to_cpu=False):
    '''
    :param model:
    :param dir:
    :param filename:
    :param step: if None. Load the latest.
    :return: the saved state dict
    '''
    import torch
    import parse
    if not step:
        files = glob.glob(os.path.join(dir, '%s.*' % filename))
        parsed = []
        for fn in files:
            r = parse.parse('{}.{:d}', fn)
            if r:
                parsed.append((r, fn))
        if not parsed:
            return None

        step, path = max(parsed, key=lambda x: x[0][1])
    else:
        path = os.path.join(dir, '%s.%d' % (filename, step))

    if os.path.isfile(path):
        if load_to_cpu:
            return torch.load(path, map_location=lambda storage, location: storage)
        else:
            return torch.load(path)

    raise Exception('Failed to load model')


def get_project_root():
    return os.path.normpath(os.path.dirname(__file__) + '/../../')


def get_gibson_asset_dir():
    return os.path.join(get_project_root(), 'rmp_nav', 'gibson', 'assets', 'dataset')


def get_data_dir():
    return os.path.join(get_project_root(), 'data')


def get_config_dir():
    return os.path.join(get_project_root(), 'configs')


def get_model_dir():
    return os.path.join(get_project_root(), 'models')


def cairo_argb_to_opencv_rgb(arr):
    argb = arr.view(dtype=np.dtype((np.uint32, {'a': (np.uint8, 3),
                                                'r': (np.uint8, 2),
                                                'g': (np.uint8, 1),
                                                'b': (np.uint8, 0)})))
    return np.stack([argb['r'], argb['g'], argb['b']], axis=2)


def cairo_argb_to_opencv_bgr(arr):
    argb = arr.view(dtype=np.dtype((np.uint32, {'a': (np.uint8, 3),
                                                'r': (np.uint8, 2),
                                                'g': (np.uint8, 1),
                                                'b': (np.uint8, 0)})))
    return np.stack([argb['b'], argb['g'], argb['r']], axis=2)
