"""
Provides a global space for accessing tuning parameters.
"""

from __future__ import print_function
from ..common.utils import get_project_root
import yaml


class Params(object):
    def __init__(self, filename=None):
        self.params = {}
        self.param_file = filename
        self.reload()

    def reload(self):
        if self.param_file is not None:
            self.params = yaml.load(open(self.param_file).read(), Loader=yaml.SafeLoader)

    def get(self, key, default=None, required=False):
        if required and key not in self.params:
            raise KeyError('%s not in params but is required' % key)
        return self.params.get(key, default)

    def put(self, key, value):
        self.params[key] = value

    def __repr__(self):
        return self.param_file + ': ' + self.params.__repr__()


import os
global_params = Params(os.path.join(get_project_root(), 'configs/global_params.yaml'))


from ..common.utils import pprint_dict
print('global params:\n%s' % pprint_dict(global_params.params))
