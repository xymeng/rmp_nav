import os
from rmp_nav.common.utils import get_model_dir
from .inference import ProgressTrackerEndToEnd


tracker_models_dir = os.path.join(get_model_dir(), 'cbe')


_model_dict = {}


def register(f):
    _model_dict[f.__name__] = f
    return f


@register
def cbe():
    tracker = ProgressTrackerEndToEnd(
        weights_file=os.path.join(tracker_models_dir, 'randfi-fmax64-noodom-normwp-dagger-jitter-v2-cf0-model.5'))
    return {
        'tracker': tracker,
    }


def get(model_name):
    return _model_dict[model_name]
