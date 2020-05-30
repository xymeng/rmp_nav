from .traj_following import TrajectoryFollowerMultiFrameDstBothProximity
from .sparsify_traj import TrajSparsifierMultiframeDst
from topological_nav.controller.inference import ControllerMultiframeDst
from rmp_nav.common.utils import get_project_root, get_model_dir


controller_models_dir = get_model_dir() + '/topological_nav/controller/'
reachability_models_dir = get_model_dir() + '/topological_nav/reachability/'


_model_dict = {}


def register(f):
    _model_dict[f.__name__] = f
    return f


@register
def model_12env_v2_future_pair_proximity_z0228(device='cuda', **kwargs):
    search_thres = kwargs.pop('search_thres', 0.92)
    follow_thres = 0.92
    motion_policy = ControllerMultiframeDst(controller_models_dir + 'multiframe_dst/gtwp-normwp-farwp-jitter-weightedloss-checkwp-nf6-interval3-dmax3-z0228-model.8', device=device)
    sparsifier = TrajSparsifierMultiframeDst(reachability_models_dir + 'multiframe_dst/farwp-jitter-difftrajprob0.4-z0228-model.7', device=device)
    follower = TrajectoryFollowerMultiFrameDstBothProximity(sparsifier, motion_policy, search_thres, follow_thres)
    ret = {
        'motion_policy': motion_policy,
        'sparsifier': sparsifier,
        'follower': follower,
        'agent': 'classic_240fov_minirccar'
    }
    ret.update(kwargs)
    return ret


def get(model_name):
    return _model_dict[model_name]
