from . import agents, agent_solvers
import numpy as np
import os
from .param_utils import Params
from .agent_factory_common import add_agent
from ..common.utils import get_project_root


@add_agent
def classic_240fov_rccar2(**kwargs):
    params = Params(os.path.join(get_project_root(), 'configs/rccar_240fov_params2.yaml'))
    solver_params = eval(params.get('solver_params', 'dict()'))

    return agents.RCCarAgentLocalLIDAR(
        params=params,
        n_depth_ray=50,
        lidar_fov=np.pi / 180 * 240,
        lidar_sensor_pos=(0.16, 0.0),
        solver=agent_solvers.CarAgentLocalClassicRMPSolver(
            params=params, **solver_params
        ),
        **kwargs
    )
