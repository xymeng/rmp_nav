import numpy as np
import os

from . import agents, agent_solvers, param_utils

from .param_utils import Params
from .agent_factory_common import add_agent
from ..common.utils import get_project_root


@add_agent
def classic_240fov_turtlebot(**kwargs):
    params = Params(os.path.join(get_project_root(), 'configs/turtlebot_240fov_params.yaml'))
    solver_params = eval(params.get('solver_params', 'dict()'))

    return agents.TurtleBotLocalLIDAR(
        params=params,
        n_depth_ray=50,
        lidar_fov=np.pi / 180 * 240,
        lidar_sensor_pos=(0.0, 0.0),
        solver=agent_solvers.TurtleBotLocalClassicRMPSolver(
            params=params, **solver_params
        ),
        **kwargs
    )
