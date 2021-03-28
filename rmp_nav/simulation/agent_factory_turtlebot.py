import numpy as np
import os

from . import agents, agent_solvers

from .param_utils import Params
from .agent_factory_common import add_agent
from ..common.utils import get_project_root


@add_agent
def turtlebot_240fov_rmp(**kwargs):
    params = Params(os.path.join(get_project_root(), 'configs/turtlebot_240fov_params.yaml'))
    solver_params = eval(params.get('solver_params', 'dict()'))
    kwargs['max_vel'] = 0.5
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
