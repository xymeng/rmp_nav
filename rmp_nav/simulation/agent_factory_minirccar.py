import numpy as np
import os

from . import agents, agent_solvers, param_utils

from .param_utils import Params
from .agent_factory_common import add_agent
from ..common.utils import get_project_root


@add_agent
def classic_240fov_minirccar(**kwargs):
    params = Params(os.path.join(get_project_root(), 'configs/minirccar_240fov_params.yaml'))
    solver_params = eval(params.get('solver_params', 'dict()'))

    return agents.RCCarAgentLocalLIDAR(
        params=params,
        n_depth_ray=50,
        lidar_fov=np.pi / 180 * 240,
        lidar_sensor_pos=(0.08, 0.0),
        solver=agent_solvers.CarAgentLocalClassicRMPSolver(
            params=params, **solver_params
        ),
        **kwargs
    )


@add_agent
def minirccar_240fov_rmp_v2(**kwargs):
    # Works better at tight spaces.
    # Maximum speed is capped at 0.5m/s
    params = Params(os.path.join(get_project_root(), 'configs/minirccar_240fov_v2.yaml'))
    solver_params = eval(params.get('solver_params', 'dict()'))
    kwargs['max_waypoint_dist'] = 0.5
    kwargs['max_vel'] = 0.5
    return agents.RCCarAgentLocalLIDAR(
        params=params,
        n_depth_ray=50,
        lidar_fov=np.pi / 180 * 240,
        lidar_sensor_pos=(0.08, 0.0),
        solver=agent_solvers.CarAgentLocalClassicRMPSolver(
            params=params, **solver_params
        ),
        **kwargs)


@add_agent
def gibson12v2_240fov_minirccar_z0228_fisheye64_metric(gpu_idx=0, **kwargs):
    params = Params(os.path.join(get_project_root(), 'configs/minirccar_240fov_params.yaml'))
    kwargs['persistent_servers'] = param_utils.global_params.get('persistent_servers', None)
    return agents.RCCarAgentLocalVisualGibson(
        params=params,
        render_resolution=128,
        output_resolution=64,
        h_fov=np.deg2rad(118.6), v_fov=np.deg2rad(106.9),
        camera_pos=(0.065, 0.00),
        camera_z=0.228,
        solver=agent_solvers.LocalVisualNeuralMetricSolverV2(
            get_project_root() + '/models/neural_rmp/gibson12v2_240fov_minirccar_z0228_fisheye64_metric/z0228-logmetric-model.18',
            n_control_points=12,
            gpu=gpu_idx),
        gpu_idx=int(os.environ.get('GIBSON_GPU', gpu_idx)),
        **kwargs
    )
