import sys

import gflags
import numpy as np

from rmp_nav.simulation import agent_factory
from rmp_nav.common.utils import get_default_persistent_server_config

from cbe.eval_tracker_common import Evaluator
from cbe import model_factory, envs


gflags.DEFINE_string('model', '', '')
gflags.DEFINE_string('agent', '', 'If specified will overwrite the default.')
gflags.DEFINE_string('env', '', '')
gflags.DEFINE_integer('start_idx', 0, '')
gflags.DEFINE_integer('n_traj', 200, '')
gflags.DEFINE_integer('n_frame', None, 'Number of frames in a trajectory.')
gflags.DEFINE_integer('frame_interval', 2, 'Distance between two adjacent frames.')
gflags.DEFINE_boolean('jitter', False, 'Jitter initial location and orientation.')
gflags.DEFINE_boolean('obstacle', False, 'Place an unseen obstacle randomly along the path.')
gflags.DEFINE_float('obstacle_offset', 0.0, '')
gflags.DEFINE_boolean('noisy_actuation', False, '')
gflags.DEFINE_integer('seed', 0, '')
gflags.DEFINE_boolean('visualize', False, '')
gflags.DEFINE_boolean('save_screenshot', False, '')
gflags.DEFINE_string('screenshot_dir', '/tmp/screenshots', '')
gflags.DEFINE_boolean('save_trace', False, '')
gflags.DEFINE_string('trace_save_dir', '/tmp/traces/', '')
gflags.DEFINE_string('persistent_servers_cfg', '', '')
gflags.DEFINE_float('clip_velocity', 0.5, '')
gflags.DEFINE_float('zoom', 1.0, '')
FLAGS = gflags.FLAGS
FLAGS(sys.argv)


if FLAGS.persistent_servers_cfg:
    persistent_servers_cfg = FLAGS.persistent_servers_cfg
else:
    persistent_servers_cfg = get_default_persistent_server_config()


model = model_factory.get(FLAGS.model)()
tracker = model['tracker']

if FLAGS.agent:
    agent_name = FLAGS.agent
else:
    agent_name = model.get('robot', 'minirccar_240fov_rmp_v2')
agent = agent_factory.agents_dict[agent_name](noisy_actuation=FLAGS.noisy_actuation)

dataset, maps = envs.get(FLAGS.env)(
    # The models have gone through several iterations. The early versions do not
    # support different demo/rollout camera_z and fovs, which we fallback to what
    # are available.
    demo_camera_z=tracker.g.get('demo_camera_z', tracker.g.camera_z),
    rollout_camera_z=tracker.g.get('rollout_camera_z', tracker.g.camera_z),

    demo_fov=(np.deg2rad(tracker.g.get('demo_hfov', tracker.g.get('hfov', 0.0))),
              np.deg2rad(tracker.g.get('demo_vfov', tracker.g.get('vfov', 0.0)))),
    rollout_fov=(np.deg2rad(tracker.g.get('rollout_hfov', tracker.g.get('hfov', 0.0))),
                 np.deg2rad(tracker.g.get('rollout_vfov', tracker.g.get('vfov', 0.0)))),

    n_frame_min=FLAGS.n_frame,
    n_frame_max=FLAGS.n_frame,
    frame_interval=FLAGS.frame_interval,
    rand_frame_interval=False,
    normalize_wp=False)

dataset._init_once(FLAGS.seed)

e = Evaluator(tracker, dataset, maps, agent,
              stochastic_step=False,
              step_interval=2,
              jitter=FLAGS.jitter,
              place_obstacle=FLAGS.obstacle,
              obstacle_offset=FLAGS.obstacle_offset,
              clip_velocity=FLAGS.clip_velocity,
              visualize=FLAGS.visualize,
              save_trace=FLAGS.save_trace,
              trace_save_dir=FLAGS.trace_save_dir,
              save_screenshot=FLAGS.save_screenshot,
              screenshot_dir=FLAGS.screenshot_dir)

e.run(start_idx=FLAGS.start_idx, n_traj=FLAGS.n_traj, seed=FLAGS.seed)
