import numpy as np
import gflags
import sys
import glob
from easydict import EasyDict
import os
from rmp_nav.neural.common.dataset import DatasetVisualGibson
from rmp_nav.common.utils import get_project_root, get_data_dir, get_gibson_asset_dir, get_config_dir
from rmp_nav.simulation import agent_factory
from topological_nav.reachability import model_factory
from topological_nav.tools import eval_envs
from topological_nav.tools.eval_traj_following_common import EvaluatorReachability


gflags.DEFINE_string('env', 'space8', '')
gflags.DEFINE_string('model', 'model_12env_v2_future_pair_proximity_z0228', '')
gflags.DEFINE_boolean('dry_run', False, '')
gflags.DEFINE_float('sparsify_thres', 0.99, '')
gflags.DEFINE_integer('start_idx', 0, '')
gflags.DEFINE_integer('n_traj', 100, '')
gflags.DEFINE_float('clip_velocity', 0.5, 'Limit the max velocity.')
gflags.DEFINE_boolean('visualize', True, '')
gflags.DEFINE_boolean('save_screenshot', False, '')
gflags.DEFINE_float('zoom', 1.0, '')
FLAGS = gflags.FLAGS
FLAGS(sys.argv)


model = model_factory.get(FLAGS.model)()
sparsifier = model['sparsifier']
motion_policy = model['motion_policy']
traj_follower = model['follower']
agent = agent_factory.agents_dict[model['agent']]()


e = EvaluatorReachability(dataset=eval_envs.make(FLAGS.env, sparsifier),
                          sparsifier=sparsifier,
                          motion_policy=motion_policy,
                          follower=traj_follower,
                          agent=agent,
                          agent_reverse=None,
                          sparsify_thres=FLAGS.sparsify_thres,
                          clip_velocity=FLAGS.clip_velocity,
                          visualize=FLAGS.visualize,
                          save_screenshot=FLAGS.save_screenshot,
                          zoom=FLAGS.zoom,
                          dry_run=FLAGS.dry_run)

e.run(start_idx=FLAGS.start_idx, n_traj=FLAGS.n_traj)
