import sys
import random
import h5py
import gflags
import yaml

from rmp_nav.simulation.map_factory import make_map
from rmp_nav.simulation import agent_factory
from rmp_nav.data_generation import gen_trajectories_common


if __name__ == '__main__':
    gflags.DEFINE_integer('num_traj', 10000, '')
    gflags.DEFINE_string('out_file', 'samples.hd5', '')
    gflags.DEFINE_string('traj_image_dir', '',
                         'An optional directory to store the trajectory visualizations.')
    gflags.DEFINE_string('map_type', '', 'Currently support "gibson" and "gibson2"')
    gflags.DEFINE_string('maps', '', 'Comma separated map names.')
    gflags.DEFINE_string('map_param', '', 'An optional YAML file containing map parameters')
    gflags.DEFINE_boolean('no_planning', False, 'Disable planner.')
    gflags.DEFINE_boolean('replan', False, 'Run the planner at every step.')
    gflags.DEFINE_boolean('destination_mode', False, 'Use pre-defined destinations as goals.')
    gflags.DEFINE_integer('min_traj_length', 200, 'Minimum trajectory length.')
    gflags.DEFINE_integer('max_traj_length', 2000, 'Maximum trajectory length.')
    gflags.DEFINE_boolean('timeout_as_failure', False, 'Skip trajectory that exceeds maximum length.')
    gflags.DEFINE_string('agent', '', 'The agent name. See agent_factory.py')
    gflags.DEFINE_integer('seed', 123456, '')

    FLAGS = gflags.FLAGS

    try:
        FLAGS(sys.argv)
    except gflags.FlagsError as e:
        print("%s\nUsage: %s ARGS\n%s" % (e, sys.argv[0], FLAGS))
        sys.exit(1)

    agent = agent_factory.agents_dict[FLAGS.agent]()
    print('using agent %s' % FLAGS.agent)

    map_names = [s.strip() for s in FLAGS.maps.split(',')]
    print('maps:', map_names)

    rng = random.Random(FLAGS.seed)
    map_param = {}
    if FLAGS.map_param != '':
        import yaml
        param = yaml.load(open(FLAGS.map_param).read(), Loader=yaml.SafeLoader)
        # Randomly choose a parameter setting
        keys = sorted(param.keys())
        for key in keys:
            v = rng.choice(param[key])
            map_param[key] = v

    maps = [make_map(FLAGS.map_type, name, **map_param) for name in map_names]
    f = h5py.File(FLAGS.out_file, 'a')
    f.attrs.create('agent', agent.name.encode('ascii'))
    f.attrs.create('map_type', FLAGS.map_type.encode('ascii'))
    f.attrs.create('map_param', yaml.dump(maps[0].get_parameters()).encode('ascii'))

    gen_trajectories_common.run(
        maps,
        f,
        FLAGS.num_traj,
        FLAGS.min_traj_length,
        FLAGS.max_traj_length,
        FLAGS.traj_image_dir,
        FLAGS.no_planning,
        FLAGS.replan,
        FLAGS.timeout_as_failure,
        agent,
        FLAGS.destination_mode,
        FLAGS.seed)
