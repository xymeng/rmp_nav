import os
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from rmp_nav.simulation.map_visualizer import FindMapVisualizer
from rmp_nav.simulation.agent_visualizers import FindAgentVisualizer


def sample_start_goal(map, rng):
    locs = map.get_reachable_locations()
    idxs = rng.choice(len(locs), 2, replace=False)
    return (map.path_coord_to_map_coord(*locs[idx]) for idx in idxs)


def sample_start_destination(map, rng):
    locs = map.get_reachable_locations()

    # Start location can be anywhere
    start_loc = locs[rng.randint(len(locs))]

    # Goal must fall within a particular region of a destination
    dests = map.get_destination_labels()
    dest_label = dests[rng.randint(len(dests))]

    dest_loc = map.destination_centroids[dest_label]
    goal_map_coord = map.path_coord_to_map_coord(*dest_loc)

    start_map_coord = map.path_coord_to_map_coord(*start_loc)

    return start_map_coord, goal_map_coord, dest_label


def gen_samples(map, name, agent, start, goal, rng, max_traj_len, no_planning,
                init_heading=None,
                align_init_heading=False,
                replan=False,
                timeout_as_failure=False,
                destination=None):
    """
    :param align_init_heading: make initial heading align with planned path if True.
    :param replan: run the planner at every step. destination must be specified.
    """
    print('map name:', name)
    print('start:', start)
    print('goal:', goal)

    if no_planning:
        waypoints = [goal]
    else:
        waypoints = map.find_path(start, goal)

    if waypoints is None:
        print('failed to find path')
        return None

    agent.reset()
    agent.set_pos(start)
    agent.set_waypoints(waypoints)

    if init_heading is not None:
        agent.set_heading(init_heading)
    elif align_init_heading:
        # Estimate rough path direction
        path_direction = np.array(waypoints[min(3, len(waypoints))]) - np.array(waypoints[0])
        agent.set_heading(np.arctan2(path_direction[1], path_direction[0]))
    else:
        agent.set_heading(float(rng.uniform(0.0, np.pi * 2.0)))

    print('heading:', agent.heading)

    samples = []

    discard = False

    while True:
        if len(samples) % 100 == 0:
            print('sample %d' % len(samples))

        # We need to set agent's state first before calling step(). Otherwise the measurement
        # will not correspond to the correct state.
        pos = np.array(agent.pos, copy=True)
        vel = np.array(agent.velocity, copy=True)
        heading = np.array(agent.heading, copy=True)
        angular_vel = np.array(agent.angular_velocity, copy=True)

        if replan:
            waypoints = map.find_path_destination(pos, destination)
            if waypoints is None:
                if agent.waypoints is not None:
                    agent.wp_idx = 0
                    print('failed to replan. use previous waypoint.')
                else:
                    print('failed to replan and no previous waypoint available. discard this trajectory.')
                    discard = True
                    break
            else:
                # Update agent's waypoints. Agent will recompute waypoint in step()
                agent.set_waypoints(waypoints)
                agent.wp_idx = 0

        agent.step(0.1)  # Call step() to get the depth value and also let the agent act.

        if agent.collide():
            print('collision occurred. discard this trajectory.')
            discard = True
            break

        depth = np.array(agent.depth_local, copy=True)

        # Note that this is independent of agent's position, so even though agent has acted, the
        # waypoint is still correct.
        # agent might post-process the waypoint, hence we use agent.goals_global[0] which contains
        # the post-processed waypoint.
        wp = agent.goals_global[0]

        assert np.max(depth) < 1e3

        samples.append((pos, depth, wp, vel, heading, angular_vel))

        if agent.stopped() or agent.reached_goal():
            break

        if len(samples) >= max_traj_len:
            print('timeout')
            write_visualization_image(samples, start, goal, agent, map, '/tmp/debug.svg')
            if timeout_as_failure:
                discard = True
            break

    if discard:
        return None

    return samples


def write_visualization_image(samples, start, goal, agent, map, out_file):
    traj = np.array([s[0] for s in samples])
    if len(traj) == 0:
        return

    fig = Figure(figsize=(18, 12))
    FigureCanvas(fig)
    ax = fig.add_subplot(111)

    # Call draw() to prevent matplotlib from complaining about draw_artist() being called before
    # initial draw.
    fig.canvas.draw()

    vis = FindMapVisualizer(map)(map, ax)
    vis.draw_map()

    agent_vis = FindAgentVisualizer(agent)(ax)
    agent_vis.draw_trajectory(traj[:, 0], traj[:, 1])

    ax.plot(start[0], start[1], 'ko', markerfacecolor='none')
    ax.plot(goal[0], goal[1], 'r*')

    ax.set_title(map.name)
    fig.savefig(out_file)


def run(maps, out_hd5_obj, num_traj, min_traj_length, max_traj_length, traj_image_dir, no_planning, replan,
        timeout_as_failure, agent, destination_mode, seed):
    dtype = np.dtype([
        ('pos', (np.float32, 2)),
        ('depth_local', (np.float32, agent.lidar_sensor.n_depth)),
        ('waypoint_global', (np.float32, 2)),
        ('velocity_global', (np.float32, 2)),
        ('heading', np.float32),
        ('angular_velocity', np.float32)
    ])

    rng_trajid = np.random.RandomState(seed)

    i = 0
    while i < num_traj:
        traj_id = rng_trajid.randint(0xffffffff)
        dset_name = '/%08x' % traj_id

        print('processing trajectory %d id: %08x' % (i, traj_id))

        if dset_name in out_hd5_obj:
            print('dataset %s exists. skipped' % dset_name)
            continue

        rng = np.random.RandomState(traj_id)

        map_idx = rng.randint(0, len(maps))

        m = maps[map_idx]

        agent.set_map(m)

        if destination_mode:
            start, goal, dest = sample_start_destination(m, rng)
        else:
            start, goal = sample_start_goal(m, rng)
            dest = None

        samples = gen_samples(
            map=m,
            name=m.name,
            agent=agent,
            start=start,
            goal=goal,
            rng=rng,
            max_traj_len=max_traj_length,
            no_planning=no_planning,
            replan=replan,
            destination=dest,
            timeout_as_failure=timeout_as_failure)

        if samples is None:
            continue

        if len(samples) < min_traj_length:
            continue

        if traj_image_dir != '':
            traj_img_filename = os.path.join(traj_image_dir, '%08x.svg' % traj_id)
            write_visualization_image(samples, start, goal, agent, m, traj_img_filename)

        dset = out_hd5_obj.create_dataset(
            dset_name,
            data=np.array(samples, dtype=dtype),
            maxshape=(None,), dtype=dtype)

        dset.attrs.create('map', m.name.encode('ascii'))
        if destination_mode:
            dset.attrs.create('destination', str(dest, 'ascii'))

        out_hd5_obj.flush()

        i += 1


def run_pairwise_destination(map, out_hd5_obj, traj_image_dir, agent, seed, initial_headings):
    """
    Generate a trajectory between all pairs of destinations.
    If there are N destinations, this will generate N x (N - 1) destinations.
    :param map:
    :param out_hd5_obj:
    :param traj_image_dir:
    :param agent:
    :param seed:
    :return:
    """
    dtype = np.dtype([
        ('pos', (np.float32, 2)),
        ('depth_local', (np.float32, agent.lidar_sensor.n_depth)),
        ('waypoint_global', (np.float32, 2)),
        ('velocity_global', (np.float32, 2)),
        ('heading', np.float32),
        ('angular_velocity', np.float32)
    ])

    rng_trajid = np.random.RandomState(seed)
    agent.set_map(map)

    def get_destination_coord(dest_label):
        dest_loc = map.destination_centroids[dest_label]
        goal_map_coord = map.path_coord_to_map_coord(*dest_loc)
        return goal_map_coord

    dests = list(map.get_destination_labels())
    print('destinations:', dests)

    count = 0

    for i in range(len(dests)):
        for j in range(len(dests)):
            if i == j:
                continue

            traj_id = rng_trajid.randint(0xffffffff)
            dset_name = '/%08x' % traj_id
            print('processing trajectory %d id: %08x' % (count, traj_id))

            if dset_name in out_hd5_obj:
                print('dataset %s exists. skipped' % dset_name)
                continue

            rng = np.random.RandomState(traj_id)

            traj_img_filename = None
            if traj_image_dir != '':
                traj_img_filename = os.path.join(traj_image_dir, '%08x.svg' % traj_id)

            start = get_destination_coord(dests[i])
            goal = get_destination_coord(dests[j])

            if initial_headings:
                init_heading = initial_headings[dests[i]]
            else:
                init_heading = None

            samples = gen_samples(
                map=map,
                name=map.name,
                agent=agent,
                start=start,
                goal=goal,
                rng=rng,
                init_heading=init_heading,
                align_init_heading=init_heading is None,
                max_traj_len=2000,
                no_planning=False,
                replan=False,
                destination=None,
                timeout_as_failure=True)

            if samples is None:
                raise ValueError("Failed to generate trajectory.")

            write_visualization_image(samples, start, goal, agent, map, traj_img_filename)

            dset = out_hd5_obj.create_dataset(
                dset_name,
                data=np.array(samples, dtype=dtype),
                maxshape=(None,), dtype=dtype)

            dset.attrs.create('map', map.name.encode('ascii'))
            dset.attrs.create('start', str(dests[i]).encode('ascii'))
            dset.attrs.create('goal', str(dests[j]).encode('ascii'))

            out_hd5_obj.flush()
            count += 1
