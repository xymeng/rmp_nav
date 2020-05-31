import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cv2
import math
import os
import numpy as np
import json

from rmp_nav.simulation.gibson_map import MakeGibsonMap
from rmp_nav.common.math_utils import rotate_2d
from rmp_nav.simulation import agent_factory, sim_renderer
from rmp_nav.simulation.agent_visualizers import FindAgentVisualizer
from rmp_nav.common.image_combiner import HStack, VStack
from rmp_nav.common.utils import get_project_root, str_to_dict, pprint_dict, get_gibson_asset_dir

from topological_nav.reachability.planning import NavGraph, NavGraphSPTM
from topological_nav.tools import eval_envs
from topological_nav.reachability.traj_following import RunnerGibsonDataset


def _make_maps(map_names):
    def load_maps(datadir, map_names, **kwargs):
        maps = []
        for name in map_names:
            maps.append(MakeGibsonMap(datadir, name, **kwargs))
        return maps
    map_names = [s.strip() for s in map_names]
    map_param = {}
    return load_maps(get_gibson_asset_dir(), map_names, **map_param)


def make_vis(map, agent):
    fig = plt.Figure(tight_layout=True)

    ax = fig.add_subplot(111)
    canvas = FigureCanvas(fig)
    canvas.draw()
    cm = matplotlib.cm.get_cmap('Dark2')

    # FIXME: hardcoded map
    vis = sim_renderer.SimRenderer(map, ax, canvas)
    vis.set_agents({
        agent.name: {
            'agent': agent,
            'visualizer': FindAgentVisualizer(agent)(
                ax,
                draw_control_point=False,
                traj_color=cm(0),
                obstacle_color='none', #cm(0),
                heading_color=cm(0),
                active_wp_color=cm(0),
                label=agent.name
            )}
    })
    vis.render(True)
    vis.render(True)
    return vis


def run_agent_online_planning(
        motion_policy, sparsifier, follower, dataset, map, agent,
        start_pos, start_heading,
        goal_pos, goal_heading,
        start_ob, goal_ob,
        goal_repr,
        nav_graph,
        traj_info_dict, vis,
        replan_dead_reckon_count=5,  # Replan after this number of dead reckon iterations
        online_planning=True,
        screenshot_dir=None,
        path_vis_dir=None):

    runner = RunnerGibsonDataset(
        dataset=dataset, map=map,
        motion_policy=motion_policy, sparsifier=sparsifier, follower=follower,
        agent=agent, agent_reverse=None, clip_velocity=0.5)

    agent.reset()
    agent.set_pos(start_pos)
    agent.set_heading(start_heading)
    agent.set_waypoints([goal_pos])

    def compute_overlap(pos1, heading1, pos2, heading2):
        if 'laser_agent' in motion_policy.g:
            return map.view_overlap(pos1, heading1, agent.lidar_sensor.fov,
                                    pos2, heading2, agent.lidar_sensor.fov, mode='lidar')
        else:
            return map.view_overlap(pos1, heading1, dataset.h_fov, pos2, heading2, dataset.h_fov,
                                    mode='plane')

    def post_step(step, path, extra, screenshot_repeat=1):
        if vis is None:
            return

        redraw = False
        vis.render(redraw, blit=False)

        nav_graph.draw_path(vis, path, traj_info_dict)
        active_anchor = runner.anchors[follower.cur_anchor_idx]

        if follower.cur_anchor_idx == len(path):
            active_wp = goal_pos
        else:
            traj_idx, anchor_idx = path[follower.cur_anchor_idx]
            active_wp = traj_info_dict[traj_idx]['samples'][anchor_idx]['pos']

        vis.plotter.scatter('active_wp', active_wp[0], active_wp[1],
                            marker='o', color='r', s=100.0)

        vis.blit()
        vis_img = cv2.cvtColor(vis.get_image(), cv2.COLOR_RGB2BGR)

        state_strs = ['search', 'follow', 'dead reckoning']

        text_y = 32
        cv2.putText(vis_img, 'step %d' % step,
                    (5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2); text_y += 32
        cv2.putText(vis_img, 'state: %s' % state_strs[follower.state],
                    (5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2); text_y += 32

        if extra['reachability'] is not None:
            cv2.putText(vis_img, 'reachability: %.2f' % extra['reachability'],
                        (5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2); text_y += 32

        vis_ob = cv2.cvtColor(
            cv2.resize(ob.transpose((1, 2, 0)), dsize=(0, 0), fx=4.0, fy=4.0),
            cv2.COLOR_RGB2BGR)

        vis_anchor = cv2.cvtColor(
            cv2.resize(active_anchor['ob_repr'].transpose((1, 2, 0)),
                       dsize=(0, 0), fx=4.0, fy=4.0),
            cv2.COLOR_RGB2BGR)

        text_ob = np.zeros((32, vis_ob.shape[1], 3), np.uint8)
        text_anchor = np.zeros((32, vis_anchor.shape[1], 3), np.uint8)

        cv2.putText(text_ob, 'observation', (5, 28), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(text_anchor, 'target', (5, 28), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        vis_img = HStack(vis_img, VStack(
            text_ob,
            (vis_ob * 255).astype(np.uint8),
            text_anchor,
            (vis_anchor * 255).astype(np.uint8)
        ))

        follower_vis = follower.make_vis_img(agent=agent)
        if follower_vis is not None:
            vis_img = VStack(follower_vis, vis_img)

        cv2.imshow('vis', vis_img)
        cv2.waitKey(1)

        nonlocal screenshot_idx
        if screenshot_dir is not None:
            for _ in range(screenshot_repeat):
                cv2.imwrite('%s/%05d.tiff' % (screenshot_dir, screenshot_idx), vis_img)
                screenshot_idx += 1

    screenshot_idx = 0
    expected_steps = 1500  # ball park value

    reached = 0.0
    step = 0

    def optimize_plan(ob, prev_prob):
        path, log_prob, _ = nav_graph.find_path(ob, goal_repr,
                                                edge_add_thres=0.5, allow_subgraph=True)
        if path is None:
            return None

        if path_vis_dir is not None:
            nav_graph.visualize_path(
                map, traj_info_dict, path,
                start_pos=agent.pos, start_heading=agent.heading,
                goal_pos=goal_pos, goal_heading=goal_heading,
                save_file=os.path.join(path_vis_dir, '%05d.svg' % step))

            cv2.imwrite(
                os.path.join(path_vis_dir, 'ob-%05d.tiff' % step),
                (cv2.cvtColor(ob.transpose(1, 2, 0), cv2.COLOR_RGB2BGR) * 255).astype(np.uint8))

            next_anchor = nav_graph.graph.nodes[path[0]]['ob_repr']
            cv2.imwrite(
                os.path.join(path_vis_dir, 'next-%05d.tiff' % step),
                (cv2.cvtColor(next_anchor.transpose(1, 2, 0), cv2.COLOR_RGB2BGR) * 255).astype(np.uint8))

            if step == 0:
                cv2.imwrite(
                    os.path.join(path_vis_dir, 'goal.tiff'),
                    (cv2.cvtColor(goal_ob.transpose(1, 2, 0), cv2.COLOR_RGB2BGR) * 255).astype(np.uint8))

        prob = math.exp(log_prob)
        if prob > prev_prob or abs(log_prob) < 1e-5:
            anchors = [nav_graph.graph.nodes[_] for _ in path]
            runner.set_anchors(anchors + [{'ob_repr': goal_ob, 'dst_repr': goal_repr}])
            return path, prob

        return None

    plan_ret = optimize_plan(start_ob, 0.0)
    if plan_ret is None:
        return 0.0

    current_path, current_prob = plan_ret
    print('optimize_plan: current_path %s' % current_path)

    trace = []

    while True:
        print('step %d' % step)
        ob, wp, extra = runner.step()

        trace.append(np.array(agent.pos, copy=True))

        if (follower.state == follower.DEAD_RECON
            and follower.dead_reckon_iter >= replan_dead_reckon_count
            and (follower.dead_reckon_iter - replan_dead_reckon_count) % 3 == 0) \
                and online_planning:

            # Lower the probability threshold if dead reckon too long.
            attenuation = 0.99 ** (follower.dead_reckon_iter - replan_dead_reckon_count)
            plan_ret = optimize_plan(ob, prev_prob=current_prob * attenuation)
            if plan_ret is None and current_path is None:
                # FIXME: never reach here because current_path will never be None.
                break
            if plan_ret is not None:
                current_path, current_prob = plan_ret

        if wp is None:
            print('failed to get a good waypoint.')
            break

        if agent.reached_goal(relax=3.0):
            print('reached')
            reached = 1.0
            break
        elif agent.collide():
            print('collide')
            reached = 0.0
            break
        else:
            overlap_ratios = compute_overlap(agent.pos, agent.heading, goal_pos, goal_heading)
            if overlap_ratios[0] > 0.7 and overlap_ratios[1] > 0.7:
                print('reached (overlap)')
                reached = 1.0
                break

        post_step(step, current_path, extra, screenshot_repeat=1)
        step += 1
        if step > expected_steps:
            print('timeout')
            break

    if step > 0:
        post_step(step, current_path, extra, screenshot_repeat=10)

    return reached


def exec_plan2(model, nav_graph, dataset, agent,
               start_pos, start_heading, goal_poses, goal_headings,
               map, traj_info_dict,
               online_planning=False,
               path_vis_file='',
               vis=None,
               screenshot_dir=None,
               **kwargs):
    """
    :param shortcut_bad_path: True to treat bad path as execution failure.
    """
    dataset.agent.set_map(map)

    dataset.agent.set_pos(start_pos)
    dataset.agent.set_heading(start_heading)
    start_ob = dataset._render_agent_view(map.name)

    def find_best_goal():
        best_pos, best_heading, best_ob, best_dst_repr, best_ll, best_path, best_extra = [None] * 7

        for pos, heading in zip(goal_poses, goal_headings):
            dataset.agent.set_pos(pos)
            dataset.agent.set_heading(heading)
            ob = dataset._render_agent_view(map.name)
            dst_repr = model['sparsifier'].get_dst_repr_single(ob)

            # edge_add_thres doesn't really matter. Using a higher value can improve efficiency
            # by not adding too many low-prob edges.
            path, log_likelihood, extra = nav_graph.find_path(
                start_ob, dst_repr, edge_add_thres=0.5, allow_subgraph=True)

            if path is None:
                continue

            if best_ll is None or log_likelihood > best_ll:
                best_pos, best_heading, best_ob, best_dst_repr, best_ll, best_path, best_extra = \
                    pos, heading, ob, dst_repr, log_likelihood, path, extra

        return best_pos, best_heading, best_ob, best_dst_repr, best_ll, best_path, best_extra

    goal_pos, goal_heading, goal_ob, goal_repr, log_likelihood, path, extra = find_best_goal()

    result = {
        'log_likelihood': log_likelihood,
        'success': False
    }

    if path is None:
        print('path not found')
    else:
        likelihood = math.exp(log_likelihood)
        print('likelihood: %.2f' % likelihood)

        if path_vis_file:
            nav_graph.visualize_path(map, traj_info_dict, path,
                                     start_pos=start_pos, start_heading=start_heading,
                                     goal_pos=goal_pos, goal_heading=goal_heading,
                                     save_file=path_vis_file, title='%.2f' % likelihood)
            print('save path visualization to %s' % path)

        print('path: %s' % (path,))
        if extra is not None:
            print('start -> %s %f' % (path[0], extra['transition_probs'][0]))
            for i in range(len(path) - 1):
                n1 = nav_graph.graph.nodes[path[i]]['ob_repr']
                n2 = nav_graph.graph.nodes[path[i + 1]]['dst_repr']
                p = model['sparsifier'].predict_reachability(n1, n2)
                print('%s -> %s %f %f' % (
                    path[i], path[i + 1],
                    math.exp(-nav_graph.graph.edges[path[i], path[i + 1]]['weight']), p))
            print('%s -> goal %f' % (path[-1], extra['transition_probs'][-1]))

        plt.switch_backend('agg')
        reached = run_agent_online_planning(
            model['motion_policy'], model['sparsifier'], model['follower'],
            dataset, map, agent,
            start_pos, start_heading, goal_pos, goal_heading,
            start_ob, goal_ob, goal_repr,
            nav_graph, traj_info_dict, vis,
            online_planning=online_planning,
            screenshot_dir=screenshot_dir,
            **kwargs)

        if reached > 0.5:
            result['success'] = True

    return result


def _jitter(pos, heading, agent, rng):
    heading_perp = rotate_2d(np.array([math.cos(heading), math.sin(heading)], np.float32),
                             math.pi / 2)

    max_try = 5
    for i in range(max_try):
        # Only jitter position sideways
        new_pos = pos + heading_perp * np.clip(rng.randn() * 0.3, -0.3, 0.3)
        new_heading = heading + rng.randn() * 0.3 * np.deg2rad(30)
        agent.set_pos(new_pos)
        agent.set_heading(new_heading)
        if not agent.collide(tolerance=-0.3):
            return new_pos, new_heading

    return pos, heading


def exec_plan_to_destination(model, nav_graph, start_pos, start_heading, goal, map, dataset, agent, **kwargs):
    traj_info_dict = nav_graph.extra['traj_info']

    goal_samples = []
    for traj in traj_info_dict.values():
        if eval(traj['attrs']['goal']) == goal:
            goal_samples.append(traj['samples'][-1])

    return exec_plan2(
        model, nav_graph, dataset, agent,
        start_pos, start_heading,
        [_['pos'] for _ in goal_samples], [_['heading'] for _ in goal_samples],
        map, nav_graph.extra['traj_info'],
        **kwargs)


if __name__ == '__main__':
    import gflags
    import sys
    from topological_nav.reachability import model_factory

    gflags.DEFINE_string('model', '', 'model name from model_factory')
    gflags.DEFINE_string('model_param', '',
                         'Comma separated key=value to overwrite default model parameters')
    gflags.DEFINE_string('graph_save_file', '', '')
    gflags.DEFINE_string('task', '', '')
    gflags.DEFINE_integer('start_idx', 0, '')
    gflags.DEFINE_integer('n_run', 100, '')
    gflags.DEFINE_string('start', None, 'Used for plan_between_dest_single task.')
    gflags.DEFINE_string('goal', None, 'Used for dest related tasks.')
    gflags.DEFINE_string('start_pos', None, 'Used for plan_to_dest_single task.')
    gflags.DEFINE_float('start_heading', None, 'In degrees. Used for plan_to_dest_single task.')
    gflags.DEFINE_float('graph_subset_ratio', 1.0, 'Use this subset of trajectories of the graph.')
    gflags.DEFINE_boolean('jitter_start', False, 'Jitter the starting position and heading.')
    gflags.DEFINE_boolean('shortcut_bad_path', False,
                          'True to treat a badly planned path as failure.')
    gflags.DEFINE_boolean('online_planning', False, '')
    gflags.DEFINE_integer('replan_dead_reckon_count', 5, 'Used for online planning.')
    gflags.DEFINE_integer('seed', 12345, '')
    gflags.DEFINE_boolean('visualize', False, '')
    gflags.DEFINE_string('screenshot_dir', None, '')
    gflags.DEFINE_string('device', 'cuda', '')

    FLAGS = gflags.FLAGS
    FLAGS(sys.argv)

    def frontend():
        sparsify_thres = 0.99
        model = model_factory.get(FLAGS.model)(device=FLAGS.device, **str_to_dict(FLAGS.model_param))
        print('model:\n%s' % pprint_dict(model))

        if 'naive' in FLAGS.model:
            nav_graph = NavGraphSPTM.from_save_file(model['sparsifier'], model['motion_policy'],
                                                    FLAGS.graph_save_file)
        else:
            nav_graph = NavGraph.from_save_file(model['sparsifier'], model['motion_policy'],
                                                sparsify_thres, FLAGS.graph_save_file)

        if FLAGS.graph_subset_ratio < 1.0:
            nav_graph.set_subset_ratio(FLAGS.graph_subset_ratio)

        env_name = nav_graph.extra['env']

        agent = agent_factory.agents_dict[model['agent']]()

        dataset = eval_envs.make(env_name, model['sparsifier'])

        map_name = dataset.map_names[0]
        map = _make_maps([map_name])[0]
        dataset._init_once(0)

        agent.set_map(map)

        vis = make_vis(map, agent) if FLAGS.visualize else None

        if FLAGS.task == 'plan_to_dest_single':
            start_pos = np.array([float(_) for _ in FLAGS.start_pos.split(',')], np.float32)
            exec_plan_to_destination(
                model, nav_graph,
                start_pos=start_pos, start_heading=np.deg2rad(FLAGS.start_heading),
                goal=eval(FLAGS.goal),
                map=map, dataset=dataset, agent=agent,
                replan_dead_reckon_count=FLAGS.replan_dead_reckon_count,
                online_planning=FLAGS.online_planning,
                path_vis_dir=None,
                path_vis_file=None,
                vis=vis,
                screenshot_dir=FLAGS.screenshot_dir)

        else:
            print('Unknown task: %s' % FLAGS.task)
            exit(0)

    frontend()
