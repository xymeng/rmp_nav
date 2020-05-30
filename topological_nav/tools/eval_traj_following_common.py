import math
import numpy as np
import cv2
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from rmp_nav.common.image_combiner import VStack, HStack
from rmp_nav.simulation import sim_renderer
from rmp_nav.simulation.agent_visualizers import FindAgentVisualizer
from rmp_nav.simulation.map_visualizer import FindMapVisualizer
from rmp_nav.simulation.gibson_map import MakeGibsonMap
from rmp_nav.simulation.map_utils import path_length
from rmp_nav.common.utils import get_gibson_asset_dir

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


class EvaluatorBase(object):
    def __init__(self,
                 dataset=None, sparsifier=None, motion_policy=None, follower=None,
                 agent=None,
                 agent_reverse=None,
                 sparsify_thres=None,
                 visualize=False,
                 min_traj_len=100,
                 wp_norm_min_clip=2.0,
                 clip_velocity=None,
                 dry_run=False,
                 save_trace=False):
        """
        :param agent_reverse:  optional agent that is used if the waypoint is behind the agent.
                               If not specified agent is used for both.
        :param sparsify_thres: threshold for sparsifying the trajectory.
        :param dry_run: If True then the evaluator will not run the agent but only sparsify the
                        the trajectories.
        :param save_trace: save a visualization image of the agent's trace and also save individual
                           observations.

        """
        super(EvaluatorBase, self).__init__()
        self.dataset = dataset
        self.sparsifier = sparsifier
        self.motion_policy = motion_policy
        self.follower = follower
        self.agent = agent
        self.agent_reverse = agent_reverse
        self.sparsify_thres = sparsify_thres
        self.wp_norm_min_clip = wp_norm_min_clip
        self.min_traj_len = min_traj_len
        self.clip_velocity = clip_velocity
        self.dry_run = dry_run

        self.map_cache = {}

        self.cur_map = None
        self.cur_traj_idx = None
        self.cur_traj = None
        self.cur_anchor_idxs = None
        self.cur_traj_imgs = None
        self.cur_anchor_imgs = None
        self.cur_extra = None
        self.cur_ob = None
        self.cur_step = None

        self.save_trace = save_trace

        self.visualize = visualize
        self.fig = None
        self.ax = None
        self.vis_dict = {}

        print('sparsify thres:', sparsify_thres)
        print('search thres:', follower.search_thres)
        print('follow thres:', follower.follow_thres)

    def _init_visualizer(self, map):
        fig = plt.Figure(tight_layout=True)
        ax = fig.add_subplot(111)
        canvas = FigureCanvas(fig)
        canvas.draw()
        cm = matplotlib.cm.get_cmap('Dark2')

        # FIXME: hardcoded map
        vis = sim_renderer.SimRenderer(map, ax, canvas)
        vis.set_agents({
            self.agent.name: {
                'agent': self.agent,
                'visualizer': FindAgentVisualizer(self.agent)(
                    ax,
                    draw_control_point=False,
                    traj_color=cm(0),
                    obstacle_color=cm(0),
                    heading_color=cm(0),
                    active_wp_color=cm(0),
                    label=self.agent.name
                )}
        })

        # FIXME: has to render twice at the beginning, otherwise the rendering won't look correct.
        vis.render(True)
        vis.render(True)
        self.vis_dict[map.name] = vis

    def _render(self, sample):
        pos = sample['pos']
        heading = sample['heading']
        self.dataset.agent.pos = pos
        self.dataset.agent.heading = heading
        return self.dataset._render_agent_view(self.cur_map.name)

    def _compute_overlap(self, pos1, heading1, pos2, heading2):
        return self.cur_map.view_overlap(pos1, heading1, self.dataset.h_fov,
                                         pos2, heading2, self.dataset.h_fov)

    def _prepare(self):
        pass

    def _post_step(self):
        pass

    def _find_good_start(self, traj):
        i = 0
        for i in range(1, len(traj)):
            dx, dy = traj[i]['pos'] - traj[i - 1]['pos']
            if math.sqrt(dx ** 2 + dy ** 2) < 0.01:
                continue
            heading = traj[i-1]['heading']
            dp = (math.cos(heading) * dx + math.sin(heading) * dy) / math.sqrt(dx ** 2 + dy ** 2)
            # print('dp', dp, 'dx', dx, 'dy', dy, 'heading', heading)
            if dp > 0.01:
                break
        return i

    def _make_runner(self, map):
        return RunnerGibsonDataset(
            self.dataset, map,
            motion_policy=self.motion_policy,
            sparsifier=self.sparsifier,
            follower=self.follower,
            agent=self.agent,
            agent_reverse=self.agent_reverse,
            wp_norm_min_clip=self.wp_norm_min_clip,
            clip_velocity=self.clip_velocity)

    def run(self, *, start_idx=0, n_traj=1, seed=None):
        """
        :param start_idx, n_traj: specify the range of trajectories to test
        :return:
        """
        agent = self.agent
        dataset = self.dataset
        sparsifier = self.sparsifier
        follower = self.follower

        dataset._init_once(0)  # FIXME: move to somewhere else?
        traj_ids = sorted(dataset.traj_ids, key=lambda _: _[1])

        total = 0
        successes = 0

        rng = np.random.RandomState(seed)

        map_cache = {}

        success_idxs = []
        failures_idxs = []
        skipped_idxs = []
        sparsification_ratios = []
        sparsification_ratios2 = []  # including contextual frames
        path_lengths = []
        completed_lengths = []

        def print_stats():
            print('avg sparse ratio: %.2f' % np.mean(sparsification_ratios))
            print('avg sparse ratio2: %.2f' % np.mean(sparsification_ratios2))
            print('path lengths: max %.2f min %.2f avg %.2f' % (
                np.max(path_lengths), np.min(path_lengths), np.mean(path_lengths)))

            if self.dry_run:
                return

            print('success rate: %.2f' % (successes / total))
            print('trajectories (successes): %s' % success_idxs)
            print('trajectories (failures): %s' % failures_idxs)
            print('trajectories (skipped): %s' % skipped_idxs)

            total_path_length = np.sum(path_lengths)
            print('total path length: %.2f' % total_path_length)

            total_completed_length = np.sum(completed_lengths)
            print('completed path length: %.2f' % total_completed_length)
            print('completion ratio: %.2f' % (total_completed_length / total_path_length))
            print('average cover rate: %.2f' %
                  np.mean(np.array(completed_lengths) / np.array(path_lengths)))

        run_idx = 0
        traj_idx = start_idx
        while run_idx < n_traj:
            dset_idx, traj_id = traj_ids[traj_idx]
            traj = dataset.fds[dset_idx][traj_id]
            map_name = traj.attrs['map'].decode('ascii')

            if map_name in map_cache:
                map = map_cache[map_name]
            else:
                map = _make_maps([map_name])[0]
                map_cache[map_name] = map
                if self.visualize:
                    self._init_visualizer(map)

            start_idx = self._find_good_start(traj)
            traj = traj[start_idx:]

            if len(traj) < self.min_traj_len:
                skipped_idxs.append(traj_idx)
                print('traj %d too short. skipped' % traj_idx)
                traj_idx += 1
                continue

            traj_points = np.array([_['pos'] for _ in traj])

            self.cur_traj_idx = traj_idx
            self.cur_traj = traj
            self.cur_map = map

            path_lengths.append(path_length([_['pos'] for _ in traj]))

            traj_imgs = [self._render(traj[_]) for _ in range(len(traj))]
            anchor_idxs = sparsifier.sparsify(traj_imgs, self.sparsify_thres)
            anchor_imgs = [traj_imgs[_] for _ in anchor_idxs]

            sparse_ratio = float(len(anchor_idxs)) / len(traj_imgs)
            print('traj %d sparsification ratio: %.2f' % (traj_idx, sparse_ratio))
            sparsification_ratios.append(sparse_ratio)
            sparsification_ratios2.append(
                sparsifier.compute_sparsification_ratio(len(traj_imgs), anchor_idxs))

            if self.dry_run:
                print_stats()
                run_idx += 1
                traj_idx += 1
                continue

            self.cur_traj_imgs = traj_imgs
            self.cur_anchor_idxs = anchor_idxs
            self.cur_anchor_imgs = anchor_imgs

            follower.reset()
            anchors = []
            for anchor_idx in anchor_idxs:
                anchors.append({
                    'ob_repr': self.sparsifier.get_ob_repr(traj_imgs, anchor_idx),
                    'dst_repr': self.sparsifier.get_dst_repr(traj_imgs, anchor_idx)
                })
            follower.set_anchors(anchors)

            # FIXME: Hack for openloop follower
            if hasattr(follower, 'set_interval'):
                follower.set_interval(int(self.sparsify_thres))

            src_pos, src_heading = traj[0]['pos'], traj[0]['heading']
            goal_pos, goal_heading = traj[-1]['pos'], traj[-1]['heading']

            agent.reset()
            agent.set_map(map)
            agent.set_pos(src_pos)
            agent.set_heading(src_heading)
            agent.set_waypoints([goal_pos])

            reached = 0.0
            early_stop = False

            self._prepare()

            runner = self._make_runner(map)

            trace = []

            closest_traj_sample_idx = 0

            for i in range(len(traj) * 2 + 100):  # Maximum 2x + 100 steps
                print('step %d' % i)
                self.cur_step = i

                trace.append(np.array(agent.pos, copy=True))

                self.cur_ob, wp, self.cur_extra = runner.step()
                if wp is None:
                    early_stop = True
                    break

                dists = np.linalg.norm(traj_points - agent.pos, axis=1)
                for j in range(closest_traj_sample_idx, len(dists)):
                    if dists[j] < 0.5:
                        closest_traj_sample_idx = j
                    else:
                        break

                if agent.reached_goal(relax=3.0):
                    print('%d: reached' % traj_idx)
                    reached = 1.0
                    early_stop = True
                    break
                elif agent.collide():
                    print('%d: collide' % traj_idx)
                    reached = 0.0
                    early_stop = True
                    break
                else:
                    overlap_ratios = self._compute_overlap(agent.pos, agent.heading,
                                                           goal_pos, goal_heading)
                    if overlap_ratios[0] > 0.7 and overlap_ratios[1] > 0.7:
                        print('%d: reached (overlap)' % traj_idx)
                        reached = 1.0
                        early_stop = True
                        break

                self._post_step()

            completed_lengths.append(path_length(traj_points[:closest_traj_sample_idx]))

            if not early_stop:
                print('%d: timeout' % traj_idx)

            total += 1
            successes += reached

            if reached > 0.5:
                success_idxs.append(traj_idx)
            else:
                failures_idxs.append(traj_idx)

            run_idx += 1
            traj_idx += 1

            print_stats()

            if self.save_trace:
                self._write_trace_visualization_image(
                    map, traj, trace, anchor_idxs, '/tmp/trace.pdf', figsize=(6, 4))
                exit(0)

        print('done.')

    def _write_trace_visualization_image(self, map, traj, trace, anchor_idxs, out_file,
                                         figsize=(18, 12), linewidth=3.0, linealpha=0.8):

        cm = matplotlib.cm.get_cmap('tab10')
        fig = Figure(figsize=figsize)
        FigureCanvas(fig)
        ax = fig.add_subplot(111)

        # Call draw() to prevent matplotlib from complaining about draw_artist() being called before
        # initial draw.
        fig.canvas.draw()

        vis = FindMapVisualizer(map)(map, ax)
        vis.draw_map()

        xs, ys = zip(*[_['pos'] for _ in traj])
        ax.plot(xs, ys, color=cm(0), linewidth=linewidth, alpha=linealpha)

        xs, ys = [xs[_] for _ in anchor_idxs], [ys[_] for _ in anchor_idxs]
        ax.scatter(xs, ys, marker='x', color=cm(0), s=50.0, alpha=0.9, linewidth=2.0)

        xs, ys = zip(*trace)
        ax.plot(xs, ys, color=cm(1), linewidth=linewidth, alpha=linealpha)

        ax.plot(xs[0], ys[0], 'o', color='r', markersize=20.0)
        ax.plot(xs[-1], ys[-1], 'r*', markersize=20.0)

        ax.set_xlim(min(xs) - 1.0, max(xs) + 1.0)
        ax.set_ylim(min(ys) - 1.0, max(ys) + 1.0)

        fig.tight_layout(pad=0)
        fig.savefig(out_file)


class EvaluatorReachability(EvaluatorBase):
    def __init__(self, save_screenshot=False, zoom=1.0, **kwargs):
        super(EvaluatorReachability, self).__init__(**kwargs)
        self.save_screenshot = save_screenshot
        self.zoom = zoom

    def _prepare(self):
        if self.cur_map.name in self.vis_dict:
            vis = self.vis_dict[self.cur_map.name]
            traj = self.cur_traj
            points = [_['pos'] for _ in traj]
            xs, ys = zip(*points)
            vis.set_limits(min(xs) - 1.0, max(xs) + 1.0, min(ys) - 1.0, max(ys) + 1.0)
            vis.render(True)
            self.screenshot_idx = 0

    def _post_step(self):
        if self.cur_map.name in self.vis_dict:
            vis = self.vis_dict[self.cur_map.name]
            traj = self.cur_traj
            anchor_idxs = self.cur_anchor_idxs
            ob = self.cur_ob
            extra = self.cur_extra
            follower = self.follower

            points = [_['pos'] for _ in traj]
            xs, ys = zip(*points)
            xs2, ys2 = zip(*[points[_] for _ in anchor_idxs])

            redraw = False
            if self.zoom != 1.0:
                x1, x2, y1, y2 = vis.get_limits()
                w, h = abs(x2 - x1), abs(y2 - y1)
                x, y = self.agent.pos
                if abs(x - (x2 + x1) / 2) / (w / 2) > 0.8 or abs(y - (y2 + y1) / 2) / (h / 2) > 0.8:
                    vis.reset_viewport()
                    vis.set_viewport(self.agent.pos[0], self.agent.pos[1], self.zoom)
                    redraw = True
            vis.render(redraw, blit=False)

            vis.plotter.plot('traj', xs, ys)
            vis.plotter.scatter('anchors', xs2, ys2, marker='x')

            active_wp = points[anchor_idxs[follower.cur_anchor_idx]]
            vis.plotter.scatter('active_wp', active_wp[0], active_wp[1],
                                marker='o', color='r', s=100.0)

            vis.blit()
            vis_img = cv2.cvtColor(vis.get_image(), cv2.COLOR_RGB2BGR)

            state_strs = ['search', 'follow', 'dead reckoning']

            text_y = 32
            cv2.putText(vis_img, 'traj %d' % self.cur_traj_idx,
                        (5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2); text_y += 32
            cv2.putText(vis_img, 'step %d' % self.cur_step,
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
                cv2.resize(self.cur_anchor_imgs[extra['anchor_idx']].transpose((1, 2, 0)),
                           dsize=(256, 256)),
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

            follower_vis = self.follower.make_vis_img()
            if follower_vis is not None:
                vis_img = VStack(follower_vis, vis_img)

            cv2.imshow('vis', vis_img)
            if self.save_screenshot:
                cv2.imwrite('/tmp/screenshots/%05d.tiff' % self.screenshot_idx, vis_img)
                self.screenshot_idx += 1

                if self.save_trace:
                    cv2.imwrite('/tmp/screenshots/ob-%05d.tiff' % self.screenshot_idx,
                                cv2.cvtColor((ob.transpose((1, 2, 0)) * 255).astype(np.uint8),
                                             cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
