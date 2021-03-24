import bisect
import os

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from rmp_nav.common.image_combiner import HStack, VStack
from rmp_nav.common.text_utils import put_text
from rmp_nav.simulation.map_utils import cum_path_length, path_length
from rmp_nav.simulation.map_visualizer import FindMapVisualizer
from rmp_nav.simulation.agent_visualizers import FindAgentVisualizer
from rmp_nav.simulation import sim_renderer


class Evaluator(object):
    def __init__(self, tracker, dataset, maps, agent,
                 clip_velocity=None,
                 jitter_overlap_thres=(0.6, 0.6),
                 stochastic_step=False, step_interval=1,
                 place_obstacle=False, obstacle_offset=0.0,
                 visualize=False, figsize=None,
                 save_screenshot=False, screenshot_dir='/tmp/screenshots',
                 jitter=False, save_trace=False, trace_save_dir=''):
        self.tracker = tracker
        self.dataset = dataset
        self.maps = maps
        self.agent = agent
        self.clip_velocity = clip_velocity
        self.stochastic_step = stochastic_step
        self.step_interval = step_interval
        self.jitter = jitter
        self.jitter_overlap_thres = jitter_overlap_thres
        self.steps_margin = 100
        self.place_obstacle = place_obstacle
        self.obstacle_offset = obstacle_offset

        self.rng = np.random.RandomState(12345)

        self.cur_map = None
        self.cur_traj_xy = None
        self.cur_start_ob = None
        self.cur_goal_ob = None
        self.cur_ob = None
        self.cur_trace = None
        self.cur_step = 0

        self.fig = None
        self.figsize = figsize
        self.ax = None
        self.vis = None
        self.vis_dict = {}

        self.save_trace = save_trace
        self.trace_save_dir = trace_save_dir

        self.visualize = visualize
        self.save_screenshot = save_screenshot
        self.screenshot_dir = screenshot_dir
        self.screenshot_idx = 0
        if self.visualize:
            for map in maps.values():
                self._init_visualizer(map)

    def _init_visualizer(self, map):
        fig = plt.Figure(tight_layout=True)
        ax = fig.add_subplot(111)
        canvas = FigureCanvas(fig)
        canvas.draw()
        vis = sim_renderer.SimRenderer(map, ax, fig.canvas)
        cm = matplotlib.cm.get_cmap('Dark2')
        agent_vis = FindAgentVisualizer(self.agent)(
            ax,
            draw_control_point=False,
            traj_color=cm(0),
            obstacle_color=cm(0),
            heading_color=cm(0),
            active_wp_color=cm(0),
            label=self.agent.name,
            show_obstacles=False,
            show_rmps=False,
        )
        vis.set_agents({
            self.agent.name: {
                'agent': self.agent,
                'visualizer': agent_vis
            }
        })

        vis.render(True)
        vis.render(True)
        self.vis_dict[map.name] = vis

    def _jitter(self, agent):
        pos, heading = np.copy(agent.pos), float(agent.heading)
        for i in range(5):
            dpos = np.clip(self.rng.randn() * 0.3, -1.0, 1.0)  # 99% prob in (-1, 1)
            dheading = self.rng.randn() * 0.3 * np.deg2rad(45)  # 99% prob in +/- 45deg.
            agent.set_pos(dpos + pos)
            agent.set_heading(dheading + heading)
            if agent.collide(tolerance=0.0, inflate=3.0):
                continue
            overlaps = self._compute_overlap(agent.map, agent.pos, agent.heading, pos, heading)
            if overlaps[0] < self.jitter_overlap_thres[0] or overlaps[1] < self.jitter_overlap_thres[1]:
                continue
            return
        agent.set_pos(pos)
        agent.set_heading(heading)

    def _find_closest_traj_sample_idx(self, pos, traj_points):
        return int(np.argmin(np.linalg.norm(pos - traj_points, 2, axis=1)))

    def _diverge(self, pos, traj_points, thres=1.0):
        idx = self._find_closest_traj_sample_idx(pos, traj_points)
        return np.linalg.norm(pos - traj_points[idx], ord=2) > thres

    def _compute_overlap(self, map, pos1, heading1, pos2, heading2):
        return map.view_overlap(pos1, heading1,
                                self.dataset.rollout_fov[0],
                                pos2, heading2,
                                self.dataset.demo_fov[0])

    def _place_obstacle(self, obj, offset, positions, headings, rng, map):
        where_to_place = list(range(len(positions)))
        rng.shuffle(where_to_place)

        for i in where_to_place:
            heading = headings[i]

            dx, dy = np.cos(heading + np.pi * 0.5), np.sin(heading + np.pi * 0.5)

            if obj == 'trashcan':
                r = 0.15
                cr = 0.6
                # Left side of the trajectory
                x, y = positions[i, 0] + dx * offset, positions[i, 1] + dy * offset
                if map.put_cylinder_obstacle(x, y, r, check_collision=True, collision_inflation_radius=cr):
                    return x, y

                # Right side of the trajectory
                x, y = positions[i, 0] - dx * offset, positions[i, 1] - dy * offset
                if map.put_cylinder_obstacle(x, y, r, check_collision=True, collision_inflation_radius=cr):
                    return x, y

        return None

    def run_single(self, sample, map, traj_idx):
        agent = self.agent
        tracker = self.tracker
        rng = self.rng

        traj_len = sample['demo_traj_len']
        print('traj_len:', traj_len)

        positions = sample['pos'][:traj_len]
        headings = sample['headings'][:traj_len]
        obs = sample['demo_obs'][:traj_len]
        start_ob = sample['demo_start_ob']
        goal_ob = sample['demo_goal_ob']

        if self.place_obstacle:
            r = self._place_obstacle('trashcan', self.obstacle_offset, positions[10: -10], headings[10: -10], rng, map)
            if r is None:
                print("failed to place obstacle")
                return {
                    'outcome': 'skipped'
                }
            if r is not None:
                self.dataset.sim_clients[map.name][0].PlaceTrashcan(r[0], r[1], 0)

        self.cur_map = map
        self.cur_traj_xy = positions
        cum_path_len = cum_path_length(positions)
        self.cur_progresses = [0.0] + (cum_path_len / cum_path_len[-1]).tolist()
        self.cur_traj_headings = headings

        self.cur_start_ob = obs[0]
        self.cur_goal_ob = obs[-1]
        self.cur_trace = []
        self.cur_progress = None
        self.cur_step = 0

        agent.reset()
        agent.set_map(map)
        agent.set_pos(positions[0])
        agent.set_heading(headings[0])

        goal_pos, goal_heading = positions[-1], headings[-1]
        agent.set_waypoints([goal_pos])

        def check_if_stuck(tolerance=0.2):
            if len(self.cur_trace) < 100:
                return False
            locs = np.array(self.cur_trace[-100:])
            mean_loc = np.mean(locs, axis=0)
            return np.max(np.linalg.norm(locs - mean_loc[None], axis=1)) < tolerance

        self._prepare()

        outcome = 'failure (timeout)'

        tracker.reset()

        if tracker.__class__.__name__ == 'ProgressTrackerRPF':
            embedding = tracker.compute_traj_embedding(obs)
        else:
            embedding = tracker.compute_traj_embedding(obs)[-1]

        if self.jitter:
            self._jitter(self.agent)

        def get_ob():
            return self.dataset.render(agent.pos, agent.heading, map.name,
                                       camera_z=tracker.g.get('rollout_camera_z', tracker.g.camera_z),
                                       h_fov=np.deg2rad(tracker.g.get('rollout_hfov', tracker.g.get('hfov', 0.0))),
                                       v_fov=np.deg2rad(tracker.g.get('rollout_vfov', tracker.g.get('vfov', 0.0))))

        def step_agent(wp, n_step):
            wp_global = agent.local_to_global(wp)
            for i in range(n_step):
                agent.step(0.1, waypoint=agent.global_to_local(wp_global), max_vel=self.clip_velocity)
                if agent.collide():
                    return False
            return True

        max_steps = traj_len * self.dataset.frame_interval * 3 + self.steps_margin
        step = 0
        while step < max_steps:
            if step % 50 == 0:
                print('step %d' % step)

            if self.stochastic_step:
                n_step = self.rng.randint(1, self.dataset.frame_interval + 1)  # Randomize step size
            else:
                n_step = self.step_interval

            ob = get_ob()
            self.cur_ob = ob

            progress, wp = tracker.step(start_ob, goal_ob, embedding, ob)
            self.cur_progress = progress

            if wp is None:
                outcome = 'failure (failed to compute wp)'
                break

            step_agent(wp, n_step)

            self.cur_trace.append(np.copy(agent.pos))

            if agent.reached_goal(relax=3.0):
                outcome = 'success (position)'
                break
            if agent.collide():
                outcome = 'failure (collide)'
                break
            elif self._diverge(agent.pos, positions):
                outcome = 'failure (diverge)'
                break
            elif check_if_stuck():
                outcome = 'failure (stuck)'
                break
            else:
                overlap_ratios = self._compute_overlap(map, agent.pos, agent.heading, goal_pos, goal_heading)
                if overlap_ratios[0] > 0.7 and overlap_ratios[1] > 0.7:
                    outcome = 'success (overlap)'
                    break

            self._post_step()
            step += n_step
            self.cur_step = step

        self._post_step(repeat=10)

        if self.save_trace:
            self._write_trace_visualization_image(
                map, positions, self.cur_trace, os.path.join(self.trace_save_dir, '%05d.pdf' % traj_idx),
                figsize=(8, 5))

            im = cv2.cvtColor(self.cur_start_ob.transpose((1, 2, 0)), cv2.COLOR_RGB2BGR)
            im = (im * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(self.trace_save_dir, '%05d-start.png' % traj_idx), im)

            im = cv2.cvtColor(self.cur_goal_ob.transpose((1, 2, 0)), cv2.COLOR_RGB2BGR)
            im = (im * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(self.trace_save_dir, '%05d-goal.png' % traj_idx), im)

        map.clear_obstacles()

        return {
            'outcome': outcome,
            'trace': self.cur_trace,
            'path_len': cum_path_len[-1],
        }

    def run(self, start_idx=0, n_traj=100, seed=12345):
        dataset = self.dataset

        rng = np.random.RandomState(seed)
        self.rng = rng

        success_idxs = []
        failures_idxs = []
        skipped_idxs = []
        spls = []

        sample_idxs = list(range(len(dataset)))
        rng.shuffle(sample_idxs)

        for run_idx in range(n_traj):
            print('run_idx', run_idx)

            # Note that sample_idx is the trajectory index. The actual sample can be different
            # due to random sampling.
            sample_idx = sample_idxs[(start_idx + run_idx) % len(dataset)]

            # Note that querying the same index of the dataset will result in different trajectory segments to be
            # sampled.
            sample = dataset[sample_idx]
            # sample = dataset[run_idx % len(dataset)]

            map_name = sample['map_name']
            print('map', map_name)

            map = self.maps[map_name]
            result = self.run_single(sample, map, run_idx)
            outcome = result['outcome']

            print('%d: %s' % (run_idx, outcome))

            if outcome == 'skipped':
                skipped_idxs.append(run_idx)
                continue

            path_len = result['path_len']
            trace_len = path_length(result['trace'])

            if outcome.startswith('success'):
                success_idxs.append(run_idx)
                spls.append(path_len / max(path_len, trace_len))
            elif outcome.startswith('failure'):
                failures_idxs.append(run_idx)
                spls.append(0.0)
            else:
                raise RuntimeError('Unknown outcome %s' % outcome)

            print('success rate: %.2f' % (len(success_idxs) / float(len(success_idxs) + len(failures_idxs))))
            print('spl: %.2f' % np.mean(spls))

        print('successes:', len(success_idxs))
        print('failures:', len(failures_idxs))

        return {'success_rate': len(success_idxs) / float(len(success_idxs) + len(failures_idxs))}

    def _write_trace_visualization_image(self, map, traj, trace, out_file,
                                         figsize=(8, 5), linewidth=3.0, linealpha=0.8):
        cm = matplotlib.cm.get_cmap('tab10')
        fig = Figure(figsize=figsize)
        FigureCanvas(fig)
        ax = fig.add_subplot(111)

        # Call draw() to prevent matplotlib from complaining about draw_artist() being called before
        # initial draw.
        fig.canvas.draw()

        vis = FindMapVisualizer(map)(map, ax)
        vis.draw_map()

        xs, ys = traj[:, 0], traj[:, 1]
        ax.plot(xs, ys, color=cm(0), linewidth=linewidth, alpha=linealpha)

        xs, ys = zip(*trace)
        ax.plot(xs, ys, color=cm(1), linewidth=linewidth, alpha=linealpha)

        ax.plot(xs[0], ys[0], 'o', color='r', markersize=20.0)
        ax.plot(xs[-1], ys[-1], 'r*', markersize=20.0)

        ax.set_xlim(min(xs) - 1.0, max(xs) + 1.0)
        ax.set_ylim(min(ys) - 1.0, max(ys) + 1.0)
        ax.set_aspect('equal')

        fig.tight_layout()
        fig.savefig(out_file)

    def _prepare(self):
        if self.cur_map.name in self.vis_dict:
            vis = self.vis_dict[self.cur_map.name]
            xy = self.cur_traj_xy
            vis.set_limits(np.min(xy[:, 0]) - 1.0, np.max(xy[:, 0]) + 1.0, np.min(xy[:, 1]) - 1.0, np.max(xy[:, 1]) + 1.0)
            vis.render(force_redraw=True)

    def _post_step(self, repeat=1):
        if not self.visualize:
            return

        vis = self.vis_dict[self.cur_map.name]

        vis.render()  # render the map and robot

        traj_xy = self.cur_traj_xy

        vis.plotter.scatter('start', traj_xy[0, 0], traj_xy[0, 1], s=100)
        vis.plotter.plot('traj', traj_xy[:, 0], traj_xy[:, 1])

        if len(self.cur_trace) > 0:
            trace = np.array(self.cur_trace)
            vis.plotter.plot('trace', trace[:, 0], trace[:, 1])

        progress_idx = min(bisect.bisect(self.cur_progresses, self.cur_progress), len(traj_xy) - 1)
        vis.plotter.scatter('progress_indicator', traj_xy[progress_idx, 0], traj_xy[progress_idx, 1], s=100)

        size = 256

        font_size = 36
        text_start = put_text('   start', font_size, (255, 255, 255), size, 48, bold=True)
        text_goal = put_text('   goal', font_size, (255, 255, 255), size, 48, bold=True)
        text_ob = put_text('   current', font_size, (255, 255, 255), size, 48, bold=True)

        font_size = 24
        text_traj_len = put_text('traj len: %d steps' % (len(traj_xy) * self.step_interval),
                                 font_size, (255, 255, 255), size, 36, bold=False)
        text_time_step = put_text('step: %d' % self.cur_step,
                                  font_size, (255, 255, 255), size, 36, bold=False)
        text_progress = put_text('progress: %.2f' % self.cur_progress,
                                 font_size, (255, 255, 255), size, 36, bold=False)

        vis_img = cv2.cvtColor(vis.get_image(), cv2.COLOR_RGB2BGR)

        def cvtimg(im):
            return (cv2.cvtColor(
                cv2.resize(im.transpose((1, 2, 0)), dsize=(size, size)),
                cv2.COLOR_RGB2BGR) * 255).astype(np.uint8)

        vis_img = HStack(vis_img, VStack(
            text_start,
            cvtimg(self.cur_start_ob),
            text_goal,
            cvtimg(self.cur_goal_ob),
            text_ob,
            cvtimg(self.cur_ob),
            text_traj_len,
            text_time_step,
            text_progress
        ))

        cv2.imshow('vis', vis_img)
        cv2.waitKey(1)

        if self.save_screenshot:
            for i in range(repeat):
                cv2.imwrite(os.path.join(self.screenshot_dir, '%05d.tiff' % self.screenshot_idx), vis_img)
                self.screenshot_idx += 1
