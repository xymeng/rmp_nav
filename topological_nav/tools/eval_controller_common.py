import cv2
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from rmp_nav.simulation import sim_renderer
from rmp_nav.simulation.agent_visualizers import FindAgentVisualizer
from rmp_nav.common.image_combiner import VStack, HStack


class EvaluatorBase(object):
    def __init__(self, motion_policy, dataset, agent, agent_reverse=None,
                 visualize=False, figsize=None):
        self.motion_policy = motion_policy
        self.dataset = dataset
        self.rng = None

        dataset._init_once(0)

        self.agent = agent
        self.agent_reverse = agent_reverse
        self.ob = None

        self.fig = None
        self.figsize = figsize
        self.ax = None
        self.vis = None

        if visualize:
            self._init_visualizer()

    def _init_visualizer(self):
        fig = plt.Figure(tight_layout=True, figsize=self.figsize)
        ax = fig.add_subplot(111)
        canvas = FigureCanvas(fig)
        canvas.draw()
        cm = matplotlib.cm.get_cmap('Dark2')

        # FIXME: assume only one map.
        vis = sim_renderer.SimRenderer(list(self.dataset.maps.values())[0], ax, canvas)
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

        self.vis = vis

    def _prepare(self, dataset_idx, traj_id, traj, map_name, src_idx, dst_idx, src_sample, dst_sample):
        raise NotImplementedError

    def _act(self, frame_idx, ob):
        raise NotImplementedError

    def _get_ob(self):
        raise NotImplementedError

    def _post_step(self):
        pass

    def run(self, start_idx=0, n_traj=100, seed=12345):
        agent = self.agent
        agent_rev = self.agent_reverse
        dataset = self.dataset

        total = 0
        successes = 0

        rng = np.random.RandomState(seed)
        self.rng = rng

        traj_idxs = list(range(len(dataset)))
        distances = rng.uniform(dataset.distance_min, dataset.distance_max, len(dataset))

        rng.shuffle(traj_idxs)
        traj_idxs = traj_idxs[start_idx: start_idx + n_traj]

        success_trajs = []
        failure_trajs = []

        for idx, traj_idx in enumerate(traj_idxs):
            distance = distances[traj_idx]
            print('%d: distance %.3f' % (idx, distance))

            dataset_idx, traj_id, traj, map_name, src_idx, dst_idx, src_sample, _ = dataset._get_sample(
                traj_idx, distance)

            self._prepare(dataset_idx, traj_id, traj, map_name, src_idx, dst_idx, src_sample, traj[dst_idx])

            rollout_steps = abs(dst_idx - src_idx)

            src_pos, src_heading = src_sample['pos'], src_sample['heading']
            goal_pos, goal_heading = traj[dst_idx]['pos'], traj[dst_idx]['heading']

            m = dataset.maps[map_name]

            agent.reset()
            agent.set_map(m)
            agent.set_pos(src_pos)
            agent.set_heading(src_heading)
            agent.set_waypoints([goal_pos])

            dataset.agent.set_pos(agent.pos)
            dataset.agent.set_heading(agent.heading)

            reached = 0.0
            early_stop = False

            self.map_name = map_name

            for i in range(rollout_steps + 50):
                ob = self._get_ob()
                self.ob = ob
                wp = self._act(i, ob)

                if agent_rev is not None:
                    if wp[0] < 0.0:
                        agent_rev.set_velocity(agent.velocity)
                        agent_rev.set_pos(agent.pos)
                        agent_rev.set_heading(agent.heading)
                        agent_rev.set_map(agent.map)
                        agent_rev.step(0.1, waypoint=wp)
                        agent.set_velocity(agent_rev.velocity)
                        agent.set_pos(agent_rev.pos)
                        agent.set_heading(agent_rev.heading)
                    else:
                        agent.step(0.1, waypoint=wp)
                else:
                    agent.step(0.1, waypoint=wp)

                if agent.reached_goal():
                    print('%d: reached' % idx)
                    reached = 1.0
                    early_stop = True
                    break
                elif agent.collide():
                    print('%d: collide' % idx)
                    reached = 0.0
                    early_stop = True
                    break
                else:
                    overlap_ratios = m.view_overlap(agent.pos, agent.heading, dataset.fov,
                                                    goal_pos, goal_heading, dataset.fov)
                    # print('overlap %.2f %.2f' % (overlap_ratios[0], overlap_ratios[1]))
                    if overlap_ratios[0] > 0.7 and overlap_ratios[1] > 0.8:
                        print('%d: reached (overlap)' % idx)
                        reached = 1.0
                        early_stop = True
                        break

                dataset.agent.set_pos(agent.pos)
                dataset.agent.set_heading(agent.heading)

                self._post_step()

            if not early_stop:
                print('%d: timeout' % idx)

            total += 1

            if reached > 0.5:
                success_trajs.append(idx)
            else:
                failure_trajs.append(idx)

            successes += reached

            print('success rate: %.2f' % (successes / total))

        print('success trajs:', success_trajs)
        print('failure trajs:', failure_trajs)


class EvaluatorMultiframeDst(EvaluatorBase):
    def __init__(self, *args, save_screenshot=False, zoom=1.0, **kwargs):
        super(EvaluatorMultiframeDst, self).__init__(*args, **kwargs)

        self.ob = None

        (self.src_img,
         self.src_position,
         self.src_heading,
         self.src_waypoint,
         self.src_velocity) = (None,) * 5

        (self.dst_imgs,
         self.dst_positions,
         self.dst_headings,
         self.dst_waypoints,
         self.dst_velocities) = (None,) * 5

        self.save_screenshot = save_screenshot
        self.screenshot_idx = 0
        self.zoom = zoom

    def _get_ob(self):
        return self.dataset._render_agent_view(self.map_name)

    def _prepare(self, dataset_idx, traj_id, traj, map_name, src_idx, dst_idx, src_sample, dst_sample):
        mp = self.motion_policy
        dataset = self.dataset

        dst_samples = []
        for i in range(mp.n_frame):
            dst_samples.append(traj[max(dst_idx - i * mp.frame_interval, 0)])
        dst_samples = dst_samples[::-1]

        if mp.g.get('future', False):
            for i in range(mp.n_frame - 1):
                dst_samples.append(traj[min(dst_idx + (i + 1) * mp.frame_interval, len(traj) - 1)])

        ([self.src_img],
         [self.src_position],
         [self.src_heading],
         [self.src_waypoint],
         [self.src_velocity]) = dataset._make(
            map_name, [src_sample], attrs=traj.attrs)

        (self.dst_imgs,
         self.dst_positions,
         self.dst_headings,
         self.dst_waypoints,
         self.dst_velocities) = dataset._make(
            map_name, dst_samples, attrs=traj.attrs)

        if self.vis is not None and self.zoom != 1.0:
            self.vis.reset_viewport()
            self.vis.set_viewport(self.agent.pos[0], self.agent.pos[1], self.zoom)

    def _post_step(self):
        vis = self.vis
        if vis is not None:
            redraw = False
            if self.zoom != 1.0:
                x1, x2, y1, y2 = vis.get_limits()
                w, h = abs(x2 - x1), abs(y2 - y1)
                x, y = self.agent.pos
                if abs(x - (x2 + x1) / 2) / (w / 2) > 0.8 or abs(y - (y2 + y1) / 2) / (h / 2) > 0.8:
                    vis.reset_viewport()
                    vis.set_viewport(self.agent.pos[0], self.agent.pos[1], self.zoom)
                    redraw = True
            vis.render(redraw, blit=True)

            vis_img = cv2.cvtColor(vis.get_image(), cv2.COLOR_RGB2BGR)

            ob_vis = np.array(self.ob).transpose((1, 2, 0))
            target_seq_vis = np.array(self.dst_imgs).transpose((0, 2, 3, 1))

            seq_vis = VStack(ob_vis, np.concatenate(target_seq_vis, axis=1))

            seq_vis = cv2.cvtColor(np.clip(seq_vis * 255, 0.0, 255.0).astype(np.uint8),
                                   cv2.COLOR_RGB2BGR)

            vis_img = VStack(seq_vis, vis_img)
            cv2.imshow('vis', vis_img)
            cv2.waitKey(1)
            if self.save_screenshot:
                cv2.imwrite('/tmp/screenshots/%05d.tiff' % self.screenshot_idx, vis_img)
                self.screenshot_idx += 1

    def _act(self, frame_idx, ob):
        mp = self.motion_policy
        self.ob = ob
        wp = mp.predict_waypoint(ob, self.dst_imgs)
        if np.linalg.norm(wp) < 0.3:
            wp = wp / np.linalg.norm(wp) * 0.3
        return wp
