import cv2
import numpy as np
import torch
from rmp_nav.common.utils import pprint_dict
from rmp_nav.common.image_combiner import VStack


class TrajectoryFollower(object):
    (SEARCH, FOLLOW, DEAD_RECON) = range(3)

    def __init__(self, sparsifier, motion_policy, search_thres, follow_thres,
                 dead_reckon_search_thres=None):
        """
        :param dead_reckon_search_thres: if None will use search_thres.
                                         Otherwise use the specified threshold.
        """
        super(TrajectoryFollower, self).__init__()

        self.sparsifier = sparsifier
        self.motion_policy = motion_policy
        self.search_thres = search_thres
        self.follow_thres = follow_thres
        self.dead_reckon_search_thres = dead_reckon_search_thres

        self.agent = None
        self.anchors = None
        self.cur_anchor_idx = None
        self.cur_wp = None
        self.dead_reckon_wp = None
        self.dead_reckon_iter = None
        self.dead_reckon_start_idx = None

        self.state = self.SEARCH

    def __repr__(self):
        return pprint_dict({
            'search_thres': self.search_thres,
            'follow_thres': self.follow_thres
        })

    def reset(self):
        self.agent = None
        # self.traj = None
        self.anchors = None
        self.cur_anchor_idx = None
        self.cur_wp = None
        self.dead_reckon_wp = None
        self.dead_reckon_iter = None
        self.dead_reckon_start_idx = None
        self.state = self.SEARCH

    def set_anchors(self, anchors):
        """
        :param anchors: a list of dicts. Each dict contains at least two keys:
                        'ob_repr': representation when used as an observation
                        'dst_repr': representation when used as a target
        :return:
        """
        self.anchors = anchors.copy()  # A shallow copy is good enough
        for anchor in self.anchors:
            anchor['dst_repr_torch'] = torch.as_tensor(np.array(anchor['dst_repr']),
                                                       device=self.sparsifier.device)
        self.cur_anchor_idx = 0

    def make_vis_img(self):
        return None

    def _compute_reachability(self, ob, anchor_idx):
        ret = self.sparsifier.predict_reachability(ob, self.anchors[anchor_idx]['dst_repr_torch'])
        return ret

    def _compute_wp(self, ob, anchor_idx):
        ret = self.motion_policy.predict_waypoint(ob, self.anchors[anchor_idx]['dst_repr_torch'])
        return ret

    def _find_best_anchor_idx(self, ob, start_idx, thres, max_ahead=None):
        if max_ahead is None:
            limit = len(self.anchors)
        else:
            limit = min(start_idx + max_ahead, len(self.anchors))

        rs = [self._compute_reachability(ob, i) for i in range(start_idx, limit)]

        best_r = 0.0
        best_idx = -1

        for idx, r in enumerate(rs):
            if r > best_r:
                best_r = r
                best_idx = idx + start_idx

        if best_r < thres:
            best_idx = -1

        print('best_anchor_r', best_r)
        return best_idx, best_r

    def _find_next_good_anchor_idx(self, ob, cur_anchor_idx, max_ahead=1):
        # TODO: it might not be a good idea to advance the anchor too far away, because it is likely
        # TODO: to be a false positive. Advance by one sounds good enough.
        # TODO: advance by one might be a bit brittle?

        for next_idx in range(cur_anchor_idx + 1,
                              min(cur_anchor_idx + max_ahead + 1, len(self.anchors))):
            r = self._compute_reachability(ob, next_idx)
            if r > self.follow_thres:
                return next_idx, r

        # Cannot advance anchor. Re-evaluate current.
        r = self._compute_reachability(ob, cur_anchor_idx)

        if r > self.follow_thres:
            return cur_anchor_idx, r

        # print('cannot find a good anchor. best_idx %d best_r %.2f' % (best_idx, best_r))
        return -1, r

    def get_next_wp(self, agent, ob):
        r = None
        target = None
        self.agent = agent

        while True:
            if self.state == self.SEARCH:
                next_idx, r = self._find_best_anchor_idx(
                    ob, self.cur_anchor_idx, self.search_thres)
                print('SEARCH anchor idx: %d reachability: %.2f' % (next_idx, r))
                if next_idx < 0:
                    wp = None
                else:
                    target = self.anchors[next_idx]['dst_repr']
                    wp = self._compute_wp(ob, next_idx)
                    self.state = self.FOLLOW
                    self.cur_anchor_idx = next_idx
                break
            elif self.state == self.FOLLOW:
                next_idx, r = self._find_next_good_anchor_idx(ob, self.cur_anchor_idx, 5)
                print('FOLLOW next idx: %d reachability: %.2f' % (next_idx, r))

                if next_idx < 0:
                    # Failed to find a good waypoint, switch to dead reckoning mode
                    self.dead_reckon_iter = 0
                    self.dead_reckon_start_idx = self.cur_anchor_idx
                    self.state = self.DEAD_RECON
                else:
                    target = self.anchors[next_idx]
                    wp = self._compute_wp(ob, next_idx)
                    self.cur_anchor_idx = next_idx
                    self.dead_reckon_iter = 0
                    break
            elif self.state == self.DEAD_RECON:
                # Try following
                idx, _ = self._find_next_good_anchor_idx(ob, self.cur_anchor_idx)
                if idx >= 0:
                    # If a good wp is available, switch to FOLLOW mode, otherwise stay at this state
                    self.state = self.FOLLOW
                    self.cur_anchor_idx = idx
                    target = self.anchors[idx]['dst_repr']
                    wp = self._compute_wp(ob, idx)
                else:
                    print('DEAD RECKON')
                    # Try searching

                    print('cur_anchor', self.cur_anchor_idx,
                          'dead_reckon_iter', self.dead_reckon_iter)

                    if self.dead_reckon_search_thres is None:
                        search_thres = self.search_thres
                    else:
                        search_thres = self.dead_reckon_search_thres

                    # Search at a radius of self.dead_reckon_iter
                    idx, r = self._find_best_anchor_idx(
                        ob,
                        max(self.dead_reckon_start_idx - self.dead_reckon_iter, 0),
                        search_thres,
                        self.dead_reckon_iter * 2)

                    if idx >= 0:
                        print('found anchor %d reachability: %.2f' % (idx, r))
                        self.state = self.FOLLOW
                        self.cur_anchor_idx = idx
                        target = self.anchors[idx]['dst_repr']
                        wp = self._compute_wp(ob, idx)
                    else:
                        self.dead_reckon_iter += 1
                        if self.dead_reckon_iter > 200:
                            # Give up
                            wp = None
                            break

                        if self.dead_reckon_iter % 3 == 0:
                            self.cur_anchor_idx = max(self.cur_anchor_idx - 1, 0)

                        wp = agent.global_to_local(self.cur_wp)
                break
            else:
                assert False

        if wp is not None:
            self.cur_wp = agent.local_to_global(wp)

        return wp, {'reachability': r,
                    'target': target,
                    'anchor_idx': self.cur_anchor_idx}


class TrajectoryFollowerMultiFrameDstBothProximity(TrajectoryFollower):
    def __init__(self, *args, **kwargs):
        super(TrajectoryFollowerMultiFrameDstBothProximity, self).__init__(*args, **kwargs)
        self.last_visited_anchor = None

    def reset(self):
        super(TrajectoryFollowerMultiFrameDstBothProximity, self).reset()
        self.last_visited_anchor = None

    def make_vis_img(self, **kwargs):
        target_imgs = self.target_seq.data.cpu().numpy()  # N x C x H x W
        target_img = np.concatenate(target_imgs, axis=2).transpose(1, 2, 0)  # H x (N x W) x C
        ob = self.ob.transpose(1, 2, 0)
        return cv2.cvtColor((VStack(ob, target_img) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    def _compute_wp(self, ob, anchor_idx):
        # These are used for visualization.
        self.ob = ob
        self.target_seq = self.anchors[anchor_idx]['dst_repr_torch']
        return super(type(self), self)._compute_wp(ob, anchor_idx)

    def _compute_proximity(self, ob, anchor_idx):
        mp = self.motion_policy
        return np.mean(mp.predict_proximity(ob, self.anchors[anchor_idx]['dst_repr_torch']))

    def _find_next_good_anchor_idx(self, ob, cur_anchor_idx, max_ahead=1):
        """
        If there are multiple anchors that are highly reachable, choose the furthest one with
        proximity >= thres.
        """
        reachable_idxs = []
        rs = []
        ps = []

        proximity_thres = 0.6
        update_last_visit_anchor_thres = 0.65

        # Update last_visited_anchor
        cur_p = self._compute_proximity(ob, cur_anchor_idx)
        print('cur anchor: %d' % cur_anchor_idx)
        print('cur proximity: %.2f' % cur_p)
        if cur_p > update_last_visit_anchor_thres:
            self.last_visited_anchor = cur_anchor_idx

        if self.last_visited_anchor is not None:
            start_idx = self.last_visited_anchor + 1
        else:
            start_idx = cur_anchor_idx + 1

        idxs = list(range(start_idx,
                          min(cur_anchor_idx + max_ahead + 1, len(self.anchors))))
        for next_idx in idxs:
            r = self._compute_reachability(ob, next_idx)
            p = self._compute_proximity(ob, next_idx)
            print('%d %.2f' % (next_idx, r))
            if r > self.follow_thres:
                reachable_idxs.append(next_idx)
                rs.append(r)
                ps.append(p)
            else:
                # We assume that anchor must be continuously reachable
                # This could help with ambiguous observations.
                break

        if len(reachable_idxs) > 0:
            print('reachable_idxs:', reachable_idxs)
            print('reachability:', rs)
            print('proximity:', ps)
            print('last_visited_anchor', self.last_visited_anchor)

            if max(ps) > proximity_thres:
                i = np.argmax(ps)
            else:
                i = -1
                if self.last_visited_anchor is not None:
                    max_ahead_anchor = self.last_visited_anchor + 1
                    if reachable_idxs[0] >= max_ahead_anchor:
                        i = 0
                    else:
                        for _ in range(len(reachable_idxs)):
                            if reachable_idxs[_] > max_ahead_anchor:
                                i = _ - 1
            print('next_anchor:', reachable_idxs[i])
            return reachable_idxs[i], rs[i]

        # Cannot advance anchor. Re-evaluate current.
        cur_r = self._compute_reachability(ob, cur_anchor_idx)
        if cur_r > self.follow_thres:
            return cur_anchor_idx, cur_r

        return -1, cur_r


class RunnerBase(object):
    def __init__(self, motion_policy, sparsifier, follower,
                 agent, agent_reverse=None,
                 wp_norm_min_clip=2.0,
                 clip_velocity=None):
        self.motion_policy = motion_policy
        self.sparsifier = sparsifier
        self.follower = follower
        self.agent = agent
        self.agent_reverse = agent_reverse
        self.wp_norm_min_clip = wp_norm_min_clip
        self.clip_velocity = clip_velocity

    def _get_ob(self):
        raise NotImplementedError

    def set_anchors(self, anchors):
        self.follower.reset()
        self.follower.set_anchors(anchors)

    @property
    def anchors(self):
        return self.follower.anchors

    def step(self):
        """
        :return: (wp, extra). wp is set to None if agent fails to act.
        """
        agent = self.agent
        agent_rev = self.agent_reverse

        ob = self._get_ob()
        wp, extra = self.follower.get_next_wp(agent, ob)

        if wp is None:
            print('Failed to compute the next waypoint. Stop here.')
            return ob, wp, extra

        if np.linalg.norm(wp) < self.wp_norm_min_clip:
            wp = wp / np.linalg.norm(wp) * self.wp_norm_min_clip

        step_kwargs = {
            'waypoint': wp,
            'max_vel': self.clip_velocity,
        }
        if agent_rev is not None:
            if wp[0] < 0.0:
                agent_rev.set_velocity(agent.velocity)
                agent_rev.set_pos(agent.pos)
                agent_rev.set_heading(agent.heading)
                agent_rev.set_map(agent.map)
                agent_rev.step(0.1, **step_kwargs)
                agent.set_velocity(agent_rev.velocity)
                agent.set_pos(agent_rev.pos)
                agent.set_heading(agent_rev.heading)
            else:
                agent.step(0.1, **step_kwargs)
        else:
            agent.step(0.1, **step_kwargs)

        return ob, wp, extra


class RunnerGibsonDataset(RunnerBase):
    def __init__(self, dataset, map, **kwargs):
        super(RunnerGibsonDataset, self).__init__(**kwargs)
        self.map = map
        self.dataset = dataset

    def _get_ob(self):
        self.dataset.agent.set_pos(self.agent.pos)
        self.dataset.agent.set_heading(self.agent.heading)
        return self.dataset._render_agent_view(self.map.name)
