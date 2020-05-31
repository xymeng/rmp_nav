import random
import math
import networkx as nx
import numpy as np
import pickle
import time
from copy import deepcopy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.patches import ArrowStyle
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from rmp_nav.simulation import sim_renderer
from rmp_nav.common.utils import pprint_dict


class NavGraphBase(object):
    def __init__(self):
        # Vertex is represented by a tuple (traj_idx, ob_idx)
        # Use node attribute to access data
        self.graph = nx.DiGraph()
        self.trajs = {}

        self.cm = matplotlib.cm.get_cmap('tab10')
        self.arrow_styles = [
            ArrowStyle('->', head_length=2.0, head_width=2.0),
            # ArrowStyle('-|>', head_length=2.0, head_width=2.0),
            ArrowStyle('-[', widthB=3.0, lengthB=2.0)
        ]
        self.line_widths = [1.0, 2.0]
        self.line_styles = ['-', '-.']

    def _get_vis_style(self, u, v):
        arrow_styles = self.arrow_styles
        line_widths = self.line_widths
        line_styles = self.line_styles
        cm = self.cm

        if u[0] == v[0]:
            return {
                'color': self.get_traj_vis_color(u[0]),
                'arrowstyle': arrow_styles[(u[0] // cm.N) % len(arrow_styles)],
                'linewidth': line_widths[(u[0] // (cm.N * len(arrow_styles))) % len(line_widths)],
                'linestyle': line_styles[(u[0] // (cm.N * len(arrow_styles) * len(line_widths))) % len(line_styles)]
            }
        else:
            return {
                'color': (0.8, 0.8, 0.8),
                'linestyle': ':',
                'linewidth': 0.5,
                'arrowstyle': ArrowStyle('-|>', head_length=1.0, head_width=1.0)
            }

    def get_traj_vis_color(self, traj_idx):
        return self.cm(traj_idx % self.cm.N)

    def _draw_edge(self, vis, u, v, traj_info_dict):
        style = self._get_vis_style(u, v)

        sample_u = traj_info_dict[u[0]]['samples'][u[1]]
        sample_v = traj_info_dict[v[0]]['samples'][v[1]]
        pos_u = sample_u['pos']
        pos_v = sample_v['pos']

        vis.plotter.fixed_arrow2('edge %d %d %d %d' % (u[0], u[1], v[0], v[1]),
                                 pos_u[0], pos_u[1], pos_v[0], pos_v[1],
                                 **style)

    def _draw_traj(self, vis, traj_idx, traj_info_dict):
        traj = self.trajs[traj_idx]
        for i in range(len(traj) - 1):
            u = traj[i]
            v = traj[i + 1]
            self._draw_edge(vis, u, v, traj_info_dict)

    def draw_path(self, vis, path, traj_info_dict):
        for i in range(len(path) - 1):
            self._draw_edge(vis, path[i], path[i + 1], traj_info_dict)

    def visualize_path(self, map, traj_info_dict, path,
                       start_pos=None, start_heading=None, goal_pos=None, goal_heading=None,
                       vis=None, save_file='', title=''):
        if vis is None:
            fig = plt.Figure(tight_layout=True)
            ax = fig.add_subplot(111)
            canvas = FigureCanvas(fig)
            canvas.draw()
            vis = sim_renderer.SimRenderer(map, ax, canvas)
            vis.render(True)
            vis.render(True)
        if start_pos is not None:
            vis.ax.scatter([start_pos[0]], [start_pos[1]],
                           marker='o', c=np.array([[0, 0.5, 0]]), s=50)
            if start_heading is not None:
                vis.plotter.fixed_arrow('start', start_pos[0], start_pos[1],
                                        math.cos(start_heading), math.sin(start_heading),
                                        color=(0, 0.5, 0))
        if goal_pos is not None:
            vis.ax.scatter([goal_pos[0]], [goal_pos[1]],
                           marker='*', c=np.array([[0.5, 0, 0]]), s=50)
            if goal_heading is not None:
                vis.plotter.fixed_arrow('goal', goal_pos[0], goal_pos[1],
                                        math.cos(goal_heading), math.sin(goal_heading),
                                        color=(0.5, 0, 0))
        for i in range(len(path) - 1):
            self._draw_edge(vis, path[i], path[i + 1], traj_info_dict)
        vis.ax.set_title(title)
        if save_file != '':
            vis.ax.get_figure().savefig(save_file)
        return vis

    def visualize(self, map, traj_info_dict, save_file='', max_traj=None,
                  annotate_starting_loc=True, figsize=None, limits=None, xticks=None, yticks=None):
        """
        :param max_traj: only draw trajecies with indices <= max_traj
        """
        fig = plt.Figure(figsize=figsize)
        ax = fig.add_subplot(111)
        canvas = FigureCanvas(fig)
        canvas.draw()

        vis = sim_renderer.SimRenderer(map, ax, canvas)
        vis.render(True)
        vis.render(True)

        g = self.graph

        vis.clear()
        vis.render(True)

        for (u, v), data in g.edges.items():
            if max_traj is not None:
                if u[0] > max_traj or v[0] > max_traj:
                    continue

            sample_u = traj_info_dict[u[0]]['samples'][u[1]]
            sample_v = traj_info_dict[v[0]]['samples'][v[1]]

            pos_u = sample_u['pos']
            pos_v = sample_v['pos']

            vis.plotter.fixed_arrow2('edge %d %d %d %d' % (u[0], u[1], v[0], v[1]),
                                     pos_u[0], pos_u[1], pos_v[0], pos_v[1],
                                     **self._get_vis_style(u, v))

        if annotate_starting_loc:
            for traj_idx, traj in self.trajs.items():
                start_node = traj[0]
                sample = traj_info_dict[start_node[0]]['samples'][start_node[1]]
                x, y = sample['pos']
                vis.plotter.text('traj %d' % traj_idx, x, y, '%d' % traj_idx,
                                 fontsize=12, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})

        if limits is not None:
            ax.set_xlim(limits[0], limits[1])
            ax.set_ylim(limits[2], limits[3])

        if xticks is not None:
            ax.set_xticks(xticks)
        if yticks is not None:
            ax.set_yticks(yticks)

        if save_file != '':
            fig.tight_layout(pad=0.1)
            fig.savefig(save_file)

        return vis.get_image()


class NavGraph(NavGraphBase):
    def __init__(self, sparsifier, motion_policy,
                 sparsify_thres, edge_add_thres, reuse_thres,
                 delta_tw=None,
                 reuse_edge_prob_mode='hard',
                 reuse_terminal_nodes=False,
                 reuse_start_nodes=True,
                 reuse_goal_nodes=True):
        """
        :param edge_add_thres: threshold for adding edges between any two nodes
        :param reuse_thres: threshold for reusing a node when adding a new trajectory
        :param delta_tw: same parameter as in SPTM
        :param reuse_edge_prob_mode: a string
                 hard: reuse edge is set to prob of 0.9999
                 actual: reuse edge is set to the actual reachability
        :param reuse_terminal_nodes: True to allow reusing start nodes and goal nodes
        :param reuse_start_nodes: applicable only when reuse_terminal_nodes is True
        :param reuse_goal_nodes: applicable only when reuse_terminal_nodes is True
        """
        super(NavGraph, self).__init__()
        self.sparsifier = sparsifier
        self.motion_policy = motion_policy
        self.sparsify_thres = sparsify_thres
        self.edge_add_thres = edge_add_thres
        self.reuse_thres = reuse_thres
        self.reuse_edge_prob_mode = reuse_edge_prob_mode
        self.reuse_terminal_nodes = reuse_terminal_nodes
        self.reuse_start_nodes = reuse_start_nodes
        self.reuse_goal_nodes = reuse_goal_nodes
        self.delta_tw = delta_tw

        # Vertex is represented by a tuple (traj_idx, ob_idx)
        # Use node attribute to access data
        self.graph = nx.DiGraph()
        self.trajs = {}
        self.subgraph = None

        # You can put anything inside and it will be saved/loaded.
        self.extra = {
            'edge_add_thres': edge_add_thres,
            'reuse_thres': reuse_thres,
            'reuse_edge_prob_mode': reuse_edge_prob_mode,
            'reuse_terminal_nodes': reuse_terminal_nodes,
            'reuse_start_nodes': reuse_start_nodes,
            'reuse_goal_nodes': reuse_goal_nodes,
            'sparsify_thres': sparsify_thres,
            'delta_tw': delta_tw
        }

    def __repr__(self):
        return pprint_dict({
            'edge_add_thres': self.edge_add_thres,
            'reuse_thres': self.reuse_thres,
            'reuse_edge_prob_mode': self.reuse_edge_prob_mode,
            'reuse_terminal_nodes': self.reuse_terminal_nodes,
            'reuse_start_nodes': self.reuse_start_nodes,
            'reuse_goal_nodes': self.reuse_goal_nodes,
            'delta_tw': self.delta_tw,
            'number of nodes': len(self.graph.nodes),
            'number of edges': len(self.graph.edges)
        })

    def save(self, out_file):
        with open(out_file, 'wb') as f:
            pickle.dump({
                'trajs': self.trajs,
                'graph': self.graph,
                'extra': self.extra
            }, f)

    @staticmethod
    def from_save_file(sparsifier, motion_policy, sparsify_thres, save_file):
        g = NavGraph(sparsifier, motion_policy, sparsify_thres, 0.0, 0.0)
        g.load(save_file)
        return g

    def load(self, save_file):
        with open(save_file, 'rb') as f:
            d = pickle.load(f)
            self.trajs = d['trajs']
            self.graph = d['graph']
            self.extra = d['extra']
            self.edge_add_thres = self.extra.get('edge_add_thres', self.edge_add_thres)
            self.reuse_thres = self.extra.get('reuse_thres', self.reuse_thres)
            self.sparsify_thres = self.extra.get('sparsify_thres', self.sparsify_thres)
            self.delta_tw = self.extra.get('delta_tw', self.delta_tw)
            self.reuse_terminal_nodes = self.extra.get('reuse_terminal_nodes', self.reuse_terminal_nodes)
            self.reuse_start_nodes = self.extra.get('reuse_start_nodes', self.reuse_start_nodes)
            self.reuse_goal_nodes = self.extra.get('reuse_goal_nodes', self.reuse_goal_nodes)
            self.reuse_edge_prob_mode = self.extra.get('reuse_edge_prob_mode', self.reuse_edge_prob_mode)

    def compute_reachability(self, ob, dst):
        return self.sparsifier.predict_reachability(ob, dst)

    def compute_reachability_batch(self, obs, dsts):
        return self.sparsifier.predict_reachability_batch(obs, dsts)

    def _find_reusable_nodes(self, graph, traj, anchor_obs, anchor_idxs):
        replacements = {}
        if len(graph.nodes) == 0:
            return replacements

        if self.reuse_terminal_nodes:
            if self.reuse_start_nodes:
                start = 0
            else:
                start = 1
            if self.reuse_goal_nodes:
                end = len(anchor_obs)
            else:
                end = len(anchor_obs) - 1
            it = range(start, end)
        else:
            it = range(1, len(anchor_obs) - 1)

        for i in it:
            obs = []
            goals = []
            vs = []
            anchor_idx = anchor_idxs[i]

            if i == 0:
                # The first node doesn't have a parent, so we use itself.
                ob_prev = self.sparsifier.get_ob_repr(traj, anchor_idxs[0])
            elif anchor_idxs[i - 1] in replacements:
                # If we have replaced previous node, use it.
                ob_prev = graph.nodes[replacements[anchor_idxs[i - 1]]]['ob_repr']
            else:
                ob_prev = self.sparsifier.get_ob_repr(traj, anchor_idxs[i - 1])

            if i == len(anchor_obs) - 1:
                # The last node doesn't have a child, so we use itself.
                ob_next_dst_repr = self.sparsifier.get_dst_repr(traj, anchor_idxs[i])
            else:
                ob_next_dst_repr = self.sparsifier.get_dst_repr(traj, anchor_idxs[i + 1])

            # We need to be careful when selecting the candidates nodes. If A(t) has been replaced
            # with B(t'), then it's not a good idea to replace A(t+1) with B(t'') where t'' < t'.
            # This causes an edge B(t') -> B(t'') to be added which is usually wrong (unless there
            # are loops in B). Here we enforce that if B(t') has been reused, then for t'' < t',
            # B(t'') cannot be reused.

            for v, data in graph.nodes.items():
                skip = False
                for reused_node in replacements.values():
                    if v[0] == reused_node[0] and v[1] <= reused_node[1]:
                        skip = True
                        break
                if skip:
                    continue

                vs.append(v)
                obs.append(ob_prev)
                goals.append(data['dst_repr'])
                obs.append(data['ob_repr'])
                goals.append(ob_next_dst_repr)

            rs = self.compute_reachability_batch(obs, goals)
            best_r1, best_r2 = 0.0, 0.0
            for j in range(len(vs)):
                r1, r2 = rs[j * 2], rs[j * 2 + 1]
                if r1 > self.reuse_thres and r2 > self.reuse_thres and r1 * r2 > best_r1 * best_r2:
                    replacements[anchor_idx] = vs[j]
                    best_r1, best_r2 = r1, r2

        return replacements

    def _find_new_edges(self, new_nodes, new_graph, old_graph, sample_limit=100):
        edges = []
        if len(old_graph.nodes) == 0:
            return edges

        uvs = []
        obs = []
        goals = []

        nodes = list(old_graph.nodes)
        random.shuffle(nodes)

        for u in new_nodes:
            ob_data = new_graph.nodes[u]
            for node in nodes[:sample_limit]:
                v, data = old_graph[node]
                uvs.append((u, v))
                obs.append(ob_data['ob_repr'])
                goals.append(data['dst_repr'])
                obs.append(data['ob_repr'])
                goals.append(ob_data['dst_repr'])

        rs = self.compute_reachability_batch(obs, goals)
        for i in range(len(uvs)):
            u, v = uvs[i]
            r1, r2 = rs[i * 2], rs[i * 2 + 1]
            if r1 > self.edge_add_thres:
                edges.append((u, v, r1))
            if r2 > self.edge_add_thres:
                edges.append((v, u, r2))

        return edges

    def _find_new_edges_sptm(self, nodes, new_g, old_g, sample_limit=100):
        # SPTM uses median, but I found median causes problems in some situations.
        # min provides stronger guarantee (too strong maybe?)
        def compute_sequence_reachability(seq1, seq2):
            assert len(seq1) == len(seq2)
            obs = [new_g.nodes[_]['ob_repr'] for _ in seq1]
            dsts = [new_g.nodes[_]['dst_repr'] for _ in seq2]
            return np.min(self.compute_reachability_batch(obs, dsts))

        def make_seq(tid, apos):
            return [self.trajs[tid][np.clip(_, 0, len(self.trajs[tid]) - 1)]
                    for _ in range(apos - self.delta_tw, apos + self.delta_tw + 1)]

        all_nodes = list(old_g.nodes)
        np.random.shuffle(all_nodes)
        edges = []

        for traj_id, anchor_idx in all_nodes[:sample_limit]:
            anchor_pos = self.trajs[traj_id].index((traj_id, anchor_idx))  # TODO: slow?
            seq1 = make_seq(traj_id, anchor_pos)
            for traj_id2, anchor_idx2 in nodes:
                anchor_pos2 = self.trajs[traj_id2].index((traj_id2, anchor_idx2))  # TODO: slow?
                seq2 = make_seq(traj_id2, anchor_pos2)
                r = compute_sequence_reachability(seq1, seq2)
                if r > self.edge_add_thres:
                    edges.append(((traj_id, anchor_idx), (traj_id2, anchor_idx2), r))
                r = compute_sequence_reachability(seq2, seq1)
                if r > self.edge_add_thres:
                    edges.append(((traj_id2, anchor_idx2), (traj_id, anchor_idx), r))
        return edges

    def set_subset_ratio(self, ratio):
        """
        Create a subgraph with a subset of trajectories of the original graph.
        :param ratio: a number in (0.0, 1.0]
        :return: None
        """
        if ratio == 1.0:
            self.subgraph = None
            return
        g = deepcopy(self.graph)
        if ratio < 1.0:
            n_traj_to_keep = int(len(self.trajs) * ratio)
            for u in self.graph.nodes:
                # remove nodes with traj ids >= n_traj_to_keep
                if u[0] >= n_traj_to_keep:
                    g.remove_node(u)
        self.subgraph = g

    def find_path(self, ob, dst_repr, edge_add_thres=0.9, edge_goal_thres=None,
                  allow_subgraph=False, cache_goal=False):
        """
        :param edge_goal_thres: if None will use edge_add_thres for edges towards goal.
        """

        use_cache_goal_graph = False
        if cache_goal and hasattr(self, 'cache_goal_graph'):
            g = deepcopy(self.cache_goal_graph)
            use_cache_goal_graph = True
        else:
            if allow_subgraph and self.subgraph is not None:
                g = deepcopy(self.subgraph)
            else:
                g = deepcopy(self.graph)

        existing_nodes = list(g.nodes.items())

        start_node = (-1, 0)
        goal_node = (-1, 1)

        start_time = time.time()

        # This is equivalent to the code below but much slower.
        #
        # for u, data in existing_nodes:
        #     r = self.compute_reachability(ob, data['dst_repr'])
        #     if r > edge_add_thres:
        #         g.add_edge(start_node, u, weight=-math.log(r))
        #     r = self.compute_reachability(data['ob_repr'], dst_repr)
        #     if r > edge_add_thres:
        #         g.add_edge(u, goal_node, weight=-math.log(r))

        if not use_cache_goal_graph:
            g.add_node(goal_node)
            # Add an edge from every node in the graph to the new goal node.
            obs = []
            dsts = []
            for u, data in existing_nodes:
                obs.append(data['ob_repr'])
                dsts.append(dst_repr)
            rs = self.compute_reachability_batch(obs, dsts)
            if edge_goal_thres is None:
                thres = edge_add_thres
            else:
                thres = edge_goal_thres
            for i in range(len(rs)):
                if rs[i] > thres:
                    print('add %s -> goal %.3f' % (existing_nodes[i][0], rs[i]))
                    g.add_edge(existing_nodes[i][0], goal_node, weight=-math.log(rs[i]))
            if cache_goal:
                self.cache_goal_graph = deepcopy(g)

        g.add_node(start_node)

        # Add an edge from current observation to every node in the graph
        obs = []
        dsts = []
        for u, data in existing_nodes:
            if u != goal_node:
                obs.append(ob)
                dsts.append(data['dst_repr'])
        rs = self.compute_reachability_batch(obs, dsts)
        for i in range(len(rs)):
            if existing_nodes[i][0] != goal_node:
                if rs[i] > edge_add_thres:
                    print('add ob -> %s %.3f' % (existing_nodes[i][0], rs[i]))
                    g.add_edge(start_node, existing_nodes[i][0], weight=-math.log(rs[i]))
        best_ob_entry = np.argmax(rs)
        print('best ob entry node: %s %.3f' % (existing_nodes[best_ob_entry][0], rs[best_ob_entry]))

        print('plan time:', time.time() - start_time)

        try:
            path = nx.shortest_path(g, source=start_node, target=goal_node, weight='weight')
        except nx.NetworkXNoPath:
            return None, None, None

        transition_probs = []

        neglog_prob = 0.0
        for i in range(len(path) - 1):
            neglogp = g.edges[path[i], path[i + 1]]['weight']
            neglog_prob += neglogp
            transition_probs.append(math.exp(-neglogp))
        path = path[1:-1]  # Exclude start and goal
        return path, -neglog_prob, {'transition_probs': transition_probs}

    def add_traj(self, traj, find_new_edge_samples=100):
        """
        :param traj: a list of observations representing the trajectory
        :return an integer identifier of this trajectory in the graph
        """
        this_idx = len(self.trajs)
        anchor_idxs = self.sparsifier.sparsify(traj, self.sparsify_thres)
        anchor_obs = [traj[_] for _ in anchor_idxs]

        old_g = self.graph
        g = deepcopy(self.graph)

        start_time = time.time()
        replacements = self._find_reusable_nodes(old_g, traj, anchor_obs, anchor_idxs)
        print('find reusable nodes time: %.2f' % (time.time() - start_time))

        self.trajs[this_idx] = []

        nodes = {}
        for anchor_idx in anchor_idxs:
            if anchor_idx not in replacements:
                nodes[anchor_idx] = (this_idx, anchor_idx)
                g.add_node((this_idx, anchor_idx),
                           ob_repr=self.sparsifier.get_ob_repr(traj, anchor_idx),
                           dst_repr=self.sparsifier.get_dst_repr(traj, anchor_idx))
            else:
                nodes[anchor_idx] = replacements[anchor_idx]

            self.trajs[this_idx].append(nodes[anchor_idx])

        reused_nodes = set(replacements.values())
        print('number of reused nodes:', len(reused_nodes),
              '(%.2f)' % (len(reused_nodes) / len(nodes)))
        print('reused_nodes: %s' % reused_nodes)

        # Add an edge for each pair of adjacent anchors
        # TODO: how should we set the probability of edges with reused nodes?
        if self.reuse_edge_prob_mode == 'hard':
            probs = [0.9999 for _ in range(len(anchor_obs) - 1)]
        elif self.reuse_edge_prob_mode == 'actual':
            # TODO: we probably should treat consecutive anchors as hard edges
            obs = [g.nodes[nodes[anchor_idxs[_]]]['ob_repr'] for _ in range(len(anchor_idxs) - 1)]
            dsts = [g.nodes[nodes[anchor_idxs[_]]]['dst_repr'] for _ in range(1, len(anchor_idxs))]
            probs = self.compute_reachability_batch(obs, dsts)
        else:
            raise RuntimeError('Unsupported reuse_edge_prob_mode %s' % self.reuse_edge_prob_mode)

        for i in range(len(anchor_obs) - 1):
            u, v = nodes[anchor_idxs[i]], nodes[anchor_idxs[i + 1]]
            assert u in g
            assert v in g
            if u[0] == v[0] and u[1] + 1 == v[1]:
                # u, v are anchors of the same trajectory but have no gap (i.e., they correspond to
                # adjacent frames). We treat it as a hard edge because v is clearly reachable from u.
                p = 0.9999
            else:
                p = probs[i]
            print('add %s -> %s' % (u, v))
            g.add_edge(u, v, weight=-math.log(p))

        new_nodes = set([_ for _ in nodes.values() if _ not in reused_nodes])
        start_time = time.time()
        if self.delta_tw is not None:
            new_edges = self._find_new_edges_sptm(new_nodes, g, old_g, sample_limit=find_new_edge_samples)
        else:
            new_edges = self._find_new_edges(new_nodes, g, old_g, sample_limit=find_new_edge_samples)
        print('find new edges time:', time.time() - start_time)
        print('number of new edges:', len(new_edges))

        for u, v, p in new_edges:
            assert u in g
            assert v in g
            assert p > self.edge_add_thres
            print('add new edge %s -> %s' % (u, v))
            g.add_edge(u, v, weight=-math.log(p))

        self.graph = g
        return this_idx


class NavGraphSPTM(NavGraphBase):
    def __init__(self, retrieval_net, motion_policy, subsample_factor=None,
                 delta_tl=None, delta_tw=None, s_shortcut=None):
        super(NavGraphSPTM, self).__init__()
        self.retrieval_net = retrieval_net
        self.motion_policy = motion_policy

        self.subsample_factor = subsample_factor
        self.delta_tl = delta_tl
        self.delta_tw = delta_tw
        self.s_shortcut = s_shortcut

        # Vertex is represented by a tuple (traj_idx, ob_idx)
        # Use node attribute to access data
        self.graph = nx.DiGraph()

        self.trajs = {}
        self.full_traj_obs = {}

        # You can put anything inside and it will be saved/loaded.
        self.extra = {
            'subsample_factor': subsample_factor,
            'delta_tl': delta_tl,
            'delta_tw': delta_tw,
            's_shortcut': s_shortcut,
        }

    def __repr__(self):
        return pprint_dict({
            'subsample_factor': self.subsample_factor,
            'delta_tl': self.delta_tl,
            'delta_tw': self.delta_tw,
            's_shortcut': self.s_shortcut,
            'number of nodes': len(self.graph.nodes),
            'number of edges': len(self.graph.edges)
        })

    def save(self, out_file):
        with open(out_file, 'wb') as f:
            pickle.dump({
                'trajs': self.trajs,
                'full_traj_obs': self.full_traj_obs,
                'graph': self.graph,
                'extra': self.extra
            }, f)

    @staticmethod
    def from_save_file(retrieval_net, motion_policy, save_file):
        g = NavGraphSPTM(retrieval_net, motion_policy)
        g.load(save_file)
        return g

    def load(self, save_file):
        with open(save_file, 'rb') as f:
            d = pickle.load(f)
            self.trajs = d['trajs']
            self.graph = d['graph']
            self.extra = d['extra']
            self.delta_tl, self.delta_tw, self.s_shortcut = [
                self.extra[_] for _ in [
                    'delta_tl', 'delta_tw', 's_shortcut']]
            self.full_traj_obs = d['full_traj_obs']
            # populate ob_repr and dst_repr node attributes
            print('populating node attributes...')
            for node in self.graph.nodes:
                traj_id, anchor_idx = node
                self.graph.nodes[node]['ob_repr'] = self.retrieval_net.get_ob_repr(
                    self.full_traj_obs[traj_id], anchor_idx)
                self.graph.nodes[node]['dst_repr'] = self.retrieval_net.get_dst_repr(
                    self.full_traj_obs[traj_id], anchor_idx)

    def compute_reachability(self, ob, dst):
        return self.retrieval_net.predict_reachability(ob, dst)

    def compute_reachability_batch(self, obs, dsts):
        return self.retrieval_net.predict_reachability_batch(obs, dsts)

    def _localize(self, ob, topk=1):
        nodes = list(self.graph.nodes)
        rs = []
        for node in nodes:
            rs.append(self.compute_reachability(ob, self.graph.nodes[node]['dst_repr']))
        closest_idxs = np.argsort(rs)[-topk:]
        # print('closest ob:', rs[closest_idx])
        return [nodes[_] for _ in closest_idxs]

    def _localize_dst(self, dst_repr, topk=1):
        nodes = list(self.graph.nodes)
        rs = []
        for node in nodes:
            rs.append(self.compute_reachability(self.graph.nodes[node]['ob_repr'], dst_repr))
        closest_idxs = np.argsort(rs)[-topk:]
        # print('closest dst:', rs[closest_idx]
        return [nodes[_] for _ in closest_idxs]

    def find_path(self, ob, dst_repr, **kwargs):
        g = deepcopy(self.graph)

        start_nodes = self._localize(ob, topk=1)
        goal_nodes = self._localize_dst(dst_repr, topk=1)

        best_path = None
        score = 0.0

        print('start nodes: %s' % start_nodes)
        print('goal nodes: %s' % goal_nodes)

        for start_node in start_nodes:
            for goal_node in goal_nodes:
                try:
                    path = nx.shortest_path(g, source=start_node, target=goal_node)
                    # print('path: %s' % path)
                    if best_path is None or len(path) < len(best_path):
                        best_path = path
                        score = -len(best_path)
                except nx.NetworkXNoPath:
                    continue

        return best_path, score, None

    def _create_traj_shortcuts(self, g, nodes):
        for i in range(len(nodes) - 1):
            u = nodes[i]
            # Add adjacent node
            v = nodes[i + 1]
            g.add_edge(u, v)
            # Add node that is delta_tl away
            v = nodes[min(i + self.delta_tl, len(nodes) - 1)]
            g.add_edge(u, v)

    def _create_shortcuts(self, old_g, new_g, nodes, sample_limit=100):
        def compute_median_sequence_reachability(seq1, seq2):
            assert len(seq1) == len(seq2)
            obs = [self.full_traj_obs[traj_id][ob_idx] for traj_id, ob_idx in seq1]
            dsts = [self.retrieval_net.get_dst_repr(self.full_traj_obs[traj_id], ob_idx)
                    for traj_id, ob_idx in seq2]
            return np.median(self.compute_reachability_batch(obs, dsts))

        all_nodes = list(old_g.nodes)
        np.random.shuffle(all_nodes)

        for traj_id, anchor_idx in all_nodes[:sample_limit]:
            seq1 = [(traj_id, np.clip(_, 0, len(self.full_traj_obs[traj_id]) - 1))
                    for _ in range(anchor_idx - self.delta_tw,
                                   anchor_idx + self.delta_tw + 1)]
            for traj_id2, anchor_idx2 in nodes:
                seq2 = [(traj_id2, np.clip(_, 0, len(self.full_traj_obs[traj_id2]) - 1))
                        for _ in range(anchor_idx2 - self.delta_tw,
                                       anchor_idx2 + self.delta_tw + 1)]
                r = compute_median_sequence_reachability(seq1, seq2)
                if r > self.s_shortcut:
                    new_g.add_edge((traj_id, anchor_idx), (traj_id2, anchor_idx2))
                r = compute_median_sequence_reachability(seq2, seq1)
                if r > self.s_shortcut:
                    new_g.add_edge((traj_id2, anchor_idx2), (traj_id, anchor_idx))

    def add_traj(self, traj, **kwargs):
        """
        :param traj: a list of observations representing the trajectory
        :return an integer identifier of this trajectory in the graph
        """
        this_idx = len(self.trajs)
        self.full_traj_obs[this_idx] = traj

        anchor_idxs = self.retrieval_net.sparsify(traj, self.subsample_factor)

        old_g = self.graph
        g = deepcopy(self.graph)

        nodes = []
        for anchor_idx in anchor_idxs:
            nodes.append((this_idx, anchor_idx))
            g.add_node((this_idx, anchor_idx))

        self._create_traj_shortcuts(g, nodes)

        start_time = time.time()
        self._create_shortcuts(old_g, g, nodes)
        print('create shortcuts time: %.2f' % (time.time() - start_time))

        self.trajs[this_idx] = nodes
        self.graph = g

        return this_idx


def update_nav_graph(graph, dataset, traj_ids, traj_info_dict,
                     max_trajs=None, sanitize=True, min_length=100, find_new_edge_samples=100):
    """
    Update graph using traj_ids from dataset. Already added trajectories are skipped.
    graph and traj_info_dict are updated.

    :param graph: current graph
    :param dataset: trajectory source
    :param traj_ids: a list of trajectory identifiers.
                     Each element is usually of the form (dataset_idx, traj_id)
    :param traj_info_dict: key: graph_traj_id value: trajectory samples from the dataset
    :param sanitize: trim the trajectory to remove bad start
    :param min_length: skip trajectory with length smaller than this.
    :return: None.
    """

    def _find_good_start(traj):
        """
        Find a good starting index (trim the trajectory if the agent backs at the beginning).
        :return: a good starting index
        """
        i = 0
        for i in range(1, len(traj)):
            dx, dy = traj[i]['pos'] - traj[i - 1]['pos']
            if math.sqrt(dx ** 2 + dy ** 2) < 0.01:
                continue
            heading = traj[i-1]['heading']
            dp = (math.cos(heading) * dx + math.sin(heading) * dy) / math.sqrt(dx ** 2 + dy ** 2)
            if dp > 0.01:
                break
        return i

    n_traj = 0
    id_mapping = graph.extra.setdefault('id_mapping', {})
    for idx, tid in enumerate(traj_ids):
        traj = dataset.locate_traj(tid)
        if sanitize:
            traj = traj[_find_good_start(traj):]
        if len(traj) < min_length:
            print('traj %s too short. skipped' % (tid,))
            continue

        if tid in id_mapping:
            print('skip traj %s' % (tid,))
        else:
            map_name = dataset.locate_traj_map(tid)
            obs = dataset.render_traj(traj, map_name=map_name)
            print('[%d/%d] add traj %s len: %d' % (idx, len(traj_ids), tid, len(obs)))
            graph_traj_id = graph.add_traj(obs, find_new_edge_samples=find_new_edge_samples)
            id_mapping[tid] = graph_traj_id

        if isinstance(traj, np.ndarray):
            # Real trace
            traj_info_dict[id_mapping[tid]] = {
                'samples': traj
            }
        else:
            # Simulation trace
            traj_info_dict[id_mapping[tid]] = {
                'samples': traj[()],  # [()] converts hd5 dataset into numpy,
                'attrs': dict(traj.attrs.items())
            }
        n_traj += 1
        if max_trajs is not None and n_traj >= max_trajs:
            break
