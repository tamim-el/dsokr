import os

import numpy as np
import networkx as nx
from joblib import Parallel, delayed


def plot_qm9(G, ax, text=None, pos=None, draw_edge_feature=True):
    edge_style_map = {"0": "-", "1": "--", "2": ":"}
    node_atom_map = {"0": "C", "1": "N", "2": "O", "3": "F"}
    node_color_map = {"0": 1, "1": 0.67, "2": 0.33, "3": 0}
    if pos is None:
        pos = nx.kamada_kawai_layout(G)

    node_labels = dict(G.nodes.data("feature"))
    for key, val in node_labels.items():
        node_labels[key] = node_atom_map[val]

    nodes_colors = []
    for _, val in list(G.nodes.data("feature")):
        nodes_colors.append(node_color_map[val])

    if draw_edge_feature:
        edge_styles = [edge_style_map[edge[-1]]
                       for edge in G.edges.data("bond")]
        nx.draw(G, pos, labels=node_labels, style=edge_styles, width=3,
                ax=ax, node_color=nodes_colors, cmap='viridis', alpha=0.6)
    else:
        nx.draw(G, pos, labels=node_labels, width=3, ax=ax,
                node_color=nodes_colors, cmap='viridis', alpha=0.6)
    ax.set_title(text)

    return pos


def from_grkl_to_dict(Y_grkl):
    Y_dict = []
    n = len(Y_grkl)
    for i in range(n):
        Y_dict_new = {}

        A = Y_grkl[i].get_adjacency_matrix()
        Y_dict_new['A'] = A

        n_vert = A.shape[0]

        F = np.zeros((n_vert, 4))
        vert_labels = Y_grkl[i].get_labels(label_type='vertex')
        for j in range(n_vert):
            F[j, vert_labels[j]] = 1
        Y_dict_new['F'] = F

        if n_vert == 1:
            E = [[[0., 0., 0., 1.]]]
        else:
            E = np.zeros((n_vert, n_vert, 3))
            E = np.dstack((E, np.ones((n_vert, n_vert))))
            edge_labels = Y_grkl[i].get_labels(label_type='edge')
            edges1, edges2 = np.where(A == 1)
            for j in range(len(edges1)):
                v1, v2 = edges1[j], edges2[j]
                E[v1, v2, edge_labels[(v1, v2)]] = 1
                E[v1, v2, 3] = 0
        Y_dict_new['E'] = E
        Y_dict.append(Y_dict_new)

    Y = np.array(Y_dict)

    return Y


def to_networkx(y, use_edge_feature=True, thres=None):
    if use_edge_feature:
        E = y['E']
        adj = np.argmax(E, axis=-1)
        idx_edge = E.shape[-1] - 1
        A = np.asarray(adj != idx_edge, dtype=int)
    else:
        A = y['A']
        A = A.copy()
        np.fill_diagonal(A, 0.0)
        A = np.where(A > thres, 1, 0)

    F = y['F']
    F = np.argmax(F, axis=1)

    rows, cols = np.where(A == 1)
    edges = list(zip(rows.tolist(), cols.tolist()))
    G = nx.Graph()
    G.add_edges_from(edges)
    G.add_nodes_from(list(range(len(F))))

    F_dic = {}
    for k, l in enumerate(F):
        F_dic[k] = str(l.item())

    nx.set_node_attributes(G, F_dic, name="feature")

    if use_edge_feature:
        E_dict = {}
        for i, j in edges:
            E_dict[(i, j)] = {"bond": str(adj[i, j])}

        nx.set_edge_attributes(G, E_dict)

    numeric_indices = [index for index in range(G.number_of_nodes())]
    node_indices = sorted([node for node in G.nodes()])
    assert numeric_indices == node_indices

    return G


def eval_graph(G_preds, G_trgts, with_edge_feature=True, n_jobs=None):
    res_total = {}

    def node_match(x, y): return x["feature"] == y["feature"]
    def edge_match(x, y): return x["bond"] == y["bond"]
    if n_jobs is None:
        n_jobs = os.cpu_count()
    if with_edge_feature:
        geds = Parallel(n_jobs=n_jobs)(
            delayed(nx.graph_edit_distance)(
                G_pred, G_trgt, node_match=node_match, edge_match=edge_match
            )
            for G_pred, G_trgt in zip(G_preds, G_trgts)
        )
    else:
        geds = Parallel(n_jobs=n_jobs)(
            delayed(nx.graph_edit_distance)(
                G_pred, G_trgt, node_match=node_match)
            for G_pred, G_trgt in zip(G_preds, G_trgts)
        )

    res_total["edit_distance"] = np.mean(geds)
    res_total["eds"] = geds

    return res_total
