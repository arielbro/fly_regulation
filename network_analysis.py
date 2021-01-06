from matplotlib import pyplot as plt
import numpy as np
import scipy
import math
import itertools
import networkx
import random
import pandas as pd
import plotting
from scipy import linalg


def read_network(path):
    df = pd.read_csv(path, sep='\t')
    df = df[['Official Symbol Interactor A', 'Official Symbol Interactor B']]
    ppi = networkx.from_pandas_edgelist(df, 'Official Symbol Interactor A',
                                        'Official Symbol Interactor B').to_undirected()
    return ppi


def intersect_ids(network, ids):
    return [name for name in network.nodes if name in ids]


def topological_analysis(network, subset, metrics, metric_names, random_iter=1000):
    for metric, name in zip(metrics, metric_names):
        topological_analysis_single_metric(network, subset, metric, name, random_iter)


def topological_analysis_single_metric(network, subset, topological_metric, metric_name, random_iter=1000):
    subset_value = topological_metric(network, subset)

    control_metric_values = []
    for i in range(random_iter):
        random_selection = np.random.choice(list(network.nodes.keys()), len(subset))
        control_value = topological_metric(network, random_selection)
        control_metric_values.append(control_value)

    # print("subset metric v.s. expected for metric {}:\n{:.3f} v.s. {:.3f}".format(metric_name, subset_value,
    #                                                                               control_average_metric))

    if not np.all(np.isnan(control_metric_values)):
        plotting.plot_val_against_control(subset_value, control_metric_values, metric_name)


def normalized_internal_edges_size(network, vertices):
    vertices = [v for v in vertices if network.has_node(v)]
    if len(vertices) == 0:
        return np.nan
    internal_edges = len(network.subgraph(vertices).copy().edges)
    normalizer = math.comb(len(vertices), 2)
    return internal_edges / float(normalizer)


def fraction_of_internal_edges(network, vertices):
    vertices = [v for v in vertices if network.has_node(v)]
    if len(vertices) == 0:
        return np.nan
    internal_edges = len(network.subgraph(vertices).copy().edges)
    total_edges = len(list(networkx.edge_boundary(network, vertices))) + internal_edges
    return internal_edges / float(total_edges)


def average_shortest_path_len(network, vertices):
    vertices = [v for v in vertices if network.has_node(v)]
    if len(vertices) == 0:
        return np.nan
    total_len = 0
    for (u, v) in itertools.combinations(vertices, 2):
        total_len += len(list(networkx.shortest_path(network, u, v)))
    return total_len / float(math.comb(len(vertices), 2))


def average_empirical_num_shortest_paths(network, vertices, n_iter=100):
    vertices = [v for v in vertices if network.has_node(v)]
    if len(vertices) == 0:
        return np.nan
    total_paths = 0
    for (u, v) in random.sample(list(itertools.combinations(vertices, 2)), min(n_iter, math.comb(len(vertices), 2))):
        total_paths += len(list(networkx.all_shortest_paths(network, u, v)))
    return total_paths / float(math.comb(len(vertices), 2))


def propagate(network, seed_values, method=None, k=None, alpha=None):
    # see review "Network propagation: a universal amplifier of genetic associations"
    # https://www.nature.com/articles/nrg.2017.38.pdf?origin=ppub
    if isinstance(seed_values, (np.ndarray, list)):
        seed_values = np.array(seed_values)
    elif isinstance(seed_values, dict):
        seed_values = np.array([seed_values.get(v, 0) for v in network.nodes])
    else:
        raise ValueError("Unknown format for seed values - {}".format(type(seed_values)))

    A = networkx.adjacency_matrix(network).todense()
    degrees = networkx.degree(network)
    D = np.diag([degrees[v] for v in network.nodes])

    # Authors differ on definitions. Here we use insulated_diffusion as the symmetric variant of RWR/pagerank.
    if method in ['heat', 'diffusion_kernel', 'heat_kernel']:
        if alpha is None:
            raise ValueError("alpha argument required for heat kernel propagation")
        W = D - A
        S = linalg.expm(-alpha * W)
    elif method in ['pagerank', 'random_walk_with_restart', 'RWR']:
        if alpha is None:
            raise ValueError("alpha argument required for random walk with restart propagation")
        inv_D_root = linalg.inv(linalg.sqrtm(D))
        W = np.matmul(np.matmul(inv_D_root, A), inv_D_root)
        S = alpha * linalg.inv(np.identity(len(network)) - (1 - alpha) * W)
    elif method == 'insulated_diffusion':
        W = np.matmul(A, linalg.inv(D))
        if alpha is None:
            raise ValueError("alpha argument required for random walk with restart propagation")
        S = alpha * linalg.inv(np.identity(len(network)) - (1 - alpha) * W)
    elif method in ['random_walk', 'RW']:
        if k is None:
            raise ValueError("k argument required for random walk")
        if k != int(k):
            raise ValueError("k argument must be integer")
        W = np.matmul(A, linalg.inv(D))
        S = np.linalg.matrix_power(W, int(k))
    else:
        raise ValueError("Unknown propagation method {}".format(method))

    return np.matmul(S, np.array(seed_values))


