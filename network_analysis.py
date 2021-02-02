from matplotlib import pyplot as plt
import numpy as np
import scipy
import math
import itertools
import networkx
import random
import pandas as pd
import plotting
from scipy import linalg, stats
from sklearn.model_selection import cross_val_score, train_test_split
import sklearn
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
import seaborn as sns


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


def average_shortest_path_len(network, vertices, no_path_val=1e10):
    vertices = [v for v in vertices if network.has_node(v)]
    if len(vertices) == 0:
        return np.nan
    total_len = 0
    for (u, v) in itertools.combinations(vertices, 2):
        try:
            total_len += len(list(networkx.shortest_path(network, u, v)))
        except networkx.NetworkXNoPath:
            total_len += no_path_val
    return total_len / float(math.comb(len(vertices), 2))


def average_empirical_shortest_path_len(network, vertices, no_path_val=1e10, n_iter=1000):
    vertices = [v for v in vertices if network.has_node(v)]
    if len(vertices) == 0:
        return np.nan
    total_len = 0
    for (u, v) in random.sample(list(itertools.combinations(vertices, 2)), min(n_iter, math.comb(len(vertices), 2))):
        try:
            total_len += len(list(networkx.shortest_path(network, u, v)))
        except networkx.NetworkXNoPath:
            total_len += no_path_val
    return total_len / float(math.comb(len(vertices), 2))


def average_empirical_num_shortest_paths(network, vertices, n_iter=1000):
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
        S = np.asarray(np.linalg.matrix_power(W, int(k)))
    else:
        raise ValueError("Unknown propagation method {}".format(method))

    return np.matmul(S, seed_values)


class PropagationEstimator(ClassifierMixin):
    """
    A predictor fitting sklearn's format for scoring propagation based completion.
    The predictor takes a matrix of a single feature - node indices, guaranteed to be subsest of the proteins
    measured in the phospho data, given as ids.
    Fitting corresponds to propagating from the nodes with ids given
    (using seed values given at initialization)
    Predicting corresponds to revealing propagated values for the ids given.
    """
    def __init__(self, network, seed_values, method=None, alpha=None, k=None, id_to_index=None, is_classifier=True):
        self.network = network
        if id_to_index is not None:
            self.id_to_index = id_to_index
        else:
            self.id_to_index = {v: u for u, v in dict(enumerate(network.nodes)).items()}
        if isinstance(seed_values, dict):
            dict_seed_values = seed_values
            seed_values = [None] * len(self.id_to_index)
            for prot_id, val in dict_seed_values.items():
                seed_values[self.id_to_index[prot_id]] = val
        self.seed_values = seed_values
        # kwargs not allowed on sklearn estimators, that was a nasty mystery bug...
        self.method = method
        self.alpha = alpha
        self.k = k
        self.pred = None
        self.is_classifier = is_classifier
        if self.is_classifier:
            self.classes_ = [0, 1]

    def get_params(self, deep=True):
        return {'network': self.network, 'id_to_index': self.id_to_index,
                'seed_values': self.seed_values, 'method': self.method, 'alpha': self.alpha, 'k': self.k}

    def set_params(self, **params):
        self.network = params['network']
        self.id_to_index = params['id_to_index']
        self.seed_values = params['seed_values']
        self.method = params['method']
        self.alpha = params['alpha']
        self.k = params['k']

    def fit(self, X=None, y=None):
        ids = X[:, 0]
        fit_seed_values = {n: self.seed_values[self.id_to_index[n]] for n in ids}

        self.pred = propagate(network=self.network, seed_values=fit_seed_values,
                              method=self.method, alpha=self.alpha, k=self.k)

    def transform(self, X):
        ids = X[:, 0]
        cur_preds = np.array([self.pred[self.id_to_index[node_id]] for node_id in ids])
        # cur_preds = np.array([1] * X.shape[0])
        return cur_preds

    def predict(self, X):
        if self.is_classifier:
            return (self.transform(X) > 0.5).astype(int)
        else:
            return self.transform(X)

    def predict_proba(self, X):
        # Will only work if seed values conform with confidence values.
        probs = np.array(self.transform(X))
        return np.array([1 - probs, probs]).transpose()


def cross_validate_propagation(network, seed_values, scoring='explained_variance', method=None, k=None, alpha=None,
                               title_prefix="", is_classifier=False):
    predictor = PropagationEstimator(network, seed_values, method=method, k=k, alpha=alpha, is_classifier=is_classifier)
    ids, vals = zip(*seed_values.items())
    X = np.array([ids]).transpose()
    y = np.array(vals)

    # TODO: currently leave-one-out
    scores = cross_val_score(predictor, X, y, scoring=scoring, cv=5)
    print("average score: {:.3f} ({:.3f} std of scores)".format(np.mean(scores), np.std(scores)))

    if scoring == 'roc_auc':
        title = title_prefix + " method:{}, k:{}, alpha:{}".format(method, k, alpha)
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
        predictor.fit(X_train, y_train)
        sklearn.metrics.plot_roc_curve(predictor, X_test, y_test, name=title)


def correlate_vertex_scores_with_distance(network, vertex_scores, score_name, plot_graph=True):
    graph_dists = networkx.all_pairs_shortest_path_length(network)
    graph_dists_list = []
    score_dists = []
    pairs_visited = set()
    for v1, node_dist_dict in graph_dists:
        if v1 not in vertex_scores:
            continue
        for v2 in node_dist_dict:
            if v1 == v2:
                continue
            if v2 not in vertex_scores:
                continue
            if ((v1, v2) in pairs_visited) or ((v2, v1) in pairs_visited):
                continue
            else:
                pairs_visited.add((v1, v2))
                graph_dists_list.append(node_dist_dict[v2])
                score_dists.append(abs(vertex_scores[v1] - vertex_scores[v2]))

    graph_dists_list = np.array(graph_dists_list)
    score_dists = np.array(score_dists)
    corr, p_value = stats.pearsonr(graph_dists_list, score_dists)
    if plot_graph:
        dist_vals = sorted(np.unique(graph_dists_list))
        box_vals = [score_dists[np.where(graph_dists_list == d)[0]] for d in dist_vals]
        plt.boxplot(box_vals)
        # sns.boxplot(box_vals)
        # plt.scatter(graph_dists_list, score_dists)
        plt.title("{} by graph distance. Correlation: {:.2f} (p-value {:.1e})".format(score_name, corr, p_value))
        plt.xlabel("shortest-path graph distance")
        plt.ylabel("{} absolute difference".format(score_name))
        plt.show()
    else:
        print("{} absolute difference to graph distance correlation: {:.2f} (p-value {:.1e})".format(score_name,
                                                                                                     corr, p_value))
