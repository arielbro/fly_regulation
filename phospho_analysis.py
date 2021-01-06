import numpy as np
from scipy import stats
import pandas as pd
from matplotlib import pyplot as plt
import plotting
from sklearn.model_selection import cross_val_score
import sklearn
from sklearn.base import BaseEstimator, RegressorMixin
from network_analysis import propagate


def read_phospho_data(path, concentration_outlier_threshold=2):
    df = pd.read_csv(path)
    df = df[['Gene name', '1-prob', 'log2 fold change',
             'Nuclear cycle 14 parent protein conc. (uM) (some proteins not measured)']]
    df.columns = ['name', 'prob', 'fold', 'concentration']
    z_scores = stats.zscore(df['concentration'], nan_policy='omit')
    df['concentration'] = np.where(z_scores < concentration_outlier_threshold, df['concentration'], np.nan)
    df = df.set_index('name')
    return df


def aggregate_peptide_values(data, agg=np.mean, verbose=True):
    if verbose:
        plotting.plot_index_duplicity_hist(data)
    return data.groupby(data.index).apply(agg)


def restrict_to_common_ids(phospho_data, network, verbose=True):
    common_ids = set(phospho_data.index) & set(network.nodes)
    if verbose:
        print("Common protein ids: {}, out of {} in phospho data and {} in network".format(
            len(common_ids), len(phospho_data.index), len(network)))
    return phospho_data[phospho_data.index.isin(common_ids)]


def get_up_down_sets(data, val_threshold, p_threshold, val_field='fold', p_field='prob', verbose=True):
    up_proteins = data.loc[(abs(data[val_field]) > val_threshold) & (data[p_field] < p_threshold)]
    down_proteins = data.loc[(data[val_field] < -val_threshold) & (data[p_field] < p_threshold)]
    if verbose:
        counts = [len(up_proteins), len(down_proteins),
                len(data) - len(up_proteins) - len(down_proteins)]
        bars = plt.bar(['up', 'down', 'rest'], counts)

        for bar, count in zip(bars, counts):
            height = bar.get_height()
            plt.text(bar.get_x() + 0.4, height + 2, count, ha='center')
        plt.title("Proportion of over and under phosphorylated proteins")
        plt.show()
    return up_proteins.index, down_proteins.index


def metric_prot_scores(metric, proteins):
    return [metric[p] for p in proteins if p in metric]


def up_down_centrality_analysis(up_proteins, down_proteins, total_proteins, metrics, metric_names):
    for metric, name in zip(metrics, metric_names):
        up_down_centrality_analysis_single_metric(up_proteins, down_proteins, total_proteins, metric, name)


def up_down_centrality_analysis_single_metric(up_proteins, down_proteins, total_proteins, metric, metric_name):
    print("up down analysis for metric {}".format(metric_name))
    up_metric = metric_prot_scores(metric, up_proteins)
    down_metric = metric_prot_scores(metric, down_proteins)
    total_metric = metric_prot_scores(metric, total_proteins)
    print("metric means: up: {:.2e}, down: {:.2e}, total: {:.2e}".format(np.mean(up_metric),
                                                                         np.mean(down_metric),
                                                                         np.mean(total_metric)))
    print("p-values (against full set): up: {:.2e}, down: {:.2e}".format(
        stats.mannwhitneyu(up_metric, total_metric)[1], stats.mannwhitneyu(down_metric, total_metric)[1]))

    print("")


class PropagationEstimator(RegressorMixin):
    """
    A predictor fitting sklearn's format for scoring propagation based completion.
    The predictor takes a matrix of a single feature - node indices, guaranteed to be subsest of the proteins
    measured in the phospho data, given as ids.
    Fitting corresponds to propagating from the nodes with ids given
    (using seed values given at initialization)
    Predicting corresponds to revealing propagated values for the ids given.
    """
    def __init__(self, network, seed_values, method=None, alpha=None, k=None, id_to_index=None):
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
        return cur_preds

    def predict(self, X):
        return self.transform(X)

    def predict_proba(self, X):
        # Will only work if seed values conform with confidence values.
        return self.transform(X)


def cross_validate_propagation(network, seed_values, scoring='explained_variance', method=None, k=None, alpha=None):
    predictor = PropagationEstimator(network, seed_values, method=method, k=k, alpha=alpha)
    ids, vals = zip(*seed_values.items())
    X = np.array([ids]).transpose()
    y = np.array(vals)
    # TODO: currently leave-one-out
    return cross_val_score(predictor, X, y, scoring=scoring, cv=5)

