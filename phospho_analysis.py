import numpy as np
from scipy import stats
import pandas as pd
from matplotlib import pyplot as plt
import random
import plotting


def read_phospho_data(path, concentration_outlier_threshold=2):
    df = pd.read_csv(path)
    if '1-prob' not in df.columns:
        df['1-prob'] = np.power(10, -df['negative_log_one_minus_prob'])
    df = df[['Gene name', '1-prob', 'log2 fold change',
             'Nuclear cycle 14 parent protein conc. (uM) (some proteins not measured)']]
    df.columns = ['name', 'prob', 'fold', 'concentration']
    z_scores = stats.zscore(df['concentration'], nan_policy='omit')
    df['concentration'] = np.where(z_scores < concentration_outlier_threshold, df['concentration'], np.nan)
    df = df.set_index('name')
    return df


def best_per_protein(prot_data):
    res = pd.Series()
    res['prob'] = np.min(prot_data['prob'])
    pos_fold = np.max(prot_data['fold'])
    if pos_fold < 0:
        pos_fold = 0
    neg_fold = np.min(prot_data['fold'])
    if neg_fold > 0:
        neg_fold = 0
    res['fold'] = pos_fold if (abs(pos_fold) > abs(neg_fold)) else neg_fold
    for col in prot_data.columns:
        if col not in ['prob', 'fold']:
            res[col] = random.sample(sorted(prot_data[col]), 1)[0]
    return res


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
    up_proteins = data.loc[(data[val_field] > val_threshold) & (data[p_field] < p_threshold)]
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
