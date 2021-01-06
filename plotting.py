from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
import networkx
import matplotlib
import matplotlib.cm
import numpy as np
import scipy
import visJS2jupyter.visJS_module


def plot_network(network, title, subtext):
    ids, degrees = list(zip(*network.degree))

    fig, ax = plt.subplots(dpi=1200)
    plt.title(title)
    networkx.draw(network, nodelist=ids,
                  node_size=[d * 0.1 + 0.1 for d in degrees],
                  node_color=[d[1] for d in network.degree],
                  cmap=plt.cm.magma,
                  width=0.05)
    plt.text(0.5, -0.15, subtext, ha='center', va='bottom', transform=ax.transAxes)
    plt.show()


def plot_network_visjs2jupyter(network, title, subtext, vals=None):
    # Taken from http://bl.ocks.org/brinrosenthal/raw/658325f6e0db7419625a31c883313e9b/
    # TODO: remember to cite papers related to modules I'm using.
    if vals is None:
        vals = {v[0]: v[1] for v in network.degree()}

    # plt.title(title)

    # map the vals to node colors
    networkx.set_node_attributes(network, vals, 'vals')
    node_to_color = visJS2jupyter.visJS_module.return_node_to_color(network, field_to_map='vals',
                                                                    cmap=matplotlib.cm.spring_r, alpha=1,
                                                                    color_max_frac=.9, color_min_frac=.1)
    # set node initial positions using networkx's spring_layout function
    pos = networkx.spring_layout(network)

    nodes_dict = [{"id": n, "color": node_to_color[n],
                   "degree": networkx.degree(network, n),
                   "x": pos[n][0] * 1000,
                   "y": pos[n][1] * 1000} for n in network.nodes
                  ]
    node_map = dict(zip(network.nodes, range(len(network.nodes))))  # map to indices for source/target in edges
    edges = list(network.edges)
    edges_dict = [{"source": node_map[edges[i][0]], "target": node_map[edges[i][1]],
                   "color": "gray", "title": 'test'} for i in range(len(edges))]

    # set some network-wide styles
    visJS2jupyter.visJS_module.visjs_network(nodes_dict, edges_dict,
                                             node_size_multiplier=7,
                                             node_size_transform='',
                                             node_color_highlight_border='red',
                                             node_color_highlight_background='#D3918B',
                                             node_color_hover_border='blue',
                                             node_color_hover_background='#8BADD3',
                                             node_font_size=25,
                                             edge_arrow_to=True,
                                             physics_enabled=True,
                                             edge_color_highlight='#8A324E',
                                             edge_color_hover='#8BADD3',
                                             edge_width=3,
                                             max_velocity=15,
                                             min_velocity=1)

    # print(subtext)
    # plt.show()


def plot_index_duplicity_hist(data):
    data.groupby(data.index)['fold'].count().hist(bins=range(20), density=True)
    plt.title("peptides per gene histogram")
    plt.show()


def plot_val_against_control(actual_val, control_vals, metric_name):
    plt.figure(figsize=(10, 4))
    n, bins, patches = plt.hist(control_vals, density=True)
    # plt.axvline(np.mean(control_metric_values), color='k', linestyle='dashed', linewidth=1)
    # plt.axvline(subset_value, color='r', linestyle='dashed', linewidth=1)
    if np.isnan(actual_val):
        print("Warning: nan value for {}. Skipping plot".format(metric_name))
    else:
        loc = np.digitize(actual_val, bins)
        # np digitize and hist bins unfortunately don't agree on the upper bound of the last bin
        if actual_val == max(bins):
            loc = len(bins) - 1
        if (loc == 0) or (loc == len(bins)):
            # print("Warning: subset value outside control histogram for metric {}".format(metric_name))
            # print(actual_val, control_vals)
            plt.axvline(actual_val, color='r', linestyle='dashed', linewidth=1)
        else:
            patches[loc - 1].set_fc('r')
    plt.title("{} hist for control sets:".format(metric_name))
    red_patch = mpatches.Patch(color='red', label='Actual value')
    plt.legend(handles=[red_patch])

    sign = 1 if (actual_val > np.mean(control_vals)) else -1
    frac_more_extreme = len([val for val in control_vals if sign * val > sign * actual_val]
                            ) / float(len(control_vals))
    plt.xlabel("{}\np-value={:.1e}".format(metric_name, frac_more_extreme))
    plt.show()


def plot_metrics_comparison(metric1, metric2, metric1_name, metric2_name, keys):
    # print("{} v.s. {}".format(prot_metric, vertex_metric))
    x, y = zip(*[(metric1[k], metric2[k]) for k in keys])
    non_nans = np.where(~(np.isnan(x) | np.isnan(y)))[0]
    x, y = np.array(x)[non_nans], np.array(y)[non_nans]
    corr, p_value = scipy.stats.spearmanr(x, y, nan_policy='omit')
    m, b = np.polyfit(x, y, 1)

    plt.scatter(x, y, s=0.3)
    plt.plot((np.min(x), np.max(x)), b + m * np.array((np.min(x), np.max(x))), '--r', linewidth=0.75)
    plt.title("{} by {}".format(metric2_name, metric1_name))
    plt.xlabel(metric1_name + "\ncorr={:.1f}, p-value={:.1e}".format(corr, p_value))
    plt.ylabel(metric2_name)
    plt.show()
