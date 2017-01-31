#!/usr/bin/env python
# coding: utf-8

import collections
import IO
import matplotlib.pyplot as plt
import numpy as np
import math
import sklearn.cross_decomposition


if __name__ == '__main__':
    IO.Log.warning('Please do not run that script, load it!')
    exit(1)


def properties_of(category):
    """Return a dictionary with keys: edge_color, face_color, marker."""
    circle, cross, diamond, triangle = 'o', 'x', 'D', '^'

    blue, dark_red, gold, = '#1F77B4', '#A00000', '#FFD700'
    green, red, orange = '#2CA02C', '#D62728', '#FF7F0E'

    if category == 'B':
        return {'edge_color': blue, 'face_color': blue, 'marker': circle}
    elif category == 'E':
        return {'edge_color': green, 'face_color': green, 'marker': cross}
    elif category == 'G':
        return {'edge_color': red, 'face_color': red, 'marker': triangle}
    elif category == 'N':
        return {'edge_color': gold, 'face_color': dark_red, 'marker': diamond}
    elif category == 'NA':
        return {'edge_color': blue, 'face_color': blue, 'marker': circle}
    elif category == 'SA':
        return {'edge_color': green, 'face_color': green, 'marker': cross}
    elif category == 'U':
        return {'edge_color': red, 'face_color': red, 'marker': triangle}
    elif category == 'WL':
        return {'edge_color': orange, 'face_color': orange, 'marker': diamond}
    else:
        raise Exception('Unknown category ' + category)


def scatter_plot(x_values, y_values, cat):
    plt.scatter(x=x_values,
                y=y_values,
                edgecolors=properties_of(cat)['edge_color'],
                marker=properties_of(cat)['marker'],
                s=40,
                c=properties_of(cat)['face_color'],
                alpha=.6,
                # linewidth=0.10,
                label=cat)

def plot_plot(x_values, y_values):
    plt.plot(x_values,
             y_values,
             color='#1F77B4',         # line color
             linestyle='solid',
             marker='D',              # do not set it to
             markerfacecolor='blue',  # marker color
             markersize=5)

def scores_plot(model, pc_x, pc_y, normalize=False):
    """Plot the scores on the specified components."""
    if pc_x == pc_y:
        IO.Log.error('Principal components must be different!')
        exit(1)

    pc_x, pc_y = min(pc_x, pc_y), max(pc_x, pc_y)

    scores_x = model.T[:, pc_x]
    scores_y = model.T[:, pc_y]
    if normalize:
        scores_x = scores_x / max(abs(scores_x))
        scores_y = scores_y / max(abs(scores_y))
    for n in range(model.T.shape[0]):
        score_pc_x = scores_x[n]
        score_pc_y = scores_y[n]
        cat = model.categories[n]
        scatter_plot(score_pc_x, score_pc_y, cat)

    ax = plt.gca()
    plt.title('Scores plot')
    plt.xlabel('PC{}'.format(pc_x + 1))
    plt.ylabel('PC{}'.format(pc_y + 1))
    plt.axvline(0, linestyle='dashed', color='black')
    plt.axhline(0, linestyle='dashed', color='black')
    if normalize:
        ax.set_xlim(-1.1,1.1)
        ax.set_ylim(-1.1,1.1)
    else:
        ax.set_xlim(model.get_loadings_scores_xy_limits(pc_x, pc_y)['x'])
        ax.set_ylim(model.get_loadings_scores_xy_limits(pc_x, pc_y)['y'])

    handles, labels = ax.get_legend_handles_labels()
    by_label = collections.OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())


def loadings_plot(model, pc_x, pc_y):
    """Plot the loadings."""
    if pc_x == pc_y:
        IO.Log.error('Principal components must be different!')
        exit(1)

    pc_x, pc_y = min(pc_x, pc_y), max(pc_x, pc_y)
    plt.scatter(x=model.P[:, pc_x],
                y=model.P[:, pc_y])

    ax = plt.gca()
    for n in range(model.P.shape[0]):
        ax.annotate(model.keys[n + 1],
                    xy=(model.P[n, pc_x], model.P[n, pc_y]),
                    xycoords='data',
                    xytext=(0, 5),
                    textcoords='offset points',
                    horizontalalignment='center',
                    verticalalignment='bottom')

    plt.title('Loadings plot')
    plt.xlabel('PC{}'.format(pc_x + 1))
    plt.ylabel('PC{}'.format(pc_y + 1))
    plt.axvline(0, linestyle='dashed', color='black')
    plt.axhline(0, linestyle='dashed', color='black')

def biplot(model, pc_x, pc_y):
    scores_plot(model, pc_x, pc_y, normalize=True)
    loadings_plot(model, pc_x, pc_y)

def explained_variance_plot(model, matrix='x'):
    """Plot the cumulative explained variance."""
    plt.title('Explained variance plot')
    plt.xlabel('Principal component number')
    plt.ylabel('Cumulative variance captured (%)')

    explained_variance = model.get_explained_variance(matrix)

    ax = plt.gca()
    ax.set_xlim(-0.5, len(explained_variance))
    ax.set_ylim(max(-2, explained_variance[0] - 2), 102)
    plt.plot(range(len(explained_variance)),
             np.cumsum(explained_variance))

def scree_plot(model, matrix='x'):
    """Plot the explained variance of the model."""
    plt.title('Scree plot')
    plt.xlabel('Principal component number')
    plt.ylabel('Eigenvalues')

    if matrix == 'x':
        eigen = model.x_eigenvalues
    elif matrix == 'y':
        eigen = model.y_eigenvalues
    else:
        IO.Log.error('[scree_plot] '
                     'Accepted values for matrix are x and y')

        return

    ax = plt.gca()
    ax.set_xlim(-0.5, len(eigen))
    ax.set_ylim(-0.5, math.ceil(eigen[0]) + 0.5)
    plot_plot(range(len(eigen)),
             eigen)

def inner_relation_plot(model, nr):
    if nr > model.nr_lv:
        Io.Log.error('[inner_relation_plot] '
                     'chosen LV must be in [0-{}]'.format(model.nr_lv))

    plt.title('Inner relation for LV {}'.format(nr))
    plt.xlabel('t{}'.format(nr))
    plt.ylabel('u{}'.format(nr))

    for i in range(model.T.shape[0]):
        cat = model.categories[i]
        scatter_plot(model.T[i, nr], model.U[i, nr], cat)

def d_plot(model):
    plt.title('Inner relation (variable b)')
    plt.xlabel('LV number')
    plt.ylabel('inner relation variable')

    for i in range(model.d.shape[0]):
        cat = model.categories[i]
        scatter_plot(i, model.d[i], cat)

def inner_relation_sklearn(model, nr):
    if nr > model.nr_lv:
        Io.Log.error('[inner_relation_plot] '
                     'chosen LV must be in [0-{}]'.format(model.nr_lv))

    X = model.dataset.copy()
    Y = model.dummy_Y.copy()

    sklearn_pls = sklearn.cross_decomposition.PLSRegression(
                  n_components=min(model.n, model.m), scale=True, max_iter=1e4, tol=1e-6, copy=True)
    sklearn_pls.fit(X, Y)

    plt.title('Inner relation for LV {} (sklearn)'.format(nr))
    plt.xlabel('t{}'.format(nr))
    plt.ylabel('u{}'.format(nr))

    for i in range(model.T.shape[0]):
        cat = model.categories[i]
        scatter_plot(sklearn_pls.x_scores_[i, nr], sklearn_pls.y_scores_[i, nr], cat)
