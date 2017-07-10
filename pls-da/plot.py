#!/usr/bin/env python3
# coding: utf-8

import collections
import math
import matplotlib.pyplot as plt
import numpy as np
import sklearn.cross_decomposition as sklCD
import IO


if __name__ == '__main__':
    IO.Log.warning('Please do not run that script, load it!')
    exit(1)


def properties_of(category, all_categories):
    """Return a dictionary with keys: edge_color, face_color, marker

       category is a unique string in all_categories (list, set or tuple)
       Raises Exception if category is not in all_categories
    """
    all_categories = sorted(list(set(all_categories)))

    if len(all_categories) < 2:
        IO.Log.warning('Too few categories to choose different colors/markers')

    if category not in all_categories:
        raise Exception('Could not choose a color/marker for {} since it '
                        'does not compare in all categories '
                        '({})'.format(category, all_categories))
    matches = (('#1F77B4', 'o'),  # blue,     circle
               ('#2CA02C', 'x'),  # green,    cross
               ('#D62728', '^'),  # red,      triangle_up
               ('#FF7F0E', 'D'),  # orange,   diamond
               ('#A00000', 's'),  # dark_red, square
               ('#FFD700', '*'),  # gold,     star
               ('#000000', '+'),  #           plus
               ('#000000', 'h'),  #           hexagon
               ('#000000', 'p'),  #           pentagon
               )
    index = all_categories.index(category) % len(matches)
    color, marker = matches[index]
    return {'edge_color': color, 'face_color': color, 'marker': marker}


def check_matrix(matrix):
    return matrix == 'x' or matrix == 'y'


def scatter_plot(x_values, y_values, cat=None, all_cat=None):
    """Draw a scatter plot using a custom color determined by the category."""

    if cat is None:
        color = 'blue'
        linecolor = '#1F77B4'
        marker = 'o'
    else:
        color = properties_of(cat, all_cat)['face_color']
        linecolor = color
        marker = properties_of(cat, all_cat)['marker']

    plt.scatter(x=x_values,
                y=y_values,
                edgecolors=linecolor,
                marker=marker,
                s=30,
                c=color,
                alpha=.6,
                # linewidth=0.10,
                label=cat)


def plot_plot(x_values, y_values, cat=None, all_cat=None):
    """Draw a plot using a custom color determined by the given category."""
    if cat is None:
        color = 'blue'
        linecolor = '#1F77B4'
    else:
        color = properties_of(cat, all_cat)['face_color']
        linecolor = color

    plt.plot(x_values,
             y_values,
             color=linecolor,         # line color
             linestyle='solid',
             marker='D',              # do not set it to
             markerfacecolor=color,  # marker color
             markersize=5)


def scores_plot(model, pc_a, pc_b, matrix='x', normalize=False):
    """Plot the scores on the specified components for the chosen matrix.

    Each point is plotted using a custom color determined by its category.

    If normalize is True the plot will be between -1 and 1.
    """
    if pc_a == pc_b:
        IO.Log.error('Principal components must be different!')
        exit(1)
    if not check_matrix(matrix):
        IO.Log.error('[scores_plot] '
                     'Accepted values for matrix are x and y')
        return

    pc_a, pc_b = min(pc_a, pc_b), max(pc_a, pc_b)

    if matrix == 'x':
        scores = model.T.copy()
    else:
        scores = model.U.copy()

    scores_a = scores[:, pc_a]
    scores_b = scores[:, pc_b]
    if normalize:
        scores_a = scores_a / max(abs(scores_a))
        scores_b = scores_b / max(abs(scores_b))
    for n in range(scores.shape[0]):
        score_pc_a = scores_a[n]
        score_pc_b = scores_b[n]
        cat = model.categories[n]
        scatter_plot(score_pc_a, score_pc_b, cat, model.categories)

    ax = plt.gca()
    plt.title('Scores plot for {}'.format(matrix))
    plt.xlabel('LV{}'.format(pc_a + 1))
    plt.ylabel('LV{}'.format(pc_b + 1))
    plt.axvline(0, linestyle='dashed', color='black')
    plt.axhline(0, linestyle='dashed', color='black')
    if normalize:
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
    else:
        ax.set_xlim(model.get_loadings_scores_xy_limits(pc_a, pc_b)['x'])
        ax.set_ylim(model.get_loadings_scores_xy_limits(pc_a, pc_b)['y'])

    handles, labels = ax.get_legend_handles_labels()
    by_label = collections.OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())


def loadings_plot(model, pc_a, pc_b, matrix='x'):
    """Plot the loadings on the specified components for the chosen matrix."""
    if pc_a == pc_b:
        IO.Log.error('Principal components must be different!')
        exit(1)

    if not check_matrix(matrix):
        IO.Log.error('[loadings_plot] '
                     'Accepted values for matrix are x and y')
        return

    pc_a, pc_b = min(pc_a, pc_b), max(pc_a, pc_b)

    if matrix == 'x':
        loadings = model.P.copy()
    else:
        loadings = model.Q.copy()

    scatter_plot(loadings[:, pc_a],
                 loadings[:, pc_b])

    ax = plt.gca()
    for n in range(loadings.shape[0]):
        ax.annotate(model.keys[n + 1],
                    xy=(loadings[n, pc_a], loadings[n, pc_b]),
                    xycoords='data',
                    xytext=(0, 5),
                    textcoords='offset points',
                    horizontalalignment='center',
                    verticalalignment='bottom')

    plt.title('Loadings plot for {}'.format(matrix))
    plt.xlabel('LV{}'.format(pc_a + 1))
    plt.ylabel('LV{}'.format(pc_b + 1))
    plt.axvline(0, linestyle='dashed', color='black')
    plt.axhline(0, linestyle='dashed', color='black')


def biplot(model, pc_a, pc_b, matrix='x'):
    """Plot both loadings and scores on the same graph."""
    if pc_a == pc_b:
        IO.Log.error('Principal components must be different!')
        exit(1)

    if not check_matrix(matrix):
        IO.Log.error('[biplot] '
                     'Accepted values for matrix are x and y')
        return

    pc_a, pc_b = min(pc_a, pc_b), max(pc_a, pc_b)

    scores_plot(model, pc_a, pc_b, normalize=True, matrix=matrix)
    loadings_plot(model, pc_a, pc_b, matrix=matrix)

    plt.title('Biplot for {}'.format(matrix))
    plt.xlabel('LV{}'.format(pc_a + 1))
    plt.ylabel('LV{}'.format(pc_b + 1))


def explained_variance_plot(model, matrix='x'):
    """Plot the cumulative explained variance for the chosen matrix."""

    if not check_matrix(matrix):
        IO.Log.error('[explained_variance_plot] '
                     'Accepted values for matrix are x and y')
        return

    plt.title('Explained variance plot for {}'.format(matrix))
    plt.xlabel('Principal component number')
    plt.ylabel('Cumulative variance captured (%)')

    explained_variance = model.get_explained_variance(matrix)

    ax = plt.gca()
    ax.set_xlim(-0.5, len(explained_variance))
    ax.set_ylim(max(-2, explained_variance[0] - 2), 102)
    plt.plot(range(len(explained_variance)),
             np.cumsum(explained_variance))


def scree_plot(model, matrix='x'):
    """Plot the explained variance of the model for the chosen matrix."""
    plt.title('Scree plot for {}'.format(matrix))
    plt.xlabel('Principal component number')
    plt.ylabel('Eigenvalues')

    if not check_matrix(matrix):
        IO.Log.error('[scree_plot] Accepted values for matrix are x and y '
                     '(got {} instead)'.format(repr(matrix)))
        raise Exception('Bad matrix parameter ({}) in '
                        'scree_plot() '.format(repr(matrix)))

    if matrix == 'x':
        eigen = model.x_eigenvalues
    else:
        eigen = model.y_eigenvalues

    ax = plt.gca()
    ax.set_xlim(-0.5, len(eigen))
    ax.set_ylim(-0.5, math.ceil(eigen[0]) + 0.5)
    plot_plot(range(len(eigen)), eigen)


def inner_relation_plot(model, nr):
    """Plot the inner relation for the chosen latent variable."""
    if nr > model.nr_lv:
        IO.Log.error('[inner_relation_plot] '
                     'chosen LV must be in [0-{}]'.format(model.nr_lv))

    plt.title('Inner relation for LV {}'.format(nr))
    plt.xlabel('t{}'.format(nr))
    plt.ylabel('u{}'.format(nr))

    for i in range(model.T.shape[0]):
        cat = model.categories[i]
        scatter_plot(model.T[i, nr], model.U[i, nr], cat, model.categories)


def data_plot(model, all_cat):
    """Plot the dataset distinguishing with colors the categories."""

    plt.title('Data by category')
    plt.xlabel('sample')
    plt.ylabel('Value')

    for i in range(model.dataset.shape[0]):
        cat = model.categories[i]
        plt.plot(range(model.dataset.shape[1]),
                 model.dataset[i],
                 color=properties_of(cat, all_cat)['face_color'],  # line color
                 linestyle='solid',
                 alpha=.5,
                 marker=properties_of(cat, all_cat)['marker'],     # do not set it to
                 markerfacecolor=properties_of(cat, all_cat)['face_color'],
                 markeredgecolor=properties_of(cat, all_cat)['edge_color'])


def modeled_Y_plot(model):
    """Plot the difference between the real categories and the modeled ones."""
    plt.title('Y calculated')
    plt.xlabel('sample')
    plt.ylabel('modeled Y')
    for j in range(model.p):
        for i in range(model.n):
            cat = model.categories[i]
            scatter_plot(i, model.Y_modeled[i, j], cat, model.categories)


def y_leverage_plot(model):
    """Plot Y residuals over the leverage."""
    plt.title('Leverage')
    plt.xlabel('leverage')
    plt.ylabel('Y residuals')
    tmp = np.linalg.inv(np.dot(model.U.T, model.U))
    leverage = np.empty(model.n)

    for j in range(model.p):
        for i in range(model.n):
            leverage[i] = model.U[i].dot(tmp).dot(model.U[i].T)
            cat = model.categories[i]
            scatter_plot(leverage[i], model.E_y[i, j], cat, model.categories)



def d_plot(model):
    plt.title('Inner relation (variable b)')
    plt.xlabel('LV number')
    plt.ylabel('inner relation variable')

    for i in range(model.d.shape[0]):
        cat = model.categories[i]
        scatter_plot(i, model.d[i], cat, model.categories)


def inner_relation_sklearn(model, nr):
    if nr > model.nr_lv:
        IO.Log.error('[inner_relation_plot] '
                     'chosen LV must be in [0-{}]'.format(model.nr_lv))

    X = model.dataset.copy()
    Y = model.dummy_Y.copy()

    sklearn_pls = sklCD.PLSRegression(n_components=min(model.n, model.m),
                                      scale=True, max_iter=1e4, tol=1e-6,
                                      copy=True)
    sklearn_pls.fit(X, Y)

    plt.title('Inner relation for LV {} (sklearn)'.format(nr))
    plt.xlabel('t{}'.format(nr))
    plt.ylabel('u{}'.format(nr))

    for i in range(model.T.shape[0]):
        cat = model.categories[i]
        scatter_plot(sklearn_pls.x_scores_[i, nr],
                     sklearn_pls.y_scores_[i, nr],
                     cat, model.categories)
