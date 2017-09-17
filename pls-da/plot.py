#!/usr/bin/env python3
# coding: utf-8

import collections
import functools
import math
import numpy as np
import scipy.stats as scipy_stats

import IO
import model


MODEL = None
TRAIN_SET = None
TEST_SET = None
STATS = None


@functools.lru_cache(maxsize=32, typed=False)
def symbol(category=None):
    """Return a dictionary with keys: hex, marker.

       On unknown category the first record is returned as default.
    """
    records = (('#1F77B4', 'o'),  # blue,     circle
               ('#2CA02C', 'x'),  # green,    cross
               ('#D62728', '^'),  # red,      triangle_up
               ('#FF7F0E', 'D'),  # orange,   diamond
               ('#A00000', 's'),  # dark_red, square
               ('#FFD700', '*'),  # gold,     star
               ('#0D0099', '.'),  # darkblue, point


               # ('#000000', '+'),  #           plus
               # ('#000000', 'h'),  #           hexagon
               # ('#000000', 'p'),  #           pentagon
               )
    index = -1
    if category in TRAIN_SET.categories:
        index = sorted(TRAIN_SET.categories).index(category) % len(records)
    return [dict(zip(('hex', 'marker'), rec)) for rec in records][index]


def update_global_model(value):
    """Set MODEL to value."""
    if isinstance(value, model.Model):
        global MODEL
        MODEL = value
    else:
        IO.Log.error('Wrong type in update_global_model() '
                     '({}).'.format(type(value)))


def update_global_train_set(value):
    """Set TRAIN_SET to value."""
    if isinstance(value, model.TrainingSet):
        global TRAIN_SET
        TRAIN_SET = value
    else:
        IO.Log.error('Wrong type in update_global_train_set() '
                     '({}).'.format(type(value)))


def update_global_test_set(value):
    """Set TEST_SET to value."""
    if isinstance(value, model.TestSet):
        global TEST_SET
        TEST_SET = value
    else:
        IO.Log.error('Wrong type in update_global_test_set() '
                     '({}).'.format(type(value)))


def update_global_statistics(value):
    """Set STATS to value."""
    if isinstance(value, model.Statistics):
        global STATS
        STATS = value
    else:
        IO.Log.error('Wrong type in update_global_statistics() '
                     '({}).'.format(type(value)))


def scatter_wrapper(ax, x_values, y_values, cat=None):
    """Draw a scatter plot using a different color/marker for each category."""
    ax.scatter(x=x_values, y=y_values,
               alpha=0.5,
               c=symbol(cat)['hex'],
               edgecolors=symbol(cat)['hex'],
               label=cat,
               # linewidth=0.1,
               marker=symbol(cat)['marker'],
               s=30)


def line_wrapper(ax, x_values, y_values, cat=None, linestyle='solid'):
    """Draw a line plot using a different color for each category."""
    ax.plot(x_values, y_values,
            alpha=0.5,
            color=symbol(cat)['hex'],
            linestyle=linestyle,
            marker=symbol(cat)['marker'],
            markerfacecolor=symbol(cat)['hex'],
            markeredgecolor=symbol(cat)['hex'],
            markersize=5.48)


def scree(ax, x=False, y=False):
    """Plot the explained variance of the model for the x or y matrix.

       Raise ValueError if x and y does not differ.
    """
    if bool(x) == bool(y):
        raise ValueError('In plot.scree() X, Y matrix flags must differ')

    eigen = MODEL.x_eigenvalues if x else MODEL.y_eigenvalues
    line_wrapper(ax, range(1, len(eigen) + 1), eigen)

    ax.set_title('Scree plot for {}'.format('X' if x else 'Y'))
    ax.set_xlabel('Number of latent variables')
    ax.set_ylabel('Eigenvalues')
    ax.set_xlim(0.5, len(eigen) + 0.5)
    ax.set_ylim(-0.3, math.ceil(eigen[0]) + 0.1)
    if len(eigen) <= 12:
        ax.set_xticks(list(range(1, len(eigen) + 1)))


def cumulative_explained_variance(ax, x=False, y=False):
    """Plot the cumulative explained variance for the x or y matrix.

       Raise ValueError if x and y does not differ.
    """
    if bool(x) == bool(y):
        raise ValueError('In plot.explained_variance() X, Y matrix flags '
                         'does not differ')

    explained_variance = model.explained_variance(MODEL, 'x' if x else 'y')
    line_wrapper(ax, range(1, len(explained_variance) + 1),
                 np.cumsum(explained_variance))

    ax.set_title('Explained variance plot for {}'.format('X' if x else 'Y'))
    ax.set_xlabel('Number of latent variables')
    ax.set_ylabel('Cumulative variance captured (%)')
    ax.set_xlim(0.5, len(explained_variance) + 0.5)
    ax.set_ylim(max(-2, explained_variance[0] - 2), 102)
    if len(explained_variance) <= 12:
        ax.set_xticks(list(range(1, len(explained_variance) + 1)))


def inner_relations(ax, num):
    """Plot the inner relations for the chosen latent variable.

       Raise ValueError if num is greater than available latent variables.
    """
    if num > MODEL.nr_lv:
        IO.Log.debug('In plot.inner_relations() num (' + str(num) + ') is '
                     'greater than MODEL.nr_lv ({})'.format(MODEL.nr_lv))
        raise ValueError('In plot.inner_relations() chosen latent variable '
                         'number ({}) is out of bounds [1:{}]'.format(
                             num, MODEL.nr_lv))

    for i in range(MODEL.T.shape[0]):
        scatter_wrapper(ax, MODEL.T[i, num - 1], MODEL.U[i, num - 1],
                        TRAIN_SET.categorical_y[i])

    ax.set_title('Inner relation for LV {}'.format(num))
    ax.set_xlabel('t{}'.format(num))
    ax.set_ylabel('u{}'.format(num))


def biplot(ax, lv_a, lv_b, x=False, y=False, normalize=True):
    """Plot loadings and scores on lv_a, lv_b components for the x or y matrix.

       Setting normalize force axes ends to -1 and 1.

       Raise ValueError if x and y does not differ.
       Raise ValueError if lv_a and lv_b are the same component.
    """
    if bool(x) == bool(y):
        raise ValueError('In plot.biplot() X, Y matrix flags must differ')
    if lv_a == lv_b:
        raise ValueError('Latent variables must be different!')

    scores(ax, lv_a, lv_b, x=x, y=y, normalize=normalize)
    loadings(ax, lv_a, lv_b, x=x, y=y)

    ax.set_title('Biplot for {}'.format('X' if x else 'Y'))
    ax.set_xlabel('LV{}'.format(lv_a))
    ax.set_ylabel('LV{}'.format(lv_b))


def scores(ax, lv_a, lv_b, x=False, y=False, normalize=False):
    """Plot the scores on the lv_a, lv_b components for the x or y matrix.

       Setting normalize force axes ends to -1 and 1.
       Points of each category have a different color/shape.

       Raise ValueError if x and y does not differ.
       Raise ValueError if lv_a and lv_b are the same component.
    """
    if bool(x) == bool(y):
        raise ValueError('In plot.scores() X, Y matrix flags must differ')
    if lv_a == lv_b:
        raise ValueError('Latent variables must be different!')

    lv_a, lv_b = min(lv_a, lv_b), max(lv_a, lv_b)

    if lv_a > MODEL.nr_lv or lv_b > MODEL.nr_lv:
        raise ValueError('In plot.scores() at least one of the chosen latent '
                         'variable numbers ({} and {}) '.format(lv_a, lv_b)
                         'is out of bounds [1:{}]'.format(MODEL.nr_lv))

    scores_matrix = MODEL.T.copy() if x else MODEL.U.copy()

    scores_a, scores_b = scores_matrix[:, lv_a - 1], scores_matrix[:, lv_b - 1]
    if normalize:
        scores_a = scores_a / max(abs(scores_a))
        scores_b = scores_b / max(abs(scores_b))

    for n in range(scores_matrix.shape[0]):
        scatter_wrapper(ax, scores_a[n], scores_b[n],
                        TRAIN_SET.categorical_y[n])

    ax.set_title('Scores plot for {}'.format('X' if x else 'Y'))
    ax.set_xlabel('LV{}'.format(lv_a))
    ax.set_ylabel('LV{}'.format(lv_b))
    ax.axvline(0, linestyle='dashed', color='black')
    ax.axhline(0, linestyle='dashed', color='black')
    if normalize:
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
    else:
        ax.set_xlim(model.integer_bounds(MODEL.P, MODEL.T, lv_a - 1))
        ax.set_ylim(model.integer_bounds(MODEL.P, MODEL.T, lv_b - 1))

    handles, labels = ax.get_legend_handles_labels()
    by_label = collections.OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())


def loadings(ax, lv_a, lv_b, x=False, y=False):
    """Plot the loadings on the lv_a, lv_b components for the x or y matrix.

       Annotate every point with the corresponding variable name.

       Raise ValueError if x and y does not differ.
       Raise ValueError if lv_a and lv_b are the same component.
    """
    if bool(x) == bool(y):
        raise ValueError('In plot.loadings() X, Y matrix flags must differ')
    if lv_a == lv_b:
        raise ValueError('Latent variables must be different!')

    lv_a, lv_b = min(lv_a, lv_b), max(lv_a, lv_b)

    if lv_a > MODEL.nr_lv or lv_b > MODEL.nr_lv:
        raise ValueError('In plot.loadings() at least one of the chosen latent'
                         ' variable numbers ({} and {}) '.format(lv_a, lv_b)
                         'is out of bounds [1:{}]'.format(MODEL.nr_lv))

    loadings_matrix = MODEL.P.copy() if x else MODEL.Q.copy()

    for n in range(loadings_matrix.shape[0]):
        line_wrapper(ax, (0, loadings_matrix[n, lv_a - 1]),
                     (0, loadings_matrix[n, lv_b - 1]), linestyle='dashed')
        ax.annotate(TRAIN_SET.header[n + 1],
                    horizontalalignment='center',
                    textcoords='offset points',
                    verticalalignment='bottom',
                    xy=(loadings_matrix[n, lv_a - 1],
                        loadings_matrix[n, lv_b - 1]),
                    xycoords='data',
                    xytext=(0, 5))

    ax.set_title('Loadings plot for {}'.format('X' if x else 'Y'))
    ax.set_xlabel('LV{}'.format(lv_a))
    ax.set_ylabel('LV{}'.format(lv_b))
    ax.axvline(0, linestyle='dashed', color='black')
    ax.axhline(0, linestyle='dashed', color='black')


def calculated_y(ax):
    """Plot the difference between the real categories and the modeled ones."""
    ax.set_title('Calculated Y')
    ax.set_xlabel('Samples')
    ax.set_ylabel('Modeled Y')
    for j in range(MODEL.p):
        for i in range(MODEL.n):
            scatter_wrapper(ax, i, MODEL.Y_modeled[i, j],
                            TRAIN_SET.categorical_y[i])


def y_predicted_y_real(ax):
    """Plot the y predicted over the y measured."""
    if STATS is None or TEST_SET is None:
        IO.Log.debug('In plot.y_predicted_y_real() STATS or TEST_SET is None')
        raise TypeError('Please run prediction')

    ax.set_xlabel('Measured Y')
    ax.set_ylabel('Predicted Y')

    for j in range(MODEL.p):
        for i in range(MODEL.m):
            scatter_wrapper(ax, STATS.y_real[i, j], STATS.y_pred[i, j],
                            TEST_SET.categories[j])


def y_predicted(ax):
    """Plot the y predicted, with color representing the current y."""
    if STATS is None or TEST_SET is None:
        IO.Log.debug('In plot.y_predicted() STATS or TEST_SET is None')
        raise TypeError('Please run prediction')

    ax.set_xlabel('Sample')
    ax.set_ylabel('Y predicted')

    for j in range(MODEL.p):
        scatter_wrapper(ax, range(STATS.y_pred.shape[0]), STATS.y_pred[:, j],
                        TEST_SET.categories[j])
        # Add a vertical line at the beginning of each category
        ax.axvline(TEST_SET.categorical_y.index(TEST_SET.categories[j]),
                   linestyle='dashed', color='gray')

    ylim = ax.get_ylim()[1]
    ax.set_xlim(0, ax.get_xlim()[1])
    for j in range(MODEL.p):
        ax.text(TEST_SET.categorical_y.index(TEST_SET.categories[j]) + 1,
                ylim - .25, TEST_SET.categories[j])


def t_square_q(ax):
    """Plot the q statistic over the Hotelling's t^2 with confidence levels."""
    ax.set_title('T^2 - Q')
    ax.set_xlabel('Hotelling\'s T^2')
    ax.set_ylabel('Q residuals')

    t_square = MODEL.t_square
    q_res = MODEL.q_residuals_x
    for i in range(MODEL.n):
        scatter_wrapper(ax, t_square[i], q_res[i], TRAIN_SET.categorical_y[i])

    t_square_confidence_level = scipy_stats.norm.interval(
        0.95, np.mean(t_square), np.std(t_square))[1]
    q_confidence_level = scipy_stats.norm.interval(
        0.95, np.mean(q_res), np.std(q_res))[1]

    ax.axvline(t_square_confidence_level, linestyle='dashed', color='black')
    ax.axhline(q_confidence_level, linestyle='dashed', color='black')

    handles, labels = ax.get_legend_handles_labels()
    by_label = collections.OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())


def y_residuals_leverage(ax):
    """Plot Y residuals over the leverage."""
    for j in range(MODEL.p):
        for i in range(MODEL.n):
            scatter_wrapper(ax, MODEL.leverage[i], MODEL.E_y[i, j],
                            TRAIN_SET.categorical_y[i])

    ax.set_xlabel('Leverage')
    ax.set_ylabel('Y residuals')


def leverage(ax):

    """Plot leverage over the sample."""
    for i in range(MODEL.n):
        scatter_wrapper(ax, i, MODEL.leverage[i],
                        TRAIN_SET.categorical_y[i])

    ax.set_xlabel('Samples')
    ax.set_ylabel('Leverage')


def q_over_leverage(ax):
    """Plot the Q statistics over the leverage."""
    q_res = MODEL.q_residuals_x
    for i in range(MODEL.n):
        scatter_wrapper(ax, q_res[i], MODEL.leverage[i],
                        TRAIN_SET.categorical_y[i])

    q_confidence_level = scipy_stats.norm.interval(
        0.95, np.mean(q_res), np.std(q_res))[1]

    ax.axvline(q_confidence_level, linestyle='dashed', color='black')

    ax.set_xlabel('Q residuals')
    ax.set_ylabel('Leverage')


def regression_coefficients(ax):
    """Plot the regression coefficients."""
    for i in range(MODEL.b.shape[0]):
        scatter_wrapper(ax, i + 1, MODEL.b[i], TRAIN_SET.categorical_y[i])

    ax.set_title('Inner relation (variable b)')
    ax.set_xlabel('Latent variable number')
    ax.set_ylabel('Inner relation variable')


def weights(ax, lv_a, lv_b):
    """Plot the weights regarding the given principal components.

       Annotate every point with the corresponding variable name.

       Raise ValueError if lv_a and lv_b are the same component.
    """
    if lv_a == lv_b:
        raise ValueError('Latent variables must be different!')

    lv_a, lv_b = min(lv_a, lv_b), max(lv_a, lv_b)

    if lv_a > MODEL.nr_lv or lv_b > MODEL.nr_lv:
        raise ValueError('In plot.weights() at least one of the chosen latent '
                         'variable numbers ({} and {}) '.format(lv_a, lv_b)
                         'is out of bounds [1:{}]'.format(MODEL.nr_lv))

    scatter_wrapper(ax, MODEL.W[:, lv_a - 1], MODEL.W[:, lv_b - 1])

    for n in range(MODEL.W.shape[0]):
        ax.annotate(TRAIN_SET.header[n + 1],
                    horizontalalignment='center',
                    textcoords='offset points',
                    verticalalignment='bottom',
                    xy=(MODEL.W[n, lv_a - 1], MODEL.W[n, lv_b - 1]),
                    xycoords='data',
                    xytext=(0, 5))

    ax.set_title('Weights plot')
    ax.set_xlabel('LV{}'.format(lv_a))
    ax.set_ylabel('LV{}'.format(lv_b))
    ax.axvline(0, linestyle='dashed', color='black')
    ax.axhline(0, linestyle='dashed', color='black')


def weights_line(ax, lv):
    """Plot all the weights used by the model."""
    if lv > MODEL.nr_lv:
        raise ValueError('In plot.weights_line() the chosen latent variable '
                         'number ({}) '.format(lv)
                         'is out of bounds [1:{}]'.format(MODEL.nr_lv))
    line_wrapper(ax, range(MODEL.W.shape[0]), MODEL.W[:, lv - 1])

    ax.set_title('Weights line plot')
    ax.set_xlabel('Samples')
    ax.set_ylabel('LV{}'.format(lv))
    ax.axvline(0, linestyle='dashed', color='black')
    ax.axhline(0, linestyle='dashed', color='black')


def data(ax):
    """Plot the dataset distinguishing with colors the categories."""
    for i in range(MODEL.n):
        line_wrapper(ax, range(MODEL.m), TRAIN_SET.x[i],
                     TRAIN_SET.categorical_y[i])

    ax.set_title('Data by category')
    ax.set_xlabel('Samples')
    ax.set_ylabel('Values')


def rmsecv_lv(ax, stats):
    """Plot the RMSECV value for the current cv."""
    if stats is None:
        IO.Log.debug('In plot.rmsecv_lv() stats is None')
        raise TypeError('Please run cross-validation')

    r = []
    for j in range(len(stats[0])):               # lv
        rss = np.zeros((stats[0][j].p))
        for i in range(len(stats)):              # split
            for k, y in enumerate(stats[i][j].rss):     # y
                rss[k] += y/len(stats[0])
        r.append(rss)

    r = np.asarray(r)
    r = np.sqrt(r.T / MODEL.n)

    for i in range(len(r)):
        line_wrapper(ax, range(1, len(r.T) + 1), r[i])

    ax.set_title('RMSECV')
    ax.set_xlabel('LV')
    ax.set_ylabel('RMSECV')


if __name__ == '__main__':
    raise SystemExit('Please do not run that script, load it!')
