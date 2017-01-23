#!/usr/bin/env python
# coding: utf-8

import collections
import PKG.io
import matplotlib.pyplot as plt


if __name__ == '__main__':
    PKG.io.Log.warning('Please do not run that script, load it!')
    exit(1)


def properties_of(category):
    """Return a dictionary with keys: edge_color, face_color, marker."""
    circle, cross, diamond, triangle = 'o', 'x', 'D', '^'

    blue, dark_red, gold, green, red = '#1F77B4', '#A00000', '#FFD700', \
                                       '#2CA02C', '#D62728'
    if category == 'B':
        return {'edge_color': blue, 'face_color': blue, 'marker': circle}
    elif category == 'E':
        return {'edge_color': green, 'face_color': green, 'marker': cross}
    elif category == 'G':
        return {'edge_color': red, 'face_color': red, 'marker': triangle}
    elif category == 'N':
        return {'edge_color': gold, 'face_color': dark_red, 'marker': diamond}
    else:
        raise Exception('Unknown category ' + category)


def scores_plot(model, pc_x, pc_y):
    """Plot the scores on the specified components."""
    if pc_x == pc_y:
        PKG.io.Log.warning('Principal components must be different!')
        exit(1)

    pc_x, pc_y = min(pc_x, pc_y), max(pc_x, pc_y)

    for n in range(model.scores.shape[0]):
        cat = model.categories[n]
        plt.scatter(x=model.scores[n, pc_x],
                    y=model.scores[n, pc_y],
                    edgecolors=properties_of(cat)['edge_color'],
                    marker=properties_of(cat)['marker'],
                    c=properties_of(cat)['face_color'],
                    label=cat)

    ax = plt.gca()
    plt.title('Scores plot')
    plt.xlabel('PC{}'.format(pc_x + 1))
    plt.ylabel('PC{}'.format(pc_y + 1))
    plt.axvline(0, linestyle='dashed', color='black')
    plt.axhline(0, linestyle='dashed', color='black')
    ax.set_xlim(model.get_loadings_scores_xy_limits(pc_x, pc_y)['x'])
    ax.set_ylim(model.get_loadings_scores_xy_limits(pc_x, pc_y)['y'])

    handles, labels = ax.get_legend_handles_labels()
    by_label = collections.OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())


def loadings_plot(model, pc_x, pc_y):
    """Plot the loadings."""
    if pc_x == pc_y:
        PKG.io.Log.warning('Principal components must be different!')
        exit(1)

    pc_x, pc_y = min(pc_x, pc_y), max(pc_x, pc_y)
    plt.scatter(x=model.loadings[:, pc_x],
                y=model.loadings[:, pc_y])

    ax = plt.gca()
    for n in range(model.loadings.shape[0]):
        ax.annotate(model.keys[n + 1],
                    xy=(model.loadings[n, pc_x], model.loadings[n, pc_y]),
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
