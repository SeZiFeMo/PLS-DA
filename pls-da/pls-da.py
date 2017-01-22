#!/usr/bin/env python3
# coding: utf-8

from collections import OrderedDict
from math import floor, ceil, sqrt
import matplotlib.pyplot as plt
import numpy as np

if __name__ != '__main__':
    print('That script should be run, not loaded!')
    exit(1)

log_level = {'DEBUG': 0,
             'INFO': 1,
             'WARNING': 2,
             'ERROR': 3,
             'CRITICAL': 4}['WARNING']

dataset = None
namespace = None

def log(record, level=0):
    """Print log message if above threshold."""
    global log_level
    if (level >= log_level):
        print('{}\n'.format(str(record).strip('\n')))

def mat2str(data, show=False, h_bar='-', v_bar='|', join='+'):
    """Print or return an ascii table."""
    try:
        if isinstance(data, (np.ndarray, np.generic)) and data.ndim == 2:
            ret = join + h_bar + h_bar * 11 * len(data[0]) + join + '\n'
            for row in data:
                ret += v_bar + ' '
                for col in row:
                    ret += '{: < 10.3e} '.format(col)
                ret += v_bar + '\n'
            ret += join + h_bar + h_bar * 11 * len(data[0]) + join + '\n'
        elif (isinstance(data, (np.ndarray, np.generic)) and data.ndim == 1) \
                or isinstance(data, (list, tuple)):
            ret = join + h_bar + h_bar * 11 * len(data) + join + '\n'
            ret += v_bar + ' '
            for cell in data:
                ret += '{: < 10.3e} '.format(cell)
            ret += v_bar + '\n'
            ret += join + h_bar + h_bar * 11 * len(data) + join + '\n'
        else:
            raise Exception('Not supported data type ({}) '
                            'in mat2str()'.format(type(data)))
    except Exception as e:
        print(e)
        ret = str(data)
    finally:
        if show:
            print(ret)
        else:
            return ret

def seq2str(seq):
    """Return string representation of a sequence."""
    return str(seq).translate({ord(d): '' for d in '{}\'[]"()'})

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

class PLS_DA(object):
    def __init__(self, csv_file):

        self.axis = 0

        self._dataset_original = None
        self.parse_csv(csv_file)
        self.dataset = np.copy(self._dataset_original)
        self.n_rows, self.n_cols = self.dataset.shape

        self.mean = None
        self.sigma = None
        self.centered = False
        self.normalized = False
        self.autoscaled = False

    def parse_csv(self, filename):
        """self.keys             = first row of labels
           self.categories       = first column of labels about wine type
           self._dataset_original = the rest of the matrix
        """

        self._dataset_original = list()
        try:
            with open(filename, 'r', encoding='iso8859') as f:
                self.keys = f.readline().strip('\n').split(';')
                for line in f.readlines():
                    line = line.strip('\n').split(';')
                    self._dataset_original.append(list(line))
        except IOError:
            print('ERROR: file \'{}\' not existent, '
                  'not readable or corrupted'.format(filename))
            exit(1)

        for d in self._dataset_original:
            if d[0].startswith('B'):
                d[0] = 'B'
            elif d[0].startswith('E'):
                d[0] = 'E'
            elif d[0].startswith('G'):
                d[0] = 'G'
            else:
                print('WARNING: unexpected wine '
                      'category ({})'.format(d['Category']))

        # Delete category column from self._dataset_original
        # and save it for future uses
        self.categories = [x[0] for x in self._dataset_original]
        self._dataset_original = [x[1:] for x in self._dataset_original]

        # Replace commas with dots
        self._dataset_original = [[float(elem.replace(',', '.'))
                                   for elem in row]
                                  for row in self._dataset_original]
        self._dataset_original = np.array(self._dataset_original)
        log('[PLS_DA::parse_csv] '
            '_dataset_original: \n{}'.format(mat2str(self._dataset_original)),
            0)

    def preprocess_mean(self, use_original=False):
        """Substitute self.dataset with its centered version."""
        dataset = self._dataset_original if use_original else self.dataset
        self.mean = dataset.mean(axis=self.axis)
        self.dataset = dataset - self.mean
        self.dummy_Y = self.dummy_Y - self.dummy_Y.mean(axis=self.axis)
        log('[PLS_DA::preprocess_mean] '
            'Centered matrix: \n{}'.format(mat2str(self.dataset)), 0)
        self.centered = True

    def preprocess_normalize(self, use_original=False):
        """Substitute self.dataset with its normalized version."""
        dataset = self._dataset_original if use_original else self.dataset
        self.sigma = dataset.std(axis=self.axis)
        self.dataset = self.dataset / self.sigma
        self.dummy_Y = self.dummy_Y - self.dummy_Y.std(axis=self.axis)
        log('[PLS_DA::preprocess_normalize] '
            'Normalized matrix: \n{}'.format(mat2str(self.dataset)), 0)
        self.normalized = True


    def preprocess_autoscale(self, use_original=False):
        """Substitute self.dataset with its autoscaled version."""
        dataset = self._dataset_original if use_original else self.dataset
        self.preprocess_mean(use_original)  # it initializes self.dataset
        self.preprocess_normalize(use_original)
        log('[PLS_DA::preprocess_autoscale] '
            'Autoscaled matrix: \n{}'.format(mat2str(self.dataset)), 0)
        self.autoscaled = True

    def get_dummy_variables(self):

        categories = set(self.categories)

        li = []
        for cat in categories:
            li.append([1.0 if c == cat else 0.0 for c in self.categories])

        self.dummy_Y = np.array(li)
        self.dummy_Y = self.dummy_Y.T

        log('[PLS_DA::get_dummy_variables] '
            'dummy Y variables: \n{}'.format(mat2str(self.dummy_Y)), 0)

    def nipals_method(self, nr_lv, tol=1e-6, max_iter=10000):
        """Find the Principal Components with the NIPALS algorithm."""
        # Start with maximal residual (matrix X)
        E_x = self.dataset.copy()  # Residuals of PC0
        E_y = self.dummy_Y.copy()
        n,m = self.dataset.shape
#        print(np.ones((n,1)).shape)
#        E_x = np.concatenate((np.ones((n,1)), E_x), axis=1)

        if self.mean is None:
            log('WARNING: no pretreatment specified and NIPALS selected', 2)

        n, m = E_x.shape
        n, p = E_y.shape
        T = np.empty((n, nr_lv))
        P = np.empty((m, nr_lv))
        W = np.empty((m, nr_lv))
        U = np.empty((n, nr_lv))
        Q = np.empty((p, nr_lv))
        b = np.empty((nr_lv))
        s_list = []

        # Loop for each possible PC
        for i in range(nr_lv):
            # Initialize u as a column of E_x with maximum variance
            max_var_index = np.argmax(np.sum(np.power(E_y, 2), axis=0))
            u = E_y[:, max_var_index].copy()

            for it in range(max_iter + 2):
                # Evaluate w as projection of u
                w = np.dot(E_x.T, u) / np.dot(u, u)
                # Normalize w
                w /= np.linalg.norm(w)
                # Evaluate t as projection of w
                t = np.dot(E_x, w)  # / np.dot(p, p)

                # Y part
                # Evaluate q as projection of t in Y
                q = np.dot(E_y.T, t) / np.dot(t, t)
                # Normalize q
                q /= np.linalg.norm(q)
                # Evaluate u_star as projection of q in Y
                u_star = np.dot(E_y, q)  # / np.dot(p, p)

                diff = u_star - u
                delta_u = np.linalg.norm(diff)
                u = u_star
                print('NIPALS iteration: {}\n'
                    '      difference: {:.5e}'.format(it, delta_u))
                #if it > 1 and delta_u < tol * np.linalg.norm(u_star):
                if it > 1 and delta_u < tol:
                    break
            else:
                print('Warning: reached max '
                      'iteration number ({})'.format(max_iter))
                print('NIPALS iteration: {}\n'
                    '      difference: {:.5e}'.format(it, delta_u))
            # Save the evaluated values
            s_list.append(np.linalg.norm(t))
            T[:, i] = t
            P[:, i] = np.dot(E_x.T, t) / np.dot(t, t)
            # regression coefficient for the inner relation
            b[i] = np.dot(u.T, t)/ np.dot(t, t)
            W[:, i] = w
            U[:, i] = u
            Q[:, i] = np.dot(E_y.T, u) / np.dot(u, u)
            E_x -= np.dot(np.row_stack(t), np.column_stack(P[:, i]))
            E_y -= b[i] * np.dot(np.row_stack(t), np.column_stack(q.T))

        self.scores = T
        self.loadings = P
        self.eigenvalues = np.power(np.array(s_list), 2) / (self.n_rows - 1)

        log('NIPALS loadings shape: {}'.format(self.loadings.shape), 1)
        log('NIPALS scores shape: {}'.format(self.scores.shape), 1)
        log('NIPALS eigenvalues: \n{}'.format(mat2str(self.eigenvalues)), 1)

        self.eig = False
        self.nipals = True
        self.svd = False

    def get_loadings_scores_xy_limits(self, pc_x, pc_y):
        """Return dict of x and y limits: {'x': (min, max), 'y': (min, max)}"""
        x_val = np.concatenate((self.loadings[:, pc_x], self.scores[:, pc_x]))
        y_val = np.concatenate((self.loadings[:, pc_y], self.scores[:, pc_y]))
        min_x, max_x = floor(np.min(x_val)), ceil(np.max(x_val))
        min_y, max_y = floor(np.min(y_val)), ceil(np.max(y_val))
        return {'x': (min_x, max_x), 'y': (min_y, max_y)}

def scores_plot(model, pc_x, pc_y):
    """Plot the scores on the specified components."""
    if pc_x == pc_y:
        print('WARNING: principal components must be different!')
        return

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
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

def loadings_plot(model, pc_x, pc_y):
    """Plot the loadings."""
    if pc_x == pc_y:
        print('Principal components must be different')
        return

    pc_x, pc_y = min(pc_x, pc_y), max(pc_x, pc_y)
    plt.scatter(x=model.loadings[:, pc_x], y=model.loadings[:, pc_y])

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
# main
pls = PLS_DA("wine.csv")
pls.get_dummy_variables()
pls.preprocess_autoscale()
pls.nipals_method(nr_lv=4)
plt.subplot(2,1,1)
scores_plot(pls, 0, 1)
plt.subplot(2,1,2)
loadings_plot(pls, 0, 1)
plt.show()
