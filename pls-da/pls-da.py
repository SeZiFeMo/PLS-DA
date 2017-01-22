#!/usr/bin/env python
# coding: utf-8

import argparse
import collections
import logging
import math
import sys

if sys.version_info < (3,):
    major, minor = sys.version_info[0], sys.version_info[1]
    print('You are using the Python interpreter {}.{}.\n'
          'Please use at least Python version 3!'.format(major, minor))
    exit(1)

for lib in ('matplotlib.pyplot', 'numpy', 'yaml'):
    try:
        exec('import ' + lib)
    except ImportError:
        print('Could not import {} library, please install it!'.format(lib))
        exit(1)

if __name__ != '__main__':
    print('Please do not load that script, run it!')
    exit(1)


args = None

def parse_args():
    """Return command line parameters."""
    global args
    if args is None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter,
            description='Script to ...\n',
            epilog='REQUIREMENTS: Python 3 (>= 3.4)\n'
                   '              NumPy   https://pypi.python.org/pypi/numpy\n'
                   '              PyYAML  http://pyyaml.org/wiki/PyYAML\n')
        parser.add_argument('-i', '--input-file',
                            default='wine.csv',
                            dest='input_file',
                            help='File with comma saved value input dataset',
                            metavar='file',
                            type=str)
        args = parser.parse_args()
    return args
parse_args()


def setup_logging(level):
    """Initialize logging for all the libraries and the script."""
    level = getattr(logging, level.upper())
    logging.basicConfig(format='[%(levelname)s]\t%(message)s', level=level)
    for l in logging.Logger.manager.loggerDict.keys():
    logging.getLogger(__name__).setLevel(level)  # current script logging
setup_logging('debug')  # called here to ensure log function will work


def mat2str(data, show=False, h_bar='-', v_bar='|', join='+'):
    """Print or return an ascii table."""
    try:
        if isinstance(data, (numpy.ndarray, numpy.generic)) and data.ndim == 2:
            ret = join + h_bar + h_bar * 11 * len(data[0]) + join + '\n'
            for row in data:
                ret += v_bar + ' '
                for col in row:
                    ret += '{: < 10.3e} '.format(col)
                ret += v_bar + '\n'
            ret += join + h_bar + h_bar * 11 * len(data[0]) + join + '\n'
        elif (isinstance(data, (numpy.ndarray, numpy.generic))
              and data.ndim == 1) or isinstance(data, (list, tuple)):
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


def log(msg='', data=None, level='debug'):
    """Print log message if above threshold."""
    logger = getattr(logging.getLogger(__name__), level)
    if data is None:
        logger(msg)
    else:
        if (isinstance(data, (numpy.ndarray, numpy.generic))
            and data.ndim in (1, 2)) or \
           isinstance(data, (list, tuple)):
            data = mat2str(data)
        else:
            data = yaml.dump(data, default_flow_style=False)
        logger(msg.rstrip('\n') + '\n    ' + data.replace('\n', '\n    '))


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
        print('WARNING: principal components must be different!')
        return

    pc_x, pc_y = min(pc_x, pc_y), max(pc_x, pc_y)

    for n in range(model.scores.shape[0]):
        cat = model.categories[n]
        matplotlib.pyplot.scatter(x=model.scores[n, pc_x],
                                  y=model.scores[n, pc_y],
                                  edgecolors=properties_of(cat)['edge_color'],
                                  marker=properties_of(cat)['marker'],
                                  c=properties_of(cat)['face_color'],
                                  label=cat)

    ax = matplotlib.pyplot.gca()
    matplotlib.pyplot.title('Scores plot')
    matplotlib.pyplot.xlabel('PC{}'.format(pc_x + 1))
    matplotlib.pyplot.ylabel('PC{}'.format(pc_y + 1))
    matplotlib.pyplot.axvline(0, linestyle='dashed', color='black')
    matplotlib.pyplot.axhline(0, linestyle='dashed', color='black')
    ax.set_xlim(model.get_loadings_scores_xy_limits(pc_x, pc_y)['x'])
    ax.set_ylim(model.get_loadings_scores_xy_limits(pc_x, pc_y)['y'])

    handles, labels = ax.get_legend_handles_labels()
    by_label = collections.OrderedDict(zip(labels, handles))
    matplotlib.pyplot.legend(by_label.values(), by_label.keys())


def loadings_plot(model, pc_x, pc_y):
    """Plot the loadings."""
    if pc_x == pc_y:
        print('Principal components must be different')
        return

    pc_x, pc_y = min(pc_x, pc_y), max(pc_x, pc_y)
    matplotlib.pyplot.scatter(x=model.loadings[:, pc_x],
                              y=model.loadings[:, pc_y])

    ax = matplotlib.pyplot.gca()
    for n in range(model.loadings.shape[0]):
        ax.annotate(model.keys[n + 1],
                    xy=(model.loadings[n, pc_x], model.loadings[n, pc_y]),
                    xycoords='data',
                    xytext=(0, 5),
                    textcoords='offset points',
                    horizontalalignment='center',
                    verticalalignment='bottom')

    matplotlib.pyplot.title('Loadings plot')
    matplotlib.pyplot.xlabel('PC{}'.format(pc_x + 1))
    matplotlib.pyplot.ylabel('PC{}'.format(pc_y + 1))
    matplotlib.pyplot.axvline(0, linestyle='dashed', color='black')
    matplotlib.pyplot.axhline(0, linestyle='dashed', color='black')


class PLS_DA(object):

    def __init__(self, csv_file=None):
        """Constructor method"""
        if csv_file is None:
            csv_file = parse_args().input_file

        self.axis = 0

        self._dataset_original = None
        self.parse_csv(csv_file)
        self.dataset = numpy.copy(self._dataset_original)
        self.n_rows, self.n_cols = self.dataset.shape

        self.mean = None
        self.sigma = None
        self.centered = False
        self.normalized = False
        self.autoscaled = False

    def parse_csv(self, filename):
        """self.keys              = first row of labels
           self.categories        = first column of labels about wine type
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
        self._dataset_original = numpy.array(self._dataset_original)
        log('[PLS_DA::parse_csv] self._dataset_original',
            self._dataset_original)

    def preprocess_mean(self, use_original=False):
        """Substitute self.dataset with its centered version."""
        dataset = self._dataset_original if use_original else self.dataset
        self.mean = dataset.mean(axis=self.axis)
        self.dataset = dataset - self.mean
        self.dummy_Y = self.dummy_Y - self.dummy_Y.mean(axis=self.axis)
        log('[PLS_DA::preprocess_mean] Centered matrix', self.dataset)
        self.centered = True

    def preprocess_normalize(self, use_original=False):
        """Substitute self.dataset with its normalized version."""
        dataset = self._dataset_original if use_original else self.dataset
        self.sigma = dataset.std(axis=self.axis)
        self.dataset = self.dataset / self.sigma
        self.dummy_Y = self.dummy_Y - self.dummy_Y.std(axis=self.axis)
        log('[PLS_DA::preprocess_normalize] Normalized matrix', self.dataset)
        self.normalized = True

    def preprocess_autoscale(self, use_original=False):
        """Substitute self.dataset with its autoscaled version."""
        dataset = self._dataset_original if use_original else self.dataset
        self.preprocess_mean(use_original)  # it initializes self.dataset
        self.preprocess_normalize(use_original)
        log('[PLS_DA::preprocess_autoscale] Autoscaled matrix', self.dataset)
        self.autoscaled = True

    def get_dummy_variables(self):

        categories = set(self.categories)

        li = []
        for cat in categories:
            li.append([1.0 if c == cat else 0.0 for c in self.categories])

        self.dummy_Y = numpy.array(li)
        self.dummy_Y = self.dummy_Y.T

        log('[PLS_DA::get_dummy_variables] dummy Y variables', self.dummy_Y)

    def nipals_method(self, nr_lv, tol=1e-6, max_iter=10000):
        """Find the Principal Components with the NIPALS algorithm."""
        # Start with maximal residual (matrix X)
        E_x = self.dataset.copy()  # Residuals of PC0
        E_y = self.dummy_Y.copy()
        n, m = self.dataset.shape
#       print(numpy.ones((n,1)).shape)
#       E_x = numpy.concatenate((numpy.ones((n,1)), E_x), axis=1)

        if self.mean is None:
            log('WARNING: no pretreatment specified and NIPALS selected',
                level='warning')

        n, m = E_x.shape
        n, p = E_y.shape
        T = numpy.empty((n, nr_lv))
        P = numpy.empty((m, nr_lv))
        W = numpy.empty((m, nr_lv))
        U = numpy.empty((n, nr_lv))
        Q = numpy.empty((p, nr_lv))
        b = numpy.empty((nr_lv))
        s_list = []

        # Loop for each possible PC
        for i in range(nr_lv):
            # Initialize u as a column of E_x with maximum variance
            max_var_index = numpy.argmax(numpy.sum(numpy.power(E_y, 2),
                                                   axis=0))
            u = E_y[:, max_var_index].copy()

            for it in range(max_iter + 2):
                # Evaluate w as projection of u
                w = numpy.dot(E_x.T, u) / numpy.dot(u, u)
                # Normalize w
                w /= numpy.linalg.norm(w)
                # Evaluate t as projection of w
                t = numpy.dot(E_x, w)  # / numpy.dot(p, p)

                # Y part
                # Evaluate q as projection of t in Y
                q = numpy.dot(E_y.T, t) / numpy.dot(t, t)
                # Normalize q
                q /= numpy.linalg.norm(q)
                # Evaluate u_star as projection of q in Y
                u_star = numpy.dot(E_y, q)  # / numpy.dot(p, p)

                diff = u_star - u
                delta_u = numpy.linalg.norm(diff)
                u = u_star
                print('NIPALS iteration: {}\n'
                      '      difference: {:.5e}'.format(it, delta_u))
                # if it > 1 and delta_u < tol * numpy.linalg.norm(u_star):
                if it > 1 and delta_u < tol:
                    break
            else:
                print('Warning: reached max '
                      'iteration number ({})'.format(max_iter))
                print('NIPALS iteration: {}\n'
                      '      difference: {:.5e}'.format(it, delta_u))
            # Save the evaluated values
            s_list.append(numpy.linalg.norm(t))
            T[:, i] = t
            P[:, i] = numpy.dot(E_x.T, t) / numpy.dot(t, t)
            # regression coefficient for the inner relation
            b[i] = numpy.dot(u.T, t) / numpy.dot(t, t)
            W[:, i] = w
            U[:, i] = u
            Q[:, i] = numpy.dot(E_y.T, u) / numpy.dot(u, u)
            E_x -= numpy.dot(numpy.row_stack(t), numpy.column_stack(P[:, i]))
            E_y -= b[i] * numpy.dot(numpy.row_stack(t),
                                    numpy.column_stack(q.T))

        self.scores = T
        self.loadings = P
        self.eigenvalues = numpy.power(numpy.array(s_list), 2) \
            / (self.n_rows - 1)

        log('NIPALS loadings shape', self.loadings.shape, level='info')
        log('NIPALS scores shape', self.scores.shape, level='info')
        log('NIPALS eigenvalues', self.eigenvalues, level='info')

        self.eig = False
        self.nipals = True
        self.svd = False

    def get_loadings_scores_xy_limits(self, pc_x, pc_y):
        """Return dict of x and y limits: {'x': (min, max), 'y': (min, max)}"""
        x_val = numpy.concatenate((self.loadings[:, pc_x],
                                   self.scores[:, pc_x]))
        y_val = numpy.concatenate((self.loadings[:, pc_y],
                                   self.scores[:, pc_y]))
        min_x = math.floor(numpy.min(x_val))
        max_x = math.ceil(numpy.max(x_val))
        min_y = math.floor(numpy.min(y_val))
        max_y = math.ceil(numpy.max(y_val))
        return {'x': (min_x, max_x), 'y': (min_y, max_y)}


pls = PLS_DA()
pls.get_dummy_variables()
pls.preprocess_autoscale()
pls.nipals_method(nr_lv=4)
matplotlib.pyplot.subplot(2, 1, 1)
scores_plot(pls, 0, 1)
matplotlib.pyplot.subplot(2, 1, 2)
loadings_plot(pls, 0, 1)
matplotlib.pyplot.show()
