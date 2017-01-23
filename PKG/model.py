#!/usr/bin/env python
# coding: utf-8

import math
import numpy as np
import PKG.io
import PKG.plot
import PKG.utility


if __name__ == '__main__':
    PKG.io.Log.warning('Please do not run that script, load it!')
    exit(1)


class PLS_DA(object):

    def __init__(self, csv_file=None):
        """Constructor method"""
        if csv_file is None:
            csv_file = PKG.utility.CLI.args().input_file

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
            PKG.io.Log.error('File \'{}\' not existent, '
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
                PKG.io.Log.warning('Unexpected wine '
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
        PKG.io.Log.debug('[PLS_DA::parse_csv] self._dataset_original',
                         self._dataset_original)

    def preprocess_mean(self, use_original=False):
        """Substitute self.dataset with its centered version."""
        dataset = self._dataset_original if use_original else self.dataset
        self.mean = dataset.mean(axis=self.axis)
        self.dataset = dataset - self.mean
        self.dummy_Y = self.dummy_Y - self.dummy_Y.mean(axis=self.axis)
        PKG.io.Log.debug('[PLS_DA::preprocess_mean] Centered matrix',
                         self.dataset)
        self.centered = True

    def preprocess_normalize(self, use_original=False):
        """Substitute self.dataset with its normalized version."""
        dataset = self._dataset_original if use_original else self.dataset
        self.sigma = dataset.std(axis=self.axis)
        self.dataset = self.dataset / self.sigma
        self.dummy_Y = self.dummy_Y - self.dummy_Y.std(axis=self.axis)
        PKG.io.Log.debug('[PLS_DA::preprocess_normalize] Normalized matrix',
                         self.dataset)
        self.normalized = True

    def preprocess_autoscale(self, use_original=False):
        """Substitute self.dataset with its autoscaled version."""
        dataset = self._dataset_original if use_original else self.dataset
        self.preprocess_mean(use_original)  # it initializes self.dataset
        self.preprocess_normalize(use_original)
        PKG.io.Log.debug('[PLS_DA::preprocess_autoscale] Autoscaled matrix',
                         self.dataset)
        self.autoscaled = True

    def get_dummy_variables(self):

        categories = set(self.categories)

        li = []
        for cat in categories:
            li.append([1.0 if c == cat else 0.0 for c in self.categories])

        self.dummy_Y = np.array(li)
        self.dummy_Y = self.dummy_Y.T

        PKG.io.Log.debug('[PLS_DA::get_dummy_variables] dummy Y variables',
                         self.dummy_Y)

    def nipals_method(self, nr_lv, tol=1e-6, max_iter=10000):
        """Find the Principal Components with the NIPALS algorithm."""
        # Start with maximal residual (matrix X)
        E_x = self.dataset.copy()  # Residuals of PC0
        E_y = self.dummy_Y.copy()
        n, m = self.dataset.shape
#       PKG.io.Log.debug('np.ones((n,1)).shape', np.ones((n,1)).shape)
#       E_x = np.concatenate((np.ones((n,1)), E_x), axis=1)

        if self.mean is None:
            PKG.io.Log.warning(
                'No pretreatment specified and NIPALS selected')

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
            max_var_index = np.argmax(np.sum(np.power(E_y, 2),
                                             axis=0))
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
                PKG.io.Log.debug('NIPALS iteration: {}\n'
                                 '       difference: '
                                 '{:.5e}'.format(it, delta_u))
                # if it > 1 and delta_u < tol * np.linalg.norm(u_star):
                if it > 1 and delta_u < tol:
                    break
            else:
                PKG.io.Log.warning('Reached max '
                                   'iteration number ({})'.format(max_iter))
                PKG.io.Log.warning('NIPALS iteration: {}\n'
                                   '       difference: '
                                   '{:.5e}'.format(it, delta_u))
            # Save the evaluated values
            s_list.append(np.linalg.norm(t))
            T[:, i] = t
            P[:, i] = np.dot(E_x.T, t) / np.dot(t, t)
            # regression coefficient for the inner relation
            b[i] = np.dot(u.T, t) / np.dot(t, t)
            W[:, i] = w
            U[:, i] = u
            Q[:, i] = np.dot(E_y.T, u) / np.dot(u, u)
            E_x -= np.dot(np.row_stack(t), np.column_stack(P[:, i]))
            E_y -= b[i] * np.dot(np.row_stack(t),
                                 np.column_stack(q.T))

        self.scores = T
        self.loadings = P
        self.eigenvalues = np.power(np.array(s_list), 2) \
            / (self.n_rows - 1)

        PKG.io.Log.info('NIPALS loadings shape', self.loadings.shape)
        PKG.io.Log.info('NIPALS scores shape', self.scores.shape)
        PKG.io.Log.info('NIPALS eigenvalues', self.eigenvalues)

        self.eig = False
        self.nipals = True
        self.svd = False

    def get_loadings_scores_xy_limits(self, pc_x, pc_y):
        """Return dict of x and y limits: {'x': (min, max), 'y': (min, max)}"""
        x_val = np.concatenate((self.loadings[:, pc_x],
                                self.scores[:, pc_x]))
        y_val = np.concatenate((self.loadings[:, pc_y],
                                self.scores[:, pc_y]))
        min_x = math.floor(np.min(x_val))
        max_x = math.ceil(np.max(x_val))
        min_y = math.floor(np.min(y_val))
        max_y = math.ceil(np.max(y_val))
        return {'x': (min_x, max_x), 'y': (min_y, max_y)}


