#!/usr/bin/env python
# coding: utf-8

import math
import numpy as np
import IO
import utility


if __name__ == '__main__':
    IO.Log.warning('Please do not run that script, load it!')
    exit(1)


class PLS_DA(object):

    allowed_categories = ('B', 'E', 'G', 'N', 'NA', 'SA', 'U', 'WL')

    def __init__(self, csv_file=None):
        """Constructor method"""
        if csv_file is None:
            csv_file = utility.CLI.args().input_file

        self.axis = 0
        self.keys, body = IO.CSV.parse(csv_file)
        IO.Log.debug('[PLS_DA::__init__] Using {} as input.'.format(csv_file))

        # Delete category column from body and save it for future uses
        self.categories = [row[0] for row in body]
        # Check all values of self.categories are admitted.
        for i, cell in enumerate(self.categories):
            for cat in ('NA', 'SA', 'U', 'WL', 'B', 'E', 'G'):
                if cell.startswith(cat):
                    self.categories[i] = cat
                    break
            else:
                IO.Log.warning('Unexpected category ({})'.format(cell))

        # The other columns of body are the dataset (matrix)
        self._dataset_original = np.array([np.array(row[1:]) for row in body])
        IO.Log.debug('[PLS_DA::__init__] self._dataset_original',
                     self._dataset_original)

        self.dataset = np.copy(self._dataset_original)
        self.n_rows, self.n_cols = self.dataset.shape

        self.mean = None
        self.sigma = None
        self.centered = False
        self.normalized = False
        self.autoscaled = False

    def preprocess_mean(self, use_original=False):
        """Substitute self.dataset with its centered version."""
        dataset = self._dataset_original if use_original else self.dataset
        self.mean = dataset.mean(axis=self.axis)
        self.dataset = dataset - self.mean
        self.dummy_Y = self.dummy_Y - self.dummy_Y.mean(axis=self.axis)
        IO.Log.debug('[PLS_DA::preprocess_mean] Centered matrix',
                     self.dataset)
        self.centered = True

    def preprocess_normalize(self, use_original=False):
        """Substitute self.dataset with its normalized version."""
        dataset = self._dataset_original if use_original else self.dataset
        self.sigma = dataset.std(axis=self.axis)
        self.dataset = self.dataset / self.sigma
        self.dummy_Y = self.dummy_Y - self.dummy_Y.std(axis=self.axis)
        IO.Log.debug('[PLS_DA::preprocess_normalize] Normalized matrix',
                     self.dataset)
        self.normalized = True

    def preprocess_autoscale(self, use_original=False):
        """Substitute self.dataset with its autoscaled version."""
        self.preprocess_mean(use_original)  # it initializes self.dataset
        self.preprocess_normalize(use_original)
        IO.Log.debug('[PLS_DA::preprocess_autoscale] Autoscaled matrix',
                     self.dataset)
        self.autoscaled = True

    def get_dummy_variables(self):

        categories = set(self.categories)

        li = []
        for cat in categories:
            li.append([1.0 if c == cat else 0.0 for c in self.categories])

        self.dummy_Y = np.array(li)
        self.dummy_Y = self.dummy_Y.T

        IO.Log.debug('[PLS_DA::get_dummy_variables] dummy Y variables',
                     self.dummy_Y)

    def nipals_method(self, nr_lv, tol=1e-6, max_iter=10000):
        """Find the Principal Components with the NIPALS algorithm."""
        # Start with maximal residual (matrix X)
        E_x = self.dataset.copy()  # Residuals of PC0
        E_y = self.dummy_Y.copy()
        n, m = self.dataset.shape
#       IO.Log.debug('np.ones((n,1)).shape', np.ones((n,1)).shape)
#       E_x = np.concatenate((np.ones((n,1)), E_x), axis=1)

        if self.mean is None:
            IO.Log.warning('No pretreatment specified and NIPALS selected')

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
                IO.Log.debug('NIPALS iteration: {}\n'
                             '       difference: '
                             '{:.5e}'.format(it, delta_u))
                # if it > 1 and delta_u < tol * np.linalg.norm(u_star):
                if it > 1 and delta_u < tol:
                    break
            else:
                IO.Log.warning('Reached max '
                               'iteration number ({})'.format(max_iter))
                IO.Log.warning('NIPALS iteration: {}\n'
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

        IO.Log.info('NIPALS loadings shape', self.loadings.shape)
        IO.Log.info('NIPALS scores shape', self.scores.shape)
        IO.Log.info('NIPALS eigenvalues', self.eigenvalues)

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
