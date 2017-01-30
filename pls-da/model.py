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
        """Constructor method.

           Raises Exception if file could not be parsed.
        """
        if csv_file is None:
            csv_file = utility.CLI.args().input_file

        self.axis = 0
        self.keys, body = IO.CSV.parse(csv_file)
        IO.Log.debug('[PLS_DA::__init__] Using {} as input'.format(csv_file))

        # Delete category column from body and save it for future uses
        self.categories = [row[0] for row in body]
        # Check all values of self.categories are admitted.
        for i, cell in enumerate(self.categories):
            for cat in self.allowed_categories:
                if cell.startswith(cat):
                    self.categories[i] = cat
                    break
            else:
                raise Exception('Unexpected category ({}) '
                                'in row number {}'.format(cell, i))

        # The other columns of body are the dataset (matrix)
        self._dataset_original = np.array([np.array(row[1:]) for row in body])
        IO.Log.debug('[PLS_DA::__init__] self._dataset_original',
                     self._dataset_original)

        self.dataset = np.copy(self._dataset_original)
        self.n, self.m = self.dataset.shape

        self.T = None
        self.P = None
        self.W = None
        self.U = None
        self.Q = None
        self.C = None
        self.B = None
        self.Y_modeled = None
        self.x_eigenvalues = None
        self.y_eigenvalues = None

        self.mean = None
        self.sigma = None
        self.centered = False
        self.normalized = False
        self.autoscaled = False

        # For each category create a vector
        self.dummy_Y = np.array([[1.0 if c == cat else 0.0
                                  for c in self.categories]
                                 for cat in set(self.categories)]).T
        IO.Log.debug('[PLS_DA::__init__] dummy Y variables', self.dummy_Y)
        self.p = self.dummy_Y.shape[1]

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
        self.dummy_Y = self.dummy_Y / self.dummy_Y.std(axis=self.axis)
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

    def nipals_method(self, nr_lv=None, tol=1e-6, max_iter=1e4):
        """Find the Principal Components with the NIPALS algorithm."""
        # Start with maximal residual (matrix X)
        E_x = self.dataset.copy()  # Residuals of PC0
        E_y = self.dummy_Y.copy()
        n, m = self.dataset.shape

        if nr_lv is None:
            nr_lv = min(n,m)


        if self.mean is None:
            IO.Log.warning('No pretreatment specified and NIPALS selected')

        n, m = E_x.shape
        n, p = E_y.shape
        self.T = np.empty((n, nr_lv))
        self.P = np.empty((m, nr_lv))
        self.W = np.empty((m, nr_lv))
        self.U = np.empty((n, nr_lv))
        self.Q = np.empty((p, nr_lv))
        self.C = np.empty((p, nr_lv))
        self.d = np.empty((nr_lv))
        s_list_x = []
        s_list_y = []

        # Loop for each possible PC
        for i in range(nr_lv):
            # Initialize u as a column of E_x with maximum variance
            max_var_index = np.argmax(np.sum(np.power(E_y, 2), axis=0))
            u = E_y[:, max_var_index].copy()

            for it in range(int(max_iter) + 2):
                # Evaluate w as projection of u in X and normalize it
                w = np.dot(E_x.T, u) / np.dot(u, u)
                w = w / np.linalg.norm(w)
                # Evaluate t as projection of w in X
                t = np.dot(E_x, w)

                # Y part
                # Evaluate c as projection of t in Y and normalize it
                c = np.dot(E_y.T, t) / np.dot(t, t)
                c = c / np.linalg.norm(c)
                # Evaluate u_star as projection of c in Y
                u_star = np.dot(E_y, c)

                diff = u_star - u
                delta_u = np.dot(diff, diff)
                if it > 1 and delta_u < tol:
                    break
                u = u_star
            else:
                IO.Log.warning('Reached max '
                               'iteration number ({})'.format(max_iter))
                IO.Log.warning('NIPALS iteration: {}\n'
                               '       difference: {:.5e}'.format(it, delta_u))
            # Save the evaluated values
            s_list_x.append(np.linalg.norm(t))
            s_list_y.append(np.linalg.norm(u))
            self.T[:, i] = t
            self.P[:, i] = np.dot(E_x.T, t) / np.dot(t, t)
            self.W[:, i] = w
            # regression coefficient for the inner relation
            self.d[i] = np.dot(u.T, t) / np.dot(t, t)
            self.C[:, i] = c
            self.U[:, i] = u
            self.Q[:, i] = np.dot(E_y.T, u) / np.dot(u, u)
            E_x = E_x - np.dot(np.row_stack(t), np.column_stack(self.P[:, i]))
            E_y = E_y - self.d[i] * np.dot(np.row_stack(t), np.column_stack(c.T))

        self.x_eigenvalues = np.power(np.array(s_list_x), 2) / (self.n - 1)
        self.y_eigenvalues = np.power(np.array(s_list_y), 2) / (self.n - 1)
        self.y_eigenvalues = self.y_eigenvalues[:min(n,p)]

        # Compute regression parameters B
        # tmp = (P'W)^{-1}
        tmp = np.linalg.inv(self.P.T.dot(self.W))
        self.B = self.W.dot(tmp).dot(self.C.T)

        IO.Log.info('NIPALS loadings shape', self.P.shape)
        IO.Log.info('NIPALS scores shape', self.T.shape)
        IO.Log.info('NIPALS x_eigenvalues', self.x_eigenvalues)

    def get_modeled_y(self):
        self.Y_modeled = self.dataset.dot(self.B)
        IO.Log.debug('Modeled Y prior to the discriminant classification',
                    self.Y_modeled)

    def get_loadings_scores_xy_limits(self, pc_x, pc_y):
        """Return dict of x and y limits: {'x': (min, max), 'y': (min, max)}"""
        x_val = np.concatenate((self.P[:, pc_x], self.T[:, pc_x]))
        y_val = np.concatenate((self.P[:, pc_y], self.T[:, pc_y]))
        min_x, max_x = math.floor(np.min(x_val)), math.ceil(np.max(x_val))
        min_y, max_y = math.floor(np.min(y_val)), math.ceil(np.max(y_val))
        return {'x': (min_x, max_x), 'y': (min_y, max_y)}

    def get_explained_variance(self, matrix='x'):
        """Return the explained variance (computed over self.eigenvalues)."""
        if matrix == 'x':
            eigen = self.x_eigenvalues
        elif matrix == 'y':
            eigen = self.y_eigenvalues
        else:
            IO.Log.error('[PCA::get_explained_variance] '
                         'Accepted values for matrix are x and y')

            return

        IO.Log.info('[PCA::get_explained_variance] '
                    'Eigenvalues for {}: \n{}'.format(matrix, eigen))
        return 100 * eigen / np.sum(eigen)
