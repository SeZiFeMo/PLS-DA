#!/usr/bin/env python3
# coding: utf-8

import math
import numpy as np
import IO
import utility

CATEGORIES = ('NA', 'SA', 'U', 'WL')


class Preprocessing(object):
    """Class to preprocess csv input data."""

    def __init__(self, input_file=None):
        """Load and parse csv input file.

           self.header      list of samples' properties (text)
           self.categories  list of samples' labels (text)
           self.dataset     list of samples' values (float)
           self.dummy_y     list of samples' labels (1 or 0)

           self.axis        axis to compute std and mean
        """
        if input_file is None:
            input_file = utility.CLI.args().input_file
        self.header, body = IO.CSV.parse(input_file)
        IO.Log.debug('Successfully parsed {} input file.'.format(input_file))

        self.categories = [row[0] for row in body]
        global CATEGORIES
        CATEGORIES = set(self.categories)

        self.dataset = np.array([np.array(row[1:]) for row in body])
        IO.Log.debug('Loaded dataset', self.dataset)

        self.dummy_y = np.array([[1.0 if c == cat else 0.0
                                  for c in self.categories]
                                 for cat in CATEGORIES]).T
        IO.Log.debug('Dummy y', self.dummy_y)

        self.axis = 0
        self._centered = False
        self._normalized = False

    @property
    def n(self):
        """Return number of rows of dataset (or of dummy y)."""
        return self.dataset.shape[0]

    @property
    def m(self):
        """Return number of columns of dataset."""
        return self.dataset.shape[1]

    @property
    def p(self):
        """Return number of columns in dummy y."""
        return self.dummy_y.shape[1]

    @property
    def centered(self):
        """Return whether dataset has been centered."""
        return self._centered

    @property
    def normalized(self):
        """Return whether dataset has been normalized"""
        return self._normalized

    @property
    def autoscaled(self):
        """Return wheter dataset has been autoscaled"""
        return self._centered and self._normalized

    @property
    def original(self):
        """Return wheter dataset has been preprocessed"""
        return not (self._centered or self._normalized)

    def center(self, quiet=False):
        """Center the dataset and the dummy y to their mean."""
        if self.centered:
            IO.Log.warning('Already centered dataset')
            return

        self.dataset = self.dataset - self.dataset.mean(axis=self.axis)
        if not quiet:
            IO.Log.debug('Centered dataset', self.dataset)

        self.dummy_y = self.dummy_y - self.dummy_y.mean(axis=self.axis)
        self._centered = True

    def normalize(self, quiet=False):
        """Normalize the dataset and the dummy y."""
        if self.normalized:
            IO.Log.warning('Already normalized dataset')
            return

        self.dataset = self.dataset / self.dataset.std(axis=self.axis)
        if not quiet:
            IO.Log.debug('Normalized dataset', self.dataset)

        self.dummy_y = self.dummy_y / self.dummy_y.std(axis=self.axis)
        self._normalized = True

    def autoscale(self):
        """Center and normalize the dataset and the dummy y."""
        if self.normalized:
            IO.Log.warning('Already autoscaled dataset')
            return

        self.center(quiet=True)
        self.normalize(quiet=True)
        IO.Log.debug('Autoscaled dataset', self.dataset)

    def empty_method(self):
        """Do not remove this method, it is needed by the GUI."""
        pass


class Nipals(object):
    """Class to compute the NIPALS method used in PLS.

       Non-linear Iterative PArtial Least Square
    """

    def __init__(self, preproc):
        """Nipals method will be applied to a Preprocessing object.

           Raises TypeError on wrong preproc data type.
        """
        if not isinstance(preproc, Preprocessing):
            raise TypeError('Argument passed to Nipals.__init__() is not a'
                            'Preprocessing object!')

        self.preproc = preproc

    @property
    def n(self):
        return self.preproc.n

    @property
    def m(self):
        return self.preproc.m

    @property
    def p(self):
        return self.preproc.p

    def run(self, nr_lv=None, tol=1e-6, max_iter=1e4):
        """Find the Principal Components with the NIPALS algorithm."""

        # Start with maximal residual (matrix X, matrix Y)
        E_x = self.preproc.dataset.copy()
        E_y = self.preproc.dummy_y.copy()

        n, m = E_x.shape
        n, p = E_y.shape

        if nr_lv is None:
            nr_lv = min(n, m)
        if nr_lv > min(n, m):
            IO.Log.warning('Too many latent variables specified. '
                           'Will use {}'.format(min(n, m)))
            nr_lv = min(n, m)
        self.nr_lv = nr_lv

        self.T = np.empty((n, nr_lv))
        self.P = np.empty((m, nr_lv))
        self.W = np.empty((m, nr_lv))
        self.U = np.empty((n, nr_lv))
        self.Q = np.empty((p, nr_lv))
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
                t = np.dot(E_x, w) / np.dot(w, w)

                # Y part
                # Evaluate q as projection of t in Y and normalize it
                q = np.dot(E_y.T, t) / np.dot(t, t)
                q = q / np.linalg.norm(q)
                # Evaluate u_star as projection of c in Y
                u_star = np.dot(E_y, q) / np.dot(q, q)

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
            p = np.dot(E_x.T, t) / np.dot(t, t)
            p_norm = np.linalg.norm(p)
            self.P[:, i] = p / p_norm
            self.T[:, i] = t * p_norm
            self.W[:, i] = w * p_norm
            self.U[:, i] = u
            self.Q[:, i] = q

            s_list_x.append(np.linalg.norm(t))
            s_list_y.append(np.linalg.norm(u))
            # regression coefficient for the inner relation
            self.d[i] = np.dot(u.T, t) / np.dot(t, t)

            # Calculate residuals
            E_x = E_x - np.dot(np.row_stack(t), np.column_stack(self.P[:, i]))
            E_y = E_y - self.d[i] * np.dot(np.row_stack(t),
                                           np.column_stack(q.T))

        self.x_eigenvalues = np.power(np.array(s_list_x), 2) / (self.n - 1)
        self.y_eigenvalues = np.power(np.array(s_list_y), 2) / (self.n - 1)

        # Compute regression parameters B
        # tmp = (P'W)^{-1}
        tmp = np.linalg.inv(self.P.T.dot(self.W))
        self.B = self.W.dot(tmp).dot(np.diag(self.d)).dot(self.Q.T)
        self.Y_modeled = self.preproc.dataset.dot(self.B)
        IO.Log.debug('Modeled Y prior to the discriminant classification',
                     self.Y_modeled)
        Y_dummy = [[1 if elem == max(row) else -1 for elem in row]
                   for row in self.Y_modeled]
        self.Y_modeled_dummy = np.array(Y_dummy)

        self.E_y = E_y
        IO.Log.info('NIPALS loadings shape', self.P.shape)
        IO.Log.info('NIPALS scores shape', self.T.shape)
        IO.Log.info('NIPALS x_eigenvalues', self.x_eigenvalues)


def integer_bounds(P, T, col):
        """Return tuple with min and max integers bounds for P[col] and T[col].
        """
        extracted = np.concatenate((P[:, col], T[:, col]))
        return math.floor(np.min(extracted)), math.ceil(np.max(extracted))


def explained_variance(model, matrix='x'):
        """Return the explained variance of model.[x|y].eigenvalues

           Raises Exception if matrix is not 'x' or 'y'
        """
        if matrix == 'x':
            eigen = model.x_eigenvalues
        elif matrix == 'y':
            eigen = model.y_eigenvalues
        else:
            raise ValueError('Bad matrix parameter ({}) in '
                             'explained_variance() '.format(repr(matrix)))

        IO.Log.info('[model.explained_variance] '
                    'Eigenvalues for {}: \n{}'.format(matrix, eigen))
        return 100 * eigen / np.sum(eigen)


if __name__ == '__main__':
    raise SystemExit('Please do not run that script, load it!')
