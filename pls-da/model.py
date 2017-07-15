#!/usr/bin/env python3
# coding: utf-8

import math
import numpy as np

import IO
import utility




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
        """Return whether dataset has been autoscaled"""
        return self._centered and self._normalized

    @property
    def original(self):
        """Return whether dataset has been preprocessed"""
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


class Model(object):
    """Save a NIPALS model and provide helper methods to access it."""

    def __init__(self, X, Y, max_lv):
        """Instantiate space for the model."""

        self.X = X
        self.Y = Y
        self.n = X.shape[0]
        self.m = X.shape[1]
        self.p = Y.shape[1]

        # max_lv is the number of lv in which the model was calculated
        # nr_lv is the number of lv used for prediction
        self.max_lv = max_lv
        self.nr_lv = max_lv

        self.T = np.empty((self.n, max_lv))
        self.P = np.empty((self.m, max_lv))
        self.W = np.empty((self.m, max_lv))
        self.U = np.empty((self.n, max_lv))
        self.Q = np.empty((self.p, max_lv))

        self.b = np.empty((max_lv))
        self.x_eigenvalues = np.empty((max_lv))
        self.y_eigenvalues = np.empty((max_lv))
        self.Y_modeled = np.empty((self.n, self.p))
        self.Y_modeled_dummy = np.empty((self.n, self.p))
        self.B = np.empty((self.m, self.p))
        self.E_x = np.empty((self.n, self.m))
        self.E_y = np.empty((self.n, self.p))

    def predict(self, test_set, nr_lv):
        """Return Y predicted over this model."""
#        T_cap = np.empty((test_set.shape[0], self.m))
#        E_x = test_set.copy()
#        Y = np.zeros((test_set.shape[0], self.p))

#        T_cap = np.dot(test_set, self.W)
        return np.dot(test_set, self.B)


class Statistics(object):
    """Calculate statistics tied only to Y over the results of a prediction."""

    def __init__(self, y_real, y_pred):
        """Save the real and the predicted Y."""

        assert y_real.shape == y_pred.shape, "Y real and Y predicted" \
                                             "must have the same dimension"
        self.y_real = y_real
        self.y_pred = y_pred

    @property
    def ess(self):
        return np.linalg.norm(self.y_pred - self.y_real.mean(axis=0),
                              axis=0)**2

    @property
    def rss(self):
        return np.linalg.norm(self.y_real - self.y_pred, axis=0)**2

    @property
    def tss(self):
        return self.ess + self.rss

    @property
    def rmsec(self):
        return np.sqrt(self.rss / self.y_real.shape[-1])

    @property
    def r_squared(self):
        r_squared = 1 - self.rss / self.tss
        assert r_squared > 0, "Negative r_squared found"
        return r_squared


def nipals(X, Y, nr_lv=None, tol=1e-6, max_iter=1e4):
    """Find the Principal Components with the NIPALS algorithm."""

    # Start with maximal residual (matrix X, matrix Y)
    E_x = X.copy()
    E_y = Y.copy()

    assert X.shape[0] == Y.shape[0], "Incompatible X and Y matrices"

    n = X.shape[0]
    m = X.shape[1]
    p = Y.shape[1]

    if nr_lv is None:
        nr_lv = min(n, m)
    if nr_lv > min(n, m):
        IO.Log.warning('Too many latent variables specified. '
                       'Will use {}'.format(min(n, m)))
        nr_lv = min(n, m)

    model = Model(X, Y, nr_lv)

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
        model.P[:, i] = p / p_norm
        model.T[:, i] = t * p_norm
        model.W[:, i] = w * p_norm
        model.U[:, i] = u
        model.Q[:, i] = q

        s_list_x.append(np.linalg.norm(t))
        s_list_y.append(np.linalg.norm(u))
        # regression coefficient for the inner relation
        model.b[i] = np.dot(u.T, t) / np.dot(t, t)

        # Calculate residuals
        E_x = E_x - np.dot(np.row_stack(t), np.column_stack(model.P[:, i]))
        E_y = E_y - model.b[i] * np.dot(np.row_stack(t),
                                        np.column_stack(q.T))

    model.x_eigenvalues = np.power(np.array(s_list_x), 2) / (model.n - 1)
    model.y_eigenvalues = np.power(np.array(s_list_y), 2) / (model.n - 1)

    # Compute regression parameters B
    # tmp = (P'W)^{-1}
    tmp = np.linalg.inv(model.P.T.dot(model.W))
    model.B = model.W.dot(tmp).dot(np.diag(model.b)).dot(model.Q.T)
    model.Y_modeled = X.dot(model.B)
    IO.Log.debug('Modeled Y prior to the discriminant classification',
                 model.Y_modeled)
    Y_dummy = [[1 if elem == max(row) else -1 for elem in row]
               for row in model.Y_modeled]
    model.Y_modeled_dummy = np.array(Y_dummy)

    model.E_y = E_y
    IO.Log.info('NIPALS loadings shape', model.P.shape)
    IO.Log.info('NIPALS scores shape', model.T.shape)
    IO.Log.info('NIPALS x_eigenvalues', model.x_eigenvalues)

    return model


def cross_validation(preproc, split, max_lv):
    """Perform a cross-validation procedure on a Preprocessing dataset.

    Return a list of dictionaries of Statistics object. Every dictionary
    corresponds to a split, while every element in a dictionary corresponds
    to the model predicted with a specific lv. The key of the dictionary
    element is the lv used for that prediction.
    """
    results = []
    for train, test in venetian_blind_split(preproc, split):
        model = nipals(*train)
        res = dict()
        for lv in range(max_lv):
            y_pred = model.predict(test[0], lv)
            res[lv] = Statistics(test[1], y_pred)
        results.append(res)
    return results


def venetian_blind_split(preproc, split):
    """Split the dataset in train and test using the venetian blind algo."""
    for offset in range(split, 0, -1):  # order result logically
        mask = np.arange(offset, preproc.n + offset) % split == 0
        test_x = preproc.dataset[mask]
        train_x = preproc.dataset[~mask]
        test_y = preproc.dummy_y[mask]
        train_y = preproc.dummy_y[~mask]

        yield ((train_x, train_y), (test_x, test_y))


def integer_bounds(P, T, col):
    """Return tuple with min and max integers bounds for P[col] and T[col]."""
    extracted = np.concatenate((P[:, col], T[:, col]))
    return math.floor(np.min(extracted)), math.ceil(np.max(extracted))


def explained_variance(model, matrix='x'):
    """Return the explained variance of model.[x|y].eigenvalues

       Raise ValueError if matrix is not 'x' or 'y'.
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
