#!/usr/bin/env python3
# coding: utf-8

import math
import numpy as np

import IO
import utility


class Dataset(object):
    """Represent a chemometrics dataset."""
    def __init__(self, input_file=None):
        """Load and parse csv input file.

           self.header      list of samples' properties (text)
           self.categorical_y  list of samples' labels (text)
           self.x           list of samples' values (float)
           self.y           list of samples' labels (1 or 0)

           self.axis        axis to compute std and mean
        """
        if input_file is not None:
            self.header, body = IO.CSV.parse(input_file)
            IO.Log.debug('Successfully parsed file {}.'.format(input_file))

            self.body = body
            self.categorical_y = [row[0] for row in body]
            self.categories = utility.get_unique_list(self.categorical_y)

            self.x = np.array([np.array(row[1:]) for row in body])
            IO.Log.debug('Loaded dataset', self.x)

            self.y = np.array([[1.0 if c == cat else 0.0
                                for c in self.categorical_y]
                               for cat in self.categories]).T
            IO.Log.debug('Dummy y', self.y)

            self.mean_x = np.zeros(self.m)
            self.mean_y = np.zeros(self.p)
            self.sigma_x = np.ones(self.m)
            self.sigma_y = np.ones(self.p)
            self.axis = 0
            self._centered = False
            self._normalized = False

    @property
    def n(self):
        """Return number of rows of x (or of dummy y)."""
        return self.x.shape[0]

    @property
    def m(self):
        """Return number of columns of x."""
        return self.x.shape[1]

    @property
    def p(self):
        """Return number of columns in dummy y."""
        return self.y.shape[1]

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


class TrainingSet(Dataset):
    """Class to preprocess csv input data."""

    def center(self, quiet=False):
        """Center the dataset and the dummy y to their mean."""
        if self.centered:
            IO.Log.warning('Already centered dataset')
            return

        self.mean_x = self.x.mean(axis=self.axis)
        self.x = self.x - self.mean_x
        if not quiet:
            IO.Log.debug('Centered x', self.x)

        self.mean_y = self.y.mean(axis=self.axis)
        self.y = self.y - self.mean_y
        self._centered = True

    def normalize(self, quiet=False):
        """Normalize the dataset and the dummy y."""
        if self.normalized:
            IO.Log.warning('Already normalized dataset')
            return

        self.sigma_x = self.x.std(axis=self.axis)
        self.x = self.x / self.sigma_x
        if not quiet:
            IO.Log.debug('Normalized dataset', self.x)

        self.sigma_y = self.y.std(axis=self.axis)
        self.y = self.y / self.sigma_y
        self._normalized = True

    def autoscale(self):
        """Center and normalize the x and the dummy y."""
        if self.normalized:
            IO.Log.warning('Already autoscaled dataset')
            return

        self.center(quiet=True)
        self.normalize(quiet=True)
        IO.Log.debug('Autoscaled dataset', self.x)

    def empty_method(self):
        """Do not remove this method, it is needed by the GUI."""
        pass


class TestSet(Dataset):
    """Model a test set."""
    def __init__(self, filename, train_set):
        super().__init__(filename)

        # apply the same transformation used on the test set
        assert isinstance(train_set,
                          TrainingSet), 'argument train_set must be of type' \
                                        'TrainingSet, is instead of ' \
                                        'type {}'.format(type(train_set))

        self.mean_x = train_set.mean_x
        self.sigma_x = train_set.sigma_x
        self.mean_y = train_set.mean_y
        self.sigma_y = train_set.sigma_y
        self.x -= self.mean_x
        self.x /= self.sigma_x
        self.y -= self.mean_y
        self.y /= self.sigma_y
        self._centered = train_set.centered
        self._normalized = train_set.normalized


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
        self._nr_lv = max_lv

        self._T = np.zeros((self.n, max_lv))
        self._P = np.zeros((self.m, max_lv))
        self._W = np.zeros((self.m, max_lv))
        self._U = np.zeros((self.n, max_lv))
        self._Q = np.zeros((self.p, max_lv))

        self._b = np.zeros(max_lv)
        self._x_eigenvalues = np.zeros(max_lv)
        self._y_eigenvalues = np.zeros(max_lv)
        self._Y_modeled = np.zeros((self.n, self.p))
        self._Y_modeled_dummy = np.zeros((self.n, self.p))

    @property
    def nr_lv(self):
        return self._nr_lv

    @nr_lv.setter
    def nr_lv(self, value):
        assert 0 < value <= self.max_lv, "Chosen latent variable" \
                                    " number {} out of bounds [0, {}]".format(
                                            value, self.max_lv)

        utility.clear_property_cache(self, 'Y_modeled')
        utility.clear_property_cache(self, 'Y_modeled_dummy')
        utility.clear_property_cache(self, 'E_x')
        utility.clear_property_cache(self, 'E_y')
        utility.clear_property_cache(self, 'B')
        utility.clear_property_cache(self, 't_square')
        utility.clear_property_cache(self, 'q_residuals_x')
        utility.clear_property_cache(self, 'leverage')

        self._nr_lv = value

    @property
    def T(self):
        return self._T[:, :self.nr_lv]

    @property
    def P(self):
        return self._P[:, :self.nr_lv]

    @property
    def W(self):
        return self._W[:, :self.nr_lv]

    @property
    def U(self):
        return self._U[:, :self.nr_lv]

    @property
    def Q(self):
        return self._Q[:, :self.nr_lv]

    @property
    def b(self):
        return self._b[:self.nr_lv]

    @property
    def x_eigenvalues(self):
        return self._x_eigenvalues[:self.nr_lv]

    @property
    def y_eigenvalues(self):
        return self._y_eigenvalues[:self.nr_lv]

    @utility.cached_property
    def Y_modeled(self):
        Y_modeled = self.X.dot(self.B)
        IO.Log.debug('Modeled Y prior to the discriminant classification',
                     Y_modeled)
        return Y_modeled

    @utility.cached_property
    def Y_modeled_dummy(self):
        dummy = [[1 if elem == max(row) else -1 for elem in row]
                 for row in self.Y_modeled]
        return np.array(dummy)

    @utility.cached_property
    def E_x(self):
        return self.X - np.dot(self.T, self.P.T)

    @utility.cached_property
    def E_y(self):
        return self.Y - (self.T.dot(np.diag(self.b))).dot(self.Q.T)

    @utility.cached_property
    def B(self):
        # Compute regression parameters B
        # tmp = (P'W)^{-1}
        tmp = np.linalg.inv(self.P.T.dot(self.W))
        return ((self.W.dot(tmp)).dot(np.diag(self.b))).dot(self.Q.T)

    @utility.cached_property
    def t_square(self):
        lambda_inv = 1 / self.x_eigenvalues
        return np.diag(self.T.dot(np.diag(lambda_inv)).dot(self.T.T))

    @utility.cached_property
    def q_residuals_x(self):
        if self.nr_lv == self.max_lv:
            IO.Log.warning('Q residuals with max number of components are 0')
        return np.diag(self.E_x.dot(self.E_x.T))

    @utility.cached_property
    def leverage(self):
        temp = np.linalg.inv(np.dot(self.U.T, self.U))
        leverage = np.empty(self.n)

        for j in range(self.p):
            for i in range(self.n):
                leverage[i] = self.U[i].dot(temp).dot(self.U[i].T)
        return leverage

    def predict(self, test_set_x):
        """Return Y predicted for the given test set over this model."""
        return np.dot(test_set_x, self.B)


class Statistics(object):
    """Calculate statistics tied only to Y over the results of a prediction."""

    def __init__(self, y_real, y_pred):
        """Save the real and the predicted Y."""

        assert y_real.shape == y_pred.shape, 'Y real and Y predicted ' \
                                             'must have the same dimension'
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
        IO.Log.debug('r_squared: ', r_squared)

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
            w /= np.linalg.norm(w)
            # Evaluate t as projection of w in X
            # t = np.dot(E_x, w) / np.dot(w, w)
            t = np.dot(E_x, w)

            # Y part
            # Evaluate q as projection of t in Y and normalize it
            # q = np.dot(E_y.T, t) / np.dot(t, t)
            q = np.dot(E_y.T, t)
            q /= np.linalg.norm(q)

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
        p = p / p_norm
        t = t * p_norm
        w = w * p_norm

        s_list_x.append(np.linalg.norm(t))
        s_list_y.append(np.linalg.norm(u))
        # regression coefficient for the inner relation
        model.b[i] = np.dot(u.T, t) / np.dot(t, t)

        # Calculate residuals
        E_x -= np.dot(np.row_stack(t), np.column_stack(p))
        E_y -= model.b[i] * np.dot(np.row_stack(t),
                                   np.column_stack(q.T))

        model.P[:, i] = p
        model.T[:, i] = t
        model.W[:, i] = w
        model.U[:, i] = u
        model.Q[:, i] = q

    model._x_eigenvalues = np.power(np.array(s_list_x), 2) / (model.n - 1)
    model._y_eigenvalues = np.power(np.array(s_list_y), 2) / (model.n - 1)

    IO.Log.info('NIPALS loadings shape', model.P.shape)
    IO.Log.info('NIPALS scores shape', model.T.shape)
    IO.Log.info('NIPALS x_eigenvalues', model.x_eigenvalues)

    return model


def cross_validation(train_set, split, sample, max_lv):
    """Perform a cross-validation procedure on a TrainingSet dataset.

    Return a list of dictionaries of Statistics object. Every dictionary
    corresponds to a split, while every element in a dictionary corresponds
    to the model predicted with a specific lv. The key of the dictionary
    element is the lv used for that prediction.
    """
    results = []
    for train, test in venetian_blind_split(train_set, split, sample):
        model = nipals(*train)
        res = dict()
        for lv in range(1, max_lv):
            model.nr_lv = lv
            y_pred = model.predict(test[0])
            res[lv] = Statistics(test[1], y_pred)
        results.append(res)
    return results


def venetian_blind_split(train_set, split, sample):
    """Split the dataset in train and test using the venetian blind algo."""

    # find the minimum divisor of split * sample greater than n
    # this is needed to be able to use the function np.roll properly
    n = train_set.n
    base = split * sample
    div = n - (n % base) + (base if n % base else 0)

    # build the set of mask used to create the first split
    masks = []
    masks.append(np.arange(0, div) % (split*sample) == 0)
    for offset in range(1, sample):
        masks.append(np.roll(masks[0], offset))

    # build the mask for the first split
    m = []
    for i in zip(*masks):
        x = False
        for e in i:
            x = x or e
        m.append(x)

    for offset in range(split):
        # Create the mask for the current split and truncate it to
        # its real length
        mask = np.roll(m, offset * sample)[:n]
        test_x = train_set.x[mask]
        train_x = train_set.x[~mask]
        test_y = train_set.y[mask]
        train_y = train_set.y[~mask]

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
