#!/usr/bin/env python3
# coding: utf-8

import copy
import math
import numpy as np
import scipy
import sklearn.cross_decomposition as sklCD
import unittest

from context import IO
from context import model
from context import create_environment

absolute_tolerance = 0.1


class test_trainingset_preprocessing(unittest.TestCase):

    matrix_A = np.array([[2.0, 3.0, 4.0, 5.0],
                         [4.0, 6.0, 8.0, 10.0],
                         [6.0, 9.0, 12.0, 15.0]])
    normalized_A = np.array([[2. / math.sqrt(8. / 3), 3. / math.sqrt(6.0000),
                              4. / math.sqrt(32 / 3), 5. / math.sqrt(50 / 3)],
                             [4. / math.sqrt(8. / 3), 6. / math.sqrt(6.0000),
                              8. / math.sqrt(32 / 3), 10 / math.sqrt(50 / 3)],
                             [6. / math.sqrt(8. / 3), 9. / math.sqrt(6.0000),
                              12 / math.sqrt(32 / 3), 15 / math.sqrt(50 / 3)]])
    matrix_3x3 = np.array([[1.00000000, 2.00000000, 3.00000000],
                           [1.0 - 1e-8, 2.0 - 1e-8, 3.0 - 1e-8],
                           [1.0 + 1e-8, 2.0 + 1e-8, 3.0 + 1e-8]])
    matrix_3x2 = np.array([[0.50000000, 0.50000000],
                           [0.5 - 1e-8, 0.5 + 1e-8],
                           [0.5 + 1e-8, 0.5 - 1e-8]])
    null_3x3 = np.array([[0.0 for c in range(3)] for r in range(3)])
    null_3x2 = np.array([[0.0 for c in range(2)] for r in range(3)])

    def setUp(self):
        self.train_set = model.TrainingSet('.train_set_synthesis.csv')

    def tearDown(self):
        self.train_set = None

    def test_TrainingSet_init(self):
        self.assertEqual(len(self.train_set.header), self.train_set.m + 1)
        self.assertEqual(len(self.train_set.categorical_y), self.train_set.n)
        self.assertEqual(len(self.train_set.y), self.train_set.n)
        self.assertEqual(len(self.train_set.y[0]),
                         len(set(self.train_set.categorical_y)))
        self.assertFalse(self.train_set.centered)
        self.assertFalse(self.train_set.normalized)
        self.assertFalse(self.train_set.autoscaled)

    def test_TrainingSet_center(self):
        self.train_set.x = self.matrix_3x3.copy()
        self.train_set.y = self.matrix_3x2.copy()

        self.train_set.center()

        self.assertTrue(self.train_set.centered)
        self.assertFalse(self.train_set.normalized)
        self.assertFalse(self.train_set.autoscaled)
        np.testing.assert_allclose(self.train_set.x, self.null_3x3,
                                   atol=absolute_tolerance)
        np.testing.assert_allclose(self.train_set.y, self.null_3x2,
                                   atol=absolute_tolerance)

    def test_TrainingSet_normalize(self):
        self.train_set.x = self.matrix_A.copy()
        self.train_set.y = self.matrix_A.copy()

        self.train_set.normalize()

        self.assertFalse(self.train_set.centered)
        self.assertTrue(self.train_set.normalized)
        self.assertFalse(self.train_set.autoscaled)
        np.testing.assert_allclose(self.train_set.x, self.normalized_A)
        np.testing.assert_allclose(self.train_set.y, self.normalized_A)

    def test_TrainingSet_autoscale(self):
        dataset_copy = self.train_set.x.copy()
        dummy_y_copy = self.train_set.y.copy()

        self.train_set.center()
        self.train_set.normalize()

        dataset_autoscaled = self.train_set.x
        dummy_y_autoscaled = self.train_set.y
        self.assertTrue(self.train_set.autoscaled)

        self.train_set.x = dataset_copy
        self.train_set.y = dummy_y_copy
        self.train_set._centered = False
        self.train_set._normalized = False

        self.train_set.autoscale()

        self.assertTrue(self.train_set.autoscaled)
        np.testing.assert_allclose(self.train_set.x, dataset_autoscaled)
        np.testing.assert_allclose(self.train_set.y, dummy_y_autoscaled)


class test_eigen_module(unittest.TestCase):

    matrix_3x3 = np.array([[1.00000000, 2.00000000, 3.00000000],
                           [1.0 - 1e-8, 2.0 - 1e-8, 3.0 - 1e-8],
                           [1.0 + 1e-8, 2.0 + 1e-8, 3.0 + 1e-8]])
    matrix_3x2 = np.array([[0.50000000, 0.50000000],
                           [0.5 - 1e-8, 0.5 + 1e-8],
                           [0.5 + 1e-8, 0.5 - 1e-8]])

    def setUp(self):
        self.train_set = model.TrainingSet('.train_set_synthesis.csv')
        self.train_set.x = self.matrix_3x3.copy()
        self.train_set.y = self.matrix_3x2.copy()

        cov_x = np.dot(self.train_set.x.T,
                       self.train_set.x) / (self.train_set.n - 1)
        cov_y = np.dot(self.train_set.y.T,
                       self.train_set.y) / (self.train_set.n - 1)

        self.nipals = model.nipals(self.train_set.x, self.train_set.y)
        self.wx, vx = scipy.linalg.eig(cov_x)
        self.wy, vy = scipy.linalg.eig(cov_y)
        self.wx = np.real(self.wx)
        self.wy = np.real(self.wy)
        self.wx[::-1].sort()
        self.wy[::-1].sort()
        self.x_variance = 100 * self.wx / np.sum(self.wx)
        self.y_variance = 100 * self.wy / np.sum(self.wy)

    def tearDown(self):
        self.train_set = None
        self.nipals = None

    def test_nipals_eigenvectors_x_eigen(self):
        np.testing.assert_allclose(self.nipals.x_eigenvalues, self.wx,
                                   atol=absolute_tolerance)

    @unittest.skip("Different algorithm")
    def test_nipals_eigenvectors_y_eigen(self):
        np.testing.assert_allclose(self.nipals.y_eigenvalues, self.wy,
                                   atol=absolute_tolerance)

    def test_nipals_eigenvectors_x_variance(self):
        np.testing.assert_allclose(self.nipals.explained_variance_x,
                                   self.x_variance, atol=absolute_tolerance)

    @unittest.skip('explained_variance_y is not the y variance')
    def test_nipals_eigenvectors_y_variance(self):
        np.testing.assert_allclose(self.nipals.explained_variance_y,
                                   self.y_variance, atol=absolute_tolerance)


class nipals_abstract(object):

    def setUp(self):
        j = 2
        self.train_set = model.TrainingSet('.train_set_synthesis.csv')

        X = np.array([[1, 1.9], [1.9, 1], [3.8, 4.2], [4, 3.6], [3.6, 4.4]])
        Y = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [0, 1]])
        self.train_set.x = X.copy()
        self.train_set.y = Y.copy()

        self.train_set.autoscale()
        # autoscale also matrices for sklearn
        X = self.train_set.x.copy()
        Y = self.train_set.y.copy()
        self.nipals = model.nipals(X, Y)

        self.sklearn_pls = sklCD.PLSRegression(n_components=j, scale=True,
                                               max_iter=1e4, tol=1e-6,
                                               copy=True)
        self.sklearn_pls.fit(X, Y)

        IO.Log.debug('NIPALS x scores', self.nipals.T)
        IO.Log.debug('sklearn x scores', self.sklearn_pls.x_scores_)
        IO.Log.debug('NIPALS x loadings', self.nipals.P)
        IO.Log.debug('sklearn x loadings', self.sklearn_pls.x_loadings_)
        IO.Log.debug('NIPALS x weights', self.nipals.W)
        IO.Log.debug('sklearn x weights', self.sklearn_pls.x_weights_)
        IO.Log.debug('NIPALS y scores', self.nipals.U)
        IO.Log.debug('sklearn y scores', self.sklearn_pls.y_scores_)
        IO.Log.debug('NIPALS y loadings', self.nipals.Q)
        IO.Log.debug('sklearn y loadings', self.sklearn_pls.y_loadings_)
        IO.Log.debug('sklearn y weights', self.sklearn_pls.y_weights_)

    def tearDown(self):
        self.train_set = None
        self.nipals = None
        self.sklearn_pls = None

    @unittest.skip("Different algorithm")
    def test_nipals_x_scores(self):
        np.testing.assert_allclose(np.absolute(self.nipals.T),
                                   np.absolute(self.sklearn_pls.x_scores_),
                                   atol=absolute_tolerance)

    def test_nipals_x_loadings(self):
        np.testing.assert_allclose(np.absolute(self.nipals.P),
                                   np.absolute(self.sklearn_pls.x_loadings_),
                                   atol=absolute_tolerance)

    def test_nipals_x_weights(self):
        np.testing.assert_allclose(np.absolute(self.nipals.W),
                                   np.absolute(self.sklearn_pls.x_weights_),
                                   atol=absolute_tolerance)

    @unittest.skip("Different algorithm")
    def test_nipals_y_scores(self):
        np.testing.assert_allclose(np.absolute(self.nipals.U),
                                   np.absolute(self.sklearn_pls.y_scores_),
                                   atol=absolute_tolerance)

    @unittest.skip("Different algorithm")
    def test_nipals_y_loadings(self):
        np.testing.assert_allclose(np.absolute(self.nipals.Q),
                                   np.absolute(self.sklearn_pls.y_loadings_),
                                   atol=absolute_tolerance)

    def test_nipals_coefficient(self):
        np.testing.assert_allclose(np.absolute(self.nipals.B),
                                   np.absolute(self.sklearn_pls.coef_),
                                   err_msg='From class '
                                           '{}'.format(type(self).__name__),
                                   atol=absolute_tolerance)

    def test_nipals_inner_relation(self):
        """The relation in U = TB + H"""
        np.testing.assert_allclose(self.nipals.U,
                                   np.dot(self.nipals.T,
                                          np.diag(self.nipals.b)),
                                   atol=1)

    def test_x_component(self):
        np.testing.assert_allclose(self.train_set.x,
                                   np.dot(self.nipals.T, self.nipals.P.T),
                                   err_msg="X != TP'", atol=absolute_tolerance)

    def test_y_component(self):
        np.testing.assert_allclose(self.train_set.y,
                                   np.dot(self.nipals.U, self.nipals.Q.T),
                                   err_msg="Y != UQ'", atol=absolute_tolerance)

    def test_p_unit_length(self):
        for i in range(self.nipals.m):
            np.testing.assert_allclose(np.linalg.norm(self.nipals.P[:, i]),
                                       1.0, atol=absolute_tolerance)

    def test_q_unit_length(self):
        for i in range(self.nipals.m):
            np.testing.assert_allclose(np.linalg.norm(self.nipals.Q[:, i]),
                                       1.0, atol=absolute_tolerance)

    @unittest.skip("Property which should hold")
    def test_t_centered_around_zero(self):
        for i in range(self.nipals.n):
            np.testing.assert_allclose(sum(self.nipals.T[i, :]), 0,
                                       atol=absolute_tolerance)

    @unittest.skip("Property which should hold")
    def test_u_centered_around_zero(self):
        for i in range(self.nipals.n):
            np.testing.assert_allclose(sum(self.nipals.U[i, :]), 0,
                                       atol=absolute_tolerance)

    @unittest.skip("Property which should hold")
    def test_w_orthogonal(self):
        for i in range(self.nipals.m):
            with self.subTest(i=i):
                for j in range(self.nipals.m):
                    with self.subTest(j=j):
                        w_i = self.nipals.W[i, :]
                        w_j = self.nipals.W[i, :]
                        norm = np.linalg.norm(w_i)**2 if i == j else 0
                        np.testing.assert_allclose(w_i.T.dot(w_j), norm,
                                                   atol=absolute_tolerance)

    @unittest.skip("Property which should hold")
    def test_t_orthogonal(self):
        for i in range(self.nipals.m):
            with self.subTest(i=i):
                for j in range(self.nipals.m):
                    with self.subTest(j=j):
                        t_i = self.nipals.T[i, :]
                        t_j = self.nipals.T[i, :]
                        norm = np.linalg.norm(t_i)**2 if i == j else 0
                        np.testing.assert_allclose(t_i.T.dot(t_j), norm,
                                                   atol=absolute_tolerance)

    def test_nr_lv_implementation_all_matrices(self):
        mdl = copy.deepcopy(self.nipals)
        n = mdl.n
        m = mdl.m
        p = mdl.p
        lv = 1
        mdl.nr_lv = lv
        self.assertEqual(mdl.T.shape, (n, lv))
        self.assertEqual(mdl.U.shape, (n, lv))
        self.assertEqual(mdl.P.shape, (m, lv))
        self.assertEqual(mdl.W.shape, (m, lv))
        self.assertEqual(mdl.Q.shape, (p, lv))
        self.assertEqual(mdl.b.shape, (lv, ))
        self.assertEqual(mdl.x_eigenvalues.shape, (lv, ))
        self.assertEqual(mdl.y_eigenvalues.shape, (lv, ))
        self.assertEqual(mdl.Y_modeled.shape, (n, p))
        self.assertEqual(mdl.Y_modeled_dummy.shape, (n, p))


class test_nipals_method(nipals_abstract, unittest.TestCase):

    @unittest.skip("Different algorithm")
    def test_nipals_y_weights(self):
        np.testing.assert_allclose(np.absolute(self.nipals.W),
                                   np.absolute(self.sklearn_pls.y_weights_),
                                   atol=absolute_tolerance)


if __name__ == '__main__':

    create_environment()

    unittest.main(failfast=False)
