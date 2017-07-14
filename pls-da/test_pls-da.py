#!/usr/bin/env python3
# coding: utf-8

import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy
import sklearn.cross_decomposition as sklCD
import unittest

import IO
import model
import plot
import utility

absolute_tolerance = 0.1


class test_IO_module(unittest.TestCase):

    csv_sample = str('CATEGORY;VAR1;VAR2;VAR3\n'
                     'E;0,5;-50;4,99\n'
                     'G;-2.5;100;2.9\n'
                     'B;15;1.23;4.56\n')
    np_array = np.array([-5, 6, -7, 8, -9])
    np_matrix = np.array([[1, 2], [3, 4], [5, 6]])
    char_list = list('0123456789ABCDEF')

    def setUp(self):
        with open('test_temporary.csv', 'w') as f:
            f.write(self.csv_sample)

    def tearDown(self):
        os.remove('test_temporary.csv')

    def test_mat2str(self):
        self.assertTrue(callable(IO.mat2str))
        self.assertIsInstance(IO.mat2str(self.np_array), str)
        self.assertIsInstance(IO.mat2str(self.np_matrix), str)
        self.assertIsInstance(IO.mat2str(self.char_list), str)
        self.assertRaises(Exception, IO.mat2str, 'A string')
        self.assertRaises(Exception, IO.mat2str, 123.456)
        self.assertRaises(Exception, IO.mat2str, None)

    def test_Log_critical(self):
        self.assertTrue(callable(IO.Log.critical))

    def test_Log_debug(self):
        self.assertTrue(callable(IO.Log.debug))

    def test_Log_error(self):
        self.assertTrue(callable(IO.Log.error))

    def test_Log_info(self):
        self.assertTrue(callable(IO.Log.info))

    def test_Log_warning(self):
        self.assertTrue(callable(IO.Log.warning))

    def test_CSV(self):
        header, body = IO.CSV.parse('test_temporary.csv')
        self.assertEqual(header, ['CATEGORY', 'VAR1', 'VAR2', 'VAR3'])
        self.assertEqual(body, [['E', 0.5, -50, 4.99],
                                ['G', -2.5, 100, 2.9],
                                ['B', 15, 1.23, 4.56]])


class test_model_module(unittest.TestCase):

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
        self.preproc = model.Preprocessing()

    def tearDown(self):
        self.preproc = None

    def test_Preprocessing_init(self):
        self.assertEqual(len(self.preproc.header), self.preproc.m + 1)
        self.assertEqual(len(self.preproc.categories), self.preproc.n)
        self.assertEqual(len(self.preproc.dummy_y), self.preproc.n)
        self.assertEqual(len(self.preproc.dummy_y[0]),
                         len(set(self.preproc.categories)))
        self.assertFalse(self.preproc.centered)
        self.assertFalse(self.preproc.normalized)
        self.assertFalse(self.preproc.autoscaled)

    def test_Preprocessing_center(self):
        self.preproc.dataset = self.matrix_3x3.copy()
        self.preproc.dummy_y = self.matrix_3x2.copy()

        self.preproc.center()

        self.assertTrue(self.preproc.centered)
        self.assertFalse(self.preproc.normalized)
        self.assertFalse(self.preproc.autoscaled)
        np.testing.assert_allclose(self.preproc.dataset, self.null_3x3,
                                   atol=absolute_tolerance)
        np.testing.assert_allclose(self.preproc.dummy_y, self.null_3x2,
                                   atol=absolute_tolerance)

    def test_Preprocessing_normalize(self):
        self.preproc.dataset = self.matrix_A.copy()
        self.preproc.dummy_y = self.matrix_A.copy()

        self.preproc.normalize()

        self.assertFalse(self.preproc.centered)
        self.assertTrue(self.preproc.normalized)
        self.assertFalse(self.preproc.autoscaled)
        np.testing.assert_allclose(self.preproc.dataset, self.normalized_A)
        np.testing.assert_allclose(self.preproc.dummy_y, self.normalized_A)

    def test_Preprocess_autoscale(self):
        dataset_copy = self.preproc.dataset.copy()
        dummy_y_copy = self.preproc.dummy_y.copy()

        self.preproc.center()
        self.preproc.normalize()

        dataset_autoscaled = self.preproc.dataset
        dummy_y_autoscaled = self.preproc.dummy_y
        self.assertTrue(self.preproc.autoscaled)

        self.preproc.dataset = dataset_copy
        self.preproc.dummy_y = dummy_y_copy
        self.preproc._centered = False
        self.preproc._normalized = False

        self.preproc.autoscale()

        self.assertTrue(self.preproc.autoscaled)
        np.testing.assert_allclose(self.preproc.dataset, dataset_autoscaled)
        np.testing.assert_allclose(self.preproc.dummy_y, dummy_y_autoscaled)


class test_eigen_module(unittest.TestCase):

    matrix_3x3 = np.array([[1.00000000, 2.00000000, 3.00000000],
                           [1.0 - 1e-8, 2.0 - 1e-8, 3.0 - 1e-8],
                           [1.0 + 1e-8, 2.0 + 1e-8, 3.0 + 1e-8]])
    matrix_3x2 = np.array([[0.50000000, 0.50000000],
                           [0.5 - 1e-8, 0.5 + 1e-8],
                           [0.5 + 1e-8, 0.5 - 1e-8]])

    def setUp(self):
        self.preproc = model.Preprocessing()
        self.preproc.dataset = self.matrix_3x3.copy()
        self.preproc.dummy_y = self.matrix_3x2.copy()

        cov_x = np.dot(self.preproc.dataset.T,
                       self.preproc.dataset) / (self.preproc.n - 1)
        cov_y = np.dot(self.preproc.dummy_y.T,
                       self.preproc.dummy_y) / (self.preproc.n - 1)

        self.nipals = model.nipals(self.preproc.dataset, self.preproc.dummy_y)
        self.wx, vx = scipy.linalg.eig(cov_x)
        self.wy, vy = scipy.linalg.eig(cov_y)
        self.wx = np.real(self.wx)
        self.wy = np.real(self.wy)
        self.wx[::-1].sort()
        self.wy[::-1].sort()
        self.x_variance = 100 * self.wx / np.sum(self.wx)
        self.y_variance = 100 * self.wy / np.sum(self.wy)

    def tearDown(self):
        self.preproc = None
        self.nipals = None

    def test_nipals_eigenvectors_x_eigen(self):
        np.testing.assert_allclose(self.nipals.x_eigenvalues, self.wx,
                                   atol=absolute_tolerance)

    @unittest.skip("Different algorithm")
    def test_nipals_eigenvectors_y_eigen(self):
        np.testing.assert_allclose(self.nipals.y_eigenvalues, self.wy,
                                   atol=absolute_tolerance)

    def test_nipals_eigenvectors_x_variance(self):
        np.testing.assert_allclose(model.explained_variance(self.nipals, 'x'),
                                   self.x_variance, atol=absolute_tolerance)

    @unittest.skip("Different algorithm")
    def test_nipals_eigenvectors_y_variance(self):
        np.testing.assert_allclose(model.explained_variance(self.nipals, 'y'),
                                   self.y_variance, atol=absolute_tolerance)


class nipals_abstract(object):

    def setUp(self):
        j = 2
        self.preproc = model.Preprocessing()

        X = np.array([[1, 1.9], [1.9, 1], [3.8, 4.2], [4, 3.6]])
        Y = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
        self.preproc.dataset = X.copy()
        self.preproc.dummy_y = Y.copy()

        self.preproc.autoscale()
        # autoscale also matrices for sklearn
        X = self.preproc.dataset.copy()
        Y = self.preproc.dummy_y.copy()
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
        self.preproc = None
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
        """The relation in U = TD + H"""
        np.testing.assert_allclose(self.nipals.U,
                                   np.dot(self.nipals.T,
                                          np.diag(self.nipals.b)),
                                   atol=1)

    def test_x_component(self):
        np.testing.assert_allclose(self.preproc.dataset,
                                   np.dot(self.nipals.T, self.nipals.P.T),
                                   err_msg="X != TP'", atol=absolute_tolerance)

    def test_y_component(self):
        np.testing.assert_allclose(self.preproc.dummy_y,
                                   np.dot(self.nipals.U, self.nipals.Q.T),
                                   err_msg="Y != UQ'", atol=absolute_tolerance)

    def test_coef(self):
        np.testing.assert_allclose(self.preproc.dummy_y,
                                   self.nipals.Y_modeled_dummy,
                                   atol=absolute_tolerance)


class test_nipals_method(nipals_abstract, unittest.TestCase):

    def calculate_nipals(self, j):
        self.nipals.run(nr_lv=j)

    @unittest.skip("Different algorithm")
    def test_nipals_y_weights(self):
        np.testing.assert_allclose(np.absolute(self.nipals.W),
                                   np.absolute(self.sklearn_pls.y_weights_),
                                   atol=absolute_tolerance)


class test_plot_module(unittest.TestCase):

    x = [-5, -4, 0, 4, 5]
    y = [0, 3, -1, 3, 10]

    def setUp(self):
        preproc = model.Preprocessing()
        plot.update_global_preproc(preproc)
        model_nipals = model.nipals(preproc.dataset, preproc.dummy_y)
        plot.update_global_model(model_nipals)

    def test_symbol(self):
        categories = list(model.CATEGORIES)
        categories.append(None)
        categories.append('')
        for cat in categories:
            cat_symbols = plot.symbol(cat)
            self.assertIsInstance(cat_symbols, dict)
            for key in ('hex', 'marker'):
                self.assertTrue(key in cat_symbols)
                self.assertIsInstance(cat_symbols[key], str)

    def test_scatter_wrapper(self):
        for cat in model.CATEGORIES:
            self.assertIsNone(plot.scatter_wrapper(plt.gca(), self.x, self.y,
                                                   cat))
            plt.clf()  # clear current figure

        self.assertRaises(ValueError, plot.scatter_wrapper, plt.gca(),
                          self.x, self.y[:-1], 'U')
        self.assertRaises(ValueError, plot.scatter_wrapper, plt.gca(),
                          'string', 123, 'U')
        self.assertRaises(ValueError, plot.scatter_wrapper, plt.gca(),
                          456, 'string', 'U')

    def test_scores_plot(self):
        self.assertRaises(ValueError, plot.scores, plt.gca(),
                          pc_a=1, pc_b=1, x=True)

    def test_loadings_plot(self):
        self.assertRaises(ValueError, plot.loadings, plt.gca(),
                          pc_a=1, pc_b=1, y=True)

    def test_check_consistency(self):
        plot.scatter_wrapper(plt.gca(), self.x, self.y)
        plot.scatter_wrapper(plt.gca(), self.x, self.y, 'U')
        plot.line_wrapper(plt.gca(), self.x, self.y)
        plot.line_wrapper(plt.gca(), self.x, self.y, 'U')
        plot.scree(plt.gca(), x=True)
        plot.cumulative_explained_variance(plt.gca(), x=True)
        plot.inner_relations(plt.gca(), 0)
        plot.biplot(plt.gca(), 0, 1, x=True)
        plot.scores(plt.gca(), 0, 1, x=True)
        plot.loadings(plt.gca(), 0, 1, x=True)
        plot.calculated_y(plt.gca())
        plot.y_residuals_leverage(plt.gca())
        plot.regression_coefficients(plt.gca())
        plot.data(plt.gca())
        plot.sklearn_inner_relations(plt.gca(), 0)


class test_utility_module(unittest.TestCase):

    def test_check_python_version(self):
        self.assertTrue(utility.check_python_version())

    def test_CLI_args(self):
        self.assertIsInstance(utility.CLI.args(), argparse.Namespace)
        self.assertTrue(hasattr(utility.CLI.args(), 'input_file'))


if __name__ == '__main__':
    unittest.main(failfast=False)
