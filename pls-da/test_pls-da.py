#!/usr/bin/env python
# coding: utf-8

import argparse
import IO
import math
import model
import numpy as np
import os
import plot
import sklearn.cross_decomposition
import unittest
import utility
import scipy


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
        self.pls_da = model.PLS_DA()

    def tearDown(self):
        self.pls_da = None

    def test_PLS_DA_init(self):
        self.assertEqual(len(self.pls_da.keys), self.pls_da.m + 1)
        self.assertEqual(len(self.pls_da.categories), self.pls_da.n)
        self.assertEqual(len(self.pls_da.dummy_Y), self.pls_da.n)
        self.assertEqual(len(self.pls_da.dummy_Y[0]),
                         len(set(self.pls_da.categories)))
        self.assertIsNone(self.pls_da.mean)
        self.assertIsNone(self.pls_da.sigma)
        self.assertFalse(self.pls_da.centered)
        self.assertFalse(self.pls_da.normalized)
        self.assertFalse(self.pls_da.autoscaled)

    def test_PLS_DA_preprocess_mean(self):
        self.pls_da.dataset = self.matrix_3x3.copy()
        self.pls_da.dummy_Y = self.matrix_3x2.copy()

        self.pls_da.preprocess_mean()

        self.assertTrue(self.pls_da.centered)
        self.assertFalse(self.pls_da.normalized)
        self.assertFalse(self.pls_da.autoscaled)
        np.testing.assert_almost_equal(self.pls_da.dataset, self.null_3x3,
                                       decimal=7)
        np.testing.assert_almost_equal(self.pls_da.dummy_Y, self.null_3x2,
                                       decimal=7)

    def test_PLS_DA_preprocess_normalize(self):
        self.pls_da.dataset = self.matrix_A.copy()
        self.pls_da.dummy_Y = self.matrix_A.copy()

        self.pls_da.preprocess_normalize()

        self.assertFalse(self.pls_da.centered)
        self.assertTrue(self.pls_da.normalized)
        self.assertFalse(self.pls_da.autoscaled)
        np.testing.assert_almost_equal(self.pls_da.dataset, self.normalized_A,
                                       decimal=7)
        np.testing.assert_almost_equal(self.pls_da.dummy_Y, self.normalized_A,
                                       decimal=7)

    def test_PLS_DA_preprocess_autoscale(self):
        dataset_copy = self.pls_da.dataset.copy()
        dummy_Y_copy = self.pls_da.dummy_Y.copy()

        self.pls_da.preprocess_mean()
        self.pls_da.preprocess_normalize()

        dataset_autoscaled = self.pls_da.dataset
        dummy_Y_autoscaled = self.pls_da.dummy_Y

        self.pls_da.dataset = dataset_copy
        self.pls_da.dummy_Y = dummy_Y_copy
        self.pls_da.centered = False
        self.pls_da.normalized = False

        self.pls_da.preprocess_autoscale()

        self.assertTrue(self.pls_da.autoscaled)
        np.testing.assert_almost_equal(self.pls_da.dataset, dataset_autoscaled,
                                       decimal=7)
        np.testing.assert_almost_equal(self.pls_da.dummy_Y, dummy_Y_autoscaled,
                                       decimal=7)

    def test_PLS_DA_nipals_method(self):
        n, j, k = 15, 10, 5
        X, Y = np.random.random((n, j)), np.random.random((n, k))
        self.pls_da.dataset = X.copy()
        self.pls_da.dummy_Y = Y.copy()

        self.pls_da.preprocess_autoscale()
        self.pls_da.nipals_method(nr_lv=j)


        sklearn_pls = sklearn.cross_decomposition.PLSRegression(
            n_components=j, scale=True, max_iter=1e4, tol=1e-6, copy=True)
        sklearn_pls.fit(X, Y)

        IO.Log.debug('NIPALS scores', self.pls_da.T)
        IO.Log.debug('sklearn scores', sklearn_pls.x_scores_)
        IO.Log.debug('NIPALS loadings', self.pls_da.P)
        IO.Log.debug('sklearn loadings', sklearn_pls.x_loadings_)

#        np.testing.assert_almost_equal(np.absolute(self.pls_da.T),
#                                       np.absolute(sklearn_pls.x_scores_),
#                                       decimal=1)
#        np.testing.assert_almost_equal(np.absolute(self.pls_da.P),
#                                       np.absolute(sklearn_pls.x_loadings_),
#                                       decimal=1)
#        np.testing.assert_almost_equal(np.absolute(self.pls_da.W),
#                                       np.absolute(sklearn_pls.x_weights_),
#                                       decimal=1)
#
#        np.testing.assert_almost_equal(np.absolute(self.pls_da.U),
#                                       np.absolute(sklearn_pls.y_scores_),
#                                       decimal=0)
        np.testing.assert_almost_equal(np.absolute(self.pls_da.Q),
                                       np.absolute(sklearn_pls.y_loadings_),
                                       decimal=0)
#        np.testing.assert_almost_equal(np.absolute(self.pls_da.B),
#                                       np.absolute(sklearn_pls.coef_),
#                                       decimal=1)


    def test_PLS_DA_eigenvectors_explained_variance(self):
        self.pls_da.dataset = self.matrix_3x3.copy()
        self.pls_da.dummy_Y = self.matrix_3x2.copy()

        cov_x = np.dot(self.pls_da.dataset.T, self.pls_da.dataset) / (self.pls_da.n - 1)
        cov_y = np.dot(self.pls_da.dummy_Y.T, self.pls_da.dummy_Y) / (self.pls_da.n - 1)

        self.pls_da.nipals_method()
        wx, vx = scipy.linalg.eig(cov_x)
        wy, vy = scipy.linalg.eig(cov_y)
        wx = np.real(wx)
        wy = np.real(wy)
        wx[::-1].sort()
        wy[::-1].sort()

        np.testing.assert_almost_equal(self.pls_da.x_eigenvalues, wx)
        np.testing.assert_almost_equal(self.pls_da.y_eigenvalues, wy)

        x_variance = 100 * wx / np.sum(wx)
        y_variance = 100 * wy / np.sum(wy)
        np.testing.assert_almost_equal(self.pls_da.get_explained_variance(), x_variance)
        np.testing.assert_almost_equal(self.pls_da.get_explained_variance('y'), y_variance)



class test_plot_module(unittest.TestCase):

    x = [-5, -4, 0, 4, 5]
    y = [0, 3, -1, 3, 10]

    def test_get_category(self):
        for cat in model.PLS_DA.allowed_categories:
            self.assertIsInstance(plot.properties_of(cat), dict)
            d = plot.properties_of(cat)
            for key in ('edge_color', 'face_color', 'marker'):
                self.assertTrue(key in d)
                self.assertIsInstance(d[key], str)
        self.assertRaises(Exception, plot.get, '')
        self.assertRaises(Exception, plot.get, 'Q')
        self.assertRaises(Exception, plot.get, None)

    def test_scatter_plot(self):
        for cat in model.PLS_DA.allowed_categories:
            self.assertIsNone(plot.scatter_plot(self.x, self.y, cat))
        self.assertRaises(ValueError, plot.scatter_plot,
                          self.x, self.y[:-1], 'E')
        self.assertRaises(ValueError, plot.scatter_plot,
                          'string', 123, 'E')
        self.assertRaises(ValueError, plot.scatter_plot,
                          456, 'string', 'E')

    def test_scores_plot(self):
        with self.assertRaises(SystemExit) as cm:
            plot.scores_plot(model=None, pc_x=1, pc_y=1)
        self.assertEqual(cm.exception.code, 1)

    def test_loadings_plot(self):
        with self.assertRaises(SystemExit) as cm:
            plot.loadings_plot(model=None, pc_x=1, pc_y=1)
        self.assertEqual(cm.exception.code, 1)


class test_utility_module(unittest.TestCase):

    def test_check_python_version(self):
        self.assertTrue(utility.check_python_version())

    def test_CLI_args(self):
        self.assertIsInstance(utility.CLI.args(), argparse.Namespace)
        self.assertTrue(hasattr(utility.CLI.args(), 'input_file'))


if __name__ == '__main__':
    IO.Log.set_level('info')
    unittest.main(failfast=True)
