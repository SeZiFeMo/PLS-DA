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
import copy

absolute_tolerance = 0.1

csv_train_sample = str('Category;PalmiticAcid;PalmitoleicAcid;StearicAcid;'
                       'OleicAcid;LinoleicAcid;EicosanoicAcid;'
                       'LinolenicAcid\n'
                       'NA;10.75;0.75;2.26;78.230011;6.72;0.36;0.6\n'
                       'NA;10.88;0.73;2.24;77.084566;7.81;0.31;0.61\n'
                       'NA;9.104569;0.54;2.46;81.129997;5.49;0.31;0.63\n'
                       'NA;10.51;0.67;2.59;77.704569;6.72;0.5;0.8\n'
                       'NA;9.104569;0.49;2.68;79.239998;6.78;0.51;0.7\n'
                       'SA;13.64;2.04;2.25;69.290001;10.84;0.21;0.5\n'
                       'SA;14.1;1.99;2.16;71.300003;9.55;0.21;0.48\n'
                       'SA;13.84;1.78;2.08;71.050003;4.56;0.29;0.67\n'
                       'SA;14.12;1.85;2.17;68.414568;12.03;0.34;0.72\n'
                       'SA;14.1;2.32;2.8;67.150002;12.33;0.32;0.6\n'
                       'U;10.85;0.7;1.8;79.550003;6.05;0.2;0.5\n'
                       'U;10.85;0.7;1.85;79.550003;6;0.25;0.55\n'
                       'U;10.9;0.6;1.9;79.5;6;0.28;0.47\n'
                       'U;10.8;0.65;1.89;79.545698;6.02;0.35;0.2\n'
                       'U;10.9;0.6;1.95;79.550003;6;0.28;0.42\n'
                       'WL;11.9;1.5;2.9;73.400002;10.2;0;0.1\n'
                       'WL;11.1;1.3;2.1;75.5;10;0;0\n'
                       'WL;10.7;1.2;2.1;76;9.845699;0;0.1\n'
                       'WL;10.1;0.9;3.5;74.800003;10.5;0.1;0.1\n'
                       'WL;10.3;1;2.3;77.400002;9;0;0\n')


csv_test_sample = str('Category;PalmiticAcid;PalmitoleicAcid;StearicAcid;'
                      'OleicAcid;LinoleicAcid;EicosanoicAcid;LinolenicAcid'
                      '\n'
                      'NA;9.66;0.57;2.4;79.514567;6.19;0.5;0.78\n'
                      'NA;11;0.61;2.35;77.274569;7.34;0.39;0.64\n'
                      'NA;10.82;0.6;2.39;77.444567;7.09;0.46;0.83\n'
                      'NA;10.36;0.59;2.35;78.68;6.61;0.3;0.62\n'
                      'SA;14.54;1.83;1.96;70.57;10.14;0.27;0.46\n'
                      'SA;13.47;1.94;1.97;72.764567;8.95;0.25;0.46\n'
                      'SA;15.09;2.09;2.57;66.470001;12.4;0.42;0.62\n'
                      'SA;12.86;1.92;2.03;71.32;10.53;0.38;0.65\n'
                      'U;11;0.55;1.98;79.050003;6;0.35;0.5\n'
                      'U;10.85;0.6;1.88;79.550003;6.02;0.3;0.5\n'
                      'U;10.75;0.68;1.95;79.545698;6.02;0.2;0.4\n'
                      'U;10.95;0.6;1.98;79.444567;6;0.38;0.34\n'
                      'WL;10.2;1;2.2;75.300003;10.3;0;0\n'
                      'WL;10.6;1.4;2.4;76.800003;8.3;0.1;0.4\n'
                      'WL;10.6;1.4;2.7;76.145697;8.8;0.1;0.2\n'
                      'WL;11.2;1.3;2.5;75.300003;9.7;0;0\n')


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

    def test_dump(self):
        pass

    def test_save_matrix(self):
        pass

    def test_load(self):
        pass


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


class test_plot_module(unittest.TestCase):

    x = [-5, -4, 0, 4, 5]
    y = [0, 3, -1, 3, 10]

    def setUp(self):
        self.train_set = model.TrainingSet('.train_set_synthesis.csv')
        plot.update_global_train_set(self.train_set)
        model_nipals = model.nipals(self.train_set.x, self.train_set.y)
        plot.update_global_model(model_nipals)
        test_set = model.TestSet('.test_set_synthesis.csv', self.train_set)
        plot.update_global_test_set(test_set)

        y_pred = model_nipals.predict(test_set.x)
        plot.update_global_statistics(model.Statistics(test_set.y, y_pred))

    def test_symbol(self):
        categories = list(self.train_set.categories)
        categories.append(None)
        categories.append('')
        for cat in categories:
            cat_symbols = plot.symbol(cat)
            self.assertIsInstance(cat_symbols, dict)
            for key in ('hex', 'marker'):
                self.assertTrue(key in cat_symbols)
                self.assertIsInstance(cat_symbols[key], str)

    def test_update_globals(self):
        pass

    def test_scatter_wrapper(self):
        for cat in self.train_set.categories:
            self.assertIsNone(plot.scatter_wrapper(plt.gca(), self.x, self.y,
                                                   cat))
            plt.clf()  # clear current figure

        self.assertRaises(ValueError, plot.scatter_wrapper, plt.gca(),
                          self.x, self.y[:-1], 'U')
        self.assertRaises(ValueError, plot.scatter_wrapper, plt.gca(),
                          'string', 123, 'U')
        self.assertRaises(ValueError, plot.scatter_wrapper, plt.gca(),
                          456, 'string', 'U')

    def test_line_wrapper(self):
        pass

    def test_scree(self):
        pass

    def test_cumulative_explained_variance(self):
        pass

    def test_inner_relation(self):
        pass

    def test_biplot(self):
        pass

    def test_scores(self):
        self.assertRaises(ValueError, plot.scores, plt.gca(),
                          lv_a=1, lv_b=1, x=True)

    def test_loadings(self):
        self.assertRaises(ValueError, plot.loadings, plt.gca(),
                          lv_a=1, lv_b=1, y=True)

    def test_calculated_y(self):
        pass

    def test_y_predicted_y_real(self):
        pass

    def test_y_predicted(self):
        pass

    def test_y_modeled_class(self):
        pass

    def test_weights(self):
        pass

    def test_weights_line(self):
        pass

    def rmsec_lv(self):
        pass

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
        plot.calculated_y(plt.gca(), 0)
        plot.y_predicted_y_real(plt.gca())
        plot.y_predicted(plt.gca())
        plot.t_square_q(plt.gca())
        plot.y_residuals_leverage(plt.gca())
        plot.leverage(plt.gca())
        plot.q_over_leverage(plt.gca())
        plot.regression_coefficients(plt.gca())
        plot.weights(plt.gca(), 0, 1)
        plot.weights_line(plt.gca(), 0)
        plot.data(plt.gca())


class test_utility_module(unittest.TestCase):

    def test_CLI_args(self):
        self.assertIsInstance(utility.CLI.args(), argparse.Namespace)

    def test_list_to_string(self):
        pass

    def test_get_unique_list(self):
        pass

    def test_cached_property(self):
        pass


if __name__ == '__main__':
    try:
        with open('.train_set_synthesis.csv', 'r') as f:
            train_file = f.read()
        if train_file != csv_train_sample:
            raise FileNotFoundError('Train file is different')
    except:
        with open('.train_set_synthesis.csv', 'w') as f:
            f.write(csv_train_sample)

    try:
        with open('.test_set_synthesis.csv', 'r') as f:
            test_file = f.read()
        if test_file != csv_test_sample:
            raise FileNotFoundError('Test file is different')
    except:
        with open('.test_set_synthesis.csv', 'w') as f:
            f.write(csv_test_sample)

    unittest.main(failfast=False)
