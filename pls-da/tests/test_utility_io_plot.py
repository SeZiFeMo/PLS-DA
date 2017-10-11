#!/usr/bin/env python3
# coding: utf-8

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import unittest

from context import IO
from context import model
from context import plot
from context import utility
from context import create_environment

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

    def test_dump(self):
        pass

    def test_save_matrix(self):
        pass

    def test_load(self):
        pass


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

    create_environment()

    unittest.main(failfast=False)
