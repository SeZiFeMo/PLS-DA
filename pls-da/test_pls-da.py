#!/usr/bin/env python
# coding: utf-8

import argparse
import IO
import model
import numpy as np
import plot
import unittest
import utility


class test_IO_module(unittest.TestCase):

    np_array = np.array([-5, 6, -7, 8, -9])
    np_matrix = np.array([[1, 2], [3, 4], [5, 6]])
    char_list = list('0123456789ABCDEF')

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
        pass


class test_model_module(unittest.TestCase):
    pass


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
        pass

    def test_loadings_plot(self):
        pass


class test_utility_module(unittest.TestCase):

    def test_check_python_version(self):
        self.assertTrue(utility.check_python_version())

    def test_CLI_args(self):
        self.assertIsInstance(utility.CLI.args(), argparse.Namespace)
        self.assertTrue(hasattr(utility.CLI.args(), 'input_file'))

if __name__ == '__main__':
    unittest.main(failfast=True)
