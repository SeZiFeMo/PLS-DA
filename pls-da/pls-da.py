#!/usr/bin/env python3
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np

if __name__ != '__main__':
    print('That script should be run, not loaded!')
    exit(1)

log_level = {'DEBUG': 0,
             'INFO': 1,
             'WARNING': 2,
             'ERROR': 3,
             'CRITICAL': 4}['WARNING']

dataset = None
namespace = None

def log(record, level=0):
    """Print log message if above threshold."""
    global log_level
    if (level >= log_level):
        print('{}\n'.format(str(record).strip('\n')))

def mat2str(data, show=False, h_bar='-', v_bar='|', join='+'):
    """Print or return an ascii table."""
    try:
        if isinstance(data, (np.ndarray, np.generic)) and data.ndim == 2:
            ret = join + h_bar + h_bar * 11 * len(data[0]) + join + '\n'
            for row in data:
                ret += v_bar + ' '
                for col in row:
                    ret += '{: < 10.3e} '.format(col)
                ret += v_bar + '\n'
            ret += join + h_bar + h_bar * 11 * len(data[0]) + join + '\n'
        elif (isinstance(data, (np.ndarray, np.generic)) and data.ndim == 1) \
                or isinstance(data, (list, tuple)):
            ret = join + h_bar + h_bar * 11 * len(data) + join + '\n'
            ret += v_bar + ' '
            for cell in data:
                ret += '{: < 10.3e} '.format(cell)
            ret += v_bar + '\n'
            ret += join + h_bar + h_bar * 11 * len(data) + join + '\n'
        else:
            raise Exception('Not supported data type ({}) '
                            'in mat2str()'.format(type(data)))
    except Exception as e:
        print(e)
        ret = str(data)
    finally:
        if show:
            print(ret)
        else:
            return ret


def seq2str(seq):
    """Return string representation of a sequence."""
    return str(seq).translate({ord(d): '' for d in '{}\'[]"()'})

class PLS_DA(object):
    def __init__(self, csv_file):
        self._dataset_original = None
        self.parse_csv(csv_file)
        self.dataset = np.copy(self._dataset_original)
        self.n_rows, self.n_cols = self.dataset.shape

    def parse_csv(self, filename):
        """self.keys             = first row of labels
           self.categories       = first column of labels about wine type
           self._dataset_original = the rest of the matrix
        """

        self._dataset_original = list()
        try:
            with open(filename, 'r', encoding='iso8859') as f:
                self.keys = f.readline().strip('\n').split(';')
                for line in f.readlines():
                    line = line.strip('\n').split(';')
                    self._dataset_original.append(list(line))
        except IOError:
            print('ERROR: file \'{}\' not existent, '
                  'not readable or corrupted'.format(filename))
            exit(1)

        for d in self._dataset_original:
            if d[0].startswith('B'):
                d[0] = 'B'
            elif d[0].startswith('E'):
                d[0] = 'E'
            elif d[0].startswith('G'):
                d[0] = 'G'
            else:
                print('WARNING: unexpected wine '
                      'category ({})'.format(d['Category']))

        # Delete category column from self._dataset_original
        # and save it for future uses
        self.categories = [x[0] for x in self._dataset_original]
        self._dataset_original = [x[1:] for x in self._dataset_original]

        # Replace commas with dots
        self._dataset_original = [[float(elem.replace(',', '.'))
                                   for elem in row]
                                  for row in self._dataset_original]
        self._dataset_original = np.array(self._dataset_original)
        log('[PLS_DA::parse_csv] '
            '_dataset_original: \n{}'.format(mat2str(self._dataset_original)),
            0)

    def preprocess_mean(self, use_original=False):
        """Substitute self.dataset with its centered version."""
        dataset = self._dataset_original if use_original else self.dataset
        self.mean = dataset.mean(axis=self.axis)
        self.dataset = dataset - self.mean
        log('[PLS_DA::preprocess_mean] '
            'Centered matrix: \n{}'.format(mat2str(self.dataset)), 0)
        self.centered = True

    def preprocess_normalize(self, use_original=False):
        """Substitute self.dataset with its normalized version."""
        dataset = self._dataset_original if use_original else self.dataset
        self.sigma = dataset.std(axis=self.axis)
        self.dataset = self.dataset / self.sigma
        log('[PLS_DA::preprocess_normalize] '
            'Normalized matrix: \n{}'.format(mat2str(self.dataset)), 0)
        self.normalized = True


    def preprocess_autoscaling(self, use_original=False):
        """Substitute self.dataset with its autoscaled version."""
        dataset = self._dataset_original if use_original else self.dataset
        self.preprocess_mean(use_original)  # it initializes self.dataset
        self.preprocess_normalize(use_original)
        log('[PLS_DA::preprocess_autoscaling] '
            'Autoscaled matrix: \n{}'.format(mat2str(self.dataset)), 0)
        self.autoscaled = True

    def get_dummy_variables(self):

        categories = set(self.categories)

        li = []
        for cat in categories:
            li.append([1 if c == cat else 0 for c in self.categories])

        self.dummy_Y = np.array(li)

        log('[PLS_DA::get_dummy_variables] '
            'dummy Y variables: \n{}'.format(mat2str(self.dummy_Y)), 0)

# main
pls = PLS_DA("wine.csv")
pls.get_dummy_variables()
