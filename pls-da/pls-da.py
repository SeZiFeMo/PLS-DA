#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
import sys

if sys.version_info < (3,):
    major, minor = sys.version_info[0], sys.version_info[1]
    print('You are using the Python interpreter {}.{}.\n'
          'Please use at least Python version 3!'.format(major, minor))
    exit(1)

for lib in ('numpy', 'yaml'):
    try:
        exec('import ' + lib)
    except ImportError:
        print('Could not import {} library, please install it!'.format(lib))
        exit(1)

if __name__ != '__main__':
    print('Please do not load that script, run it!')
    exit(1)


args = None
dataset = None


def parse_args():
    """Return command line parameters."""
    global args
    if args is None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter,
            description='Script to ...\n',
            epilog='REQUIREMENTS: Python 3 (>= 3.4)\n'
                   '              NumPy   https://pypi.python.org/pypi/numpy\n'
                   '              PyYAML  http://pyyaml.org/wiki/PyYAML\n')
        parser.add_argument('-i', '--input-file',
                            default='wine.csv',
                            dest='input_file',
                            help='File with comma saved value input dataset',
                            metavar='file',
                            type=str)
        args = parser.parse_args()
    return args
parse_args()


def setup_logging(level):
    """Initialize logging for all the libraries and the script."""
    level = getattr(logging, level.upper())
    logging.basicConfig(format='[%(levelname)s]\t%(message)s', level=level)
    for l in logging.Logger.manager.loggerDict.keys():
    logging.getLogger(__name__).setLevel(level)  # current script logging
setup_logging('debug')  # called here to ensure log function will work


def mat2str(data, show=False, h_bar='-', v_bar='|', join='+'):
    """Print or return an ascii table."""
    try:
        if isinstance(data, (numpy.ndarray, numpy.generic)) and data.ndim == 2:
            ret = join + h_bar + h_bar * 11 * len(data[0]) + join + '\n'
            for row in data:
                ret += v_bar + ' '
                for col in row:
                    ret += '{: < 10.3e} '.format(col)
                ret += v_bar + '\n'
            ret += join + h_bar + h_bar * 11 * len(data[0]) + join + '\n'
        elif (isinstance(data, (numpy.ndarray, numpy.generic))
              and data.ndim == 1) or isinstance(data, (list, tuple)):
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


def log(msg='', data=None, level='debug'):
    """Print log message if above threshold."""
    logger = getattr(logging.getLogger(__name__), level)
    if data is None:
        logger(msg)
    else:
        if (isinstance(data, (numpy.ndarray, numpy.generic))
            and data.ndim in (1, 2)) or \
           isinstance(data, (list, tuple)):
            data = mat2str(data)
        else:
            data = yaml.dump(data, default_flow_style=False)
        logger(msg.rstrip('\n') + '\n    ' + data.replace('\n', '\n    '))


class PLS_DA(object):

    def __init__(self, csv_file=None):
        """Constructor method"""
        if csv_file is None:
            csv_file = parse_args().input_file

        self._dataset_original = None
        self.parse_csv(csv_file)
        self.dataset = numpy.copy(self._dataset_original)
        self.n_rows, self.n_cols = self.dataset.shape

    def parse_csv(self, filename):
        """self.keys              = first row of labels
           self.categories        = first column of labels about wine type
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
        self._dataset_original = numpy.array(self._dataset_original)
        log('[PLS_DA::parse_csv] self._dataset_original',
            self._dataset_original)

    def preprocess_mean(self, use_original=False):
        """Substitute self.dataset with its centered version."""
        dataset = self._dataset_original if use_original else self.dataset
        self.mean = dataset.mean(axis=self.axis)
        self.dataset = dataset - self.mean
        log('[PLS_DA::preprocess_mean] Centered matrix', self.dataset)
        self.centered = True

    def preprocess_normalize(self, use_original=False):
        """Substitute self.dataset with its normalized version."""
        dataset = self._dataset_original if use_original else self.dataset
        self.sigma = dataset.std(axis=self.axis)
        self.dataset = self.dataset / self.sigma
        log('[PLS_DA::preprocess_normalize] Normalized matrix', self.dataset)
        self.normalized = True

    def preprocess_autoscaling(self, use_original=False):
        """Substitute self.dataset with its autoscaled version."""
        dataset = self._dataset_original if use_original else self.dataset
        self.preprocess_mean(use_original)  # it initializes self.dataset
        self.preprocess_normalize(use_original)
        log('[PLS_DA::preprocess_autoscaling] Autoscaled matrix', self.dataset)
        self.autoscaled = True

    def get_dummy_variables(self):

        categories = set(self.categories)

        li = []
        for cat in categories:
            li.append([1 if c == cat else 0 for c in self.categories])

        self.dummy_Y = numpy.array(li)

        log('[PLS_DA::get_dummy_variables] dummy Y variables', self.dummy_Y)

pls = PLS_DA()
pls.get_dummy_variables()
