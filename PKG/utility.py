#!/usr/bin/env python
# coding: utf-8

import argparse
import PKG.io
import sys

if __name__ == '__main__':
    PKG.io.Log.warning('Please do not run that script, load it!')
    exit(1)


def check_python_version():
    if sys.version_info < (3,):
        maj, min = sys.version_info[0], sys.version_info[1]
        PKG.io.Log.warning('You are using the Python interpreter {}.{}.\n'
                           'Please use at least Python version 3!'.format(maj,
                                                                          min))
        exit(1)
    else:
        return True


class CLI(object):

    _args = None
    _description = 'Script to ...\n'
    _epilog = 'REQUIREMENTS: Python 3 (>= 3.4)\n'                             \
        '              NumPy   https://pypi.python.org/pypi/numpy\n'          \
        '              PyYAML  http://pyyaml.org/wiki/PyYAML\n'

    def args(self=None):
        if CLI._args is None:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.RawTextHelpFormatter,
                description=CLI._description,
                epilog=CLI._epilog)
            parser.add_argument('-i', '--input-file',
                                default='wine.csv',
                                dest='input_file',
                                help='File with comma saved value dataset',
                                metavar='file',
                                type=str)
            CLI._args = parser.parse_args()
        return CLI._args
