#!/usr/bin/env python3
# coding: utf-8

""" PLS-DA is a project about the Partial least squares Discriminant Analysis
    on a given dataset.'
    PLS-DA is a project developed for the Processing of Scientific Data exam
    at University of Modena and Reggio Emilia.
    Copyright (C) 2017  Serena Ziviani, Federico Motta
    This file is part of PLS-DA.
    PLS-DA is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    any later version.
    PLS-DA is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with PLS-DA.  If not, see <http://www.gnu.org/licenses/>.
"""

__authors__ = "Serena Ziviani, Federico Motta"
__copyright__ = "PLS-DA  Copyright (C)  2017"
__license__ = "GPL3"


import argparse
from functools import update_wrapper


if __name__ == '__main__':
    raise SystemExit('Please do not run that script, load it!')


def list_to_string(seq, prec=6, separator=', '):
    ret = ''
    for num in seq:
        ret += '{0:.{1}f}'.format(num, prec) + separator
    return ret.rstrip(separator)


def get_unique_list(seq):
    """Return a list with duplicates removed, preserving order."""
    seen = set()
    return [x for x in seq if x not in seen and not seen.add(x)]


class CLI(object):

    _args = None
    _description = str('This program has been developed for the Processing of '
                       'Scientific Data exam\n(EDS), at Physics, Informatics '
                       'and Mathematics departement (FIM) of\nUniversity of '
                       'Modena and Reggio Emilia (UNIMORE) in Italy.\n\n'
                       'Its main purpose is to conduct a "Partial least '
                       'squares Discriminant\nAnalysis" (PLS-DA) on a given '
                       'dataset.')

    _epilog = str('REQUIREMENTS: Python 3 (>= 3.5.3)\n' + ' ' * 14 +
                  'Matplotlib http://matplotlib.org\n' + ' ' * 14 +
                  'NumPy      http://pypi.python.org/pypi/numpy\n' + ' ' * 14 +
                  'PyQt5      http://pypi.python.org/pypi/PyQt5\n' + ' ' * 14 +
                  'PyYAML     http://pyyaml.org/wiki/PyYAML\n' + ' ' * 14 +
                  'SciPy      http://www.scipy.org\n' +
                  '\n\n' +
                  'Copyright (C) 2017 Serena Ziviani, Federico Motta\n\n' +
                  'This program is free software: you can redistribute it ' +
                  'and/or modify it under\nthe terms of the GNU General ' +
                  'Public License as published by the Free Software\n' +
                  'Foundation, either version 3 of the License, or any later ' +
                  'version.\n\n' +
                  'This program is distributed in the hope that it will be ' +
                  'useful, but WITHOUT\nANY WARRANTY; without even the ' +
                  'implied warranty of MERCHANTABILITY or FITNESS\nFOR A ' +
                  'PARTICULAR PURPOSE. See the GNU General Public License for ' +
                  'more details.\n\n' +
                  'You should have received a copy of the GNU General Public ' +
                  'License along with\nthis program. If not, see ' +
                  'http://www.gnu.org/licenses/.')

    def args(self=None):
        if CLI._args is None:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.RawTextHelpFormatter,
                description=CLI._description,
                epilog=CLI._epilog)
            group = parser.add_mutually_exclusive_group()
            group.add_argument('-q', '--quiet',
                               action='count',
                               default=0,
                               help='Set logging to WARNING, ERROR or '
                                    'CRITICAL (-q|-qq|-qqq)')
            group.add_argument('-v', '--verbose',
                               action='store_true',
                               help='Set logging to DEBUG '
                                    '(default level is INFO)')
            CLI._args = parser.parse_args()
        return CLI._args



_CLASS_CACHE_ATTR_NAME = '_class_cached_properties'
_OBJ_CACHE_ATTR_NAME = '_cached_properties'


def cached_property(fn):
    def _cached_property(self):
        return _get_property_value(fn, self, _OBJ_CACHE_ATTR_NAME)
    return property(update_wrapper(_cached_property, fn))


def set_property_cache(obj, name, value):
    cache = _get_cache(obj)
    cache[name] = value
    setattr(obj, _OBJ_CACHE_ATTR_NAME, cache)


def clear_property_cache(obj, name):
    cache = _get_cache(obj)
    if name in cache:
        del cache[name]


def is_property_cached(obj, name):
    cache = _get_cache(obj)
    return name in cache


def _get_cache(obj, cache_attr_name=_OBJ_CACHE_ATTR_NAME):
    return getattr(obj, cache_attr_name, {})


def _update_cache(obj, cache_attr_name, cache_key, result):
    cache = _get_cache(obj, cache_attr_name)
    cache[cache_key] = result
    setattr(obj, cache_attr_name, cache)


def _get_property_value(fn, obj, cache_attr_name, cache_false_results=True):
    cache = _get_cache(obj, cache_attr_name)
    cache_key = fn.__name__

    if cache_key in cache:
        return cache[cache_key]

    result = fn(obj)
    if result is not None or cache_false_results:
        _update_cache(obj, cache_attr_name, cache_key, result)

    return result
