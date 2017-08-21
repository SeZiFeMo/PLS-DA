#!/usr/bin/env python3
# coding: utf-8

import argparse
import sys
import IO
from functools import update_wrapper

if __name__ == '__main__':
    raise SystemExit('Please do not run that script, load it!')


def check_python_version():
    if sys.version_info < (3,):
        major, minor, *__ = sys.version_info
        IO.Log.warning('You are using the Python interpreter {}.{}.\n'
                       'Please use at least Python version 3!'.format(major,
                                                                      minor))
        exit(1)
    else:
        return True


def get_unique_list(seq):
    """Return a list with duplicates removed, preserving order."""
    seen = set()
    return [x for x in seq if x not in seen and not seen.add(x)]


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
            group = parser.add_mutually_exclusive_group()
            parser.add_argument('-i', '--input-file',
                                default='datasets/olive_training.csv',
                                dest='input_file',
                                help='File with comma saved value dataset',
                                metavar='file',
                                type=str)
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
