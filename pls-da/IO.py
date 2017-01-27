#!/usr/bin/env python
# coding: utf-8

import logging
import numpy as np
import yaml


def mat2str(data, h_bar='-', v_bar='|', join='+'):
    """Return an ascii table."""
    try:
        if isinstance(data, (np.ndarray, np.generic)) and data.ndim == 2:
            ret = join + h_bar + h_bar * 11 * len(data[0]) + join + '\n'
            for row in data:
                ret += v_bar + ' '
                for col in row:
                    ret += '{: < 10.3e} '.format(col)
                ret += v_bar + '\n'
            ret += join + h_bar + h_bar * 11 * len(data[0]) + join + '\n'
        elif (isinstance(data, (np.ndarray, np.generic))
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
        Log.warning(e)
        ret = str(data)
    finally:
        return ret


class Log(object):

    __default = 'debug'
    __initialized = False
    __name = 'PLS_DA'

    def __log(msg='', data=None, level=None):
        """Print log message if above threshold."""
        if level is None:
            level = Log.__default

        if not Log.__initialized:
            logging_level = getattr(logging, Log.__default.upper())
            logging.basicConfig(format='[%(levelname)s]\t%(message)s',
                                level=logging_level)
            for l in logging.Logger.manager.loggerDict.keys():
                logging.getLogger(l).setLevel(logging.INFO)

            # current script / package logging
            logging.getLogger(Log.__name).setLevel(logging_level)
            Log.__initialized = True

        logger = getattr(logging.getLogger(Log.__name), level)
        if data is None:
            logger(msg.replace('\n', '\n    '))
        else:
            if (isinstance(data, (np.ndarray, np.generic))
                and data.ndim in (1, 2)) or \
               isinstance(data, (list, tuple)):
                data = mat2str(data)
            else:
                data = yaml.dump(data, default_flow_style=False)
            logger(msg.rstrip('\n') + '\n    ' + data.replace('\n', '\n    '))

    def critical(msg='', data=None):
        return Log.__log(msg=msg, data=data, level='critical')

    def debug(msg='', data=None):
        return Log.__log(msg=msg, data=data, level='debug')

    def error(msg='', data=None):
        return Log.__log(msg=msg, data=data, level='error')

    def info(msg='', data=None):
        return Log.__log(msg=msg, data=data, level='info')

    def warning(msg='', data=None):
        return Log.__log(msg=msg, data=data, level='warning')


class CSV(object):

    def parse(filename, encoding='iso8859', separator=';'):
        """Return the header (list) and the body of a table (list of lists)"""
        header, body = list(), list()
        try:
            with open(filename, 'r', encoding=encoding) as f:
                header = f.readline().strip('\n').split(separator)

                for line in f.readlines():
                    row = line.strip('\n').split(separator)
                    body.append(list(row))
        except IOError:
            Log.error('File {} not existent, not readable '
                      'or corrupted.'.format(filename))
            exit(1)
        else:
            if len(header) < 1 or len(body) < 1:
                Log.error('Too few columns or rows in {}'.format(filename))
                exit(1)
            for i, row in enumerate(body):
                if len(row) != len(header):
                    Log.error('Bad number of columns in {} body row'.format(i))
                    exit(1)

            for i, row in enumerate(body):
                for j, cell in enumerate(row):
                    try:
                        val = float(cell.replace(',', '.'))
                    except ValueError:
                        continue
                    else:
                        body[i][j] = val
        return header, body


if __name__ == '__main__':
    Log.warning('Please do not run that script, load it!')
    exit(1)
