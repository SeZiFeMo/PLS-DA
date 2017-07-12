#!/usr/bin/env python3
# coding: utf-8

import logging
import numpy as np
import utility
import yaml


def dump(plsda_model, workspace):
    """Creates in workspace directory a csv file for each plsda_model matrix.
    """
    Log.error('Not yet implemented IO.dump(plsda_model, workspace)')


def load(workspace):
    """Return a plsda_model from csv file in workspace.
    """
    Log.error('Not yet implemented IO.load(workspace)')
    return None


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
            ret += join + h_bar + h_bar * 11 * len(data[0]) + join
        elif (isinstance(data, (np.ndarray, np.generic))
              and data.ndim == 1) or isinstance(data, (list, tuple)):
            ret = join + h_bar + h_bar * 11 * len(data) + join + '\n'
            ret += v_bar + ' '
            for cell in data:
                ret += '{: < 10.3e} '.format(cell)
            ret += v_bar + '\n'
            ret += join + h_bar + h_bar * 11 * len(data) + join
        else:
            raise Exception('Not supported data type ({}) '
                            'in mat2str()'.format(type(data)))
    except Exception as e:
        Log.warning(e)
        ret = str(data)
    finally:
        return ret


class Log(object):

    __default = 'critical' if utility.CLI.args().quiet >= 3 else \
                'error' if utility.CLI.args().quiet == 2 else \
                'warning' if utility.CLI.args().quiet == 1 else \
                'debug' if utility.CLI.args().verbose else \
                'info'
    __initialized = False
    __name = 'PLS_DA'

    @staticmethod
    def __log(msg='', data=None, level=None):
        """Print log message if above threshold."""
        if level is None:
            level = Log.__default

        if not Log.__initialized:
            logging_level = getattr(logging, Log.__default.upper())
            logging.basicConfig(format='[%(levelname)-8s] %(message)s',
                                level=logging_level)
            for l in logging.Logger.manager.loggerDict.keys():
                logging.getLogger(l).setLevel(logging.INFO)

            # current script / package logging
            logging.getLogger(Log.__name).setLevel(logging_level)
            Log.__initialized = True

        logger = getattr(logging.getLogger(Log.__name), level)
        my_new_line = '\n[{:<8}]     '.format(level.upper())
        if data is None:
            logger(msg.replace('\n', my_new_line))
        else:
            if (isinstance(data, (np.ndarray, np.generic))
                    and data.ndim in (1, 2)) or \
                    isinstance(data, (list, tuple)):
                data = mat2str(data)
            else:
                data = yaml.dump(data, default_flow_style=False)
                data = data.replace('\n...', '').rstrip('\n')
            logger(msg.rstrip('\n') + my_new_line
                   data.replace('\n', my_new_line))

    @staticmethod
    def critical(msg='', data=None):
        return Log.__log(msg=msg, data=data, level='critical')

    @staticmethod
    def debug(msg='', data=None):
        return Log.__log(msg=msg, data=data, level='debug')

    @staticmethod
    def error(msg='', data=None):
        return Log.__log(msg=msg, data=data, level='error')

    @staticmethod
    def info(msg='', data=None):
        return Log.__log(msg=msg, data=data, level='info')

    @staticmethod
    def set_level(level):
        if not isinstance(level, str):
            Log.error('Log.set_level() takes a string as argumenent, not a '
                      '{}'.format(type(level)))
            return
        if level not in ('critical', 'debug', 'error', 'info', 'warning'):
            Log.error('Bad level ({}) in Log.set_level()'.format(level))
            return
        Log.__default = level
        Log.__initialized = False

    @staticmethod
    def warning(msg='', data=None):
        return Log.__log(msg=msg, data=data, level='warning')


class CSV(object):

    @staticmethod
    def parse(filename, encoding='iso8859', separator=';'):
        """Return the header (list) and the body of a table (list of lists).

           Raises Exception on input error or on malformed content.
        """
        header, body = list(), list()
        try:
            with open(filename, 'r', encoding=encoding) as f:
                header = f.readline().strip('\n').split(separator)

                for line in f.readlines():
                    row = line.strip('\n').split(separator)
                    body.append(list(row))
        except IOError:
            raise Exception('File {} not existent, not readable '
                            'or corrupted.'.format(filename))
        else:
            if len(header) < 1 or len(body) < 1:
                raise Exception('Too few columns or rows in '
                                '{}'.format(filename))
            for i, row in enumerate(body):
                if len(row) != len(header):
                    raise Exception('Bad number of columns in '
                                    '{} body row'.format(i))

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
    raise SystemExit('Please do not run that script, load it!')
